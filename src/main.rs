mod config;
mod napcat;
mod openai;
mod util;

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::Context;
use config::{Config, get_cainbot_exclusive_groups_file_path, load_or_create_config};
use napcat::NapcatClient;
use openai::{ChatMessage, OpenAiCompatClient};
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Mutex;

const SKIP_TEXT: &str = "【SKIP】";
const MIN_CAINBOT_EXCLUSIVE_GROUPS_HEARTBEAT_SECONDS: u64 = 5;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct MemoryFile {
    #[serde(default)]
    #[serde(rename = "全局")]
    global: GlobalMemory,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct GlobalMemory {
    #[serde(default)]
    #[serde(rename = "设定")]
    settings: Vec<String>,
    #[serde(default)]
    #[serde(rename = "群记忆")]
    group_memory: HashMap<String, String>,
    #[serde(default)]
    #[serde(rename = "知识缓存")]
    knowledge_cache: HashMap<String, String>,
    #[serde(default)]
    #[serde(rename = "知识搜索")]
    knowledge_search: HashMap<String, String>,
    #[serde(default)]
    #[serde(rename = "人物关系")]
    relationships: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize)]
struct HistoryEntry {
    role: String,
    sender: String,
    user_id: String,
    text: String,
    time: String,
}

struct AppState {
    root_dir: PathBuf,
    config_path: PathBuf,
    memory_path: PathBuf,
    knowledge_dir: PathBuf,
    config: Mutex<Config>,
    memory: Mutex<MemoryFile>,
    static_knowledge: Mutex<HashMap<String, String>>,
    message_history: Mutex<HashMap<String, VecDeque<HistoryEntry>>>,
    group_locks: Mutex<HashMap<String, Arc<GroupGate>>>,
    openai: Mutex<OpenAiCompatClient>,
    napcat: NapcatClient,
    cainbot_sync_state: Mutex<CainbotSyncState>,
}

#[derive(Debug, Default)]
struct CainbotSyncState {
    last_signature: Option<String>,
    last_write_at_ms: u64,
}

struct GroupGate {
    lock: Mutex<()>,
    pending: AtomicUsize,
}

impl GroupGate {
    fn new() -> Self {
        Self {
            lock: Mutex::new(()),
            pending: AtomicUsize::new(0),
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let root_dir = std::env::current_dir().context("failed to get current dir")?;
    let data_dir = root_dir.join("data");
    let config_path = data_dir.join("config.json");
    let memory_path = data_dir.join("memory.json");
    let knowledge_dir = data_dir.join("Knowledge");

    let config = load_or_create_config(&config_path)?;
    let napcat = NapcatClient::new(config.napcat.clone())?;
    let openai = OpenAiCompatClient::new(config.ai.clone())?;
    let memory = load_or_create_memory(&memory_path)?;
    let static_knowledge = load_static_knowledge(&knowledge_dir)?;

    let state = Arc::new(AppState {
        root_dir,
        config_path,
        memory_path,
        knowledge_dir,
        config: Mutex::new(config),
        memory: Mutex::new(memory),
        static_knowledge: Mutex::new(static_knowledge),
        message_history: Mutex::new(HashMap::new()),
        group_locks: Mutex::new(HashMap::new()),
        openai: Mutex::new(openai),
        napcat,
        cainbot_sync_state: Mutex::new(CainbotSyncState::default()),
    });

    sync_cainbot_exclusive_groups_file(&state).await;
    spawn_cainbot_exclusive_groups_heartbeat(Arc::clone(&state));
    util::info("NapCatAIChatAssassin Rust 版已启动。");

    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);

    let runner = {
        let state = Arc::clone(&state);
        async move {
            state
                .napcat
                .run_event_loop(|event| {
                    let state = Arc::clone(&state);
                    async move {
                        tokio::spawn(async move {
                            if let Err(error) = handle_event(state, event).await {
                                util::warn(&format!("事件处理失败: {error}"));
                            }
                        });
                        Ok(())
                    }
                })
                .await
        }
    };

    tokio::select! {
        result = runner => result,
        _ = &mut shutdown => {
            util::info("收到退出信号");
            Ok(())
        }
    }
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

async fn handle_event(state: Arc<AppState>, event: Value) -> anyhow::Result<()> {
    if event.get("post_type").and_then(Value::as_str) != Some("message") {
        return Ok(());
    }
    if event.get("message_type").and_then(Value::as_str) != Some("group") {
        return Ok(());
    }
    let user_id = get_str(&event, "user_id");
    let self_id = get_str(&event, "self_id");
    if !user_id.is_empty() && user_id == self_id {
        return Ok(());
    }

    reload_runtime(&state).await?;

    let group_id = get_str(&event, "group_id");
    if group_id.is_empty() {
        return Ok(());
    }
    let gate = get_group_lock(&state, &group_id).await;
    let missed_initial = gate.pending.fetch_add(1, Ordering::SeqCst) > 0;
    let _guard = gate.lock.lock().await;
    let missed = if gate.pending.load(Ordering::SeqCst) == 1 {
        false
    } else {
        missed_initial
    };
    let result = handle_group_message(state, event, missed).await;
    gate.pending.fetch_sub(1, Ordering::SeqCst);
    result
}

async fn handle_group_message(state: Arc<AppState>, event: Value, missed: bool) -> anyhow::Result<()> {
    let config = { state.config.lock().await.clone() };
    let group_id = get_str(&event, "group_id");
    let self_id = get_str(&event, "self_id");
    if group_id.is_empty() || !is_group_enabled(&config, &group_id) {
        return Ok(());
    }

    let message_text = render_message(event.get("message"), event.get("raw_message").and_then(Value::as_str));
    let message_text = message_text.replace("[OP:image]", "");
    if should_ignore(&config, &message_text) {
        return Ok(());
    }

    let sender_name = event
        .get("sender")
        .and_then(Value::as_object)
        .and_then(|sender| {
            sender
                .get("card")
                .and_then(Value::as_str)
                .filter(|v| !v.trim().is_empty())
                .or_else(|| sender.get("nickname").and_then(Value::as_str))
        })
        .map(str::to_string)
        .unwrap_or_else(|| {
            let uid = user_id_value(&event);
            if uid.is_empty() { "群友".to_string() } else { uid }
        });
    let summary = util::build_message_summary(&message_text);
    append_history(
        &state,
        &config,
        &group_id,
        HistoryEntry {
            role: "user".to_string(),
            sender: sender_name,
            user_id: user_id_value(&event),
            text: summary,
            time: util::now_iso(),
        },
    )
    .await;

    if missed {
        util::info(&format!("MISSED - {}", message_text.trim()));
        return Ok(());
    }

    if !should_reply_by_rule(&config, &message_text, &self_id) {
        return Ok(());
    }

    let reply_messages = build_reply_messages(&state, &config, &group_id, &self_id, &message_text).await?;
    let reply_text = {
        let mut client = state.openai.lock().await;
        client
            .complete(
                &reply_messages,
                non_empty(Some(&config.ai.reply_model)),
                None,
                Some(config.ai.max_tokens),
            )
            .await
    };
    let reply_text = match reply_text {
        Ok(text) => text,
        Err(error) => {
            util::warn(&format!("AI 回复失败: {error}"));
            return Ok(());
        }
    };
    if reply_text.trim().is_empty() || reply_text.trim() == SKIP_TEXT {
        return Ok(());
    }
    let final_text: String = reply_text.chars().take(config.bot.max_message_length).collect();
    if final_text.trim().is_empty() {
        return Ok(());
    }

    let delay = sample_reply_delay(&config);
    tokio::time::sleep(Duration::from_secs_f64(delay)).await;
    state
        .napcat
        .send_group_message(
            &group_id,
            final_text.trim(),
            event.get("message_id").map(value_to_string).as_deref(),
        )
        .await?;
    append_history(
        &state,
        &config,
        &group_id,
        HistoryEntry {
            role: "assistant".to_string(),
            sender: "Cain".to_string(),
            user_id: self_id,
            text: final_text.trim().to_string(),
            time: util::now_iso(),
        },
    )
    .await;

    if config.bot.record_memory {
        let state = Arc::clone(&state);
        let group_id = group_id.clone();
        let model = config.ai.memory_model.clone();
        tokio::spawn(async move {
            if let Err(error) = update_group_memory(state, &group_id, &model).await {
                util::warn(&format!("更新群记忆失败: {error}"));
            }
        });
    }

    Ok(())
}

async fn reload_runtime(state: &Arc<AppState>) -> anyhow::Result<()> {
    let config = load_or_create_config(&state.config_path)?;
    let memory = load_or_create_memory(&state.memory_path)?;
    let static_knowledge = load_static_knowledge(&state.knowledge_dir)?;
    {
        let mut guard = state.config.lock().await;
        *guard = config.clone();
    }
    {
        let mut guard = state.memory.lock().await;
        *guard = memory;
    }
    {
        let mut guard = state.static_knowledge.lock().await;
        *guard = static_knowledge;
    }
    {
        let mut guard = state.openai.lock().await;
        *guard = OpenAiCompatClient::new(config.ai.clone())?;
    }
    shrink_history(state, config.bot.history_size).await;
    sync_cainbot_exclusive_groups_file(state).await;
    Ok(())
}

async fn shrink_history(state: &Arc<AppState>, history_size: usize) {
    let mut histories = state.message_history.lock().await;
    for history in histories.values_mut() {
        while history.len() > history_size {
            history.pop_front();
        }
    }
}

async fn append_history(state: &Arc<AppState>, config: &Config, group_id: &str, entry: HistoryEntry) {
    let mut histories = state.message_history.lock().await;
    let history = histories
        .entry(group_id.to_string())
        .or_insert_with(|| VecDeque::with_capacity(config.bot.history_size));
    history.push_back(entry);
    while history.len() > config.bot.history_size {
        history.pop_front();
    }
}

async fn build_reply_messages(
    state: &Arc<AppState>,
    config: &Config,
    group_id: &str,
    self_id: &str,
    current_text: &str,
) -> anyhow::Result<Vec<ChatMessage>> {
    let memory = state.memory.lock().await.clone();
    let long_memory = memory.global.group_memory.get(group_id).cloned().unwrap_or_default();
    let selected_knowledge = build_selected_knowledge(state, &memory, group_id).await;
    let mut prompt_parts = vec![
        config.bot.persona_prompt.clone(),
        format!("你当前所在群号：{group_id}"),
        format!("你的 QQ 号：{self_id}"),
        "如果你不想参与当前对话，必须只输出“【SKIP】”。".to_string(),
        "你可以参考最近上下文和本群记忆决定是否接话。".to_string(),
        format!("本群长期记忆：{}", if long_memory.is_empty() { "暂无" } else { &long_memory }),
    ];
    if !selected_knowledge.is_null() && selected_knowledge != json!({}) {
        prompt_parts.push(format!("命中的知识与关系：{}", serde_json::to_string(&selected_knowledge)?));
    }
    Ok(vec![
        ChatMessage {
            role: "system".to_string(),
            content: prompt_parts.join("\n\n"),
        },
        ChatMessage {
            role: "user".to_string(),
            content: format!(
                "最近共享上下文：\n{}\n\n本次最新消息：\n{}",
                build_timeline(state, group_id, 20).await,
                current_text
            ),
        },
    ])
}

async fn build_selected_knowledge(state: &Arc<AppState>, memory: &MemoryFile, group_id: &str) -> Value {
    let timeline_entries = get_history_entries(state, group_id, 20).await;
    if timeline_entries.is_empty() {
        return json!({});
    }
    let static_knowledge = state.static_knowledge.lock().await.clone();
    let mut selected = serde_json::Map::new();
    let mut selected_search = serde_json::Map::new();
    let sources = vec![
        ("知识缓存", memory.global.knowledge_cache.clone(), 0.1_f64),
        ("知识库", static_knowledge, 0.15_f64),
        ("知识搜索", memory.global.knowledge_search.clone(), 0.1_f64),
    ];
    for (source_name, source, rate) in sources {
        for (keyword, content) in source {
            for entry in &timeline_entries {
                let rank = util::get_recommend_rank(&keyword, &entry.text, 1000, rate);
                if util::get_recommend_match(rank, 1000) {
                    util::info(&format!("PEAK UP - [{source_name}] {keyword} ({rank})"));
                    selected_search.insert(keyword.clone(), Value::String(content.clone()));
                    break;
                }
            }
        }
    }
    if !selected_search.is_empty() {
        selected.insert("知识搜索".to_string(), Value::Object(selected_search));
    }
    if !memory.global.relationships.is_empty() {
        let mut selected_relationships = serde_json::Map::new();
        for (user_key, relation) in &memory.global.relationships {
            let mut hit = false;
            for entry in &timeline_entries {
                if entry.user_id == *user_key {
                    hit = true;
                    break;
                }
                if let Some(array) = relation.as_array() {
                    if let Some(first) = array.first() {
                        let aliases = if let Some(items) = first.as_array() {
                            items.iter().filter_map(Value::as_str).map(str::to_string).collect::<Vec<_>>()
                        } else if let Some(alias) = first.as_str() {
                            vec![alias.to_string()]
                        } else {
                            Vec::new()
                        };
                        if aliases.iter().any(|alias| entry.text.to_lowercase().contains(&alias.to_lowercase())) {
                            hit = true;
                            break;
                        }
                    }
                }
            }
            if hit {
                selected_relationships.insert(user_key.clone(), relation.clone());
            }
        }
        if !selected_relationships.is_empty() {
            selected.insert("人物关系".to_string(), Value::Object(selected_relationships));
        }
    }
    Value::Object(selected)
}

async fn update_group_memory(state: Arc<AppState>, group_id: &str, model_override: &str) -> anyhow::Result<()> {
    let timeline = build_timeline(&state, group_id, 20).await;
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "你负责把一个群最近聊天压缩成 120 字以内的长期记忆。不要流水账，只保留对后续聊天有价值的信息。只输出记忆文本。".to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: timeline,
        },
    ];
    let text = {
        let config = state.config.lock().await.clone();
        let mut client = state.openai.lock().await;
        client
            .complete(
                &messages,
                non_empty(Some(model_override)).or(non_empty(Some(&config.ai.memory_model))),
                Some(0.3),
                Some(config.ai.max_tokens),
            )
            .await?
    };
    let mut memory = state.memory.lock().await;
    memory.global.group_memory.insert(group_id.to_string(), text);
    util::write_json_pretty(&state.memory_path, &*memory)?;
    Ok(())
}

async fn build_timeline(state: &Arc<AppState>, group_id: &str, limit: usize) -> String {
    let entries = get_history_entries(state, group_id, limit).await;
    if entries.is_empty() {
        return "(暂无上下文)".to_string();
    }
    entries
        .iter()
        .enumerate()
        .map(|(index, item)| format!("{}. [{}] {}: {}", index + 1, item.time, item.sender, item.text))
        .collect::<Vec<_>>()
        .join("\n")
}

async fn get_history_entries(state: &Arc<AppState>, group_id: &str, limit: usize) -> Vec<HistoryEntry> {
    let histories = state.message_history.lock().await;
    histories
        .get(group_id)
        .map(|items| items.iter().rev().take(limit).cloned().collect::<Vec<_>>())
        .unwrap_or_default()
        .into_iter()
        .rev()
        .collect()
}

async fn get_group_lock(state: &Arc<AppState>, group_id: &str) -> Arc<GroupGate> {
    let mut locks = state.group_locks.lock().await;
    locks
        .entry(group_id.to_string())
        .or_insert_with(|| Arc::new(GroupGate::new()))
        .clone()
}

fn spawn_cainbot_exclusive_groups_heartbeat(state: Arc<AppState>) {
    tokio::spawn(async move {
        loop {
            let heartbeat_seconds = {
                let config = state.config.lock().await.clone();
                cainbot_exclusive_groups_heartbeat_seconds(&config)
            };
            tokio::time::sleep(Duration::from_secs(heartbeat_seconds)).await;
            sync_cainbot_exclusive_groups_file(&state).await;
        }
    });
}

async fn sync_cainbot_exclusive_groups_file(state: &Arc<AppState>) {
    if let Err(error) = sync_cainbot_exclusive_groups_file_inner(state).await {
        util::warn(&format!("同步 CainBot 互斥群文件失败: {error}"));
    }
}

async fn sync_cainbot_exclusive_groups_file_inner(state: &Arc<AppState>) -> anyhow::Result<()> {
    let config = state.config.lock().await.clone();
    if !config.integration.write_cainbot_exclusive_groups {
        return Ok(());
    }
    let signature = build_cainbot_exclusive_groups_signature(&config);
    let heartbeat_ms = cainbot_exclusive_groups_heartbeat_seconds(&config).saturating_mul(1000);
    let now_ms = current_time_ms();
    let target_path = get_cainbot_exclusive_groups_file_path(&state.root_dir, &config);
    let mut sync_state = state.cainbot_sync_state.lock().await;
    let should_write = sync_state.last_signature.as_ref() != Some(&signature)
        || !target_path.exists()
        || now_ms.saturating_sub(sync_state.last_write_at_ms) >= heartbeat_ms;
    if !should_write {
        return Ok(());
    }
    let payload = build_cainbot_exclusive_groups_payload(&config, util::now_iso());
    util::write_json_pretty_atomic(&target_path, &payload)?;
    sync_state.last_signature = Some(signature);
    sync_state.last_write_at_ms = now_ms;
    Ok(())
}

fn build_cainbot_exclusive_groups_signature(config: &Config) -> String {
    let (mode, group_ids) = build_cainbot_exclusive_groups_scope(config);
    format!("{mode}:{}", group_ids.join(","))
}

fn build_cainbot_exclusive_groups_payload(config: &Config, updated_at: String) -> Value {
    let (mode, group_ids) = build_cainbot_exclusive_groups_scope(config);
    json!({
        "version": 1,
        "source": "NapCatAIChatAssassin",
        "updatedAt": updated_at,
        "mode": mode,
        "groupIds": group_ids,
    })
}

fn build_cainbot_exclusive_groups_scope(config: &Config) -> (String, Vec<String>) {
    let enabled_groups = config
        .bot
        .enabled_groups
        .iter()
        .map(|item| item.trim().to_string())
        .filter(|item| !item.is_empty())
        .collect::<Vec<_>>();
    let mode = if enabled_groups.iter().any(|item| item == "all") {
        "all".to_string()
    } else {
        "list".to_string()
    };
    let group_ids = if mode == "all" {
        Vec::<String>::new()
    } else {
        enabled_groups.into_iter().filter(|item| item != "all").collect::<Vec<_>>()
    };
    (mode, group_ids)
}

fn cainbot_exclusive_groups_heartbeat_seconds(config: &Config) -> u64 {
    config
        .integration
        .cainbot_exclusive_groups_heartbeat_seconds
        .max(MIN_CAINBOT_EXCLUSIVE_GROUPS_HEARTBEAT_SECONDS)
}

fn current_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_millis() as u64)
        .unwrap_or_default()
}

fn is_group_enabled(config: &Config, group_id: &str) -> bool {
    config.bot.enabled_groups.iter().any(|item| item == "all" || item == group_id)
}

fn should_ignore(config: &Config, message_text: &str) -> bool {
    let trimmed = message_text.trim();
    trimmed.is_empty() || config.bot.ignore_prefixes.iter().any(|prefix| trimmed.starts_with(prefix))
}

fn should_reply_by_rule(config: &Config, message_text: &str, self_id: &str) -> bool {
    if config.bot.mention_reply && !self_id.is_empty() && message_text.contains(&format!("[OP:at,id={self_id}]")) {
        return true;
    }
    if config
        .bot
        .reply_keywords
        .iter()
        .any(|keyword| !keyword.trim().is_empty() && message_text.contains(keyword))
    {
        return true;
    }
    let probability = config.bot.reply_probability.clamp(0.0, 1.0);
    rand::rng().random_bool(probability)
}

fn sample_reply_delay(config: &Config) -> f64 {
    let low = config.bot.reply_delay_seconds.first().copied().unwrap_or(0.8);
    let high = config.bot.reply_delay_seconds.get(1).copied().unwrap_or(low);
    rand::rng().random_range(low.min(high)..=low.max(high))
}

fn render_message(message: Option<&Value>, raw_message: Option<&str>) -> String {
    if let Some(message) = message {
        if let Some(text) = message.as_str() {
            return text.to_string();
        }
        if let Some(items) = message.as_array() {
            let mut parts = Vec::new();
            for segment in items {
                let seg_type = segment.get("type").and_then(Value::as_str).unwrap_or_default();
                let data = segment.get("data").and_then(Value::as_object);
                match seg_type {
                    "text" => {
                        if let Some(text) = data.and_then(|item| item.get("text")).and_then(Value::as_str) {
                            parts.push(text.to_string());
                        }
                    }
                    "at" => {
                        if let Some(qq) = data.and_then(|item| item.get("qq")).map(value_to_string) {
                            parts.push(format!("[OP:at,id={qq}]"));
                        }
                    }
                    "image" => parts.push("[OP:image]".to_string()),
                    _ => {}
                }
            }
            let rendered = parts.join("");
            if !rendered.trim().is_empty() {
                return rendered;
            }
        }
    }
    raw_message.unwrap_or_default().to_string()
}

fn load_or_create_memory(path: &Path) -> anyhow::Result<MemoryFile> {
    if path.exists() {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read memory: {}", path.display()))?;
        let memory = serde_json::from_str::<MemoryFile>(&text)
            .with_context(|| format!("failed to parse memory: {}", path.display()))?;
        Ok(memory)
    } else {
        let memory = MemoryFile::default();
        util::write_json_pretty(path, &memory)?;
        Ok(memory)
    }
}

fn load_static_knowledge(path: &Path) -> anyhow::Result<HashMap<String, String>> {
    std::fs::create_dir_all(path)?;
    let mut merged = HashMap::new();
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        let text = match std::fs::read_to_string(entry.path()) {
            Ok(text) => text,
            Err(error) => {
                util::warn(&format!("加载知识库[{name}]失败: {error}"));
                continue;
            }
        };
        match serde_json::from_str::<HashMap<String, String>>(&text) {
            Ok(map) => {
                util::info(&format!("已加载知识库[{name}]"));
                merged.extend(map);
            }
            Err(error) => util::warn(&format!("加载知识库[{name}]失败: {error}")),
        }
    }
    util::info(&format!("已加载知识库共[{}]条", merged.len()));
    Ok(merged)
}

fn get_str(value: &Value, key: &str) -> String {
    value.get(key).map(value_to_string).unwrap_or_default()
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Number(number) => number.to_string(),
        Value::Bool(flag) => flag.to_string(),
        _ => String::new(),
    }
}

fn user_id_value(event: &Value) -> String {
    get_str(event, "user_id")
}

fn non_empty<'a>(value: Option<&'a str>) -> Option<&'a str> {
    value.and_then(|item| {
        let trimmed = item.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    })
}
