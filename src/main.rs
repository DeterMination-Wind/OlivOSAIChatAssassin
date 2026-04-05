mod config;
mod mission;
mod napcat;
mod openai;
mod tools;
mod util;

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{Context, bail};
use config::{Config, get_cainbot_exclusive_groups_file_path, load_or_create_config};
use napcat::NapcatClient;
use openai::{ChatMessage, OpenAiCompatClient};
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tools::{ToolExecutor, ToolRuntimeContext, build_tool_system_prompt, extract_tool_request};

const SKIP_TEXT: &str = "【SKIP】";
const MEMORY_SKIP_TEXT: &str = "【NO_MEMORY_UPDATE】";
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
    #[serde(default)]
    #[serde(rename = "群上下文前缀")]
    timeline_group_prefixes: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize)]
struct HistoryEntry {
    role: String,
    sender: String,
    user_id: String,
    text: String,
    time: String,
    message_id: Option<String>,
    reply_to_message_id: Option<String>,
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
    missed_backlog: Mutex<HashMap<String, usize>>,
    group_locks: Mutex<HashMap<String, Arc<GroupGate>>>,
    openai: Mutex<OpenAiCompatClient>,
    napcat: NapcatClient,
    tool_executor: ToolExecutor,
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

#[derive(Debug, Clone)]
struct ReplyRuleDecision {
    should_reply: bool,
    reason: String,
    probability: f64,
    roll: Option<f64>,
    backlog_bonus: f64,
    group_bonus: f64,
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
        root_dir: root_dir.clone(),
        config_path: config_path.clone(),
        memory_path,
        knowledge_dir,
        config: Mutex::new(config),
        memory: Mutex::new(memory),
        static_knowledge: Mutex::new(static_knowledge),
        message_history: Mutex::new(HashMap::new()),
        missed_backlog: Mutex::new(HashMap::new()),
        group_locks: Mutex::new(HashMap::new()),
        openai: Mutex::new(openai),
        tool_executor: ToolExecutor::new(root_dir.clone(), config_path.clone(), napcat.clone())?,
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
    let message_image_urls = extract_message_image_urls(event.get("message"));
    let current_message_id = event
        .get("message_id")
        .map(value_to_string)
        .filter(|value| !value.trim().is_empty());
    let current_reply_to_message_id = extract_reply_to_message_id(event.get("message"));
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
            sender: sender_name.clone(),
            user_id: user_id_value(&event),
            text: summary.clone(),
            time: util::now_iso(),
            message_id: current_message_id.clone(),
            reply_to_message_id: current_reply_to_message_id.clone(),
        },
    )
    .await;

    if missed {
        let backlog = record_missed_backlog(&state, &group_id).await;
        util::info(&format!(
            "MISSED - group={group_id} reason=group-gate-busy backlog={backlog} text={}",
            summary
        ));
        return Ok(());
    }

    let missed_backlog = peek_missed_backlog(&state, &group_id).await;
    let rule_decision = should_reply_by_rule(&config, &group_id, &message_text, &self_id, missed_backlog);
    if !rule_decision.should_reply {
        util::info(&format!(
            "SKIP-RULE - group={group_id} reason={} probability={:.3} roll={} backlog_bonus={:.3} group_bonus={:.3} text={}",
            rule_decision.reason,
            rule_decision.probability,
            format_roll(rule_decision.roll),
            rule_decision.backlog_bonus,
            rule_decision.group_bonus,
            summary
        ));
        return Ok(());
    }
    let missed_backlog = take_missed_backlog(&state, &group_id).await;

    let reply_messages = build_reply_messages(
        &state,
        &config,
        &group_id,
        &self_id,
        missed_backlog,
        &sender_name,
        current_message_id.as_deref(),
        current_reply_to_message_id.as_deref(),
        &message_text,
        &message_image_urls,
    )
    .await?;
    let reply_text = generate_reply_with_tools(
        &state,
        &config,
        ToolRuntimeContext {
            group_id: group_id.clone(),
            current_image_urls: message_image_urls.clone(),
        },
        reply_messages,
        non_empty(Some(&config.ai.reply_model)),
    )
    .await;
    let reply_text = match reply_text {
        Ok(text) => text,
        Err(error) => {
            util::warn(&format!("AI 回复失败: {error}"));
            return Ok(());
        }
    };
    if let Some(skip_reason) = classify_model_skip_reason(&reply_text) {
        util::info(&format!(
            "SKIP-AI - group={group_id} reason={skip_reason} probability={:.3} roll={} backlog_bonus={:.3} group_bonus={:.3} text={}",
            rule_decision.probability,
            format_roll(rule_decision.roll),
            rule_decision.backlog_bonus,
            rule_decision.group_bonus,
            summary
        ));
        return Ok(());
    }
    if let Some(mission) = mission::extract_mission_request(&reply_text) {
        let launch_text = mission::build_mission_launch_text(&mission);
        let launch_message_id = state
            .napcat
            .send_group_message(&group_id, &launch_text, current_message_id.as_deref())
            .await?;
        append_history(
            &state,
            &config,
            &group_id,
            HistoryEntry {
                role: "assistant".to_string(),
                sender: "Cain".to_string(),
                user_id: self_id.clone(),
                text: launch_text.clone(),
                time: util::now_iso(),
                message_id: launch_message_id,
                reply_to_message_id: current_message_id.clone(),
            },
        )
        .await;
        tokio::spawn({
            let state = Arc::clone(&state);
            let group_id = group_id.clone();
            let requester = sender_name.clone();
            let request_text = message_text.clone();
            let reply_to_message_id = current_message_id.clone();
            let config = config.clone();
            async move {
                if let Err(error) = mission::run_mission_and_report(
                    state,
                    config,
                    group_id,
                    self_id,
                    requester,
                    request_text,
                    reply_to_message_id,
                    mission,
                )
                .await
                {
                    util::warn(&format!("MISSION 执行失败: {error}"));
                }
            }
        });
        return Ok(());
    }
    let final_text: String = reply_text.chars().take(config.bot.max_message_length).collect();
    if final_text.trim().is_empty() {
        return Ok(());
    }

    let delay = sample_reply_delay(&config);
    tokio::time::sleep(Duration::from_secs_f64(delay)).await;
    let sent_message_id = state
        .napcat
        .send_group_message(
            &group_id,
            final_text.trim(),
            current_message_id.as_deref(),
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
            message_id: sent_message_id,
            reply_to_message_id: current_message_id.clone(),
        },
    )
    .await;

    if config.bot.record_memory
        && should_attempt_group_memory_update(&state, &group_id, &message_text, final_text.trim()).await
    {
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
    missed_backlog: usize,
    current_sender: &str,
    current_message_id: Option<&str>,
    current_reply_to_message_id: Option<&str>,
    current_text: &str,
    current_image_urls: &[String],
) -> anyhow::Result<Vec<ChatMessage>> {
    let memory = state.memory.lock().await.clone();
    let long_memory = memory.global.group_memory.get(group_id).cloned().unwrap_or_default();
    let selected_knowledge = build_selected_knowledge(state, &memory, group_id).await;
    let rendered_timeline = render_timeline(state, group_id, 20).await;
    let current_display_id = current_message_id
        .and_then(|id| rendered_timeline.message_id_to_display_id.get(id))
        .map(String::as_str);
    let current_reply_display_id = current_reply_to_message_id
        .and_then(|id| rendered_timeline.message_id_to_display_id.get(id))
        .map(String::as_str);
    let mut prompt_parts = vec![
        config.bot.persona_prompt.clone(),
        format!("你当前所在群号：{group_id}"),
        format!("你的 QQ 号：{self_id}"),
        "上下文每行格式：[id:PPP-N[,reply:PPP-M]]名字:内容。PPP 为本群唯一三位前缀；reply 表示该消息回复了哪条。不要在对外发言中输出 [id:...] 标签。".to_string(),
        "默认不要插话。只有在别人明确 @ 你、直接追问你、当前轮明显在问你、或你补一句能显著提高信息价值时才回复；其余情况一律输出【SKIP】。".to_string(),
        "如果决定跳过，必须只输出“【SKIP】”，不要带解释、不要带标点、不要输出 [SKIP]、skip 或别的变体。".to_string(),
        "默认不用 Markdown：不要标题、列表、引用、代码块、加粗、反引号。除非用户明确要代码、命令或结构化内容，否则用普通纯文本短句回复。".to_string(),
        "你可以参考最近上下文和本群记忆决定是否接话，但不能因为看懂了上下文就硬插话。".to_string(),
        format!("本群长期记忆：{}", if long_memory.is_empty() { "暂无" } else { &long_memory }),
    ];
    if missed_backlog > 0 {
        prompt_parts.push(format!(
            "刚才有 {missed_backlog} 条消息因排队未被立即处理。它们只是补充上下文，不代表你必须补说；只有在现在开口确实更合适时才回复。"
        ));
    }
    let tool_prompt = build_tool_system_prompt(config);
    if !tool_prompt.trim().is_empty() {
        prompt_parts.push(tool_prompt);
    }
    if !selected_knowledge.is_null() && selected_knowledge != json!({}) {
        prompt_parts.push(format!("命中的知识与关系：{}", serde_json::to_string(&selected_knowledge)?));
    }
    if !current_image_urls.is_empty() {
        prompt_parts.push(format!(
            "本次最新消息附带了 {} 张图片。如需理解图片内容、截图、报错界面或图片中的文字，可调用 read_image 读取当前图片。",
            current_image_urls.len()
        ));
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
                rendered_timeline.text,
                format_timeline_line(current_display_id, current_reply_display_id, current_sender, current_text),
            ),
        },
    ])
}

async fn generate_reply_with_tools(
    state: &Arc<AppState>,
    config: &Config,
    runtime: ToolRuntimeContext,
    mut messages: Vec<ChatMessage>,
    model_override: Option<&str>,
) -> anyhow::Result<String> {
    let max_rounds = config.tools.max_rounds.max(1);
    for _round in 0..max_rounds {
        let reply_text = {
            let mut client = state.openai.lock().await;
            client
                .complete(
                    &messages,
                    model_override,
                    None,
                    Some(config.ai.max_tokens),
                )
                .await?
        };
        let normalized = normalize_model_reply(&reply_text);
        if normalized.is_empty() || normalized == SKIP_TEXT {
            return Ok(normalized);
        }
        let Some(tool_request) = extract_tool_request(&normalized) else {
            return Ok(normalized);
        };
        let tool_result = state
            .tool_executor
            .execute(config, &runtime, tool_request)
            .await
            .unwrap_or_else(|error| format!("工具执行失败：{error}"));
        messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: normalized,
        });
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: format!("工具执行结果：\n{tool_result}"),
        });
    }
    bail!("工具调用轮数超限")
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
    let existing_memory = {
        let memory = state.memory.lock().await;
        memory.global.group_memory.get(group_id).cloned().unwrap_or_default()
    };
    let timeline = build_timeline(&state, group_id, 12).await;
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: concat!(
                "你负责维护一个群的长期记忆。",
                "只记录对后续聊天仍然有价值的稳定信息，例如长期偏好、持续中的计划、明确约定、反复提到的固定背景、稳定关系。",
                "不要记录一次性闲聊、短期情绪、临时排查步骤、已经结束的话题、口头禅、无意义玩笑。",
                "如果最近对话没有新增的长期价值，必须只输出【NO_MEMORY_UPDATE】。",
                "如果需要更新，则输出新的完整长期记忆，120字以内，不要分点，不要流水账。"
            ).to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: format!(
                "当前长期记忆：\n{}\n\n最近对话：\n{}",
                if existing_memory.is_empty() { "暂无" } else { &existing_memory },
                timeline
            ),
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
    let text = text.trim().to_string();
    if text.is_empty() || text == MEMORY_SKIP_TEXT || text == existing_memory {
        return Ok(());
    }
    if !looks_like_persistent_memory(&text) {
        return Ok(());
    }
    let mut memory = state.memory.lock().await;
    memory.global.group_memory.insert(group_id.to_string(), text);
    util::write_json_pretty(&state.memory_path, &*memory)?;
    Ok(())
}

async fn should_attempt_group_memory_update(
    state: &Arc<AppState>,
    group_id: &str,
    latest_user_text: &str,
    latest_reply_text: &str,
) -> bool {
    if !looks_like_persistent_memory(latest_user_text) && !looks_like_persistent_memory(latest_reply_text) {
        return false;
    }
    let recent_entries = get_history_entries(state, group_id, 8).await;
    if recent_entries.len() < 4 {
        return false;
    }
    let meaningful_count = recent_entries
        .iter()
        .filter(|entry| looks_like_persistent_memory(&entry.text))
        .count();
    let total_chars = recent_entries
        .iter()
        .map(|entry| entry.text.chars().count())
        .sum::<usize>();
    meaningful_count >= 3 && total_chars >= 80
}

async fn build_timeline(state: &Arc<AppState>, group_id: &str, limit: usize) -> String {
    render_timeline(state, group_id, limit).await.text
}

#[derive(Debug, Clone)]
struct RenderedTimeline {
    text: String,
    message_id_to_display_id: HashMap<String, String>,
}

async fn render_timeline(state: &Arc<AppState>, group_id: &str, limit: usize) -> RenderedTimeline {
    let entries = get_history_entries(state, group_id, limit).await;
    if entries.is_empty() {
        return RenderedTimeline {
            text: "(暂无上下文)".to_string(),
            message_id_to_display_id: HashMap::new(),
        };
    }

    let prefix = get_or_assign_timeline_group_prefix(state, group_id).await;
    let mut message_id_to_display_id = HashMap::new();
    for (index, item) in entries.iter().enumerate() {
        let display_id = format!("{prefix}-{}", index + 1);
        if let Some(message_id) = item.message_id.as_ref().filter(|value| !value.trim().is_empty()) {
            message_id_to_display_id.insert(message_id.clone(), display_id);
        }
    }

    let lines = entries
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let display_id = format!("{prefix}-{}", index + 1);
            let reply_display_id = item
                .reply_to_message_id
                .as_ref()
                .and_then(|id| message_id_to_display_id.get(id))
                .map(String::as_str);
            format_timeline_line(Some(display_id.as_str()), reply_display_id, &item.sender, &item.text)
        })
        .collect::<Vec<_>>();

    RenderedTimeline {
        text: lines.join("\n"),
        message_id_to_display_id,
    }
}

async fn get_or_assign_timeline_group_prefix(state: &Arc<AppState>, group_id: &str) -> String {
    let mut memory = state.memory.lock().await;
    let prefixes = &mut memory.global.timeline_group_prefixes;

    let mut used = HashSet::new();
    for (gid, value) in prefixes.iter() {
        if gid == group_id {
            continue;
        }
        if let Some(prefix) = normalize_group_prefix(value) {
            used.insert(prefix);
        }
    }

    let existing_raw = prefixes.get(group_id).cloned().unwrap_or_default();
    if let Some(existing) = normalize_group_prefix(&existing_raw) {
        if !used.contains(&existing) {
            if existing_raw != existing {
                prefixes.insert(group_id.to_string(), existing.clone());
                let _ = util::write_json_pretty(&state.memory_path, &*memory);
            }
            return existing;
        }
    }

    let candidate = (1u16..=999u16)
        .map(|value| format!("{value:03}"))
        .find(|value| !used.contains(value))
        .unwrap_or_else(|| "000".to_string());
    prefixes.insert(group_id.to_string(), candidate.clone());
    let _ = util::write_json_pretty(&state.memory_path, &*memory);
    candidate
}

fn normalize_group_prefix(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.len() == 3 && trimmed.chars().all(|ch| ch.is_ascii_digit()) {
        return Some(trimmed.to_string());
    }
    if let Ok(number) = trimmed.parse::<u16>() {
        if (1..=999).contains(&number) {
            return Some(format!("{number:03}"));
        }
    }
    None
}

fn format_timeline_line(
    display_id: Option<&str>,
    reply_display_id: Option<&str>,
    sender: &str,
    text: &str,
) -> String {
    let sender = sender.trim();
    let text = text.trim().replace('\r', " ").replace('\n', " ");
    let Some(display_id) = display_id.filter(|value| !value.trim().is_empty()) else {
        return format!("{sender}:{text}");
    };
    let mut header = format!("[id:{display_id}");
    if let Some(reply_display_id) = reply_display_id.filter(|value| !value.trim().is_empty()) {
        header.push_str(&format!(",reply:{reply_display_id}"));
    }
    header.push(']');
    format!("{header}{sender}:{text}")
}

fn normalize_model_reply(reply_text: &str) -> String {
    let normalized = reply_text
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .replace("[SKIP]", SKIP_TEXT)
        .replace("[skip]", SKIP_TEXT)
        .trim()
        .to_string();
    if normalized.is_empty() {
        return normalized;
    }
    let skip_like = normalized
        .trim_end_matches(['。', '.', '!', '！', '?', '？', '…', ' '])
        .trim();
    if skip_like.eq_ignore_ascii_case("skip")
        || skip_like.eq_ignore_ascii_case("[skip]")
        || skip_like == SKIP_TEXT
    {
        return SKIP_TEXT.to_string();
    }
    normalized
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

async fn record_missed_backlog(state: &Arc<AppState>, group_id: &str) -> usize {
    let mut backlog = state.missed_backlog.lock().await;
    let entry = backlog.entry(group_id.to_string()).or_insert(0);
    *entry = entry.saturating_add(1).min(32);
    *entry
}

async fn peek_missed_backlog(state: &Arc<AppState>, group_id: &str) -> usize {
    let backlog = state.missed_backlog.lock().await;
    backlog.get(group_id).copied().unwrap_or(0)
}

async fn take_missed_backlog(state: &Arc<AppState>, group_id: &str) -> usize {
    let mut backlog = state.missed_backlog.lock().await;
    backlog.remove(group_id).unwrap_or(0)
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

fn should_reply_by_rule(
    config: &Config,
    group_id: &str,
    message_text: &str,
    self_id: &str,
    missed_backlog: usize,
) -> ReplyRuleDecision {
    let group_bonus = group_reply_probability_bonus(config, group_id);
    if config.bot.mention_reply && !self_id.is_empty() && message_text.contains(&format!("[OP:at,id={self_id}]")) {
        return ReplyRuleDecision {
            should_reply: true,
            reason: "mention".to_string(),
            probability: 1.0,
            roll: None,
            backlog_bonus: 0.0,
            group_bonus,
        };
    }
    if config
        .bot
        .reply_keywords
        .iter()
        .any(|keyword| !keyword.trim().is_empty() && message_text.contains(keyword))
    {
        return ReplyRuleDecision {
            should_reply: true,
            reason: "reply-keyword".to_string(),
            probability: 1.0,
            roll: None,
            backlog_bonus: 0.0,
            group_bonus,
        };
    }
    let mut probability = config.bot.reply_probability.clamp(0.0, 1.0);
    if group_bonus != 0.0 {
        probability = (probability + group_bonus).clamp(0.0, 1.0);
    }
    let mut backlog_bonus = 0.0;
    if missed_backlog > 0 {
        backlog_bonus = 0.08 * missed_backlog.min(3) as f64;
        probability = (probability + backlog_bonus).min(0.35_f64.max(probability));
    }
    let roll = rand::rng().random::<f64>();
    ReplyRuleDecision {
        should_reply: roll < probability,
        reason: if roll < probability {
            "probability-hit".to_string()
        } else {
            "probability-roll-miss".to_string()
        },
        probability,
        roll: Some(roll),
        backlog_bonus,
        group_bonus,
    }
}

fn group_reply_probability_bonus(config: &Config, group_id: &str) -> f64 {
    let normalized_group_id = group_id.trim();
    config
        .bot
        .reply_probability_bonus_by_group
        .iter()
        .find_map(|(configured_group_id, bonus)| {
            if configured_group_id.trim() == normalized_group_id {
                Some((*bonus).clamp(-1.0, 1.0))
            } else {
                None
            }
        })
        .unwrap_or(0.0)
}

fn format_roll(roll: Option<f64>) -> String {
    roll.map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "-".to_string())
}

fn classify_model_skip_reason(reply_text: &str) -> Option<&'static str> {
    let trimmed = reply_text.trim();
    if trimmed.is_empty() {
        Some("empty-reply")
    } else if trimmed == SKIP_TEXT {
        Some("model-returned-skip")
    } else {
        None
    }
}

fn sample_reply_delay(config: &Config) -> f64 {
    let low = config.bot.reply_delay_seconds.first().copied().unwrap_or(0.8);
    let high = config.bot.reply_delay_seconds.get(1).copied().unwrap_or(low);
    rand::rng().random_range(low.min(high)..=low.max(high))
}

fn render_message(message: Option<&Value>, raw_message: Option<&str>) -> String {
    if let Some(message) = message {
        if let Some(text) = message.as_str() {
            return sanitize_raw_message_text(text);
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
    sanitize_raw_message_text(raw_message.unwrap_or_default())
}

fn sanitize_raw_message_text(text: &str) -> String {
    strip_op_markers(text, &["record", "video"])
}

fn strip_op_markers(text: &str, segment_types: &[&str]) -> String {
    let mut output = text.to_string();
    for segment_type in segment_types {
        let marker = format!("[OP:{segment_type}");
        while let Some(start) = output.find(&marker) {
            let Some(relative_end) = output[start..].find(']') else {
                output.truncate(start);
                break;
            };
            let end = start + relative_end + 1;
            output.replace_range(start..end, "");
        }
    }
    output
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

fn extract_message_image_urls(message: Option<&Value>) -> Vec<String> {
    let mut urls = Vec::new();
    let Some(items) = message.and_then(Value::as_array) else {
        return urls;
    };
    for segment in items {
        let seg_type = segment.get("type").and_then(Value::as_str).unwrap_or_default();
        if seg_type != "image" {
            continue;
        }
        let data = segment.get("data").and_then(Value::as_object);
        let Some(data) = data else {
            continue;
        };
        for key in ["url", "file", "src"] {
            if let Some(value) = data.get(key).and_then(Value::as_str) {
                let trimmed = value.trim();
                if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
                    urls.push(trimmed.to_string());
                    break;
                }
            }
        }
    }
    urls
}

fn extract_reply_to_message_id(message: Option<&Value>) -> Option<String> {
    let Some(items) = message.and_then(Value::as_array) else {
        return None;
    };
    for segment in items {
        let seg_type = segment.get("type").and_then(Value::as_str).unwrap_or_default();
        if seg_type != "reply" {
            continue;
        }
        let data = segment.get("data").and_then(Value::as_object);
        let Some(data) = data else {
            continue;
        };
        if let Some(value) = data.get("id").or_else(|| data.get("message_id")).or_else(|| data.get("messageId")) {
            let id = value_to_string(value);
            if !id.trim().is_empty() {
                return Some(id);
            }
        }
    }
    None
}

fn looks_like_persistent_memory(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed == SKIP_TEXT || trimmed == MEMORY_SKIP_TEXT {
        return false;
    }
    let normalized = trimmed
        .replace("[OP:image]", "")
        .replace('\n', " ")
        .replace('\r', " ");
    let non_space_chars = normalized.chars().filter(|ch| !ch.is_whitespace()).count();
    if non_space_chars < 12 {
        return false;
    }
    let meaningful_chars = normalized
        .chars()
        .filter(|ch| ch.is_alphanumeric() || ('\u{4e00}'..='\u{9fff}').contains(ch))
        .count();
    meaningful_chars >= 8
}
