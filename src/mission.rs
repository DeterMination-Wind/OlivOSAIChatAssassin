use std::sync::Arc;

use anyhow::bail;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::task::JoinSet;
use tokio::time::{Duration, sleep};

use crate::config::Config;
use crate::openai::{ChatMessage, OpenAiCompatClient};
use crate::tools::{ToolRuntimeContext, build_tool_system_prompt, extract_tool_request};
use crate::{AppState, HistoryEntry, SKIP_TEXT, append_history, build_selected_knowledge, build_timeline, non_empty, normalize_model_reply, util};

const MISSION_PREFIXES: [&str; 2] = ["[MISSION]", "【MISSION】"];
const STEP_MODEL_CANDIDATES: [&str; 4] = ["gpt-5.4", "gpt-5.3-codex", "gpt-5.4-mini", "gpt-5.2"];
const MISSION_STEP_MAX_ROUNDS: u32 = 20;

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct MissionPlan {
    #[serde(rename = "Topic", alias = "topic", default)]
    topic: String,
    #[serde(rename = "Steps", alias = "steps", default)]
    steps: Vec<MissionStep>,
}

#[derive(Debug, Clone, Deserialize)]
struct MissionStep {
    #[serde(rename = "Step", alias = "step", default)]
    step: String,
    #[serde(rename = "Goal", alias = "goal", default)]
    goal: String,
    #[serde(rename = "Todo", alias = "todo", alias = "task", default)]
    todo: String,
    #[serde(rename = "SleepMs", alias = "sleep_ms", default)]
    sleep_ms: Option<u64>,
}

#[derive(Debug, Clone)]
struct MissionStepResult {
    index: usize,
    name: String,
    goal: String,
    status: String,
    output: String,
    model: String,
}

pub(crate) fn extract_mission_request(text: &str) -> Option<MissionPlan> {
    let source = text.trim();
    for marker in MISSION_PREFIXES {
        let Some(position) = source.find(marker) else {
            continue;
        };
        let raw = source[position + marker.len()..].trim();
        let start = raw.find('{')?;
        let end = raw.rfind('}')?;
        if end < start {
            continue;
        }
        let plan = serde_json::from_str::<MissionPlan>(&raw[start..=end]).ok()?;
        if mission_plan_is_valid(&plan) {
            return Some(plan);
        }
    }
    None
}

pub(crate) fn build_mission_launch_text(plan: &MissionPlan) -> String {
    let topic = trim_inline(&plan.topic, 22);
    if topic.is_empty() {
        "这事我分步查一下，等我一下。".to_string()
    } else {
        format!("这事我分步查一下：{topic}。")
    }
}

pub(crate) async fn run_mission_and_report(
    state: Arc<AppState>,
    config: Config,
    group_id: String,
    self_id: String,
    requester: String,
    request_text: String,
    reply_to_message_id: Option<String>,
    mission: MissionPlan,
) -> anyhow::Result<()> {
    let timeline = build_timeline(&state, &group_id, 12).await;
    let memory = state.memory.lock().await.clone();
    let selected_knowledge = build_selected_knowledge(&state, &memory, &group_id).await;
    let step_results = execute_mission_plan(
        Arc::clone(&state),
        config.clone(),
        group_id.clone(),
        requester.clone(),
        request_text.clone(),
        timeline,
        selected_knowledge,
        mission.clone(),
    )
    .await;
    let handoff = summarize_mission_handoff(&config, &mission, &request_text, &step_results)
        .await
        .unwrap_or_else(|error| {
            util::warn(&format!("MISSION 汇总失败，回退到规则摘要：{error}"));
            build_fallback_handoff(&mission, &step_results)
        });
    let rendered = render_final_chat_reply(&config, &mission, &request_text, &handoff)
        .await
        .unwrap_or_else(|error| {
            util::warn(&format!("MISSION 聊天改写失败，直接使用 handoff：{error}"));
            handoff.clone()
        });
    let normalized = normalize_model_reply(&rendered);
    let final_text = if normalized.is_empty() || normalized == SKIP_TEXT {
        trim_inline(&handoff, config.bot.max_message_length)
    } else {
        normalized.chars().take(config.bot.max_message_length).collect::<String>()
    };
    if final_text.trim().is_empty() {
        bail!("MISSION 最终回复为空");
    }
    let message_id = state
        .napcat
        .send_group_message(&group_id, final_text.trim(), reply_to_message_id.as_deref())
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
            message_id,
            reply_to_message_id,
        },
    )
    .await;
    Ok(())
}

async fn execute_mission_plan(
    state: Arc<AppState>,
    config: Config,
    group_id: String,
    requester: String,
    request_text: String,
    timeline: String,
    selected_knowledge: Value,
    mission: MissionPlan,
) -> Vec<MissionStepResult> {
    let mut results = Vec::new();
    let mut batch = Vec::new();
    for (index, step) in mission.steps.iter().cloned().enumerate() {
        if is_sleep_step(&step) {
            if !batch.is_empty() {
                results.extend(
                    execute_step_batch(
                        Arc::clone(&state),
                        config.clone(),
                        group_id.clone(),
                        requester.clone(),
                        request_text.clone(),
                        timeline.clone(),
                        selected_knowledge.clone(),
                        mission.topic.clone(),
                        std::mem::take(&mut batch),
                    )
                    .await,
                );
            }
            let sleep_ms = step.sleep_ms.unwrap_or(0).min(60_000);
            if sleep_ms > 0 {
                util::info(&format!("MISSION sleep - {}ms", sleep_ms));
                sleep(Duration::from_millis(sleep_ms)).await;
            }
            continue;
        }
        batch.push((index, step));
    }
    if !batch.is_empty() {
        results.extend(
            execute_step_batch(
                state,
                config,
                group_id,
                requester,
                request_text,
                timeline,
                selected_knowledge,
                mission.topic.clone(),
                batch,
            )
            .await,
        );
    }
    results.sort_by_key(|item| item.index);
    results
}

async fn execute_step_batch(
    state: Arc<AppState>,
    config: Config,
    group_id: String,
    requester: String,
    request_text: String,
    timeline: String,
    selected_knowledge: Value,
    topic: String,
    batch: Vec<(usize, MissionStep)>,
) -> Vec<MissionStepResult> {
    let mut join_set = JoinSet::new();
    for (index, step) in batch {
        let state = Arc::clone(&state);
        let config = config.clone();
        let group_id = group_id.clone();
        let requester = requester.clone();
        let request_text = request_text.clone();
        let timeline = timeline.clone();
        let selected_knowledge = selected_knowledge.clone();
        let topic = topic.clone();
        join_set.spawn(async move {
            execute_mission_step(
                state,
                config,
                group_id,
                requester,
                request_text,
                timeline,
                selected_knowledge,
                topic,
                index,
                step,
            )
            .await
        });
    }
    let mut results = Vec::new();
    while let Some(joined) = join_set.join_next().await {
        match joined {
            Ok(result) => results.push(result),
            Err(error) => results.push(MissionStepResult {
                index: usize::MAX,
                name: "join".to_string(),
                goal: String::new(),
                status: "failed".to_string(),
                output: format!("step task join failed: {error}"),
                model: "runtime".to_string(),
            }),
        }
    }
    results
}

async fn execute_mission_step(
    state: Arc<AppState>,
    config: Config,
    group_id: String,
    requester: String,
    request_text: String,
    timeline: String,
    selected_knowledge: Value,
    topic: String,
    index: usize,
    step: MissionStep,
) -> MissionStepResult {
    let name = step_name(&step, index);
    util::info(&format!("MISSION step start - {}", name));
    let messages = build_mission_step_messages(
        &config,
        &topic,
        &requester,
        &request_text,
        &timeline,
        &selected_knowledge,
        index,
        &step,
    );
    let runtime = ToolRuntimeContext {
        group_id,
        current_image_urls: Vec::new(),
    };
    let mut last_error = String::new();
    for model in STEP_MODEL_CANDIDATES {
        let mut client = match OpenAiCompatClient::new(config.ai.clone()) {
            Ok(client) => client,
            Err(error) => {
                last_error = error.to_string();
                break;
            }
        };
        match generate_reply_with_tools_with_client(
            &state,
            &config,
            &runtime,
            &mut client,
            messages.clone(),
            Some(model),
        )
        .await
        {
            Ok(text) if !text.trim().is_empty() && text.trim() != SKIP_TEXT => {
                util::info(&format!("MISSION step done - {} via {}", name, model));
                return MissionStepResult {
                    index,
                    name,
                    goal: step.goal.clone(),
                    status: "ok".to_string(),
                    output: trim_inline(&text, 800),
                    model: model.to_string(),
                };
            }
            Ok(_) => {
                last_error = "step 返回空结果或 SKIP".to_string();
            }
            Err(error) => {
                last_error = error.to_string();
            }
        }
    }
    util::warn(&format!("MISSION step failed - {}: {}", name, last_error));
    MissionStepResult {
        index,
        name,
        goal: step.goal.clone(),
        status: "failed".to_string(),
        output: trim_inline(&last_error, 800),
        model: "fallback_exhausted".to_string(),
    }
}

fn build_mission_step_messages(
    config: &Config,
    topic: &str,
    requester: &str,
    request_text: &str,
    timeline: &str,
    selected_knowledge: &Value,
    index: usize,
    step: &MissionStep,
) -> Vec<ChatMessage> {
    let mut prompt_parts = vec![
        "你是 Cain 的 MISSION 子执行器，只负责当前这一个 step。".to_string(),
        "目标是实际完成/核实这一小步，并给出事实结果；做不到就明确说卡点，不要编造。".to_string(),
        "默认不用 Markdown，不要标题、列表、代码块、引用。".to_string(),
        "你可以先直接访问某个明确网页再决定下一步；如果还不知道具体页面，再按站点选择搜索工具。".to_string(),
        "需要上网找资料时：通用网页优先用 bing_search；GitHub 项目优先用 github_search；知乎内容优先用 zhihu_search；拿到候选链接后再用 fetch_web_page 继续抓取内容。".to_string(),
        "如果用户明显指定了 GitHub、知乎或某个明确网址，就先走对应站点搜索或直接抓该页面，不要绕远路。".to_string(),
        format!("Mission Topic：{topic}"),
        format!("Step 序号：{}", index + 1),
        format!("Step 名称：{}", step_name(step, index)),
    ];
    if !step.goal.trim().is_empty() {
        prompt_parts.push(format!("Step 目标：{}", step.goal.trim()));
    }
    if !step.todo.trim().is_empty() {
        prompt_parts.push(format!("Step 待办：{}", step.todo.trim()));
    }
    let tool_prompt = build_tool_system_prompt(config);
    if !tool_prompt.trim().is_empty() {
        prompt_parts.push(tool_prompt);
    }
    if !selected_knowledge.is_null() && selected_knowledge != &json!({}) {
        prompt_parts.push(format!("命中的知识与关系：{}", serde_json::to_string(selected_knowledge).unwrap_or_default()));
    }
    vec![
        ChatMessage {
            role: "system".to_string(),
            content: prompt_parts.join("\n\n"),
        },
        ChatMessage {
            role: "user".to_string(),
            content: format!(
                "请求者：{requester}\n原始请求：{request_text}\n最近上下文：\n{timeline}\n\n请只完成当前 step，并返回这一小步的结果。"
            ),
        },
    ]
}

async fn generate_reply_with_tools_with_client(
    state: &Arc<AppState>,
    config: &Config,
    runtime: &ToolRuntimeContext,
    client: &mut OpenAiCompatClient,
    mut messages: Vec<ChatMessage>,
    model_override: Option<&str>,
) -> anyhow::Result<String> {
    let max_rounds = MISSION_STEP_MAX_ROUNDS;
    for _round in 0..max_rounds {
        let reply_text = client
            .complete(
                &messages,
                model_override,
                Some(0.6),
                Some(config.ai.max_tokens),
            )
            .await?;
        let normalized = normalize_model_reply(&reply_text);
        if normalized.is_empty() || normalized == SKIP_TEXT {
            return Ok(normalized);
        }
        let Some(tool_request) = extract_tool_request(&normalized) else {
            return Ok(normalized);
        };
        let tool_result = state
            .tool_executor
            .execute(config, runtime, tool_request)
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
    bail!("MISSION step 工具调用轮数超限")
}

async fn summarize_mission_handoff(
    config: &Config,
    mission: &MissionPlan,
    request_text: &str,
    step_results: &[MissionStepResult],
) -> anyhow::Result<String> {
    let mut client = OpenAiCompatClient::new(config.ai.clone())?;
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "你负责把 mission 执行结果压缩成给 Cain 聊天 AI 的内部 handoff。只写纯文本，不要 Markdown，不要提模型、agent、MISSION 内部机制。先说结论，再补关键卡点或下一步，控制在 120 字内。".to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: format!(
                "Mission Topic：{}\n原始请求：{}\nStep 结果：\n{}",
                mission.topic,
                request_text,
                format_step_results(step_results)
            ),
        },
    ];
    let text = client
        .complete(
            &messages,
            non_empty(Some(&config.ai.model)),
            Some(0.4),
            Some(220),
        )
        .await?;
    Ok(trim_inline(&text, 160))
}

async fn render_final_chat_reply(
    config: &Config,
    mission: &MissionPlan,
    request_text: &str,
    handoff: &str,
) -> anyhow::Result<String> {
    let mut client = OpenAiCompatClient::new(config.ai.clone())?;
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: format!(
                "{}\n\n你现在是在 MISSION 完成后回群里一句自然的话。不要提 MISSION、步骤、agent、模型、内部系统。默认不用 Markdown。",
                config.bot.persona_prompt
            ),
        },
        ChatMessage {
            role: "user".to_string(),
            content: format!(
                "原始请求：{}\nMission Topic：{}\n内部汇总：{}\n\n请用 Cain 的口吻给群里一句自然回复。",
                request_text, mission.topic, handoff
            ),
        },
    ];
    client
        .complete(
            &messages,
            non_empty(Some(&config.ai.reply_model)).or(non_empty(Some(&config.ai.model))),
            Some(0.7),
            Some(config.ai.max_tokens.min(220)),
        )
        .await
}

fn build_fallback_handoff(mission: &MissionPlan, step_results: &[MissionStepResult]) -> String {
    let mut success = Vec::new();
    let mut failed = Vec::new();
    for item in step_results {
        let line = format!("{}：{}", item.name, trim_inline(&item.output, 48));
        if item.status == "ok" {
            success.push(line);
        } else {
            failed.push(line);
        }
    }
    let mut parts = vec![format!("{} 已分步处理。", trim_inline(&mission.topic, 24))];
    if !success.is_empty() {
        parts.push(format!("已确认：{}", success.join("；")));
    }
    if !failed.is_empty() {
        parts.push(format!("未完成：{}", failed.join("；")));
    }
    trim_inline(&parts.join(" "), 160)
}

fn format_step_results(step_results: &[MissionStepResult]) -> String {
    step_results
        .iter()
        .map(|item| {
            let mut line = format!(
                "{}. {} [{} via {}]",
                item.index + 1,
                item.name,
                item.status,
                item.model
            );
            if !item.goal.trim().is_empty() {
                line.push_str(&format!(" goal={}", trim_inline(&item.goal, 48)));
            }
            line.push_str(&format!(" result={}", trim_inline(&item.output, 220)));
            line
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn mission_plan_is_valid(plan: &MissionPlan) -> bool {
    !plan.topic.trim().is_empty() && plan.steps.iter().any(step_is_meaningful)
}

fn step_is_meaningful(step: &MissionStep) -> bool {
    !step.step.trim().is_empty() || !step.goal.trim().is_empty() || !step.todo.trim().is_empty() || step.sleep_ms.unwrap_or(0) > 0
}

fn is_sleep_step(step: &MissionStep) -> bool {
    step.sleep_ms.unwrap_or(0) > 0
        && step.todo.trim().is_empty()
        && step.goal.trim().is_empty()
        && (step.step.trim().is_empty() || step.step.trim().eq_ignore_ascii_case("sleep"))
}

fn step_name(step: &MissionStep, index: usize) -> String {
    if !step.step.trim().is_empty() {
        step.step.trim().to_string()
    } else {
        format!("step-{}", index + 1)
    }
}

fn trim_inline(text: &str, limit: usize) -> String {
    text.replace('\r', " ")
        .replace('\n', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(limit)
        .collect()
}
