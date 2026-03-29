use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use reqwest::Client;
use serde::Serialize;
use serde_json::{Value, json};

use crate::config::AiConfig;

const RETRYABLE_HTTP_STATUS: [u16; 8] = [408, 409, 425, 429, 500, 502, 503, 504];
const RESPONSE_FALLBACK_HTTP_STATUS: [u16; 10] = [400, 404, 405, 408, 409, 415, 425, 429, 500, 502];

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug)]
pub struct OpenAiCompatClient {
    http: Client,
    config: AiConfig,
    cooldown_until: Option<Instant>,
    cooldown_reason: String,
    retryable_failure_streak: u32,
    transport_suppressed_until: HashMap<String, Instant>,
}

impl OpenAiCompatClient {
    pub fn new(config: AiConfig) -> anyhow::Result<Self> {
        let http = Client::builder()
            .timeout(Duration::from_millis(config.request_timeout_ms.max(5000)))
            .build()?;
        Ok(Self {
            http,
            config,
            cooldown_until: None,
            cooldown_reason: String::new(),
            retryable_failure_streak: 0,
            transport_suppressed_until: HashMap::new(),
        })
    }

    pub async fn complete(
        &mut self,
        messages: &[ChatMessage],
        model: Option<&str>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> anyhow::Result<String> {
        self.validate()?;
        if let Some(until) = self.cooldown_until {
            if Instant::now() < until {
                bail!("聊天接口暂时不可用，已进入冷却：{}", self.cooldown_reason);
            }
        }
        let model = model
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .unwrap_or(&self.config.model)
            .to_string();
        let temperature = temperature.unwrap_or(self.config.temperature);
        let max_tokens = max_tokens.unwrap_or(self.config.max_tokens);
        let transports = if self.is_cc_switch_proxy() {
            vec!["responses", "chat"]
        } else {
            vec!["chat", "responses"]
        };
        let mut last_error: Option<anyhow::Error> = None;
        for (index, transport) in transports.iter().enumerate() {
            if let Some(until) = self.transport_suppressed_until.get(*transport) {
                if Instant::now() < *until {
                    continue;
                }
            }
            let result = match *transport {
                "chat" => self.complete_via_chat(messages, &model, temperature, max_tokens).await,
                _ => self.complete_via_responses(messages, &model, temperature, max_tokens).await,
            };
            match result {
                Ok(text) => {
                    self.retryable_failure_streak = 0;
                    self.cooldown_until = None;
                    self.cooldown_reason.clear();
                    return Ok(text);
                }
                Err(error) => {
                    if *transport == "chat" && self.is_cc_switch_proxy() && should_fallback_transport(&error) {
                        self.transport_suppressed_until.insert(
                            "chat".to_string(),
                            Instant::now() + Duration::from_secs(600),
                        );
                        crate::util::warn(&format!("聊天接口 chat 暂时熔断 600 秒：{error}"));
                    }
                    if index < transports.len() - 1 && should_fallback_transport(&error) {
                        crate::util::warn(&format!(
                            "聊天接口 {transport} 不稳定，切换到 {}：{error}",
                            transports[index + 1]
                        ));
                    } else {
                        last_error = Some(error);
                        break;
                    }
                    last_error = Some(error);
                }
            }
        }
        let last_error = last_error.unwrap_or_else(|| anyhow::anyhow!("unknown chat backend error"));
        if is_retryable_error(&last_error) {
            self.retryable_failure_streak += 1;
            if self.retryable_failure_streak >= self.config.failure_cooldown_threshold {
                self.cooldown_until = Some(Instant::now() + Duration::from_millis(self.config.failure_cooldown_ms));
                self.cooldown_reason = last_error.to_string();
            }
        } else {
            self.retryable_failure_streak = 0;
        }
        Err(last_error)
    }

    fn validate(&self) -> anyhow::Result<()> {
        if self.config.api_base.trim().is_empty() {
            bail!("chat.baseUrl 未配置");
        }
        if self.config.model.trim().is_empty() {
            bail!("chat.model 未配置");
        }
        Ok(())
    }

    fn is_cc_switch_proxy(&self) -> bool {
        let base = self.config.api_base.to_lowercase();
        base.contains("127.0.0.1:15721/v1") || base.contains("localhost:15721/v1")
    }

    fn build_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::from([("Content-Type".to_string(), "application/json".to_string())]);
        if !self.config.api_key.trim().is_empty() {
            headers.insert("Authorization".to_string(), format!("Bearer {}", self.config.api_key.trim()));
        }
        if self.is_cc_switch_proxy() {
            headers.insert("Connection".to_string(), "close".to_string());
        }
        headers
    }

    fn build_model_candidates(&self, model: &str) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        for candidate in std::iter::once(model.to_string()).chain(self.config.failover_models.clone()) {
            if seen.insert(candidate.clone()) {
                result.push(candidate.clone());
            }
            match candidate.as_str() {
                "gpt-5-codex-mini" => {
                    let alias = "gpt-5.1-codex-mini".to_string();
                    if seen.insert(alias.clone()) {
                        result.push(alias);
                    }
                }
                "gpt-5-codex" => {
                    let alias = "gpt-5.1-codex".to_string();
                    if seen.insert(alias.clone()) {
                        result.push(alias);
                    }
                }
                _ => {}
            }
        }
        result
    }

    async fn complete_via_chat(
        &self,
        messages: &[ChatMessage],
        model: &str,
        temperature: f32,
        max_tokens: u32,
    ) -> anyhow::Result<String> {
        let headers = self.build_headers();
        let candidates = self.build_model_candidates(model);
        let mut last_error = None;
        for (index, candidate) in candidates.iter().enumerate() {
            let body = json!({
                "model": candidate,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": false
            });
            match self.request_json("chat/completions", body, &headers).await {
                Ok(payload) => {
                    let text = extract_chat_text(&payload);
                    if !text.is_empty() {
                        return Ok(text);
                    }
                    last_error = Some(anyhow::anyhow!("聊天接口未返回可用文本"));
                }
                Err(error) => {
                    if index < candidates.len() - 1 && should_fallback_transport(&error) {
                        crate::util::warn(&format!(
                            "聊天接口 chat 当前模型 {candidate} 不稳定，切换到 {}：{error}",
                            candidates[index + 1]
                        ));
                    }
                    last_error = Some(error);
                }
            }
        }
        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("chat transport failed")))
    }

    async fn complete_via_responses(
        &self,
        messages: &[ChatMessage],
        model: &str,
        temperature: f32,
        max_tokens: u32,
    ) -> anyhow::Result<String> {
        let headers = self.build_headers();
        let candidates = self.build_model_candidates(model);
        let structured = build_responses_input(messages);
        let flattened = build_flattened_input(messages);
        let mut variants = Vec::new();
        if !structured.is_empty() {
            variants.push(json!({ "input": structured }));
        }
        if !flattened.is_empty() && !self.is_cc_switch_proxy() {
            variants.push(json!({ "input": flattened }));
        }
        if variants.is_empty() {
            bail!("聊天接口未提供可发送内容");
        }
        let mut last_error = None;
        for variant in variants {
            for (index, candidate) in candidates.iter().enumerate() {
                let body = json!({
                    "model": candidate,
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                });
                let mut merged = body.as_object().cloned().unwrap_or_default();
                if let Some(map) = variant.as_object() {
                    for (key, value) in map {
                        merged.insert(key.clone(), value.clone());
                    }
                }
                match self.request_json("responses", Value::Object(merged), &headers).await {
                    Ok(payload) => {
                        let text = extract_responses_text(&payload);
                        if !text.is_empty() {
                            return Ok(text);
                        }
                        last_error = Some(anyhow::anyhow!("聊天接口未返回可用文本"));
                    }
                    Err(error) => {
                        if index < candidates.len() - 1 && should_fallback_transport(&error) {
                            crate::util::warn(&format!(
                                "聊天接口 responses 当前模型 {candidate} 不稳定，切换到 {}：{error}",
                                candidates[index + 1]
                            ));
                        }
                        last_error = Some(error);
                    }
                }
            }
        }
        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("responses transport failed")))
    }

    async fn request_json(
        &self,
        path: &str,
        payload: Value,
        headers: &HashMap<String, String>,
    ) -> anyhow::Result<Value> {
        let url = format!("{}/{}", self.config.api_base.trim_end_matches('/'), path.trim_start_matches('/'));
        let mut attempt = 0;
        loop {
            attempt += 1;
            let mut request = self.http.post(&url);
            for (key, value) in headers {
                request = request.header(key, value);
            }
            let response = request.json(&payload).send().await;
            match response {
                Ok(response) => {
                    let status = response.status().as_u16();
                    if response.status().is_client_error() || response.status().is_server_error() {
                        let text = response.text().await.unwrap_or_default();
                        let error = anyhow::anyhow!("聊天接口返回 HTTP {status}：{}", normalize_error_text(&text));
                        if attempt < self.config.retry_attempts && RETRYABLE_HTTP_STATUS.contains(&status) {
                            crate::util::warn(&format!(
                                "聊天接口请求异常，准备重试（{attempt}/{}）：{error}",
                                self.config.retry_attempts
                            ));
                            tokio::time::sleep(Duration::from_millis(self.config.retry_delay_ms * attempt as u64)).await;
                            continue;
                        }
                        return Err(error);
                    }
                    return response.json::<Value>().await.context("failed to decode backend json");
                }
                Err(error) => {
                    if attempt < self.config.retry_attempts && is_retryable_text(&error.to_string()) {
                        crate::util::warn(&format!(
                            "聊天接口请求异常，准备重试（{attempt}/{}）：{error}",
                            self.config.retry_attempts
                        ));
                        tokio::time::sleep(Duration::from_millis(self.config.retry_delay_ms * attempt as u64)).await;
                        continue;
                    }
                    return Err(error.into());
                }
            }
        }
    }
}

fn normalize_error_text(text: &str) -> String {
    if let Ok(value) = serde_json::from_str::<Value>(text) {
        if let Some(message) = value
            .get("error")
            .and_then(|item| item.get("message"))
            .and_then(Value::as_str)
            .or_else(|| value.get("message").and_then(Value::as_str))
            .or_else(|| value.get("detail").and_then(Value::as_str))
        {
            return message.trim().chars().take(400).collect();
        }
    }
    text.replace(['\r', '\n'], " ").chars().take(240).collect()
}

fn should_fallback_transport(error: &anyhow::Error) -> bool {
    is_retryable_error(error)
        || RESPONSE_FALLBACK_HTTP_STATUS
            .iter()
            .any(|status| error.to_string().contains(&format!("HTTP {status}")))
}

fn is_retryable_error(error: &anyhow::Error) -> bool {
    RETRYABLE_HTTP_STATUS
        .iter()
        .any(|status| error.to_string().contains(&format!("HTTP {status}")))
        || is_retryable_text(&error.to_string())
}

fn is_retryable_text(message: &str) -> bool {
    let message = message.to_lowercase();
    ["timeout", "timed out", "network", "socket", "econnreset", "enotfound", "eai_again"]
        .iter()
        .any(|keyword| message.contains(keyword))
}

fn extract_chat_text(payload: &Value) -> String {
    payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .and_then(|item| item.get("message"))
        .and_then(|item| item.get("content"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn extract_responses_text(payload: &Value) -> String {
    if let Some(text) = payload.get("output_text").and_then(Value::as_str) {
        return text.trim().to_string();
    }
    if let Some(output) = payload.get("output").and_then(Value::as_array) {
        let text = output
            .iter()
            .filter_map(|block| block.get("content").and_then(Value::as_array))
            .flat_map(|items| items.iter())
            .filter_map(|item| {
                item.get("text")
                    .and_then(Value::as_str)
                    .or_else(|| item.get("output_text").and_then(Value::as_str))
                    .or_else(|| item.get("value").and_then(Value::as_str))
            })
            .collect::<String>();
        if !text.trim().is_empty() {
            return text.trim().to_string();
        }
    }
    String::new()
}

fn build_responses_input(messages: &[ChatMessage]) -> Vec<Value> {
    messages
        .iter()
        .filter(|message| !message.content.trim().is_empty())
        .map(|message| {
            json!({
                "role": message.role,
                "content": [
                    {
                        "type": "input_text",
                        "text": message.content.trim()
                    }
                ]
            })
        })
        .collect()
}

fn build_flattened_input(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .filter(|message| !message.content.trim().is_empty())
        .map(|message| format!("{}:\n{}", message.role.to_uppercase(), message.content.trim()))
        .collect::<Vec<_>>()
        .join("\n\n")
}
