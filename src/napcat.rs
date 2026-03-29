use std::time::Duration;

use anyhow::{Context, bail};
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::Value;

use crate::config::NapcatConfig;
use crate::util::info;

#[derive(Clone)]
pub struct NapcatClient {
    http: Client,
    config: NapcatConfig,
}

impl NapcatClient {
    pub fn new(config: NapcatConfig) -> anyhow::Result<Self> {
        let http = Client::builder()
            .timeout(Duration::from_millis(config.request_timeout_ms.max(5000)))
            .build()?;
        Ok(Self { http, config })
    }

    pub async fn send_group_message(
        &self,
        group_id: &str,
        text: &str,
        reply_to_message_id: Option<&str>,
    ) -> anyhow::Result<()> {
        let mut message = Vec::<Value>::new();
        if let Some(id) = reply_to_message_id {
            message.push(serde_json::json!({"type":"reply","data":{"id":id}}));
        }
        message.push(serde_json::json!({"type":"text","data":{"text":text}}));
        self.call(
            "send_group_msg",
            serde_json::json!({"group_id": group_id, "message": message}),
        )
        .await?;
        Ok(())
    }

    pub async fn run_event_loop<F, Fut>(&self, mut on_event: F) -> anyhow::Result<()>
    where
        F: FnMut(Value) -> Fut,
        Fut: std::future::Future<Output = anyhow::Result<()>>,
    {
        let mut backoff = 2u64;
        loop {
            match self.run_event_stream(&mut on_event).await {
                Ok(()) => backoff = 2,
                Err(error) => {
                    crate::util::warn(&format!("NapCat SSE 连接断开: {error}"));
                    tokio::time::sleep(Duration::from_secs(backoff)).await;
                    backoff = (backoff * 2).min(30);
                }
            }
        }
    }

    async fn run_event_stream<F, Fut>(&self, on_event: &mut F) -> anyhow::Result<()>
    where
        F: FnMut(Value) -> Fut,
        Fut: std::future::Future<Output = anyhow::Result<()>>,
    {
        let url = join_url(&self.config.event_base_url, &self.config.event_path);
        let mut request = self.http.get(url).header("Accept", "text/event-stream");
        for (key, value) in &self.config.headers {
            request = request.header(key, value);
        }
        let response = request
            .send()
            .await
            .context("failed to connect NapCat SSE")?;
        if response.status().is_client_error() || response.status().is_server_error() {
            bail!("NapCat SSE 返回 HTTP {}", response.status());
        }
        info("NapCat SSE 已连接。");
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(pos) = buffer.find("\n\n") {
                let block = buffer[..pos].to_string();
                buffer = buffer[pos + 2..].to_string();
                if let Some(event) = parse_sse_block(&block) {
                    on_event(event).await?;
                }
            }
        }
        bail!("NapCat SSE 连接已结束")
    }

    async fn call(&self, action: &str, payload: Value) -> anyhow::Result<Value> {
        let url = join_url(&self.config.base_url, action);
        let mut request = self.http.post(url).header("Content-Type", "application/json");
        for (key, value) in &self.config.headers {
            request = request.header(key, value);
        }
        let response = request.json(&payload).send().await?;
        if response.status().is_client_error() || response.status().is_server_error() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            bail!("NapCat API {action} 返回 HTTP {status}: {}", truncate(&text, 300));
        }
        let payload: Value = response.json().await?;
        if let Some(status) = payload.get("status").and_then(Value::as_str) {
            if status != "ok" {
                bail!("NapCat API {action} 失败: {status}");
            }
        }
        if let Some(retcode) = payload.get("retcode").and_then(Value::as_i64) {
            if retcode != 0 {
                bail!("NapCat API {action} retcode={retcode}");
            }
        }
        Ok(payload.get("data").cloned().unwrap_or(payload))
    }
}

fn join_url(base: &str, path: &str) -> String {
    format!("{}/{}", base.trim_end_matches('/'), path.trim_start_matches('/'))
}

fn parse_sse_block(block: &str) -> Option<Value> {
    let mut lines = Vec::new();
    for line in block.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            lines.push(rest.trim_start().to_string());
        }
    }
    if lines.is_empty() {
        return None;
    }
    match serde_json::from_str::<Value>(&lines.join("\n")) {
        Ok(value) => Some(value),
        Err(_) => {
            crate::util::warn("收到无法解析的 SSE 数据。");
            None
        }
    }
}

fn truncate(text: &str, limit: usize) -> String {
    text.chars().take(limit).collect()
}
