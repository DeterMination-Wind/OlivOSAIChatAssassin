use std::path::Path;

use anyhow::Context;
use chrono::Local;
use serde::Serialize;

pub fn now_iso() -> String {
    Local::now().format("%Y-%m-%dT%H:%M:%S%:z").to_string()
}

pub fn info(message: &str) {
    println!("[INFO] {message}");
}

pub fn warn(message: &str) {
    eprintln!("[WARN] {message}");
}

pub fn write_json_pretty<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create parent dir: {}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(value)?;
    std::fs::write(path, text).with_context(|| format!("failed to write file: {}", path.display()))?;
    Ok(())
}

pub fn write_json_pretty_atomic<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create parent dir: {}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(value)?;
    let file_name = path.file_name().and_then(|item| item.to_str()).unwrap_or("data.json");
    let temp_path = path.with_file_name(format!(".{file_name}.tmp-{}", std::process::id()));
    std::fs::write(&temp_path, text)
        .with_context(|| format!("failed to write temp file: {}", temp_path.display()))?;
    if path.exists() {
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to replace existing file: {}", path.display()));
            }
        }
    }
    std::fs::rename(&temp_path, path)
        .with_context(|| format!("failed to replace file: {}", path.display()))?;
    Ok(())
}

pub fn build_message_summary(message: &str) -> String {
    let normalized = message
        .replace('\r', " ")
        .replace('\n', " ")
        .replace("[OP:image]", "[图片]")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let trimmed = normalized.trim();
    if trimmed.is_empty() {
        "(无可读文本)".to_string()
    } else {
        trimmed.chars().take(360).collect()
    }
}

pub fn get_recommend_rank(word1_in: &str, word2_in: &str, gate_rank: i32, rate: f64) -> i32 {
    let word1 = word1_in.to_lowercase();
    let word2 = word2_in.to_lowercase();
    if word1.is_empty() || word2.is_empty() {
        return gate_rank + 1;
    }
    if word1.chars().count() > word2.chars().count() {
        return gate_rank + 2;
    }
    let a: Vec<char> = word1.chars().collect();
    let b: Vec<char> = word2.chars().collect();
    let a_len = a.len();
    let b_len = b.len();
    let find_flag = if word2.contains(&word1) { 0 } else { 1 };

    let mut dp1 = vec![vec![0usize; a_len + 1]; b_len + 1];
    for i in 1..=a_len {
        for j in 1..=b_len {
            if a[i - 1] == b[j - 1] {
                dp1[j][i] = dp1[j - 1][i - 1] + 1;
            } else {
                dp1[j][i] = dp1[j - 1][i].max(dp1[j][i - 1]);
            }
        }
    }
    let lcs_rank = dp1[b_len][a_len] as i32;

    let mut dp2 = vec![vec![0usize; a_len + 1]; b_len + 1];
    for (i, item) in dp2[0].iter_mut().enumerate() {
        *item = i;
    }
    for (j, row) in dp2.iter_mut().enumerate() {
        row[0] = j;
    }
    for i in 1..=a_len {
        for j in 1..=b_len {
            if a[i - 1] == b[j - 1] {
                dp2[j][i] = dp2[j - 1][i - 1];
            } else {
                dp2[j][i] = (dp2[j - 1][i - 1].min(dp2[j - 1][i]).min(dp2[j][i - 1])) + 1;
            }
        }
    }
    let distance_rank = dp2[b_len][a_len] as i32;
    let mut rank = find_flag * ((b_len as i32) * ((a_len as i32) - lcs_rank) + distance_rank + 1);
    rank = (((rank * rank) / (a_len as i32)).max(0)) / (b_len as i32);
    if rank >= ((a_len * b_len) as f64 * rate) as i32 {
        rank += gate_rank;
    }
    rank
}

pub fn get_recommend_match(rank: i32, gate_rank: i32) -> bool {
    rank < gate_rank
}
