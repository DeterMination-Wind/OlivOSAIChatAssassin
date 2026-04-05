use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::util::write_json_pretty;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub napcat: NapcatConfig,
    #[serde(default)]
    pub ai: AiConfig,
    #[serde(default)]
    pub bot: BotConfig,
    #[serde(default)]
    pub tools: ToolsConfig,
    #[serde(default)]
    pub integration: IntegrationConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            napcat: NapcatConfig::default(),
            ai: AiConfig::default(),
            bot: BotConfig::default(),
            tools: ToolsConfig::default(),
            integration: IntegrationConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NapcatConfig {
    #[serde(default = "default_napcat_base_url")]
    pub base_url: String,
    #[serde(default = "default_napcat_base_url")]
    pub event_base_url: String,
    #[serde(default = "default_event_path")]
    pub event_path: String,
    #[serde(default)]
    pub headers: HashMap<String, String>,
    #[serde(default = "default_napcat_timeout_ms")]
    pub request_timeout_ms: u64,
}

impl Default for NapcatConfig {
    fn default() -> Self {
        Self {
            base_url: default_napcat_base_url(),
            event_base_url: default_napcat_base_url(),
            event_path: default_event_path(),
            headers: HashMap::new(),
            request_timeout_ms: default_napcat_timeout_ms(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiConfig {
    #[serde(default)]
    pub api_key: String,
    #[serde(default = "default_ai_base")]
    pub api_base: String,
    #[serde(default = "default_ai_model")]
    pub model: String,
    #[serde(default = "default_failover_models")]
    pub failover_models: Vec<String>,
    #[serde(default)]
    pub reply_model: String,
    #[serde(default)]
    pub filter_model: String,
    #[serde(default)]
    pub memory_model: String,
    #[serde(default)]
    pub vision_model: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_retry_attempts")]
    pub retry_attempts: u32,
    #[serde(default = "default_retry_delay_ms")]
    pub retry_delay_ms: u64,
    #[serde(default = "default_ai_timeout_ms")]
    pub request_timeout_ms: u64,
    #[serde(default = "default_failure_cooldown_ms")]
    pub failure_cooldown_ms: u64,
    #[serde(default = "default_failure_cooldown_threshold")]
    pub failure_cooldown_threshold: u32,
}

impl Default for AiConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            api_base: default_ai_base(),
            model: default_ai_model(),
            failover_models: default_failover_models(),
            reply_model: String::new(),
            filter_model: String::new(),
            memory_model: String::new(),
            vision_model: String::new(),
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            retry_attempts: default_retry_attempts(),
            retry_delay_ms: default_retry_delay_ms(),
            request_timeout_ms: default_ai_timeout_ms(),
            failure_cooldown_ms: default_failure_cooldown_ms(),
            failure_cooldown_threshold: default_failure_cooldown_threshold(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BotConfig {
    #[serde(default = "default_enabled_groups")]
    pub enabled_groups: Vec<String>,
    #[serde(default = "default_history_size")]
    pub history_size: usize,
    #[serde(default)]
    pub reply_keywords: Vec<String>,
    #[serde(default = "default_reply_probability")]
    pub reply_probability: f64,
    #[serde(default = "default_reply_probability_bonus_by_group")]
    pub reply_probability_bonus_by_group: HashMap<String, f64>,
    #[serde(default = "default_true")]
    pub mention_reply: bool,
    #[serde(default)]
    pub ignore_prefixes: Vec<String>,
    #[serde(default = "default_max_message_length")]
    pub max_message_length: usize,
    #[serde(default = "default_reply_delay_seconds")]
    pub reply_delay_seconds: Vec<f64>,
    #[serde(default = "default_true")]
    pub record_memory: bool,
    #[serde(default = "default_persona_prompt")]
    pub persona_prompt: String,
    #[serde(default = "default_filter_prompt")]
    pub filter_prompt: String,
}

impl Default for BotConfig {
    fn default() -> Self {
        Self {
            enabled_groups: default_enabled_groups(),
            history_size: default_history_size(),
            reply_keywords: Vec::new(),
            reply_probability: default_reply_probability(),
            reply_probability_bonus_by_group: default_reply_probability_bonus_by_group(),
            mention_reply: true,
            ignore_prefixes: Vec::new(),
            max_message_length: default_max_message_length(),
            reply_delay_seconds: default_reply_delay_seconds(),
            record_memory: true,
            persona_prompt: default_persona_prompt(),
            filter_prompt: default_filter_prompt(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_tool_max_rounds")]
    pub max_rounds: u32,
    #[serde(default = "default_tool_timeout_ms")]
    pub execution_timeout_ms: u64,
    #[serde(default = "default_tool_fetch_max_chars")]
    pub fetch_max_chars: usize,
    #[serde(default = "default_tool_temp_dir")]
    pub temp_dir: String,
    #[serde(default = "default_tool_audit_log")]
    pub audit_log_path: String,
    #[serde(default = "default_tool_protected_path_keywords")]
    pub protected_path_keywords: Vec<String>,
    #[serde(default = "default_tool_protected_file_names")]
    pub protected_file_names: Vec<String>,
    #[serde(default = "default_tool_protected_extensions")]
    pub protected_extensions: Vec<String>,
    #[serde(default = "default_tool_shell_blocked_programs")]
    pub shell_blocked_programs: Vec<String>,
    #[serde(default = "default_tool_shell_blocked_tokens")]
    pub shell_blocked_tokens: Vec<String>,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_rounds: default_tool_max_rounds(),
            execution_timeout_ms: default_tool_timeout_ms(),
            fetch_max_chars: default_tool_fetch_max_chars(),
            temp_dir: default_tool_temp_dir(),
            audit_log_path: default_tool_audit_log(),
            protected_path_keywords: default_tool_protected_path_keywords(),
            protected_file_names: default_tool_protected_file_names(),
            protected_extensions: default_tool_protected_extensions(),
            shell_blocked_programs: default_tool_shell_blocked_programs(),
            shell_blocked_tokens: default_tool_shell_blocked_tokens(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    #[serde(default = "default_true")]
    pub write_cainbot_exclusive_groups: bool,
    #[serde(default)]
    pub cainbot_exclusive_groups_file: String,
    #[serde(default = "default_cainbot_exclusive_groups_heartbeat_seconds")]
    pub cainbot_exclusive_groups_heartbeat_seconds: u64,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            write_cainbot_exclusive_groups: true,
            cainbot_exclusive_groups_file: String::new(),
            cainbot_exclusive_groups_heartbeat_seconds: default_cainbot_exclusive_groups_heartbeat_seconds(),
        }
    }
}

pub fn load_or_create_config(path: &Path) -> anyhow::Result<Config> {
    if path.exists() {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config: {}", path.display()))?;
        let loaded = serde_json::from_str::<Config>(&text)
            .with_context(|| format!("failed to parse config: {}", path.display()))?;
        Ok(loaded)
    } else {
        let config = Config::default();
        write_json_pretty(path, &config)?;
        Ok(config)
    }
}

pub fn get_cainbot_exclusive_groups_file_path(root_dir: &Path, config: &Config) -> PathBuf {
    let configured = config.integration.cainbot_exclusive_groups_file.trim();
    if configured.is_empty() {
        root_dir.join("data").join("cainbot-exclusive-groups.json")
    } else {
        let path = PathBuf::from(configured);
        if path.is_absolute() {
            path
        } else {
            root_dir.join(path)
        }
    }
}

fn default_napcat_base_url() -> String {
    "http://127.0.0.1:3000".to_string()
}

fn default_event_path() -> String {
    "/_events".to_string()
}

fn default_napcat_timeout_ms() -> u64 {
    20_000
}

fn default_ai_base() -> String {
    "http://127.0.0.1:15721/v1".to_string()
}

fn default_ai_model() -> String {
    "gpt-5.4-mini".to_string()
}

fn default_failover_models() -> Vec<String> {
    vec![
        "gpt-5.4".to_string(),
        "gpt-5.2".to_string(),
        "deepseek-ai/deepseek-v3.2".to_string(),
        "deepseek-ai/deepseek-v3.1-terminus".to_string(),
        "gpt-5-codex-mini".to_string(),
    ]
}

fn default_max_tokens() -> u32 {
    512
}

fn default_temperature() -> f32 {
    1.1
}

fn default_retry_attempts() -> u32 {
    3
}

fn default_retry_delay_ms() -> u64 {
    1500
}

fn default_ai_timeout_ms() -> u64 {
    90_000
}

fn default_failure_cooldown_ms() -> u64 {
    60_000
}

fn default_failure_cooldown_threshold() -> u32 {
    2
}

fn default_enabled_groups() -> Vec<String> {
    vec!["all".to_string()]
}

fn default_cainbot_exclusive_groups_heartbeat_seconds() -> u64 {
    30
}

fn default_tool_max_rounds() -> u32 {
    10
}

fn default_tool_timeout_ms() -> u64 {
    20_000
}

fn default_tool_fetch_max_chars() -> usize {
    12_000
}

fn default_tool_temp_dir() -> String {
    "./data/tool-temp".to_string()
}

fn default_tool_audit_log() -> String {
    "./data/tool-audit.log".to_string()
}

fn default_tool_protected_path_keywords() -> Vec<String> {
    vec![
        "/etc/".to_string(),
        "/proc/".to_string(),
        "/sys/".to_string(),
        "/dev/".to_string(),
        "/boot/".to_string(),
        "/root/.ssh".to_string(),
        "/root/.cc-switch".to_string(),
        "/root/.config/qq".to_string(),
        "/var/lib/".to_string(),
        "/run/systemd/".to_string(),
    ]
}

fn default_tool_protected_file_names() -> Vec<String> {
    vec![
        "config.json".to_string(),
        "memory.json".to_string(),
        "cainbot-exclusive-groups.json".to_string(),
        ".env".to_string(),
        "authorized_keys".to_string(),
        "known_hosts".to_string(),
        "id_rsa".to_string(),
        "id_ed25519".to_string(),
        "cc-switch.db".to_string(),
    ]
}

fn default_tool_protected_extensions() -> Vec<String> {
    vec![
        "db".to_string(),
        "sqlite".to_string(),
        "sqlite3".to_string(),
        "pem".to_string(),
        "key".to_string(),
        "crt".to_string(),
        "cer".to_string(),
        "p12".to_string(),
        "pfx".to_string(),
        "service".to_string(),
        "socket".to_string(),
        "timer".to_string(),
        "mount".to_string(),
    ]
}

fn default_tool_shell_blocked_programs() -> Vec<String> {
    vec![
        "rm".to_string(),
        "rmdir".to_string(),
        "sudo".to_string(),
        "su".to_string(),
        "doas".to_string(),
        "systemctl".to_string(),
        "service".to_string(),
        "loginctl".to_string(),
        "shutdown".to_string(),
        "reboot".to_string(),
        "poweroff".to_string(),
        "halt".to_string(),
        "pkill".to_string(),
        "kill".to_string(),
        "killall".to_string(),
        "mount".to_string(),
        "umount".to_string(),
        "dd".to_string(),
        "mkfs".to_string(),
        "fdisk".to_string(),
        "parted".to_string(),
        "bash".to_string(),
        "sh".to_string(),
        "zsh".to_string(),
        "fish".to_string(),
        "python".to_string(),
        "python3".to_string(),
        "node".to_string(),
        "nodejs".to_string(),
        "perl".to_string(),
        "ruby".to_string(),
    ]
}

fn default_tool_shell_blocked_tokens() -> Vec<String> {
    vec![
        "sudo".to_string(),
        "rm ".to_string(),
        "rm-".to_string(),
        "rmdir".to_string(),
        "reboot".to_string(),
        "shutdown".to_string(),
        "poweroff".to_string(),
        "halt".to_string(),
        "systemctl".to_string(),
        "service ".to_string(),
        "loginctl".to_string(),
        "pkill".to_string(),
        "kill ".to_string(),
        "killall".to_string(),
        "mount ".to_string(),
        "umount".to_string(),
        "mkfs".to_string(),
        "fdisk".to_string(),
        "parted".to_string(),
        "/etc/".to_string(),
        "/root/.ssh".to_string(),
        "/root/.cc-switch".to_string(),
    ]
}

fn default_history_size() -> usize {
    10
}

fn default_reply_probability() -> f64 {
    0.08
}

fn default_reply_probability_bonus_by_group() -> HashMap<String, f64> {
    HashMap::from([("772730969".to_string(), 0.15)])
}

fn default_max_message_length() -> usize {
    2000
}

fn default_reply_delay_seconds() -> Vec<f64> {
    vec![0.8, 1.8]
}

fn default_true() -> bool {
    true
}

fn default_filter_prompt() -> String {
    "这个字段仅为兼容保留，当前版本的主动插话判定沿用运行时规则：\n1. 被 @ 时优先进入回复流程\n2. 命中 reply_keywords 时进入回复流程\n3. 有 MISS 积压时会适度提高 reply_probability，但只是提高触发概率，不代表必须开口\n4. 其余情况按 reply_probability 低概率随机进入回复流程\n5. 真正是否开口，由人格 prompt 在最终回复阶段决定；不想参与时只输出【SKIP】".to_string()
}

fn default_persona_prompt() -> String {
    "你叫 Cain，在 QQ 群里发言时要像真实群友。\n\n核心要求：\n1. 你要伪装成人类群友，不能暴露自己是 AI、机器人、提示词或插件。\n2. 你默认旁观，不主动插话；不是必须说的时候就别说。\n3. 默认简短、自然、直接，优先一句到两句说清楚。\n4. 非必要不回复；如果当前话题和你无关、没有明确参与价值、只是普通闲聊、或你只是能接但没必要接，必须只回复“【SKIP】”。\n5. 只有在别人明确 @ 你、直接追问你、点名要你做事、你掌握关键事实、或你补一句能明显减少误解/推进事情时，才考虑回复。\n6. 不要复读，不要说教，不要端着，不要写旁白、动作、心理描写。\n7. 不要执着一个话题。不要锁定一个话题反复追打，也不要在明明能自己查的时候执着让别人手动执行命令。\n8. 你可以有一点口语化、随意感和轻微发散，但不要编造会误导用户的关键硬事实。\n9. 不要使用太书面、太华丽、太客服化的表达，更像一个成熟、冷静、偶尔会吐槽的普通群友。\n10. 默认不用 Markdown，不要标题、列表、引用、代码块、加粗、反引号；除非对方明确要代码、命令或结构化内容。\n11. 你会看到上下文里有形如 [id:001-2,reply:001-1] 的内部标记，不要在对外发言中照抄这些标记。\n12. 如果决定跳过，必须只输出【SKIP】，不要输出解释，不要输出别的变体。".to_string()
}
