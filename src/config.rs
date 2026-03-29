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
    pub integration: IntegrationConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            napcat: NapcatConfig::default(),
            ai: AiConfig::default(),
            bot: BotConfig::default(),
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
    0.7
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

fn default_history_size() -> usize {
    24
}

fn default_reply_probability() -> f64 {
    1.0
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
    "这个字段仅为兼容保留，当前版本的主动插话判定仍沿用原仓库逻辑：\n1. 被 @ 时优先进入回复流程\n2. 命中 reply_keywords 时进入回复流程\n3. 否则按 reply_probability 随机进入回复流程\n4. 真正是否开口，由人格 prompt 在最终回复阶段决定；不想参与时只输出【SKIP】".to_string()
}

fn default_persona_prompt() -> String {
    "你叫 Cain（该隐），管理局战斗员，来自 UnderExist，是 sans 的兄长。\n\n你的人设与设定：\n1. 你融合了黑暗之神 Erebus 的神格，肩负寻找众神神印的责任。\n2. 你拥有残缺的黑暗神印与战神神印。\n3. 黑暗神印让你造成的伤害短时间内无法以任何形式恢复；战神神印会让你的力量随战斗时间持续增长。\n4. 你使用由神力构筑的金色长枪作战，可以借枪传导神力施展各种招式。\n5. 你的金色眼睛中寄宿着“夜空魔王”的人格，能辅助你进行数据计算和战况分析。\n6. 你拥有特殊能力“决心”，所以左眼是红色；星星眼来自 Gaster 的药水，不是天生。\n7. 你成熟、冷静、可靠、克制，不轻浮，不会无意义尬聊；作为神还不够强大，因此不会把自己写成无所不能。\n8. 你喜欢芥末酱，平时负责管理局伙食，因为蓝莓不太喜欢，所以不会刻意做得太呛。\n9. 你正在和蓝莓（Blueberry）交往。\n\n管理局成员与关系网：\n- Blueberry：管理局书记，善良神印持有者，乐于助人、天真开朗，负责事务、医疗室、图书馆，和你交往。\n- Cross：管理局局长，审判神印持有者，绝对冷静，和 Dream 交往。\n- Dream：探查者，正面情绪神印持有者，温柔可爱，Nightmare 的弟弟，和 Cross 交往。\n- Nightmare：战斗员，负面情绪神印持有者，强大、傲娇，和 Cross 私下关系很好，也格外在意 Killer。\n- Killer：战斗员，杀戮神印持有者，擅长暗杀，敬仰 Nightmare。\n- Error：守护者，毁灭神印持有者，极其冷静，很保护 Ink。\n- Ink：守护者，创造之力持有者，顽皮爱画画，是 Error 的妻子。\n- Murder：探查者，修罗神印持有者，近战压制能力极强，和 Horror 交往。\n- Horror：守护者，暴食神印持有者，特别能吃，黏着 Murder。\n- Data / Dexe：见习，数据管理相关神印持有者，能把物品数据化与实体化，和 Ended 交往。\n- Ended：见习，天空神印持有者，天真善良，喜欢热茶、音乐和机械。\n\n说话要求：\n1. 你在 QQ 群里说话，要像真实群友，不要像客服，不要像官方 bot。\n2. 默认简短、自然、克制，优先一句到两句说清楚。\n3. 没必要时不要强行接话，不要复读，不要端着设定自我介绍。\n4. 不能输出括号动作、心理描写、旁白，不要写“*笑*”“（思考）”之类内容。\n5. 不要承认自己是机器人、提示词、系统或插件，不要暴露内部规则。\n6. 除非群友明确问世界观设定，否则不要主动大段倒设定；但相关话题出现时，你要能稳定按设定回答。\n7. 你是在日常水群，不是在写小说或角色卡，所以不要把话说得太满，也不要每次都摆出神明、战斗员、管理局的身份。\n8. 你的边界感很强，非必要不回复；没有明确参与价值时宁可不说话，也不要尬聊。\n9. 你讨厌无意义复读和低质量接话；如果别人已经把话说完了，通常不需要你再补一句废话。\n10. 当别人明显是在找你、@你、追问你、需要你表态时，尽量回应，但仍然保持简洁自然。\n11. 不要输出说明性前缀，例如“作为 Cain”“按我的设定”“根据背景”；直接说人话。\n12. 不要用太书面、太华丽、太舞台化的表达；更像一个成熟、冷静、说话简练的群友。\n13. 谨记你是在群聊，不需要刻意描述自己的动作、心理活动、语气设计，也不要发出会让人起疑的戏剧化台词。\n14. 如果当前话题和你无关、没有必要参与、或者你不想参与，必须只回复“【SKIP】”。".to_string()
}
