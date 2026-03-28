# OlivOSAIChatAssassin

这个仓库已改成 `NapCat` 版，不再依赖 `OlivOS`。

当前实现思路：

- 保留原项目的“群里拟人插话”目标
- 人设改成 `Cain（该隐）` 的管理局世界观设定
- 主动插话判定仍保留原仓库逻辑：`@`、关键词、随机概率先决定是否进入回复流程，最终是否开口仍由人格 prompt 决定
- 模型调用层补上 CainBot 第 6 点里的关键能力：`chat/completions` / `responses` 双通道兼容、自动回退、重试、冷却

## 运行

```bash
pip install -e .
python -m OlivOSAIChatAssassin
```

首次运行会在仓库根目录创建：

- `data/config.json`
- `data/memory.json`

## 配置

`data/config.json` 默认包含三部分：

- `napcat`: NapCat HTTP API 和 SSE 事件流配置
- `ai`: OpenAI 兼容接口配置
- `bot`: 群启用范围、人格 prompt、主动回复筛选 prompt 等

关键字段：

- `napcat.base_url`: NapCat HTTP API 地址
- `napcat.event_base_url`: NapCat SSE 地址
- `napcat.event_path`: 一般填 `/_events`
- `ai.api_base`: OpenAI 兼容接口根地址，建议带 `/v1`
- `ai.reply_model`: 主回复模型，可留空继承 `model`
- `ai.memory_model`: 群长期记忆总结模型，可留空继承 `model`
- `bot.enabled_groups`: 生效群号列表，填 `["all"]` 表示所有群
- `bot.persona_prompt`: Cain 的人格与世界观 prompt
- `bot.reply_probability`: 非 `@` 且未命中关键词时，进入回复流程的随机概率

## 行为

- 被 `@` 或命中 `reply_keywords` 时优先回复
- 否则按 `reply_probability` 决定是否进入回复流程
- 决定回复后，再用 Cain 人设 prompt 生成最终回复
- 如果模型输出 `【SKIP】`，则本轮不发言
- 每次成功回复后会异步更新本群长期记忆

## 说明

- `app.json` 仍在仓库里，但现在只是历史文件，不参与运行
- 当前版本重点是把宿主从 `OlivOS` 改到 `NapCat`，并把模型链路升级到更稳的 OpenAI 兼容层
