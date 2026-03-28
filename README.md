# NapCat AI Chat Assassin

基于 NapCat 的智能群聊对话机器人，具有拟人化的回复能力和长期记忆功能。

## 概述

这是一个基于 [lunzhiPenxil/OlivOSAIChatAssassin](https://github.com/lunzhiPenxil/OlivOSAIChatAssassin) 的 NapCat 分支版本。项目已从 OlivOS 插件形态迁移为 NapCat HTTP + SSE 常驻服务，提供更稳定可靠的群聊AI对话体验。

## 特性

- 🎭 **角色扮演**: 采用 Cain（该隐）的管理局世界观设定，具有独特的人格
- 🤖 **智能回复**: 支持 @ 提及、关键词触发和随机概率触发多种回复方式
- 🔄 **模型容错**: 多模型自动回退机制，确保服务稳定性
- 💾 **长期记忆**: 自动维护群聊上下文记忆，提供连贯的对话体验
- ⚡ **高性能**: 基于 NapCat 的 HTTP + SSE 架构，响应迅速

## 快速开始

### 安装依赖

```bash
pip install -e .
```

### 启动服务

```bash
python -m OlivOSAIChatAssassin
```

首次运行会在项目根目录创建配置文件和记忆文件：
- `data/config.json` - 配置文件
- `data/memory.json` - 记忆数据

## 配置说明

编辑 `data/config.json` 进行配置：

### NapCat 配置
```json
{
  "napcat": {
    "base_url": "NapCat HTTP API 地址",
    "event_base_url": "NapCat SSE 地址",
    "event_path": "/_events"
  }
}
```

### AI 模型配置
```json
{
  "ai": {
    "api_base": "OpenAI 兼容接口地址（建议包含 /v1）",
    "model": "gpt-5.4-mini",
    "failover_models": ["gpt-5.4", "gpt-5.2", "deepseek-ai/deepseek-v3.2"],
    "reply_model": "主回复模型（可选）",
    "memory_model": "记忆总结模型（可选）"
  }
}
```

### 机器人配置
```json
{
  "bot": {
    "enabled_groups": ["群号列表"],
    "persona_prompt": "Cain 角色设定 prompt",
    "reply_probability": 0.3
  }
}
```

## 工作原理

1. **触发机制**:
   - 被 @ 提及时优先回复
   - 命中预设关键词时触发回复
   - 随机概率触发（由 `reply_probability` 控制）

2. **回复生成**:
   - 使用 Cain 角色设定 prompt 生成拟人化回复
   - 模型可输出 `【SKIP】` 跳过本次回复
   - 支持多模型自动回退确保可用性

3. **记忆管理**:
   - 每次成功回复后异步更新群聊长期记忆
   - 记忆用于维持对话连贯性

## 许可证

本项目基于 [AGPL-3.0](LICENSE) 许可证开源。
