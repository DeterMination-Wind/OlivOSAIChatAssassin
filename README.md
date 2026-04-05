# NapCatAIChatAssassin Rust

[![Rust](https://img.shields.io/badge/rust-runtime-orange)](Cargo.toml)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)

NapCatAIChatAssassin 的 Rust 运行时仓库。项目已从旧的 OlivOS 插件形态迁移为 NapCat HTTP + SSE 常驻服务，Rust 是正式运行与发布入口。

## 功能概览

- 拟人化群聊回复
- `@` 提及、关键词、概率触发三种回复机制
- OpenAI 兼容接口，多模型自动回退
- 群聊长期记忆
- 工具调用与受限执行
- 与 CainBot 的互斥群文件联动
- MISSION 工作流：把长期任务、延时任务和本机调查任务拆成多步执行后再汇总回复

## 获取源码

```bash
git clone https://github.com/DeterMination-Wind/OlivOSAIChatAssassin.git
cd OlivOSAIChatAssassin
```

## 环境要求

- Rust stable
- NapCat OneBot HTTP + SSE
- OpenAI 兼容接口

## 开发运行

```bash
cargo run
```

## 正式构建

```bash
cargo build --release
```

Windows 打包：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-release.ps1
```

## 运行时文件

首次运行会在项目根目录创建：

- `data/config.json`
- `data/memory.json`

仓库只保留源码与知识库模板；`data/` 下其他运行期文件默认不提交。

## 配置说明

编辑 `data/config.json`：

### NapCat

```json
{
  "napcat": {
    "base_url": "http://127.0.0.1:3000",
    "event_base_url": "http://127.0.0.1:3000",
    "event_path": "/_events"
  }
}
```

### AI

```json
{
  "ai": {
    "api_base": "OpenAI 兼容接口地址",
    "model": "gpt-5.4-mini",
    "failover_models": ["gpt-5.4", "gpt-5.3-codex", "gpt-5.4-mini", "gpt-5.2"]
  }
}
```

### 机器人行为

```json
{
  "bot": {
    "enabled_groups": ["群号列表"],
    "reply_probability": 0.3,
    "persona_prompt": "角色设定 prompt"
  }
}
```

### CainBot 联动

```json
{
  "integration": {
    "write_cainbot_exclusive_groups": true,
    "cainbot_exclusive_groups_file": "./data/cainbot-exclusive-groups.json"
  }
}
```

## 工作流说明

### 普通聊天

- 被 `@` 时优先回复
- 命中关键词时触发
- 也可按概率触发主动回复

### MISSION 工作流

- 当请求属于长期任务、延时任务、事实调查、或本机环境相关任务时，模型会先产出 `[MISSION]` 计划
- 系统按步骤并行/串行执行，再汇总成最终聊天回复
- 相关知识模板位于 `data/Knowledge/mission-workflow.json`

## 入口与发布

- 正式运行入口：`cargo run`
- 正式发布入口：`cargo build --release`
- Windows 发布脚本：`scripts/build-release.ps1`
- Python 入口仅保留兼容与迁移用途，不再是正式运行入口

## 许可证

[AGPL-3.0](LICENSE)
