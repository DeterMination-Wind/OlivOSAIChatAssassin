#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -x "./napcat-aichat-assassin-rs" ]; then
    exec ./napcat-aichat-assassin-rs
fi

if [ -x "./target/release/napcat-aichat-assassin-rs" ]; then
    exec ./target/release/napcat-aichat-assassin-rs
fi

exec /usr/bin/python3 -m OlivOSAIChatAssassin
