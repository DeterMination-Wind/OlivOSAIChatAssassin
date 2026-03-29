import os
import subprocess
import sys


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    release_binary = os.path.join(repo_root, 'target', 'release', 'napcat-aichat-assassin-rs.exe')
    debug_binary = os.path.join(repo_root, 'target', 'debug', 'napcat-aichat-assassin-rs.exe')

    for candidate in (release_binary, debug_binary):
        if os.path.exists(candidate):
            raise SystemExit(subprocess.call([candidate]))

    message = (
        'Python 入口已降级为兼容跳板，当前仓库的正式运行入口是 Rust 二进制。\n'
        '请先执行 `cargo build --release` 或 `cargo run`。\n'
        f'仓库路径: {repo_root}'
    )
    print(message, file=sys.stderr)
    raise SystemExit(1)


if __name__ == '__main__':
    main()
