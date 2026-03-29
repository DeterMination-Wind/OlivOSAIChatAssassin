import os
import subprocess
import sys


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    binary_names = (
        'napcat-aichat-assassin-rs.exe',
        'napcat-aichat-assassin-rs',
    )
    candidate_dirs = (
        repo_root,
        os.path.join(repo_root, 'target', 'release'),
        os.path.join(repo_root, 'target', 'debug'),
    )

    for base_dir in candidate_dirs:
        for binary_name in binary_names:
            candidate = os.path.join(base_dir, binary_name)
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
