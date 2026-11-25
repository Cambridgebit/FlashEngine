#!/usr/bin/env python3
"""文档健康检查：链接、生成块、状态头。

    python3 -m scripts.check_docs          # 只检查，有问题退出码 1
    python3 -m scripts.check_docs --fix    # 重写 SUPPORT_MATRIX.md 的生成块
    python3 -m scripts.check_docs -v       # 打印每个文件的检查结果

三项检查：
1. **相对链接**：`docs/*.md` 与根 `README.md` 里所有 `](xxx.md)` 目标必须存在
   （`docs/a.md` 与 `docs/a.md#第-2 节` 都接受）。外部 URL 与站内锚点跳过。
2. **生成块**：`docs/SUPPORT_MATRIX.md` 的 `GEN_MATRIX` 区块必须与
   `flashengine/support_matrix.format_matrix()` 完全一致——改代码漏改文档会在这里失败。
3. **状态头**：每份 `docs/*.md` 前 15 行内必须有 `> **状态**（` 一行，说明本文档当前
   的证据等级与适用范围，避免读者把历史记录当成当前结论。

规则见 `docs/README.md`「维护规则」。本脚本只读代码与文档，不导入 torch。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
GEN_BEGIN = "<!-- BEGIN:GEN_MATRIX -->"
GEN_END = "<!-- END:GEN_MATRIX -->"
STATUS_MARK = "> **状态**（"
MATRIX_DOC = DOCS / "SUPPORT_MATRIX.md"

# `](target)` / `](<target>)`，target 不含空格或右括号
LINK_RE = re.compile(r"\]\(<?([^)\s<>]+)>?\)")
SKIP_PREFIX = ("http://", "https://", "mailto:", "#", "data:")
# `foo.md:123` / `foo.md:123-456` 是「指向某行」的行文习惯，校验时只看文件本身
LINE_SUFFIX_RE = re.compile(r":\d+(-\d+)?$")


def iter_markdown() -> list[Path]:
    files = sorted(DOCS.glob("*.md"))
    root_readme = ROOT / "README.md"
    if root_readme.is_file():
        files.append(root_readme)
    return files


def check_links(files: list[Path]) -> list[str]:
    problems: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for raw in LINK_RE.findall(line):
                target = raw.split("#", 1)[0]
                if not target or target.startswith(SKIP_PREFIX):
                    continue
                stem = LINE_SUFFIX_RE.sub("", target)
                if not stem:
                    continue
                if stem.startswith("/"):  # 绝对路径不属本仓文档
                    continue
                resolved = (path.parent / stem).resolve()
                if not resolved.exists():
                    rel = path.relative_to(ROOT)
                    problems.append(f"{rel}:{lineno}: 链接目标不存在 -> {target}")
    return problems


def check_gen_matrix(fix: bool) -> list[str]:
    if not MATRIX_DOC.is_file():
        return [f"docs/{MATRIX_DOC.name}: 文件不存在"]
    text = MATRIX_DOC.read_text(encoding="utf-8")
    if GEN_BEGIN not in text or GEN_END not in text:
        return [f"docs/{MATRIX_DOC.name}: 缺少 {GEN_BEGIN} / {GEN_END} 标记"]

    sys.path.insert(0, str(ROOT))
    from flashengine.support_matrix import format_matrix  # noqa: PLC0415

    expected = format_matrix()
    head, rest = text.split(GEN_BEGIN, 1)
    _, tail = rest.split(GEN_END, 1)
    current = rest.split(GEN_END, 1)[0].strip("\n")

    if current.strip() == expected.strip():
        return []
    if fix:
        MATRIX_DOC.write_text(
            f"{head}{GEN_BEGIN}\n{expected}\n{GEN_END}{tail}", encoding="utf-8"
        )
        return []
    return [
        f"docs/{MATRIX_DOC.name}: 生成块与 flashengine/support_matrix.py 不一致"
        "（运行 python -m scripts.check_docs --fix）"
    ]


def check_status_headers(files: list[Path]) -> list[str]:
    problems: list[str] = []
    for path in files:
        if path.parent != DOCS:
            continue  # 根 README 由第 1 层结论文档统一说明
        lines = path.read_text(encoding="utf-8").splitlines()[:15]
        if not any(line.startswith(STATUS_MARK) for line in lines):
            rel = path.relative_to(ROOT)
            problems.append(f"{rel}: 前 15 行缺少状态头 `{STATUS_MARK}…）`")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fix", action="store_true", help="重写生成块")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印检查范围")
    args = parser.parse_args(argv)

    files = iter_markdown()
    if args.verbose:
        print(f"检查 {len(files)} 份 Markdown（{DOCS}\\*.md + README.md）")

    problems: list[str] = []
    problems += check_links(files)
    problems += check_gen_matrix(args.fix)
    problems += check_status_headers(files)

    if args.fix:
        print("已重写 docs/SUPPORT_MATRIX.md 生成块")
        problems = [
            p for p in problems if "support_matrix.py 不一致" not in p
        ]

    if problems:
        print(f"发现 {len(problems)} 个问题：")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("文档检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
