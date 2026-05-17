#!/usr/bin/env python3
r"""Post-process a latexdiff output to suppress strike-through deletions,
leaving only the new (blue) text visible.

latexdiff wraps deletions in ``\DIFdelbegin ... \DIFdelend`` and insertions in
``\DIFaddbegin ... \DIFaddend``.  Removing the former produces a clean
"new text only" PDF suitable for a project page.
"""

import argparse
import re
from pathlib import Path


# Match \DIFdelbegin not preceded by '\providecommand{' or '\newcommand{', so
# we don't eat the macro definitions latexdiff inserts in the preamble.
DEL_BLOCK = re.compile(
    r"(?<!\\providecommand\{)(?<!\\newcommand\{)"
    r"\\DIFdelbegin\b.*?\\DIFdelend\b\s*",
    re.DOTALL,
)
DEL_BLOCK_FL = re.compile(
    r"(?<!\\providecommand\{)(?<!\\newcommand\{)"
    r"\\DIFdelbeginFL\b.*?\\DIFdelendFL\b\s*",
    re.DOTALL,
)


def clean(text: str) -> str:
    # Split off the preamble so the regexes only run on the document body.
    if r"\begin{document}" in text:
        head, body = text.split(r"\begin{document}", 1)
    else:
        head, body = "", text
    body = DEL_BLOCK.sub("", body)
    body = DEL_BLOCK_FL.sub("", body)
    return head + (r"\begin{document}" + body if head else body)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inp", type=Path, help="diff.tex from latexdiff")
    p.add_argument("out", type=Path, help="output cleaned .tex")
    p.add_argument("--strip-add", action="store_true",
                   help="also remove \\DIFadd markers so the new text is "
                        "rendered without colour highlighting")
    args = p.parse_args()
    text = args.inp.read_text()
    text = clean(text)
    if args.strip_add:
        text = re.sub(r"\\DIFaddbegin\s*", "", text)
        text = re.sub(r"\\DIFaddend\s*", "", text)
        text = re.sub(r"\\DIFadd\{(.*?)\}", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"\\DIFaddbeginFL\s*", "", text)
        text = re.sub(r"\\DIFaddendFL\s*", "", text)
        text = re.sub(r"\\DIFaddFL\{(.*?)\}", r"\1", text, flags=re.DOTALL)
    args.out.write_text(text)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
