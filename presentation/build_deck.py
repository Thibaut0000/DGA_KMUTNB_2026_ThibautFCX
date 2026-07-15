"""Build the self-contained final presentation.

    python presentation/build_deck.py

Reads deck_src.html, replaces every {{IMG:name}} placeholder with a base64
data-URI of results/figures/<name>.png, and writes final_presentation.html
(single file, portable, no external dependencies).
"""
from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).with_name("deck_src.html")
OUT = Path(__file__).with_name("final_presentation.html")
FIGS = ROOT / "results" / "figures"


def main() -> int:
    html = SRC.read_text(encoding="utf-8")
    names = sorted(set(re.findall(r"\{\{IMG:([\w-]+)\}\}", html)))
    missing = [n for n in names if not (FIGS / f"{n}.png").exists()]
    if missing:
        print("missing figures:", ", ".join(missing))
        print("run the matching scripts/ to regenerate them first")
        return 1
    total = 0
    for n in names:
        raw = (FIGS / f"{n}.png").read_bytes()
        total += len(raw)
        uri = "data:image/png;base64," + base64.b64encode(raw).decode()
        html = html.replace(f"{{{{IMG:{n}}}}}", uri)
    OUT.write_text(html, encoding="utf-8")
    print(f"embedded {len(names)} figures ({total/1024:.0f} KB raw) -> {OUT} "
          f"({OUT.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
