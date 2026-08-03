#!/usr/bin/env python3
"""Per-name accuracy across bench runs — the metric WER cannot give you.

Mean WER moves for every reason at once: a name misspelled, a number written `9h` instead of
`neuf heures`, a word genuinely misheard. On a corpus this short each of those is worth several
percent, so a WER delta between two runs says *something changed* and not *the hotword list
worked*. This counts the only thing the biasing question is about: for each proper noun, how many
of its occurrences came back spelled the way the reference spells it.

Matching is case-insensitive on the token but exact on the letters, because the entire failure
under test is `Loup` vs `Lou` — folding those together would score the bug as a pass.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

NAMES = [
    "Loup",
    "Thibault",
    "Rose",
    "Colombe",
    "Olivier",
    "Pierre",
    "Mathilde",
    "Fabien",
    "Grenoble",
    "Chamonix",
    "Annecy",
]


def tokens(text: str) -> list[str]:
    folded = unicodedata.normalize("NFC", text).casefold().replace("’", "'")
    return re.sub(r"[^\w\s'-]", " ", folded, flags=re.UNICODE).split()


def main() -> int:
    runs = [json.loads(line) for line in Path(sys.argv[1]).read_text("utf8").splitlines() if line.strip()]
    width = max(len(n) for n in NAMES)
    labels = [r["label"] for r in runs]
    print(f"{'name'.ljust(width)}  " + "  ".join(f"{lab:>18}" for lab in labels))
    for name in NAMES:
        needle = name.casefold()
        cells = []
        for run in runs:
            hits = total = 0
            for s in run["samples"]:
                want = tokens(s["reference"]).count(needle)
                if want == 0:
                    continue
                total += want
                hits += min(want, tokens(s["text"]).count(needle))
            cells.append("—".rjust(18) if total == 0 else f"{hits}/{total}".rjust(18))
        print(f"{name.ljust(width)}  " + "  ".join(cells))
    print()
    print(f"{'mean WER'.ljust(width)}  " + "  ".join(f"{r['mean_wer']:>17.2%}" for r in runs))
    print(f"{'bias_mode'.ljust(width)}  " + "  ".join(f"{r['bias_mode']:>18}" for r in runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
