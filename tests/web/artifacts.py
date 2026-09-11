"""生成物の同一性チェック（リファクタリングの安全網）。

docs/ の points.bin（3 ビュー）、shards/*.json（101 片）、各 index.html に埋め込まれた manifest（`var M = {...};`）、
sw.js のデータ版のハッシュを記録・照合する。整理の前後で、これらは 1 バイトも変わってはいけない
（index.html 自体は JS が変わるので対象外。manifest は同一であるべき）。

使い方:
    uv run python tests/web/artifacts.py record   # 基準を tests/web/baseline/artifacts.json に書く
    uv run python tests/web/artifacts.py check    # 現在の docs/ を基準と照合（違えば終了コード 1）
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
BASELINE = Path(__file__).resolve().parent / "baseline" / "artifacts.json"


def sha(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:16]


def collect() -> dict[str, str]:
    out: dict[str, str] = {}
    for view in ("map2d", "map3d", "globe"):
        out[f"{view}/points.bin"] = sha((DOCS / view / "points.bin").read_bytes())
        html = (DOCS / view / "index.html").read_text(encoding="utf-8")
        m = re.search(r"^var M = (\{.*\});$", html, re.M)
        assert m, f"{view}: manifest が見つからない"
        manifest = json.loads(m.group(1))
        out[f"{view}/manifest"] = sha(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode())
        out[f"{view}/manifest.n"] = str(manifest["n"])
        out[f"{view}/manifest.traces"] = str(len(manifest["traces"]))
    shards = sorted((DOCS / "shards").glob("*.json"))
    h = hashlib.sha1()
    for f in shards:
        h.update(f.name.encode()); h.update(f.read_bytes())
    out["shards"] = f"{len(shards)} files {h.hexdigest()[:16]}"
    sw = (DOCS / "sw.js").read_text(encoding="utf-8")
    m = re.search(r"ka-data-([0-9a-f]+)", sw)
    out["sw.js/data-version"] = m.group(1) if m else "?"
    return out


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    cur = collect()
    if mode == "record":
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(cur, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"記録: {BASELINE}")
        for k, v in cur.items():
            print(f"  {k}: {v}")
        return
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    bad = 0
    for k in sorted(set(base) | set(cur)):
        a, b = base.get(k), cur.get(k)
        ok = a == b
        bad += not ok
        print(f"{'OK' if ok else 'NG'}  {k}: {b}" + ("" if ok else f"  (基準 {a})"))
    print(f"\n{len(cur) - bad}/{len(cur)} 一致")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
