"""キーワード地名の山域に、LLM（Claude）で一段抽象的な地名を付ける。

キーワード法（compute_placenames*.py）は著者キーワードの中から集中度の高い語を選ぶので、どうしても専門語になる
（「膵癌・腫瘍免疫」）。上位概念（「がんの免疫療法」）は語彙にないので選べない。そこで山域ごとに課題の本文
（埋め込みに使ったタイトル＋キーワード＋概要）を LLM に読ませ、領域全体を包含する短い名詞句を付ける（2026-09-18）。

抽出: 山域の課題から、中心重み（その位置の密度 ÷ 峰の密度。山の形に沿った不定形の重み）を確率にして k 件を無作為抽出
（乱数の種は固定）。中心の課題が多く、裾野の課題が少し混じる。件数は上限だけ決める（粗い階層 25、細かい階層 15）。
細かい階層には親（粗い階層）の名前を渡して、親の中での違いを名前に出させる。

出力: data/processed/placenames_llm_<map>_L<level>.json（モデル・日付・プロンプト版・抽出した課題番号・使用トークンを保存。
サイトはこの JSON を読むだけなので、方式の差し替えはファイルの差し替え）。

使い方:
    uv run python scripts/name_places_llm.py --map 2d --level 0            # 粗い階層 42 か所
    uv run python scripts/name_places_llm.py --map sphere --level 0        # 球面の粗い階層
    uv run python scripts/name_places_llm.py --map 2d --level 1 --parent data/processed/placenames_llm_2d_L0.json
    オプション: --limit 5（試し）、--dry-run（プロンプトを表示して呼ばない）、--k 25、--workers 4
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

MODEL = "claude-opus-5"
PRICE = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0)}  # $/M tokens (入力, 出力)
PROMPT_VERSION = "2026-09-18a"
K_BY_LEVEL = [25, 15]
MAX_CHARS = 600   # 1 課題の本文の上限（概要が長いものを切る）

SCHEMA = {
    "type": "object",
    "properties": {
        "name_ja": {"type": "string", "description": "日本語の地名。4〜12 文字の名詞句"},
        "name_en": {"type": "string", "description": "英語の地名。2〜5 語"},
        "rationale": {"type": "string", "description": "その名前にした根拠。1 文"},
        "mixed": {"type": "boolean", "description": "領域が 2 つ以上の異なる主題に分かれ、1 つの名前では無理があるとき true"},
        "sub_themes": {"type": "array", "items": {"type": "string"}, "description": "mixed のとき、主な主題を 2〜3 個"},
    },
    "required": ["name_ja", "name_en", "rationale", "mixed", "sub_themes"],
    "additionalProperties": False,
}

SYSTEM = (
    "あなたは学術地図の編集者です。研究課題の集まり（地図上で研究が密集している一つの領域）に、地図に載せる地名を付けます。"
    "地名は、個々の専門用語を並べるのではなく、領域全体を一段抽象化して表す短い名詞句にします。"
)


def load_env() -> None:
    for line in Path(".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def load_places(map_kind: str, level: int):
    if map_kind == "2d":
        meta = json.loads(Path("data/processed/placenames_2d.json").read_text(encoding="utf-8"))
        basins = pl.read_parquet("data/interim/placenames_basins.parquet")
        lv = meta["levels"][level]
        tag = f"{lv['sigma']:g}"
    else:
        meta = json.loads(Path("data/processed/placenames_sphere.json").read_text(encoding="utf-8"))
        basins = pl.read_parquet("data/interim/placenames_basins_sphere.parquet")
        lv = meta["levels"][level]
        tag = f"{lv['sigma']:.3g}"
    return meta, lv, basins, tag


def members_of(place: dict, basins: pl.DataFrame, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """山域（または区画）に属する課題の行番号と中心重み。"""
    if "sector_of" in place:
        m = (basins[f"sector_s{tag}"] == place["id"]).to_numpy()
    else:
        m = (basins[f"basin_s{tag}"] == place["id"]).to_numpy()
    idx = np.nonzero(m)[0]
    wt = basins[f"wt_s{tag}"].to_numpy()[idx].astype(float)
    return idx, wt


def sample(idx: np.ndarray, wt: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(idx) <= k:
        return idx
    p = np.clip(wt, 1e-6, None)
    return rng.choice(idx, size=k, replace=False, p=p / p.sum())


def build_prompt(place: dict, texts: list[str], n_total: int, level: int, n_places: int, parent: str | None) -> str:
    if level == 0:
        scope = f"これは粗い階層の地名で、地図全体を約 {n_places} か所に分けたうちの 1 つです。広い範囲を包含する、やや大きな括りの名前にしてください。"
    else:
        scope = (f"これは細かい階層の地名で、親の領域「{parent}」の一部です。親の中でこの領域が何を扱っているかが分かる名前にしてください。"
                 if parent else f"これは細かい階層の地名で、地図全体を約 {n_places} か所に分けたうちの 1 つです。")
    body = "\n\n".join(f"--- {i + 1} ---\n{t}" for i, t in enumerate(texts))
    return (
        f"科研費の採択課題 {n_total:,} 件を研究内容の近さで並べた地図があります。"
        f"その中の一つの領域（{place['n']:,} 件）に地名を付けてください。{scope}\n\n"
        f"以下は、その領域から中心付近を重く無作為抽出した {len(texts)} 件の課題です（タイトル／キーワード／採択時の研究概要）。"
        "裾野の課題も少し混じっています。\n\n"
        f"{body}\n\n"
        "地名の条件:\n"
        "- 日本語で 4〜12 文字の名詞句。領域全体を一段抽象化して表す（例: 「がんの免疫療法」「脱炭素の材料化学」「地域社会と災害」）。"
        "個々の専門用語を「・」で並べるのは避ける\n"
        "- 領域内の大半の課題を包含し、かつ隣の領域と区別がつく程度に具体的\n"
        "- 公式の審査区分の名称（「医歯薬学」「人文学」「工学」など）をそのまま名前にしない\n"
        "- 英語名（2〜5 語）と、その名前にした根拠を 1 文\n"
        "- 課題が 2 つ以上の異なる主題に分かれていて 1 つの名前では無理があるときは mixed を true にし、sub_themes に主題を挙げる"
        "（その場合も name_ja には最も多い主題の名前を入れる）"
    )


def call(client, model: str, prompt: str) -> tuple[dict, dict]:
    kwargs = dict(
        model=model, max_tokens=2000, system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "medium"},
    )
    r = client.messages.create(**kwargs)
    if r.stop_reason == "refusal":
        raise RuntimeError(f"refusal: {getattr(r, 'stop_details', None)}")
    text = next(b.text for b in r.content if b.type == "text")
    return json.loads(text), dict(input=r.usage.input_tokens, output=r.usage.output_tokens)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", choices=["2d", "sphere"], required=True)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--parent", type=Path, default=None, help="粗い階層の LLM 命名 JSON（細かい階層に親の名前を渡す）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    k = args.k or K_BY_LEVEL[min(args.level, len(K_BY_LEVEL) - 1)]

    meta, lv, basins, tag = load_places(args.map, args.level)
    corpus = pl.read_parquet("data/processed/corpus.parquet", columns=["award_number", "text"])
    assert bool((corpus["award_number"] == basins["award_number"]).all()), "行順が違う"
    texts_all = corpus["text"].to_list()
    awards = basins["award_number"].to_list()
    n_total = corpus.height

    parent_name: dict[int, str] = {}
    if args.parent and args.level > 0:
        pj = json.loads(args.parent.read_text(encoding="utf-8"))
        pname = {p["id"]: p["name_ja"] for p in pj["places"]}
        _, lv0, _, tag0 = load_places(args.map, 0)
        b0 = basins[f"basin_s{tag0}"].to_numpy()
        s0 = basins[f"sector_s{tag0}"].to_numpy() if f"sector_s{tag0}" in basins.columns else np.zeros(len(b0), dtype=int)
        for p in lv["places"]:
            idx, _ = members_of(p, basins, tag)
            key = np.where(s0[idx] > 0, s0[idx], b0[idx])   # 区画があれば区画 id、なければ山域 id
            top = int(np.bincount(key).argmax())
            parent_name[p["id"]] = pname.get(top, "")

    places = sorted(lv["places"], key=lambda q: -q["n"])
    if args.limit:
        places = places[: args.limit]
    jobs = []
    for p in places:
        idx, wt = members_of(p, basins, tag)
        pick = sample(idx, wt, k, seed=42 + p["id"])
        texts = [texts_all[i][:MAX_CHARS] for i in pick]
        prompt = build_prompt(p, texts, n_total, args.level, len(lv["places"]), parent_name.get(p["id"]))
        jobs.append((p, [awards[i] for i in pick], prompt))

    if args.dry_run:
        p, picks, prompt = jobs[0]
        print(prompt[:3000]); print(f"...\n[{len(jobs)} か所, 先頭のプロンプト {len(prompt):,} 文字]")
        return

    load_env()
    import anthropic
    client = anthropic.Anthropic()

    def run(job):
        p, picks, prompt = job
        try:
            out, usage = call(client, args.model, prompt)
        except Exception as e:  # 1 件の失敗で全体を止めない
            out, usage = dict(name_ja="", name_en="", rationale=f"ERROR: {e}", mixed=False, sub_themes=[]), dict(input=0, output=0)
        return dict(id=p["id"], n=p["n"], words=p["words"], parent=parent_name.get(p["id"]), sample=picks, usage=usage, **out)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(run, jobs))

    tin = sum(r["usage"]["input"] for r in results); tout = sum(r["usage"]["output"] for r in results)
    pin, pout = PRICE.get(args.model, (0, 0))
    cost = tin / 1e6 * pin + tout / 1e6 * pout
    out_path = Path(f"data/processed/placenames_llm_{args.map}_L{args.level}.json")
    payload = dict(map=args.map, level=args.level, sigma=lv["sigma"], model=args.model, prompt_version=PROMPT_VERSION,
                   created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), k=k, max_chars=MAX_CHARS,
                   usage=dict(input=tin, output=tout, cost_usd=round(cost, 3)), places=results)
    if not args.limit:
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(results)} か所 | tokens in {tin:,} / out {tout:,} | 約 ${cost:.2f} | {'→ ' + str(out_path) if not args.limit else '（--limit のため保存せず）'}\n")
    print(f"{'件数':>6}  {'キーワード地名':<28} {'LLM の地名':<16} {'英語':<34} 備考")
    for r in results:
        flag = ("mixed: " + "／".join(r["sub_themes"])) if r["mixed"] else ""
        print(f"{r['n']:>6}  {'・'.join(r['words']):<28} {r['name_ja']:<16} {r['name_en']:<34} {flag}")
    errs = [r for r in results if r["rationale"].startswith("ERROR")]
    if errs:
        print(f"\n失敗 {len(errs)} 件: " + "; ".join(r["rationale"][:80] for r in errs))


if __name__ == "__main__":
    sys.exit(main())
