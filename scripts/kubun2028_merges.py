"""令和 10 年度審査区分表の改正で統合・吸収される小区分が、内容（Ruri 埋め込み）の上で本当に近かったかを検証する（2026-10-26）。

問い: 制度が統合した組は、同じ中区分の他の組と比べても特に近かったか。境界は曖昧だったか。吸収先はデータ上の最近傍と一致するか。
対象: 2019–2025 年度開始で審査区分（小区分）が 1 つだけの課題（169,500 件）。計算は 768 次元（平均を引く）と kNN グラフ、
地図座標は表示のみ。

出力:
  reports/kubun2028/merge_adjacency.md         数表（重心距離の順位、近傍リフト、分類精度、吸収先の按分）
  reports/figures/kubun2028_chu<NN>.png       中区分ごとに「全体地図の切り出し」と「その中区分だけで UMAP を再計算」を並べた図

使い方: uv run python scripts/kubun2028_merges.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Hiragino Sans"
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"
FIG = Path("reports/figures")
OUT = Path("reports/kubun2028/merge_adjacency.md")
K_NB = 30
SEED = 42

# 改正の組（旧コード）。kind: pair=対の統合, absorb=吸収, split=分割吸収。target は新表での行き先（説明用）。
# 出所: data/reference/shinsa_kubun_r10_mext_20260729.txt 参考 2 新旧対照表（p.142〜）
GROUPS = [
    dict(chu=2, title="中区分 2 文学・言語学", merges=[
        dict(kind="absorb", src="02020", dst=["02050"], note="中国文学 → 文学一般（新 02030）"),
        dict(kind="pair", src="02030", dst=["02040"], note="英文学・英語圏文学 ＋ ヨーロッパ文学 → 英語圏文学およびヨーロッパ文学（新 02020）"),
        dict(kind="pair", src="02070", dst=["02080"], note="日本語学 ＋ 英語学 → 日本語学および英語学（新 02050）"),
    ]),
    dict(chu=5, title="中区分 5 法学", merges=[
        dict(kind="pair", src="05010", dst=["05050"], note="基礎法学 ＋ 刑事法学 → 基礎法学および刑事法学（新 05010）"),
        dict(kind="split", src="05030", dst=["05020", "05060"], note="国際法学 → 公法学（国際公法）と民事法学（国際私法）に分割吸収"),
        dict(kind="pair", src="05040", dst=["05070"], note="社会法学 ＋ 新領域法学 → 社会法学および新領域法学（新 05040）"),
    ]),
    dict(chu=7, title="中区分 7 経済学・経営学", merges=[
        dict(kind="pair", src="07010", dst=["07020"], note="理論経済学 ＋ 経済学説・経済思想 → 理論経済学および経済学史（新 07010）"),
    ]),
    dict(chu=12, title="中区分 12 解析学・応用数学", merges=[
        dict(kind="pair", src="12030", dst=["12040"], note="数学基礎 ＋ 応用数学・統計数学 → 数学基礎、応用数学および統計数学（新 12030）"),
    ]),
    dict(chu=39, title="中区分 39 生産環境農学（＋吸収先 64040 環境）", extra=["64040"], merges=[
        dict(kind="absorb", src="39060", dst=["64040"], note="生物資源保全学（大区分 F）→ 自然共生システム（大区分 K→新 M、中区分 64）"),
    ]),
    dict(chu=46, title="中区分 46 神経系の科学", merges=[
        dict(kind="split", src="46030", dst=["46020", "46010"], note="神経機能学 → システム・個体レベルの神経科学（旧 神経形態学）を主に、分子・細胞レベル（旧 神経科学一般）にも"),
    ]),
    dict(chu=51, title="中区分 51 脳の活動と病態", merges=[
        dict(kind="absorb", src="51020", dst=["51010"], note="認知脳科学 → 脳科学一般（旧 基盤脳科学）。病態神経科学は病態脳科学に改称"),
    ]),
    dict(chu=90, title="中区分 90 人間医工学（＋吸収先 59010 リハ・58010 医療管理）", extra=["59010", "58010"], merges=[
        dict(kind="split", src="90150", dst=["59010", "90110"], note="医療福祉工学 → リハビリテーション科学および医療福祉（中区分 59）と生体機械工学（旧 生体医工学）に分割吸収"),
        dict(kind="absorb", src="90140", dst=["58010"], note="医療技術評価学 → 医療管理学、医療系社会学および医療技術評価（中区分 58）"),
    ]),
]


def footnote(fig, n):
    fig.text(0.01, 0.005,
             "出典: KAKEN：科学研究費助成事業データベース（国立情報学研究所）のデータを編集・加工 | "
             f"対象: 2019–2025年度開始・審査区分が1つの採択課題 {n:,}件 | 埋め込み: cl-nagoya/ruri-v3-310m | "
             "改正: 文科省 科研費審査部会 2026-07-29 決定（令和10年度公募から）| 作成: KAKEN-ATLAS (26K15524)", color=MUTED, fontsize=8)


def main() -> None:
    t0 = time.time()
    emb_df = pl.read_parquet("data/processed/embeddings.parquet")
    award = emb_df["award_number"]
    emb = np.ascontiguousarray(np.stack(emb_df["embedding"].to_numpy()).astype(np.float32))
    n_all = len(award)
    awards = pl.read_parquet("data/interim/awards.parquet", columns=["award_number", "shokubun_codes"])
    corpus = pl.read_parquet("data/processed/corpus.parquet", columns=["award_number", "title", "category", "start_fy"])
    meta = pl.DataFrame({"award_number": award}).join(corpus, on="award_number", how="left").join(awards, on="award_number", how="left")
    sho = np.array([c[0] if (c is not None and len(c) == 1) else "" for c in meta["shokubun_codes"].to_list()])
    table = (pl.read_csv("data/reference/kubun_table.csv", schema_overrides={"sho_code": pl.Utf8})
             .select(["sho_code", "sho_name_ja", "chu_code"]).unique(subset=["sho_code"], keep="first"))
    name = dict(zip(table["sho_code"], table["sho_name_ja"]))
    chu_of = dict(zip(table["sho_code"], table["chu_code"]))
    xy = pl.read_parquet("data/processed/umap2d_nn15_md0.1.parquet").select(["award_number", "c0", "c1"])
    xy = pl.DataFrame({"award_number": award}).join(xy, on="award_number", how="left").select(["c0", "c1"]).to_numpy()
    knn = np.load("data/interim/knn50.npz")["idx"][:, :K_NB]
    single = sho != ""
    n_single = int(single.sum())
    print(f"読み込み {time.time() - t0:.0f}s, 単一区分 {n_single:,}")

    # 平均を引いた埋め込み（距離・分類用）と PCA 50
    mu = emb.mean(axis=0, keepdims=True)
    Xc = emb - mu
    pca = PCA(n_components=50, svd_solver="randomized", random_state=SEED).fit(Xc[np.random.default_rng(SEED).choice(n_all, 40000, replace=False)])
    X50 = pca.transform(Xc).astype(np.float32)

    def centroid(codes):
        idx = np.where(np.isin(sho, codes))[0]
        c = Xc[idx].mean(axis=0)
        return c / np.linalg.norm(c), idx

    lines = [f"# 審査区分表 2028 改正の統合・吸収は内容の上で近かったか（{n_single:,} 件、2026-10-26）\n",
             "計算は 768 次元（全体平均を引いた Ruri 埋め込み）と kNN グラフ（k=30）。地図座標は図の左パネルのみ。\n",
             "- **重心順位**: 統合元から見て、同じ中区分の小区分のうち統合相手が何番目に近いか（1 が最近傍）／括弧内は全 306 小区分での順位。",
             "- **近傍リフト**: 統合元の課題の近傍 30 件（自区分を除く）に占める相手区分の割合 ÷ 相手区分の全体シェア。1 なら無関係、大きいほど隣接。",
             "- **最多近傍**: 統合元の課題の近傍（自区分を除く）で最も多い小区分。制度の行き先と一致するか。",
             "- **分類精度**: 統合元 vs 相手を PCA 50 次元のロジスティック回帰で分ける 5 分割 CV の balanced accuracy。0.5 が区別不能。中区分内の全ペアの中での順位も示す（1 が最も区別しにくい）。\n"]

    for g in GROUPS:
        chu = g["chu"]
        codes = sorted([c for c in name if chu_of[c] == chu and (sho == c).any()])
        extra = g.get("extra", [])
        panel_codes = codes + [e for e in extra if e not in codes]
        lines.append(f"\n## {g['title']}\n")
        lines.append("| 小区分 | 名称 | n |\n|---|---|---|")
        for c in panel_codes:
            lines.append(f"| {c} | {name[c]} | {(sho == c).sum():,} |")

        # 中区分内の全ペアの分類精度（対照）
        pair_acc = {}
        for i, a in enumerate(panel_codes):
            for b in panel_codes[i + 1:]:
                ia, ib = np.where(sho == a)[0], np.where(sho == b)[0]
                Xp = np.vstack([X50[ia], X50[ib]]); yp = np.r_[np.zeros(len(ia)), np.ones(len(ib))]
                clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
                pair_acc[(a, b)] = pair_acc[(b, a)] = float(cross_val_score(clf, Xp, yp, cv=5, scoring="balanced_accuracy").mean())
        # 重心
        cents = {c: centroid([c])[0] for c in panel_codes}
        all_cents = {c: centroid([c])[0] for c in name if (sho == c).any()}

        lines.append("\n| 統合 | 型 | 重心順位（中区分内／全体） | 近傍リフト | 最多近傍（上位 3） | 分類精度（中区分内順位） |\n|---|---|---|---|---|---|")
        split_notes = []
        KIND = {"pair": "対の統合", "absorb": "吸収", "split": "分割吸収"}
        for m in g["merges"]:
            src = m["src"]; isrc = np.where(sho == src)[0]
            # 近傍の他区分分布
            nb = sho[knn[isrc]].ravel()
            nb = nb[(nb != "") & (nb != src)]
            vals, cnts = np.unique(nb, return_counts=True)
            order = np.argsort(-cnts)
            top3 = "、".join(f"{name[v]}({v}) {cnts[o] / cnts.sum():.0%}" for o, v in zip(order[:3], vals[order[:3]]))
            others = single & (sho != src)
            rows = []
            for dst in m["dst"]:
                d_all = {c: 1 - float(cents[src] @ all_cents[c]) for c in all_cents if c != src}
                rank_all = sorted(d_all, key=d_all.get).index(dst) + 1
                d_chu = {c: d_all[c] for c in panel_codes if c != src}
                rank_chu = sorted(d_chu, key=d_chu.get).index(dst) + 1
                share_nb = (nb == dst).mean()
                share_all = (sho[others] == dst).mean()
                lift = share_nb / share_all if share_all > 0 else float("nan")
                acc = pair_acc[(src, dst)]
                accs_sorted = sorted({pair_acc[p] for p in pair_acc})
                acc_rank = accs_sorted.index(acc) + 1
                rows.append((dst, rank_chu, rank_all, lift, acc, acc_rank))
            cell = lambda f: "<br>".join(f(r) for r in rows)  # noqa: E731
            lines.append(f"| {m['note']} | {KIND[m['kind']]} | " + cell(lambda r: f"{name[r[0]]}: {r[1]}／{r[2]}") + " | " +
                         cell(lambda r: f"{r[3]:.1f}×") + f" | {top3} | " + cell(lambda r: f"{r[4]:.2f}（{r[5]}/{len(set(pair_acc.values()))}）") + " |")
            if m["kind"] == "split":
                a, b = m["dst"][0], m["dst"][1]
                lean = []
                for i in isrc:
                    v = sho[knn[i]]; na, nbb = (v == a).sum(), (v == b).sum()
                    lean.append(0 if na > nbb else 1 if nbb > na else -1)
                lean = np.array(lean)
                split_notes.append(f"按分（{name[src]} の各課題の近傍 30 件でどちらが多いか）: {name[a]} 側 {np.mean(lean == 0):.0%}、"
                                   f"{name[b]} 側 {np.mean(lean == 1):.0%}、同数・どちらも無し {np.mean(lean == -1):.0%}")
        for note in split_notes:
            lines.append("\n" + note)
        # 全ペアの分類精度表
        lines.append("\n中区分内の全ペアの分類精度（低いほど混ざっている）:\n")
        hdr = "| | " + " | ".join(c for c in panel_codes) + " |"
        lines.append(hdr + "\n|---|" + "---|" * len(panel_codes))
        for a in panel_codes:
            lines.append(f"| {a} {name[a][:10]} | " + " | ".join("—" if a == b else f"{pair_acc[(a, b)]:.2f}" for b in panel_codes) + " |")

        # 図: 左 全体地図の切り出し、右 その中区分だけで UMAP
        from umap import UMAP
        idx_panel = np.where(np.isin(sho, panel_codes))[0]
        loc = UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(emb[idx_panel])
        merged = {m["src"] for m in g["merges"]} | {d for m in g["merges"] for d in m["dst"]}
        cmap = plt.get_cmap("tab20")
        colors = {c: cmap(i % 20) for i, c in enumerate(panel_codes)}
        fig, axes = plt.subplots(1, 2, figsize=(18, 8.5), facecolor=SURFACE)
        ax = axes[0]
        ax.scatter(xy[:, 0], xy[:, 1], s=0.3, c="#d9d8d3", linewidths=0, rasterized=True)
        pts = xy[idx_panel]
        lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
        pad = 0.15 * (hi - lo)
        for c in panel_codes:
            ii = idx_panel[sho[idx_panel] == c]
            ax.scatter(xy[ii, 0], xy[ii, 1], s=6 if c in merged else 3, c=[colors[c]], alpha=0.9 if c in merged else 0.45,
                       linewidths=0, label=f"{c} {name[c]}" + (" ★" if c in merged else ""))
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(SURFACE)
        ax.set_title("全体地図（2D UMAP）の該当領域。灰は他の課題", color=INK, loc="left")
        ax = axes[1]
        for c in panel_codes:
            jj = sho[idx_panel] == c
            ax.scatter(loc[jj, 0], loc[jj, 1], s=9 if c in merged else 5, c=[colors[c]], alpha=0.9 if c in merged else 0.45, linewidths=0)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(SURFACE)
        ax.set_title(f"この中区分 {len(idx_panel):,} 件だけで UMAP を再計算（教師なし、cosine）", color=INK, loc="left")
        for s in list(axes[0].spines.values()) + list(axes[1].spines.values()):
            s.set_visible(False)
        axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.02), ncol=2, fontsize=8, frameon=False, markerscale=3)
        notes = "\n".join("★ " + m["note"] for m in g["merges"])
        fig.suptitle(f"{g['title']}: 統合・吸収される小区分は内容の上で隣だったか", color=INK, x=0.01, ha="left", fontsize=14)
        fig.text(0.52, 0.02, notes, color=INK, fontsize=9, va="bottom")
        footnote(fig, n_single)
        fig.tight_layout(rect=(0, 0.13, 1, 0.95))
        out = FIG / f"kubun2028_chu{chu:02d}.png"
        fig.savefig(out, dpi=130, facecolor=SURFACE); plt.close(fig)
        print(f"{g['title']} 完了 {time.time() - t0:.0f}s → {out}")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"書き出し {OUT}")


if __name__ == "__main__":
    main()
