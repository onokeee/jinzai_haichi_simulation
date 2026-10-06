"""
NAND工場 メンテナンス担当 最適配置シミュレータ
------------------------------------------------
NANDフラッシュメモリ専用の工場で、製品の世代ごとに専用のラインがあります。
過去の実績データ（スキル別の担当人数・不良数・稼働率）から
「担当を増やすと、不良や装置停止がどれくらい減るか」を世代ごとに学習し、
工場全体の生産数が最大になる人員配置を計算します。

起動:  streamlit run app.py
"""
import io
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

st.set_page_config(page_title="メンテ人員 最適配置シミュレータ", page_icon="🏭", layout="wide")

# スキル区分ごとの「1人でこなせるメンテの量」（中堅 = 1 とした想定。画面には出さない）
SKILLS = {"新人": 0.6, "中堅": 1.0, "ベテラン": 1.4}
SKILL_NAMES = list(SKILLS.keys())
SKILL_FACTORS = np.array(list(SKILLS.values()))
SKILL_COLORS = {"新人": "#9ecae1", "中堅": "#3182bd", "ベテラン": "#08519c"}
MIN_PER_LINE = 1  # 各ラインに最低1人は配置する
MAX_PER_SKILL = 20  # スキルごとの人数の上限（計算時間を短く保つため）

HISTORY_COLUMNS = ["年月", "世代", "装置台数", *SKILL_NAMES, "投入数", "稼働率(%)", "不良数"]
BREAKDOWN_COLORS = {"生産数": "#2ca02c", "不良数": "#d62728", "停止で作れなかった数": "#bbbbbb"}
PATTERN_COLORS = px.colors.qualitative.Set2


# ---------------------------------------------------------------------------
# サンプル実績データ（架空）
# ---------------------------------------------------------------------------
@st.cache_data
def make_sample_history(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # 真のパラメータ（学習で当てにいく値）。新しい世代ほど工程が難しく、装置の調子の影響を受けやすい
    generations = [
        # 世代, 装置台数, 月間投入数(枚), 不良率(放置), 不良率(下限), 稼働率(放置), 稼働率(上限), 効き目
        ("第1世代", 4, 6000, 0.12, 0.010, 0.75, 0.98, 2.8),
        ("第2世代", 7, 9000, 0.18, 0.015, 0.65, 0.97, 2.2),
        ("第3世代", 12, 15000, 0.25, 0.020, 0.55, 0.97, 1.8),
        ("第4世代", 10, 12000, 0.32, 0.030, 0.50, 0.96, 1.5),
        ("第5世代", 6, 7200, 0.45, 0.040, 0.45, 0.95, 1.4),
    ]
    rows = []
    months = pd.period_range("2024-10", periods=24, freq="M")
    for name, mach, cap, d0, dmin, u0, umax, k in generations:
        for m in months:
            counts = rng.multinomial(int(rng.integers(1, max(4, mach))), [0.3, 0.45, 0.25])
            x = float(counts @ SKILL_FACTORS) / mach
            dr = dmin + (d0 - dmin) * np.exp(-k * x)
            up = umax - (umax - u0) * np.exp(-k * x)
            dr = float(np.clip(dr * rng.normal(1, 0.08), 0, 1))
            up = float(np.clip(up + rng.normal(0, 0.015), 0, 1))
            inp = int(cap * rng.uniform(0.95, 1.05))
            rows.append({
                "年月": str(m), "世代": name, "装置台数": mach,
                **dict(zip(SKILL_NAMES, map(int, counts))),
                "投入数": inp, "稼働率(%)": round(up * 100, 1), "不良数": int(inp * up * dr),
            })
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


# ---------------------------------------------------------------------------
# モデル（内部計算）
#   x = スキルを考えたメンテの量 ÷ 装置台数（装置1台あたりにかけられるメンテ）
#   不良率(x)  = 下限 + (放置時 - 下限) × exp(-k x)
#   稼働率(x)  = 上限 - (上限 - 放置時) × exp(-k x)
#   生産数     = 投入数 - 停止で作れなかった数 - 不良数
# ---------------------------------------------------------------------------
def defect_curve(x, d0, dmin, k):
    return dmin + (d0 - dmin) * np.exp(-k * x)


def uptime_curve(x, u0, umax, k):
    return umax - (umax - u0) * np.exp(-k * x)


@st.cache_data
def fit_models(hist: pd.DataFrame) -> pd.DataFrame:
    out = []
    for gen, g in hist.groupby("世代", sort=False):
        people = g[SKILL_NAMES].sum(axis=1).to_numpy(float)
        work = g[SKILL_NAMES].to_numpy(float) @ SKILL_FACTORS
        x = work / g["装置台数"].to_numpy(float)
        up = (g["稼働率(%)"] / 100).to_numpy(float)
        dr = (g["不良数"] / (g["投入数"] * up).clip(lower=1)).to_numpy(float)
        try:
            (d0, dmin, kd), _ = curve_fit(defect_curve, x, dr, p0=[0.3, 0.02, 1.5],
                                          bounds=([0, 0, 0.05], [1, 0.5, 20]), maxfev=20000)
        except Exception:
            d0, dmin, kd = 0.3, 0.03, 1.5
        try:
            (u0, umax, ku), _ = curve_fit(uptime_curve, x, up, p0=[0.5, 0.95, 1.5],
                                          bounds=([0, 0.5, 0.05], [1, 1, 20]), maxfev=20000)
        except Exception:
            u0, umax, ku = 0.5, 0.95, 1.5
        out.append({
            "世代": gen,
            "装置台数": int(g["装置台数"].iloc[-1]),
            "月間投入数": int(g["投入数"].mean()),
            # いつもの人員構成での「1人あたりのメンテの量」（傾向の線を引くときに使う）
            "1人あたりの量": float(work.sum() / people.sum()) if people.sum() > 0 else 1.0,
            "不良率_放置": d0, "不良率_下限": dmin, "k_不良": kd,
            "稼働率_放置": u0, "稼働率_上限": umax, "k_稼働": ku,
        })
    return pd.DataFrame(out)


def work_of(row) -> float:
    """スキル別人数 → スキルを考えたメンテの量"""
    return float(np.dot(row, SKILL_FACTORS))


def simulate_line(p, work: float) -> dict:
    x = work / max(p["装置台数"], 1)
    dr = defect_curve(x, p["不良率_放置"], p["不良率_下限"], p["k_不良"])
    up = uptime_curve(x, p["稼働率_放置"], p["稼働率_上限"], p["k_稼働"])
    processed = p["月間投入数"] * up
    return {
        "稼働率": up, "不良率": dr,
        "生産数": processed * (1 - dr), "不良数": processed * dr,
        "停止で作れなかった数": p["月間投入数"] * (1 - up),
    }


@st.cache_data(show_spinner=False)
def optimize(params: pd.DataFrame, staff: dict) -> np.ndarray:
    """動的計画法：ラインを1本ずつ増やしながら「使った人数ごとの最大の生産数」を表にしていく。
    全通りを調べたのと同じ、本当に一番よい配置が求まる。"""
    recs = params.to_dict("records")
    shape = tuple(staff[s] + 1 for s in SKILL_NAMES)
    choices = np.array(list(np.ndindex(*shape)))  # 1ラインに置く人数の組（新人, 中堅, ベテラン）
    choices = choices[choices.sum(axis=1) >= MIN_PER_LINE]
    best = np.full(shape, -np.inf)  # best[u] = これまでのラインに u 人を使ったときの最大の生産数
    best[(0,) * len(shape)] = 0.0
    picks = []
    for p in recs:
        gains = simulate_line(p, choices @ SKILL_FACTORS)["生産数"]
        new, pick = np.full(shape, -np.inf), np.zeros(shape, dtype=int)
        for j, (d, gain) in enumerate(zip(choices, gains)):
            dst = tuple(slice(k, None) for k in d)
            src = tuple(slice(0, n - k) for n, k in zip(shape, d))
            cand = best[src] + gain
            better = cand > new[dst]
            new[dst][better] = cand[better]
            pick[dst][better] = j
        best = new
        picks.append(pick)

    # 最後のラインから順に、選んだ人数の組をたどって配置を復元
    alloc = np.zeros((len(recs), len(shape)), dtype=int)
    u = tuple(n - 1 for n in shape)
    for i in range(len(recs) - 1, -1, -1):
        alloc[i] = choices[picks[i][u]]
        u = tuple(np.subtract(u, alloc[i]))
    return alloc


def result_table(params: pd.DataFrame, alloc: np.ndarray) -> pd.DataFrame:
    rows = []
    for p, row in zip(params.to_dict("records"), alloc):
        rows.append({"世代": p["世代"], "装置台数": int(p["装置台数"]),
                     **dict(zip(SKILL_NAMES, map(int, row))), "合計人数": int(row.sum()),
                     **simulate_line(p, work_of(row))})
    return pd.DataFrame(rows)


def count_allocations(staff: dict, n_lines: int) -> float:
    """スキル別の人数を n ラインに割り振る方法の数（各ライン1人以上）"""
    f = np.zeros([staff[s] + 1 for s in SKILL_NAMES])
    f[(0,) * f.ndim] = 1.0
    for _ in range(n_lines):
        g = f
        for ax in range(f.ndim):
            g = g.cumsum(axis=ax)
        f = g - f  # 前のラインまでの割り振り ＋ このラインに1人以上
    return float(f[(-1,) * f.ndim])


def fmt_count(c: float) -> str:
    for unit, name in ((1e16, "京"), (1e12, "兆"), (1e8, "億"), (1e4, "万")):
        if c >= unit:
            v = c / unit
            return f"約{v:,.1f}{name}通り" if v < 10 else f"約{v:,.0f}{name}通り"
    return f"{c:,.0f}通り"


def relation_figure(hist: pd.DataFrame, params: pd.DataFrame) -> go.Figure:
    """ラインごとに「担当人数 × 不良数」の散布図（実績）と、学習した傾向の線"""
    recs = params.to_dict("records")
    cols = min(3, len(recs))
    rows = math.ceil(len(recs) / cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[p["世代"] for p in recs],
                        horizontal_spacing=0.07, vertical_spacing=0.16)
    for i, p in enumerate(recs):
        r, c = i // cols + 1, i % cols + 1
        g = hist[hist["世代"] == p["世代"]]
        people = g[SKILL_NAMES].sum(axis=1)
        xs = np.arange(1, max(int(people.max()) + 2, 6) + 1)
        fig.add_scatter(x=people, y=g["不良数"], mode="markers", name="実績（1点＝1か月）",
                        marker=dict(color="#7f7f7f", size=8, opacity=0.55),
                        customdata=g["年月"], hovertemplate="%{customdata}<br>担当 %{x}人<br>不良 %{y:,.0f}枚<extra></extra>",
                        legendgroup="実績", showlegend=(i == 0), row=r, col=c)
        trend = [simulate_line(p, n * p["1人あたりの量"])["不良数"] for n in xs]
        fig.add_scatter(x=xs, y=trend, mode="lines", name="学習した傾向", line=dict(color="#d62728", width=3),
                        hovertemplate="担当 %{x}人 → 不良 約 %{y:,.0f}枚<extra></extra>",
                        legendgroup="傾向", showlegend=(i == 0), row=r, col=c)
    fig.update_xaxes(title_text="担当人数（人）", dtick=1)
    fig.update_yaxes(title_text="不良数（枚/月）", rangemode="tozero")
    fig.update_layout(height=300 * rows + 40, margin=dict(l=0, r=0, t=80, b=0),
                      legend=dict(orientation="h", yref="container", yanchor="top", y=0.99, x=0))
    return fig


def fmt_int(v):
    return f"{v:,.0f}"


# ---------------------------------------------------------------------------
# 画面
# ---------------------------------------------------------------------------
st.title("🏭 NAND工場 メンテナンス担当 最適配置シミュレータ")
st.caption("NAND フラッシュメモリ専用の工場で、製品の世代ごとに専用のラインがあります。"
           "メンテナンス担当が少ないと、装置の停止や不良が増えてしまいます。※データはすべて架空のサンプルです。")
st.markdown("📂 **過去の実績データ**　➡　🧠 **人数と不良の関係を学習**　➡　🎯 **生産数が最大になる人員配置を計算**")

with st.sidebar:
    st.header("📂 実績データ")
    src = st.radio("データの選び方", ["サンプルデータを使う", "CSVをアップロード"])
    hist = make_sample_history()
    if src == "CSVをアップロード":
        up_file = st.file_uploader("実績CSV（UTF-8 / Shift-JIS）", type="csv")
        if up_file is not None:
            raw = up_file.getvalue()
            for enc in ("utf-8-sig", "cp932"):
                try:
                    hist = pd.read_csv(io.BytesIO(raw), encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            missing = [c for c in HISTORY_COLUMNS if c not in hist.columns]
            if missing:
                st.error(f"列が足りません: {missing}")
                st.stop()
        else:
            st.info("アップロードされるまではサンプルを表示します")
    all_lines = list(dict.fromkeys(hist["世代"]))
    st.caption(f"{len(all_lines)} ライン × {hist['年月'].nunique()} か月分の実績をもとに計算しています。")
    st.download_button("📥 CSVひな形（サンプル）をダウンロード",
                       make_sample_history().to_csv(index=False).encode("utf-8-sig"),
                       "sample_history.csv", "text/csv")

params_all = fit_models(hist)
n_all = len(all_lines)

tab_sim, tab_cmp = st.tabs(["🎯 シミュレーション", "⚖️ パターン比較"])

# ---------------- シミュレーション ----------------
with tab_sim:
    st.subheader("条件を入れると、生産数が最大になる人員配置を計算します")
    c0, c1, c2, c3 = st.columns(4)
    n_lines = c0.number_input("ライン数", 1, n_all, n_all, step=1, help="上から何世代ぶんのラインを使うか")
    staff = {}
    for col, (s, default) in zip((c1, c2, c3), {"新人": 5, "中堅": 10, "ベテラン": 5}.items()):
        staff[s] = col.number_input(f"{s}（人）", 0, MAX_PER_SKILL, default, step=1)
    total_staff = sum(staff.values())
    lines = all_lines[:n_lines]
    st.caption(f"対象：{'、'.join(lines)}　／　メンテ担当 合計 {total_staff} 人　／　"
               "各ラインに最低1人は配置します。ベテランほど1人でこなせるメンテの量が多い想定です。")

    if total_staff < MIN_PER_LINE * n_lines:
        st.error(f"人数が足りません：{n_lines} ラインに最低1人ずつ、{MIN_PER_LINE * n_lines} 人以上が必要です。")
    else:
        params = params_all.iloc[:n_lines]
        with st.spinner("最適な配置を計算中…"):
            res = result_table(params, optimize(params, staff))

        m1, m2, m3 = st.columns(3)
        m1.metric("生産数（枚/月）", fmt_int(res["生産数"].sum()))
        m2.metric("不良数（枚/月）", fmt_int(res["不良数"].sum()))
        m3.metric("停止で作れなかった数（枚/月）", fmt_int(res["停止で作れなかった数"].sum()))
        st.caption(f"💡 人の配置のしかたは全部で {fmt_count(count_allocations(staff, n_lines))}。"
                   "その中から、生産数が最大になる配置をコンピュータが探し出しました。")

        left, right = st.columns(2)
        with left:
            st.markdown("**👥 おすすめの人員配置**")
            long = res.melt(id_vars="世代", value_vars=SKILL_NAMES, var_name="スキル", value_name="人数")
            fig = px.bar(long, y="世代", x="人数", color="スキル", orientation="h",
                         color_discrete_map=SKILL_COLORS, category_orders={"スキル": SKILL_NAMES})
            fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None,
                              legend=dict(orientation="h", y=-0.2))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown("**📦 ラインごとの1か月の内訳**")
            comp = res[["世代", *BREAKDOWN_COLORS]].melt(id_vars="世代", var_name="内訳", value_name="枚数")
            fig = px.bar(comp, y="世代", x="枚数", color="内訳", orientation="h",
                         color_discrete_map=BREAKDOWN_COLORS)
            fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None,
                              legend=dict(orientation="h", y=-0.2))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**📋 ラインごとの詳細**")
        show = res[["世代", "装置台数", *SKILL_NAMES, "合計人数", "稼働率", "不良率", *BREAKDOWN_COLORS]].copy()
        show["稼働率"] *= 100
        show["不良率"] *= 100
        st.dataframe(
            show.style.format({"稼働率": "{:.1f}%", "不良率": "{:.2f}%",
                               **{c: "{:,.0f} 枚" for c in BREAKDOWN_COLORS}}),
            hide_index=True, use_container_width=True,
        )
        st.caption("生産数 ＝ 投入数 − 停止で作れなかった数 − 不良数。稼働率・不良率は実績から学習した予想値です。")

        with st.expander("📊 もとにした実績データ：担当人数と不良数の関係"):
            st.plotly_chart(relation_figure(hist, params), use_container_width=True)
            st.caption("灰色の点 ＝ 過去の実績（1点が1か月）、線 ＝ 実績から学習した傾向。"
                       "担当が少ない月ほど不良が多い、という関係をもとに計算しています。")

# ---------------- パターン比較 ----------------
with tab_cmp:
    st.subheader("条件の違うパターンを並べて、生産数を比べます")
    st.caption("表は書き換えたり、行を追加・削除したりできます。パターンごとに、生産数が最大になる人員配置を計算します。")
    default_patterns = pd.DataFrame([
        {"パターン": "A：今の体制", "ライン数": n_all, "新人": 5, "中堅": 10, "ベテラン": 5},
        {"パターン": "B：中堅を4人増やす", "ライン数": n_all, "新人": 5, "中堅": 14, "ベテラン": 5},
        {"パターン": "C：新人3人がベテランに成長", "ライン数": n_all, "新人": 2, "中堅": 10, "ベテラン": 8},
        {"パターン": "D：ベテラン2人が退職", "ライン数": n_all, "新人": 5, "中堅": 10, "ベテラン": 3},
    ])
    patterns = st.data_editor(
        default_patterns, num_rows="dynamic", hide_index=True, use_container_width=True,
        column_config={
            "パターン": st.column_config.TextColumn(default="新しいパターン"),
            "ライン数": st.column_config.NumberColumn(min_value=1, max_value=n_all, step=1, default=n_all),
            **{s: st.column_config.NumberColumn(f"{s}（人）", min_value=0, max_value=MAX_PER_SKILL, step=1, default=d)
               for s, d in {"新人": 5, "中堅": 10, "ベテラン": 5}.items()},
        },
        key="patterns",
    )

    summary, alloc_rows, skipped, seen = [], [], [], set()
    for i, r in patterns.reset_index(drop=True).iterrows():
        name = str(r["パターン"]).strip() if pd.notna(r["パターン"]) else ""
        name = name or f"パターン{i + 1}"
        if name in seen:
            name = f"{name}（{i + 1}）"
        seen.add(name)
        if r[["ライン数", *SKILL_NAMES]].isna().any():
            skipped.append(name)
            continue
        n = int(min(max(r["ライン数"], 1), n_all))
        pat_staff = {s: int(min(max(r[s], 0), MAX_PER_SKILL)) for s in SKILL_NAMES}
        if sum(pat_staff.values()) < MIN_PER_LINE * n:
            skipped.append(name)
            continue
        p = params_all.iloc[:n]
        res_p = result_table(p, optimize(p, pat_staff))
        summary.append({"パターン": name, "ライン数": n, "合計人数": sum(pat_staff.values()),
                        **{c: res_p[c].sum() for c in BREAKDOWN_COLORS}})
        alloc_rows += [{"パターン": name, "世代": row["世代"], "人数": row["合計人数"]}
                       for row in res_p.to_dict("records")]

    if skipped:
        st.warning("計算できなかったパターン：" + "、".join(skipped) + "（空欄がある、または人数がライン数より少ない）")
    if not summary:
        st.info("パターンを1行以上入力してください。")
    else:
        sdf = pd.DataFrame(summary)
        sdf[list(BREAKDOWN_COLORS)] = sdf[list(BREAKDOWN_COLORS)].round()
        sdf["生産数の差"] = sdf["生産数"] - sdf["生産数"].iloc[0]
        colors = {name: PATTERN_COLORS[i % len(PATTERN_COLORS)] for i, name in enumerate(sdf["パターン"])}
        labels = [f"{v:,.0f} 枚" + ("（基準）" if i == 0 else f"（{d:+,.0f}）")
                  for i, (v, d) in enumerate(zip(sdf["生産数"], sdf["生産数の差"]))]

        st.markdown("**🏆 パターンごとの生産数（枚/月）**")
        fig = px.bar(sdf, y="パターン", x="生産数", color="パターン", orientation="h",
                     color_discrete_map=colors, text=labels)
        fig.update_traces(textposition="inside", insidetextanchor="end")
        fig.update_layout(height=90 + 55 * len(sdf), margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                          xaxis_title="生産数（枚/月）", yaxis_title=None)
        fig.update_yaxes(categoryorder="array", categoryarray=list(sdf["パターン"])[::-1])  # 表と同じく上から順に
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            sdf[["パターン", "ライン数", "合計人数", "生産数", "生産数の差", "不良数", "停止で作れなかった数"]]
            .style.format({"合計人数": "{:,.0f} 人", "生産数の差": "{:+,.0f} 枚",
                           **{c: "{:,.0f} 枚" for c in BREAKDOWN_COLORS}}),
            hide_index=True, use_container_width=True,
        )
        st.caption("生産数の差は、いちばん上のパターンと比べた差です。")

        st.markdown("**👥 パターンごとの、おすすめの人員配置（ラインごとの人数）**")
        fig = px.bar(pd.DataFrame(alloc_rows), x="世代", y="人数", color="パターン", barmode="group",
                     color_discrete_map=colors, text="人数")
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None,
                          yaxis_title="人数（人）", legend=dict(orientation="h", y=-0.15, title=None))
        st.plotly_chart(fig, use_container_width=True)
