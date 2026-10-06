"""
メンテナンス担当 最適配置シミュレータ
------------------------------------------------
NANDフラッシュメモリ工場（製品の世代ごとに3つのライン）を題材に、
過去の実績データ（スキル別の担当人数・不良数・稼働率）から
「担当を増やすと、不良や装置停止がどれくらい減るか」をラインごとに学習し、
生産数（または金額）が最大になる人員配置を計算します。

起動:  streamlit run app.py
"""
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

st.set_page_config(page_title="メンテナンス担当 最適配置シミュレータ", page_icon="🏭", layout="wide")

# スキル区分ごとの「1人でこなせるメンテの量」（中堅 = 1 とした想定。画面には出さない）
SKILLS = {"新人": 0.6, "中堅": 1.0, "ベテラン": 1.4}
SKILL_NAMES = list(SKILLS.keys())
SKILL_FACTORS = np.array(list(SKILLS.values()))
SKILL_COLORS = {"新人": "#9ecae1", "中堅": "#3182bd", "ベテラン": "#08519c"}
MIN_PER_LINE = 1  # 各ラインに最低1人は配置する
MAX_PER_SKILL = 10  # スキルごとの人数の上限（合計30人まで。計算時間を短く保つため）
DEFAULT_STAFF = {"新人": 4, "中堅": 8, "ベテラン": 3}
DEFAULT_PRICES = {"第1世代": 60, "第2世代": 100, "第3世代": 150}  # ウエハ1枚の単価（万円）

HISTORY_COLUMNS = ["年月", "世代", "装置台数", *SKILL_NAMES, "投入数", "稼働率(%)", "不良数"]
BREAKDOWN_COLORS = {"生産数": "#2ca02c", "不良数": "#d62728", "停止で作れなかった数": "#bbbbbb"}
UNITS = {"生産数": ("枚/月", 1.0, ",.0f"), "金額": ("億円/月", 1e-4, ",.1f")}  # 単位, 表示倍率, 書式

# サンプル実績データを作るときの「本当の」性質（学習で当てにいく値）
GENERATIONS = [
    # 世代, 装置台数, 月間投入数(枚), 不良率(放置), 不良率(下限), 稼働率(放置), 稼働率(上限), 効き目
    ("第1世代", 6, 9000, 0.15, 0.010, 0.70, 0.98, 9.0),   # 成熟した世代：安定していて不良が出にくい
    ("第2世代", 12, 15000, 0.25, 0.020, 0.55, 0.97, 7.0),  # 主力の世代：装置が多く、生産量も一番多い
    ("第3世代", 8, 9000, 0.40, 0.040, 0.45, 0.95, 5.0),   # 最新の世代：工程が難しく、不良が出やすい
]


# ---------------------------------------------------------------------------
# サンプル実績データ（架空）
# ---------------------------------------------------------------------------
@st.cache_data
def make_sample_history(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    months = pd.period_range("2024-10", periods=24, freq="M")
    for name, mach, cap, d0, dmin, u0, umax, k in GENERATIONS:
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


def simulate_line(p, work):
    """1ラインの予想。work（スキルを考えたメンテの量）は数値でも配列でもよい"""
    x = work / max(p["装置台数"], 1)
    dr = defect_curve(x, p["不良率_放置"], p["不良率_下限"], p["k_不良"])
    up = uptime_curve(x, p["稼働率_放置"], p["稼働率_上限"], p["k_稼働"])
    processed = p["月間投入数"] * up
    produced = processed * (1 - dr)
    return {
        "稼働率": up, "不良率": dr,
        "生産数": produced, "不良数": processed * dr,
        "停止で作れなかった数": p["月間投入数"] * (1 - up),
        "金額": produced * p["単価"],  # 万円
    }


@st.cache_data(show_spinner=False)
def solve_all(params: pd.DataFrame, total: int, objective: str):
    """動的計画法：ラインを1本ずつ増やしながら、合計 total 人以下のすべての人員構成
    （新人, 中堅, ベテラン）について「生産数（または金額）が最大になる配置」を一度に求める。
    全通りを調べたのと同じ、本当に一番よい答えになる。"""
    recs = params.to_dict("records")
    shape = (total + 1,) * len(SKILL_NAMES)
    # 1ラインに置く人数の組（新人, 中堅, ベテラン）
    choices = np.array([d for d in np.ndindex(*shape) if MIN_PER_LINE <= sum(d) <= total])
    best = np.full(shape, -np.inf)  # best[u] = これまでのラインに u 人を使ったときの最大値
    best[(0,) * len(shape)] = 0.0
    picks = []
    for p in recs:
        gains = simulate_line(p, choices @ SKILL_FACTORS)[objective]
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
    return best, picks, choices


def backtrack(picks, choices, team) -> np.ndarray:
    """最後のラインから順に、選んだ人数の組をたどって配置を復元"""
    alloc = np.zeros((len(picks), len(SKILL_NAMES)), dtype=int)
    u = tuple(team)
    for i in range(len(picks) - 1, -1, -1):
        alloc[i] = choices[picks[i][u]]
        u = tuple(np.subtract(u, alloc[i]))
    return alloc


def result_table(params: pd.DataFrame, alloc: np.ndarray) -> pd.DataFrame:
    rows = []
    for p, row in zip(params.to_dict("records"), alloc):
        rows.append({"世代": p["世代"], "装置台数": int(p["装置台数"]),
                     **dict(zip(SKILL_NAMES, map(int, row))), "合計人数": int(row.sum()),
                     **simulate_line(p, float(row @ SKILL_FACTORS))})
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


def growth_path(comp) -> list:
    """内訳 comp と同じ比率を保ちながら1人ずつ増やしたときの、1人目〜全員のチーム"""
    total = sum(comp)
    counts = [0] * len(comp)
    path = []
    for x in range(1, total + 1):
        s = max(range(len(comp)), key=lambda j: x * comp[j] / total - counts[j])  # 比率に一番足りないスキル
        counts[s] += 1
        path.append(tuple(counts))
    return path


def team_label(team) -> str:
    return "・".join(f"{s}{n}" for s, n in zip(SKILL_NAMES, team))


def composition_figure(best, comps, current, x_min: int, metric: str) -> go.Figure:
    """内訳ごとに、同じ比率で人数を増やしたときの最大値の折れ線（細線＝全通り、太線＝一番よい内訳と今の内訳）"""
    unit, scale, fmt = UNITS[metric]
    xs = list(range(x_min, sum(current) + 1))

    def series(comp):
        path = growth_path(comp)
        return [best[path[x - 1]] * scale for x in xs]

    gx, gy, gt = [], [], []
    for comp in comps:
        gx += xs + [None]
        gy += series(comp) + [None]
        gt += [team_label(comp)] * len(xs) + [None]
    fig = go.Figure()
    fig.add_scatter(x=gx, y=gy, text=gt, mode="lines", name=f"内訳ごとの線（全{len(comps)}通り）",
                    line=dict(color="rgba(140,140,140,0.35)", width=1),
                    hovertemplate="%{text}<br>%{x}人 → %{y:" + fmt + "} " + unit + "<extra></extra>")
    top = max(comps, key=lambda c: best[c])
    for comp, name, color in ((top, "一番よい内訳", "#2ca02c"), (current, "今の内訳", "#ff7f0e")):
        fig.add_scatter(x=xs, y=series(comp), mode="lines+markers", name=f"{name}（{team_label(comp)}）",
                        line=dict(color=color, width=4), marker=dict(size=6),
                        hovertemplate=name + "<br>%{x}人 → %{y:" + fmt + "} " + unit + "<extra></extra>")
    fig.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.15),
                      xaxis=dict(title="人数（人）", dtick=1), yaxis=dict(title=f"{metric}（{unit}）"))
    return fig


def relation_figure(hist: pd.DataFrame, params: pd.DataFrame) -> go.Figure:
    """ラインごとに「担当人数 × 生産数」の散布図（実績）と、学習した傾向の線"""
    recs = params.to_dict("records")
    cols = min(3, len(recs))
    rows = math.ceil(len(recs) / cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[p["世代"] for p in recs],
                        horizontal_spacing=0.07, vertical_spacing=0.16)
    for i, p in enumerate(recs):
        r, c = i // cols + 1, i % cols + 1
        g = hist[hist["世代"] == p["世代"]]
        people = g[SKILL_NAMES].sum(axis=1)
        produced = g["投入数"] * g["稼働率(%)"] / 100 - g["不良数"]  # 実績の生産数
        xs = np.arange(1, max(int(people.max()) + 2, 6) + 1)
        fig.add_scatter(x=people, y=produced, mode="markers", name="実績（1点＝1か月）",
                        marker=dict(color="#7f7f7f", size=8, opacity=0.55),
                        customdata=g["年月"], hovertemplate="%{customdata}<br>担当 %{x}人<br>生産 %{y:,.0f}枚<extra></extra>",
                        legendgroup="実績", showlegend=(i == 0), row=r, col=c)
        trend = [simulate_line(p, n * p["1人あたりの量"])["生産数"] for n in xs]
        fig.add_scatter(x=xs, y=trend, mode="lines", name="学習した傾向", line=dict(color="#2ca02c", width=3),
                        hovertemplate="担当 %{x}人 → 生産 約 %{y:,.0f}枚<extra></extra>",
                        legendgroup="傾向", showlegend=(i == 0), row=r, col=c)
    fig.update_xaxes(title_text="担当人数（人）", dtick=1)
    fig.update_yaxes(title_text="生産数（枚/月）")
    fig.update_layout(height=300 * rows + 40, margin=dict(l=0, r=0, t=80, b=0),
                      legend=dict(orientation="h", yref="container", yanchor="top", y=0.99, x=0))
    return fig


def fmt_count(c: float) -> str:
    for unit, name in ((1e16, "京"), (1e12, "兆"), (1e8, "億"), (1e4, "万")):
        if c >= unit:
            v = c / unit
            return f"約{v:,.1f}{name}通り" if v < 10 else f"約{v:,.0f}{name}通り"
    return f"{c:,.0f}通り"


def fmt_int(v):
    return f"{v:,.0f}"


# ---------------------------------------------------------------------------
# 画面
# ---------------------------------------------------------------------------
hist = make_sample_history()
params_fit = fit_models(hist)
lines = list(params_fit["世代"])
n_lines = len(lines)

st.title("🏭 メンテナンス担当 最適配置シミュレータ")
st.caption("※データはすべて架空のサンプルです。")

st.subheader("メンテ担当の人数を入れると、最適な人員配置を計算します")
staff = {}
for col, (s, default) in zip(st.columns(len(SKILL_NAMES)), DEFAULT_STAFF.items()):
    staff[s] = col.number_input(f"{s}（人）", 0, MAX_PER_SKILL, default, step=1)
total_staff = sum(staff.values())
current = tuple(staff[s] for s in SKILL_NAMES)

metric = st.radio("最大にする指標（グラフの縦軸）", list(UNITS), horizontal=True)
params = params_fit.copy()
params["単価"] = [float(DEFAULT_PRICES.get(line, 100)) for line in lines]
if metric == "金額":
    for col, (i, line) in zip(st.columns(n_lines), enumerate(lines)):
        params.loc[i, "単価"] = col.number_input(f"{line}の単価（万円/枚）", 0, 10000,
                                                 int(params.loc[i, "単価"]), step=10)
st.caption(f"メンテ担当 合計 {total_staff} 人　／　各ラインに最低1人は配置します。"
           "ベテランほど1人でこなせるメンテの量が多い想定です。")

if total_staff < MIN_PER_LINE * n_lines:
    st.error(f"人数が足りません：{n_lines} ラインに最低1人ずつ、{MIN_PER_LINE * n_lines} 人以上が必要です。")
    st.stop()

with st.spinner("最適な配置を計算中…"):
    best, picks, choices = solve_all(params, total_staff, metric)
res = result_table(params, backtrack(picks, choices, current))

cards = [("生産数（枚/月）", fmt_int(res["生産数"].sum())),
         ("不良数（枚/月）", fmt_int(res["不良数"].sum())),
         ("停止で作れなかった数（枚/月）", fmt_int(res["停止で作れなかった数"].sum()))]
if metric == "金額":
    cards.insert(0, ("金額（億円/月）", f"{res['金額'].sum() / 1e4:,.1f}"))
for col, (label, value) in zip(st.columns(len(cards)), cards):
    col.metric(label, value)
st.caption(f"💡 人の配置のしかたは全部で {fmt_count(count_allocations(staff, n_lines))}。"
           f"その中から、{metric}が最大になる配置をコンピュータが探し出しました。")

left, right = st.columns(2)
with left:
    st.markdown("**👥 おすすめの人員配置**")
    long = res.melt(id_vars="世代", value_vars=SKILL_NAMES, var_name="スキル", value_name="人数")
    fig = px.bar(long, y="世代", x="人数", color="スキル", orientation="h",
                 color_discrete_map=SKILL_COLORS, category_orders={"スキル": SKILL_NAMES})
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None,
                      legend=dict(orientation="h", y=-0.25))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.markdown("**📦 ラインごとの1か月の内訳**")
    comp = res[["世代", *BREAKDOWN_COLORS]].melt(id_vars="世代", var_name="内訳", value_name="枚数")
    fig = px.bar(comp, y="世代", x="枚数", color="内訳", orientation="h",
                 color_discrete_map=BREAKDOWN_COLORS)
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None,
                      legend=dict(orientation="h", y=-0.25))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

comps = [(a, b, total_staff - a - b) for a in range(total_staff + 1) for b in range(total_staff + 1 - a)]
st.markdown(f"**📈 人数と{metric}の関係（{total_staff}人の内訳 全{len(comps)}通り）**")
st.plotly_chart(composition_figure(best, comps, current, n_lines, metric), use_container_width=True)
st.caption(f"細い線は、新人・中堅・ベテランの内訳ごとに、同じ比率のまま人数を増やしたときの{metric}"
           "（各人数で一番よい配置にした場合）。人数を増やすと伸びがだんだん小さくなり、やがて頭打ちになります。")

st.markdown("**📋 ラインごとの詳細**")
show = res[["世代", "装置台数", *SKILL_NAMES, "合計人数", "稼働率", "不良率", *BREAKDOWN_COLORS]].copy()
show["稼働率"] *= 100
show["不良率"] *= 100
formats = {"稼働率": "{:.1f}%", "不良率": "{:.2f}%", **{c: "{:,.0f} 枚" for c in BREAKDOWN_COLORS}}
if metric == "金額":
    show["金額"] = res["金額"] / 1e4
    formats["金額"] = "{:,.1f} 億円"
st.dataframe(show.style.format(formats), hide_index=True, use_container_width=True)
st.caption("生産数 ＝ 投入数 − 停止で作れなかった数 − 不良数。稼働率・不良率は実績から学習した予想値です。")

with st.expander("📊 もとにした実績データ：担当人数と生産数の関係"):
    st.plotly_chart(relation_figure(hist, params), use_container_width=True)
    st.caption("灰色の点 ＝ 過去の実績（1点が1か月）、線 ＝ 実績から学習した傾向。"
               "担当が多い月ほど生産数が多く、ある人数を超えると増え方が小さくなる、という関係をもとに計算しています。")
