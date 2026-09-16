"""Streamlit 交互式舆情监控看板。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nlp_engine import SentimentAnalyzer  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402
from src.utils import load_config  # noqa: E402

st.set_page_config(page_title="舆情监控看板", page_icon="📡", layout="wide")
config = load_config()


@st.cache_data(ttl=300)
def load_dashboard(keyword: str, hours: int) -> dict[str, pd.DataFrame]:
    return run_pipeline(keyword, hours, mode="mock")


st.title("📡 实时社交媒体情绪分析与舆情监控")
st.caption("演示模式：使用可重复生成的模拟社交媒体数据；生产环境请接入合规的数据源。")

with st.sidebar:
    st.header("查询参数")
    keyword = st.text_input("监控关键词", value="新能源汽车")
    hours = st.slider("回溯小时数", min_value=12, max_value=168, value=48, step=12)
    if st.button("刷新分析"):
        load_dashboard.clear()

data = load_dashboard(keyword, hours)
processed = data["processed"]
metrics = data["metrics"]
forecast = data["forecast"]
analyzer = SentimentAnalyzer(config)

if processed.empty:
    st.warning("当前没有可展示的数据。")
    st.stop()

negative_ratio = float(processed["sentiment"].eq("negative").mean())
recent_count = float(metrics["post_count"].tail(1).iloc[0])
baseline = float(metrics["post_count"].mean())
negative_limit = float(config["forecast"]["negative_alert_ratio"])
heat_limit = float(config["forecast"]["heat_alert_multiplier"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("帖子总量", f"{len(processed):,}")
col2.metric("负面占比", f"{negative_ratio:.1%}", delta=f"阈值 {negative_limit:.0%}")
col3.metric("平均情绪分", f"{processed['sentiment_score'].mean():.2f}")
col4.metric("最近小时热度", f"{recent_count:.0f}", delta=f"基线 {baseline:.0f}")

if negative_ratio >= negative_limit or (baseline and recent_count >= baseline * heat_limit):
    st.error("⚠️ 触发舆情预警：请人工核查负面内容和热度来源。")
else:
    st.success("✅ 当前指标未触发规则型预警。")

left, right = st.columns(2)
with left:
    sentiment_counts = processed["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
    sentiment_counts["sentiment"] = sentiment_counts["sentiment"].map(
        {"positive": "正面", "neutral": "中性", "negative": "负面"}
    )
    st.subheader("情绪分布")
    st.plotly_chart(px.pie(sentiment_counts, names="sentiment", values="count", hole=0.35), use_container_width=True)

with right:
    st.subheader("小时热度与 24 小时预测")
    st.plotly_chart(px.line(forecast, x="hour", y="post_count", color="kind", markers=True), use_container_width=True)

left, right = st.columns([1, 2])
with left:
    st.subheader("核心观点关键词")
    keywords = analyzer.keywords(processed["text"], top_n=int(config["nlp"]["top_keywords"]))
    st.dataframe(pd.DataFrame(keywords, columns=["关键词", "次数"]), hide_index=True, use_container_width=True)
with right:
    st.subheader("来源与情绪")
    source_counts = processed.groupby(["source", "sentiment"], as_index=False).size().rename(columns={"size": "count"})
    st.plotly_chart(px.bar(source_counts, x="source", y="count", color="sentiment", barmode="group"), use_container_width=True)

st.subheader("最新舆情明细")
display_columns = ["published_at", "source", "text", "sentiment", "sentiment_score", "engagement"]
st.dataframe(processed.sort_values("published_at", ascending=False)[display_columns].head(100), hide_index=True, use_container_width=True)

