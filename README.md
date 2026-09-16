# 实时社交媒体情绪分析与舆情监控系统

一个面向突发事件、品牌危机和产品运营场景的端到端舆情分析示例项目。项目采用 Cookiecutter Data Science 风格组织代码，支持模拟数据演示，也预留了真实 API、Redis 和 Transformer 模型的扩展接口。

## 功能概览

- 模拟社交媒体采集：生成不同来源、不同情绪的帖子，并按 URL 去重
- 数据清洗：空白归一化、去重、无效文本过滤
- 情感分析：轻量中英文词典模型，输出 `positive`、`neutral`、`negative` 和 `score`
- 趋势预测：基于最近 24 小时热度的线性趋势预测未来 24 小时
- Streamlit 看板：情绪分布、热度趋势、关键词、预警信息和原始数据
- 测试覆盖：采集去重、情感分类、关键词提取、趋势预测
- 工程化配置：YAML 配置、环境变量示例、Conda 环境文件、GitHub Actions

## 快速开始

```bash
git clone <your-repository-url>
cd social-sentiment-monitor
python -m venv .venv
# Windows PowerShell
.\\.venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline --keyword "新能源汽车" --hours 48
streamlit run app/main.py
```

首次运行会在 `data/03_processed/` 生成处理后的 CSV，并在 `reports/tables/` 生成汇总数据。

## 目录结构

```text
.
├── app/main.py                 # Streamlit 看板
├── config/config.yaml          # 默认配置
├── data/01_raw                 # 原始采集数据
├── data/02_intermediate        # 中间处理数据
├── data/03_processed           # NLP 处理后的数据
├── models/                     # 模型文件目录
├── notebooks/                  # EDA 与模型原型说明
├── reports/                    # 图表与汇总表
├── src/                        # 采集、NLP、预测和流水线
└── tests/                      # 单元测试
```

## 真实数据接入

默认使用 `--mode mock`，避免 API 密钥和平台合规问题。接入真实数据时，在 `src/scraper.py` 的 `SocialMediaCollector.fetch_from_api` 中替换 HTTP 请求，并将返回结果统一映射为 `source`、`author`、`text`、`url`、`published_at`、`engagement` 字段。

API 密钥建议通过环境变量注入，不要写入 Git。Redis 不可用时，当前示例会回退到进程内 URL 集合，便于本地演示。

## 测试与部署

```bash
pytest -q
ruff check .
```

Streamlit Community Cloud 和 Hugging Face Spaces 均可使用根目录的 `requirements.txt` 部署，入口为 `app/main.py`。生产环境请增加数据库或对象存储、消息队列、人工标注评估、告警通知、权限审计和隐私合规处理。

## 免责声明

本项目用于工程演示和原型验证。模拟数据不代表真实舆情；真实部署时请遵守目标平台服务条款、隐私保护和数据抓取相关法律法规。情绪分类结果不应作为唯一的危机决策依据。

