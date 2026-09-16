# 项目汇总：实时社交媒体情绪分析与舆情监控系统

## 1. 项目定位

本项目针对突发事件、品牌危机和产品运营中的舆情监控需求，提供从数据采集、清洗、情感分类、趋势预测到可视化预警的完整原型。系统默认运行在模拟数据模式，不依赖第三方 API，即可完成本地演示。

## 2. 技术方案

| 层次 | 实现 | 可扩展方向 |
| --- | --- | --- |
| 数据采集 | 模拟社交媒体帖子、URL 去重、字段标准化 | 新闻 API、平台官方 API、爬虫、Redis |
| 数据处理 | 文本清洗、重复数据删除、时间字段标准化 | Kafka、Flink、Spark Streaming |
| NLP 分析 | 中英文轻量情感词典、极性分数、关键词统计 | BERT、RoBERTa、情感微调模型 |
| 趋势预测 | 小时级热度聚合、线性趋势基线 | ARIMA、Prophet、LSTM |
| 可视化 | Streamlit + Plotly | 多租户、权限、告警中心 |
| 工程化 | 配置文件、测试、GitHub Actions、部署文档 | Docker、CI/CD、监控与日志 |

## 3. 关键输出

- 明细数据：`data/03_processed/sentiment_posts.csv`
- 小时级趋势与预测：`reports/tables/hourly_forecast.csv`
- Streamlit 交互看板：运行 `streamlit run app/main.py`
- 单元测试：运行 `pytest -q`

## 4. 预警逻辑

看板根据负面占比超过配置阈值，或最近一小时热度超过历史均值指定倍数来提示风险。预警是规则型提示，不替代人工研判。

## 5. 生产化路线

1. 使用消息队列承接实时数据，把采集和分析解耦。
2. 使用 Redis/Bloom Filter 做分布式去重，使用数据库或对象存储保存原始数据。
3. 用人工标注数据评估 Precision、Recall、F1，并替换轻量词典模型。
4. 采用 ARIMA/Prophet/LSTM 与滚动回测比较预测效果。
5. 将告警发送到企业微信、钉钉、邮件或 PagerDuty，并记录确认状态。
6. 增加数据脱敏、访问控制、审计日志、限流和平台合规检查。

## 6. 运行命令

```bash
pip install -r requirements.txt
python -m src.pipeline --keyword "新能源汽车" --hours 48
streamlit run app/main.py
pytest -q
```

