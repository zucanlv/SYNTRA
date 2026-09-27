# Human Annotation for LLM Evaluation

本仓库提供一个本地标注网站，用于评估 LLM 对 `(query, document)` pair 的 0–3 相关性评分。样本包含 MSMARCO、Text2SQL 和 TheoremQA-Theorems，每个数据集 50 条，共 150 条。

网站只在本机运行。每次选择分数后，程序立即更新本地 CSV；关闭网站或重启电脑不会丢失已保存的进度。

## 标注者：开始标注

需要 Python 3.10 或更高版本，无需安装第三方 Python 包。

```bash
git clone <REPOSITORY_URL>
cd human-annotation-llm-eval
python app.py --annotator annotator_1
```

请将 `annotator_1` 替换为管理员分配的 ID。程序会打开浏览器；如果浏览器没有自动打开，请访问终端显示的本地地址。

页面按照以下顺序展示数据集：

1. MSMARCO
2. Text2SQL
3. TheoremQA-Theorems

每个区块内部的样本顺序由 annotator ID 确定。三名标注者看到相同的 150 条数据，但区内顺序不同。

每条 query 和 document 均同时展示英文原文与中文人工翻译。中文仅用于辅助理解；评分时请以英文原文、SQL 和数学公式为准。Text2SQL 的 document 是 SQL 代码，因此中文区域会说明 SQL 的实际语义，并保留 SQL 关键字和字段名。

## 如何评分

页面左侧完整显示该数据集运行时实际使用的英文 annotation instruction。请直接根据 instruction 判断，并在右侧选择一个分数：

| 分数 | 页面标签 |
| ---: | --- |
| 0 | Easy negative |
| 1 | Hard negative |
| 2 | Positive |
| 3 | Strong positive |

人工标注只需选择分数，无需输出 instruction 中要求 LLM 生成的 JSON 或 reasoning。

可使用鼠标或以下快捷键：

- `0`–`3`：保存对应分数
- `←` / `→`：上一条 / 下一条
- `U`：跳到下一条未标注样本

只有服务器确认 CSV 已写入后，页面才会前进。若页面显示“保存失败”，当前样本不会前进；请保留页面并重试。

## 保存与恢复

分数实时保存到：

```text
annotations/<annotator_id>.csv
```

请勿在标注过程中手工编辑该文件。停止网站时在终端按 `Ctrl+C`。再次运行相同命令即可恢复进度：

```bash
python app.py --annotator annotator_1
```

如果默认端口被占用，可指定其他端口：

```bash
python app.py --annotator annotator_1 --port 8877
```

## 完成与提交

页面显示 `150 / 150` 后，在仓库目录运行：

```bash
python scripts/validate_annotations.py annotations/annotator_1.csv
```

看到 `"valid": true` 后，将该 CSV 发送给实验管理员。不要提交其他标注者的文件，也不要重命名 CSV；文件名必须与 annotator ID 一致。

## 管理员：合并三份结果

收齐三个 CSV 后运行：

```bash
python scripts/merge_annotations.py \
  submissions/annotator_1.csv \
  submissions/annotator_2.csv \
  submissions/annotator_3.csv \
  --output merged_annotations.csv
```

合并结果只包含三名标注者的原始分数，不计算投票结果或 ground truth。

## 管理员：投票与 LLM 一致性分析

当前三份标注和私有 LLM answer key 可直接运行：

```bash
python scripts/analyze_agreement.py
```

分析将 `2/3` 合并为 `positive`、`1` 映射为 `hard_negative`、`0` 映射为
`easy_negative`，并以三人多数票生成最终人工标签。三分类出现三方各一票时，
使用有序类别的中位数 `hard_negative`，同时标记为需要人工复核。二分类分析再将
两种 negative 合并。

结果输出到 `results/`：

- `human_consensus.csv`：逐样本人工最终标签、原始三人分数和投票状态；
- `llm_human_disagreements.csv`：LLM 与人工三分类不一致的样本；
- `agreement_report.json`：总体、分数据集及排除三方平票后的三分类/二分类指标。

## 管理员：重新抽样

标注者无需执行本节。准备数据需要 `ijson`，并会在 `private/` 中生成 answer key 和审计 manifest：

```bash
python -m pip install -r requirements-admin.txt
python scripts/prepare_samples.py \
  --dataset msmarco=/private/path/to/msmarco/Annotated_Main.json \
  --dataset text2sql=/private/path/to/text2sql/Annotated_Main.json \
  --dataset theoremqa-theorems=/private/path/to/theoremqa/Annotated_Main.json \
  --sample-count 50 \
  --seed 20260819
```

`private/` 已被 Git 忽略。不要提交、复制或发送其中的文件给标注者。

## 测试

```bash
python -m unittest discover -s tests -v
python -m py_compile app.py annotation_app/*.py scripts/*.py
```

## 数据盲化

公开的 `data/samples.json` 每条记录只包含 `sample_id`、`dataset`、`query` 和 `document`。仓库不包含 LLM score、classification、reasoning、检索 score、rank、源 `doc_id` 或 `is_original`。

数学公式使用仓库内置的 KaTeX v0.18.1 离线渲染。KaTeX 按 MIT License 分发，许可证见 `web/vendor/LICENSE.katex.txt`。
# syntra-human-anno
