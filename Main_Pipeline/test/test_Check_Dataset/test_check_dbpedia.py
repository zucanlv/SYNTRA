from datasets import load_dataset
import json
import random

# 加载 default、queries、corpus
cache_dir = "/data/share/project/shared_datasets/mteb"
ds_default = load_dataset("mteb/dbpedia", "default", cache_dir=cache_dir)
ds_queries = load_dataset("mteb/dbpedia", "queries", cache_dir=cache_dir)
ds_corpus = load_dataset("mteb/dbpedia", "corpus", cache_dir=cache_dir)

# 建立 _id -> 整行 的映射，便于按 id 查找
queries_list = ds_queries["queries"]
corpus_list = ds_corpus["corpus"]
query_by_id = {row["_id"]: row for row in queries_list}
corpus_by_id = {row["_id"]: row for row in corpus_list}

# 从 test 中筛出 score==2；在最多 5 条里保证至少 1 条 query 以 '?' 结尾
test_split = ds_default["test"]
score_2_ds = test_split.filter(lambda row: row["score"] == 2)
if len(score_2_ds) == 0:
    raise RuntimeError("test 中没有 score==2 的样本，请检查数据或标签含义。")


def _query_ends_with_question(row):
    return query_by_id[row["query-id"]]["text"].rstrip().endswith("?")


rows = score_2_ds.to_list()
with_question = [r for r in rows if _query_ends_with_question(r)]
without_question = [r for r in rows if not _query_ends_with_question(r)]
if not with_question:
    raise RuntimeError(
        "score==2 的样本中没有 query 以 '?' 结尾的项，无法按要求抽样。"
    )

rng = random.Random(123)
rng.shuffle(with_question)
rng.shuffle(without_question)

n_take = min(5, len(rows))
selected = [with_question[0]]
seen = {(selected[0]["query-id"], selected[0]["corpus-id"])}
pool = with_question[1:] + without_question
for row in pool:
    if len(selected) >= n_take:
        break
    key = (row["query-id"], row["corpus-id"])
    if key in seen:
        continue
    seen.add(key)
    selected.append(row)

cols = score_2_ds.column_names
first_five = {c: [r[c] for r in selected] for c in cols}
n_rows = len(selected)

print("=" * 80)
print(
    f"Default 数据集 test 集合 score==2 随机 {n_rows} 条（含至少 1 条 query 以 ? 结尾）："
    "query 与 corpus 具体内容"
)
print("=" * 80)

# 用于 Few_Shot_Example.py 兼容格式的列表
apps_few_shot_format = []

for i in range(n_rows):
    query_id = first_five["query-id"][i]
    corpus_id = first_five["corpus-id"][i]
    score = first_five["score"][i]

    query_row = query_by_id[query_id]
    corpus_row = corpus_by_id[corpus_id]

    # 收集 Few_Shot_Example 兼容格式：query + Positive(corpus text)
    apps_few_shot_format.append({
        "query": query_row["text"],
        "Positive": corpus_row["text"]
    })

    print(f"\n【第 {i+1} 条】")
    print(f"  query-id:  {query_id}")
    print(f"  corpus-id: {corpus_id}")
    print(f"  score:     {score}")
    print("-" * 60)
    print("  QUERY 内容:")
    print(f"    {query_row['text']}")
    print("-" * 60)
    print("  CORPUS 内容:")
    print(f"    title: {corpus_row.get('title', '')}")
    text = corpus_row["text"]
    print(f"    text:  {text}")
    print("=" * 80)

# 输出 Few_Shot_Example.py 中 "apps" 兼容的 Python 结构（可直接复制到 Few_Shot_Example.py）
print("\n" + "=" * 80)
print("Few_Shot_Example 兼容格式（可复制到 Few_Shot_Example.py）：")
print("=" * 80)
print('    "dbpedia": ', end="")
print(json.dumps(apps_few_shot_format, indent=4, ensure_ascii=False))
print("=" * 80)


'''
================================================================================

【第 1 条】
  query-id:  QALD2_tr-59
  corpus-id: <dbpedia:Jimmy_Settle>
  score:     2.0
------------------------------------------------------------
  QUERY 内容:
    Give me all people with first name Jimmy.
------------------------------------------------------------
  CORPUS 内容:
    title: Jimmy Settle
    text:  Jimmy Settle was an English professional footballer. A fast-paced inside or outside right, he could have chosen sprinting if he had not taken up football.Settle played for Bolton and Bury before joining Everton in 1899, with whom he won the FA Cup in 1906. Settle was Football League Division One's leading goalscorer for the 1901-02 season with 18 goals, the lowest of the highest totals achieved in the English top-flight to date.
================================================================================


【第 5 条】
  query-id:  INEX_XER-138
  corpus-id: <dbpedia:Forillon_National_Park>
  score:     2.0
------------------------------------------------------------
  QUERY 内容:
    National Parks East Coast Canada US
------------------------------------------------------------
  CORPUS 内容:
    title: Forillon National Park
    text:  Forillon National Park, one of 42 national parks and park reserves across Canada, is located at the outer tip of the Gaspé Peninsula of Quebec and covers 244 km2 (94 sq mi). Created in 1970, Forillon was the first national park in Quebec. The park includes forests, sea coast, salt marshes, sand dunes, cliffs, and the Eastern End of the Appalachians. The park includes nesting colonies of sea birds and whales, seals, black bears, moose, and other woodland animals.
================================================================================

================================================================================
Few_Shot_Example 兼容格式（可复制到 Few_Shot_Example.py）：
================================================================================
    "dbpedia": [
    {
        "query": "Give me all people with first name Jimmy.",
        "Positive": "Jimmy Settle was an English professional footballer. A fast-paced inside or outside right, he could have chosen sprinting if he had not taken up football.Settle played for Bolton and Bury before joining Everton in 1899, with whom he won the FA Cup in 1906. Settle was Football League Division One's leading goalscorer for the 1901-02 season with 18 goals, the lowest of the highest totals achieved in the English top-flight to date."
    },
    {
        "query": "National Parks East Coast Canada US",
        "Positive": "Forillon National Park, one of 42 national parks and park reserves across Canada, is located at the outer tip of the Gaspé Peninsula of Quebec and covers 244 km2 (94 sq mi). Created in 1970, Forillon was the first national park in Quebec. The park includes forests, sea coast, salt marshes, sand dunes, cliffs, and the Eastern End of the Appalachians. The park includes nesting colonies of sea birds and whales, seals, black bears, moose, and other woodland animals."
    }
]
================================================================================












================================================================================
Default 数据集 test 集合 score==2 随机 5 条：query 与 corpus 具体内容
================================================================================

【第 1 条】
  query-id:  QALD2_tr-91
  corpus-id: <dbpedia:Wing_Wah>
  score:     2.0
------------------------------------------------------------
  QUERY 内容:
    Which organizations were founded in 1950?
------------------------------------------------------------
  CORPUS 内容:
    title: Wing Wah
    text:  Wing Wah (Chinese: 榮華) is a Hong Kong based restaurant chain and food manufacturer owned by Wing Wah Food Manufactory Limited (榮華食品製造 業有限公司). The company is most renowned for its mooncakes, and also produces: Chinese sausage, cakes, and teas.
================================================================================

【第 2 条】
  query-id:  INEX_XER-95
  corpus-id: <dbpedia:Forrest_Gump>
  score:     2.0
------------------------------------------------------------
  QUERY 内容:
    Tom Hanks movies where he plays a leading role.
------------------------------------------------------------
  CORPUS 内容:
    title: Forrest Gump
    text:  Forrest Gump is a 1994 American epic romantic-comedy-drama film based on the 1986 novel of the same name by Winston Groom. The film was directed by Robert Zemeckis and stars Tom Hanks, Robin Wright, Gary Sinise, Mykelti Williamson, and Sally Field.
================================================================================

【第 3 条】
  query-id:  SemSearch_LS-33
  corpus-id: <dbpedia:List_of_proposed_provinces_and_territories_of_Canada>
  score:     2.0
------------------------------------------------------------
  QUERY 内容:
    provinces and territories of Canada
------------------------------------------------------------
  CORPUS 内容:
    title: List of proposed provinces and territories of Canada
    text:  Since Canadian Confederation in 1867, there have been several proposals for new Canadian provinces and territories. Since 1982, the current Constitution of Canada requires an amendment ratified by seven provincial legislatures representing at least half of the national population for the creation of a new province while the creation of a new territory requires only an act of Parliament.
================================================================================

【第 4 条】
  query-id:  QALD2_tr-41
  corpus-id: <dbpedia:Ribadeo_FC>
  score:     2.0
------------------------------------------------------------
  QUERY 内容:
    Give me all soccer clubs in Spain.
------------------------------------------------------------
  CORPUS 内容:
    title: Ribadeo FC
    text:  Ribadeo Fútbol Club is a Spanish football team based in Ribadeo, Province of Lugo, in the autonomous community of Galicia. Founded in 1913 it currently plays in Tercera División – Group 1, holding home games at Estadio Municipal Pepe Barrera, which has a capacity of 2,500 spectators.
================================================================================

【第 5 条】
  query-id:  SemSearch_LS-39
  corpus-id: <dbpedia:Idalium>
  score:     2.0
------------------------------------------------------------
  QUERY 内容:
    ten ancient Greek city-kingdoms of Cyprus
------------------------------------------------------------
  CORPUS 内容:
    title: Idalium
    text:  Idalion or Idalium (Greek: Ιδάλιον, Idalion) was an ancient city in Cyprus, in modern Dali, Nicosia District. The city was founded on the copper trade in the 3rd millennium BCE. Its name in the 8th century BCE was "Ed-di-al" as it appears on the Sargon Stele of 707 BCE, and a little later on the nl:Prism of Esarhaddon.
================================================================================

================================================================================
Few_Shot_Example 兼容格式（可复制到 Few_Shot_Example.py）：
================================================================================
    "dbpedia": [

    {
        "query": "Tom Hanks movies where he plays a leading role.",
        "Positive": "Forrest Gump is a 1994 American epic romantic-comedy-drama film based on the 1986 novel of the same name by Winston Groom. The film was directed by Robert Zemeckis and stars Tom Hanks, Robin Wright, Gary Sinise, Mykelti Williamson, and Sally Field."
    },
    {
        "query": "provinces and territories of Canada",
        "Positive": "Since Canadian Confederation in 1867, there have been several proposals for new Canadian provinces and territories. Since 1982, the current Constitution of Canada requires an amendment ratified by seven provincial legislatures representing at least half of the national population for the creation of a new province while the creation of a new territory requires only an act of Parliament."
    },
    {
        "query": "ten ancient Greek city-kingdoms of Cyprus",
        "Positive": "Idalion or Idalium (Greek: Ιδάλιον, Idalion) was an ancient city in Cyprus, in modern Dali, Nicosia District. The city was founded on the copper trade in the 3rd millennium BCE. Its name in the 8th century BCE was \"Ed-di-al\" as it appears on the Sargon Stele of 707 BCE, and a little later on the nl:Prism of Esarhaddon."
    }
]
================================================================================
'''