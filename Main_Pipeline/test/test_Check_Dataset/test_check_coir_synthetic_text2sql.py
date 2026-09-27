#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""查看 CoIR-Retrieval/synthetic-text2sql 的 5 个 golden query 和 golden positive。"""

from datasets import load_from_disk


DATASET_DIR = "/data/share/project/shared_datasets/DSA/datasets/CoIR-Retrieval__synthetic-text2sql"
LIMIT = 5


def build_index(dataset):
    return {row["_id"]: row for row in dataset}


def first_split(dataset_dict, preferred=("test", "valid", "train")):
    for split in preferred:
        if split in dataset_dict:
            return split, dataset_dict[split]
    split = next(iter(dataset_dict))
    return split, dataset_dict[split]


def print_golden_examples(limit=LIMIT):
    qrels = load_from_disk(f"{DATASET_DIR}/default")
    queries = load_from_disk(f"{DATASET_DIR}/queries")["queries"]
    corpus = load_from_disk(f"{DATASET_DIR}/corpus")["corpus"]

    query_by_id = build_index(queries)
    corpus_by_id = build_index(corpus)
    split_name, qrel_split = first_split(qrels)

    print(f"数据集: CoIR-Retrieval/synthetic-text2sql")
    print(f"qrels split: {split_name}")
    print("=" * 100)

    shown = 0
    for qrel in qrel_split:
        if qrel["score"] <= 0:
            continue

        query = query_by_id.get(qrel["query-id"])
        positive = corpus_by_id.get(qrel["corpus-id"])
        if query is None or positive is None:
            continue

        shown += 1
        print(f"第{shown}条")
        print(f"query-id: {qrel['query-id']}")
        print(f"golden query:\n{query['text']}")
        if query.get("context"):
            print(f"query context:\n{query['context']}")
        print(f"corpus-id: {qrel['corpus-id']}")
        print(f"golden positive:\n{positive['text']}")
        print("-" * 100)

        if shown >= limit:
            break

    if shown == 0:
        print("没有找到 score > 0 且 query/corpus 都能匹配到的样例。")


if __name__ == "__main__":
    print_golden_examples()

'''
```python
    "coir-synthetic-text2sql": [
            {
                    "query": "Which team has the highest number of wins in the 'basketball_games' table?",
                    "Positive": "SELECT name AS team, MAX(home_team_wins + away_team_wins) AS highest_wins FROM (SELECT name, CASE WHEN home_team = team_id AND home_team_score > away_team_score THEN 1 ELSE 0 END + CASE WHEN away_team = team_id AND away_team_score > home_team_score THEN 1 ELSE 0 END AS home_team_wins, CASE WHEN home_team = team_id AND home_team_score < away_team_score THEN 1 ELSE 0 END + CASE WHEN away_team = team_id AND away_team_score < home_team_score THEN 1 ELSE 0 END AS away_team_wins FROM basketball_teams JOIN basketball_games ON basketball_teams.team_id = basketball_games.home_team OR basketball_teams.team_id = basketball_games.away_team) AS subquery GROUP BY name;"
            },
            {
                    "query": "Insert new data into the 'cosmetic_ingredients' table for a vegan eyeshadow product by brand 'Ara' with ingredients 'Mica', 'Iron Oxide', 'Titanium Dioxide' and 'Zinc Oxide'.",
                    "Positive": "INSERT INTO cosmetic_ingredients (ingredient_id, product_name, brand_name, ingredient_type) VALUES (NULL, 'Vegan Eyeshadow', 'Ara', 'Ingredient'); INSERT INTO cosmetic_ingredients (ingredient_id, product_name, brand_name, ingredient_type, ingredient_name) SELECT ingredient_id, 'Vegan Eyeshadow', 'Ara', 'Ingredient', 'Mica' FROM cosmetic_ingredients WHERE ingredient_name = 'Mica' UNION ALL SELECT NULL, 'Vegan Eyeshadow', 'Ara', 'Ingredient', 'Iron Oxide' UNION ALL SELECT NULL, 'Vegan Eyeshadow', 'Ara', 'Ingredient', 'Titanium Dioxide' UNION ALL SELECT NULL, 'Vegan Eyeshadow', 'Ara', 'Ingredient', 'Zinc Oxide';"
            },
            {
                    "query": "Identify unions in New York with the highest increase in wage increases in collective bargaining contracts compared to the previous contract.",
                    "Positive": " SELECT u.name, u.state, c.wage_increase, c.contract_end, (SELECT wage_increase FROM CollectiveBargaining cb WHERE cb.contract_end < c.contract_end AND cb.union_id = c.union_id ORDER BY contract_end DESC LIMIT 1) AS previous_wage_increase FROM UnionMembers u JOIN UnionNegotiations n ON u.union_id = n.union_id JOIN CollectiveBargaining c ON u.union_id = c.union_id WHERE u.state = 'NY' ORDER BY c.wage_increase - (SELECT wage_increase FROM CollectiveBargaining cb WHERE cb.contract_end < c.contract_end AND cb.union_id = c.union_id ORDER BY contract_end DESC LIMIT 1) DESC LIMIT 10; "
            },
            {
                    "query": "Show the number of organic skincare products sold per month, displayed as pivoted data.",
                    "Positive": "SELECT EXTRACT(MONTH FROM sale_date) AS month, brand, SUM(CASE WHEN product_subcategory = 'Cleanser' THEN sale_count ELSE 0 END) AS Cleanser, SUM(CASE WHEN product_subcategory = 'Toner' THEN sale_count ELSE 0 END) AS Toner, SUM(CASE WHEN product_subcategory = 'Serum' THEN sale_count ELSE 0 END) AS Serum, SUM(CASE WHEN product_subcategory = 'Moisturizer' THEN sale_count ELSE 0 END) AS Moisturizer FROM product_labels_v4 WHERE product_subcategory IN ('Cleanser', 'Toner', 'Serum', 'Moisturizer') AND product_label = 'Organic' GROUP BY EXTRACT(MONTH FROM sale_date), brand;"
            },
            {
                    "query": "What is the total number of investigative journalism articles published in the last 3 months, and what percentage of the total publications do they represent?",
                    "Positive": "SELECT COUNT(*) AS total_investigative_articles FROM publications WHERE genre = 'investigative journalism' AND publication_date >= DATEADD(month, -3, GETDATE());SELECT COUNT(*) AS total_publications FROM publications;SELECT (total_investigative_articles * 100.0 / total_publications) AS percentage FROM (SELECT COUNT(*) AS total_investigative_articles FROM publications WHERE genre = 'investigative journalism' AND publication_date >= DATEADD(month, -3, GETDATE())) AS investigative_articles, (SELECT COUNT(*) AS total_publications FROM publications) AS total_publications;"
            }
    ],
```
'''
