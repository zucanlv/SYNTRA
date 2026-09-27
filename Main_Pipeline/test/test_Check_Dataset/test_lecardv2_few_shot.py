import json
import sys
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = Path(
    "/data/share/project/shared_datasets/DSA/datasets/law/LeCaRDv2"
)
sys.path.insert(0, str(PIPELINE_DIR))

from Few_Shot_Example import Few_Shot_Example


class LeCaRDv2FewShotTest(unittest.TestCase):
    def test_examples_include_four_same_charge_and_one_cross_charge_pair(self):
        examples = Few_Shot_Example["lecardv2"]

        known_charges = set()
        with (DATASET_DIR / "corpus.jsonl").open(encoding="utf-8") as source:
            for line in source:
                text = json.loads(line)["text"]
                known_charges.update(text.rstrip().split("\n")[-1].split(","))

        same_charge_count = 0
        cross_charge_count = 0
        for example in examples:
            query_charges = {
                charge
                for charge in known_charges
                if charge and charge in example["query"][:220]
            }
            positive_charges = set(
                example["Positive"].rstrip().split("\n")[-1].split(",")
            )
            self.assertTrue(query_charges, "could not identify query charge")
            self.assertTrue(positive_charges, "could not identify positive charge")
            if query_charges.isdisjoint(positive_charges):
                cross_charge_count += 1
            else:
                same_charge_count += 1

        self.assertEqual(same_charge_count, 4)
        self.assertEqual(cross_charge_count, 1)

    def test_examples_are_official_relevant_pairs(self):
        examples = Few_Shot_Example["lecardv2"]
        self.assertEqual(len(examples), 5)

        query_ids_by_text = {}
        with (DATASET_DIR / "queries.jsonl").open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                query_ids_by_text.setdefault(record["text"], set()).add(record["_id"])

        corpus_ids_by_text = {}
        with (DATASET_DIR / "corpus.jsonl").open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                corpus_ids_by_text.setdefault(record["text"], set()).add(record["_id"])

        relevant_pairs = set()
        with (DATASET_DIR / "qrels" / "test.jsonl").open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                if record["score"] > 0:
                    relevant_pairs.add((record["query-id"], record["corpus-id"]))

        for example in examples:
            self.assertEqual(set(example), {"query", "Positive"})
            query_ids = query_ids_by_text.get(example["query"], set())
            corpus_ids = corpus_ids_by_text.get(example["Positive"], set())
            self.assertTrue(query_ids, "few-shot query is not in official queries.jsonl")
            self.assertTrue(corpus_ids, "few-shot positive is not in official corpus.jsonl")
            self.assertTrue(
                any(
                    (query_id, corpus_id) in relevant_pairs
                    for query_id in query_ids
                    for corpus_id in corpus_ids
                ),
                "few-shot pair is not marked relevant by official qrels",
            )


if __name__ == "__main__":
    unittest.main()
