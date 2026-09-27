"""Stream and blindly sample annotated query-document pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar


T = TypeVar("T")


class SamplingDataError(ValueError):
    """Raised when an annotated source record is internally inconsistent."""


def reservoir_sample(items: Iterable[T], k: int, rng: Any) -> tuple[list[T], int]:
    """Select ``k`` items uniformly without materializing the input stream."""
    if k < 0:
        raise ValueError("sample size must be non-negative")
    reservoir: list[T] = []
    seen = 0
    for item in items:
        seen += 1
        if len(reservoir) < k:
            reservoir.append(item)
            continue
        replacement = rng.randrange(seen)
        if replacement < k:
            reservoir[replacement] = item
    if seen < k:
        raise SamplingDataError(f"requested {k} pairs, but source has only {seen}")
    return reservoir, seen


def iter_source_pairs(path: str | Path, dataset: str) -> Iterator[dict[str, Any]]:
    """Yield valid annotated pairs from one pipeline output without loading it all."""
    try:
        import ijson
    except ImportError as exc:  # pragma: no cover - administrator environment guard
        raise RuntimeError(
            "sample preparation requires ijson; the annotation website does not"
        ) from exc

    source_path = Path(path)
    with source_path.open("rb") as handle:
        for result_index, result in enumerate(ijson.items(handle, "results.item")):
            if not isinstance(result, dict):
                raise SamplingDataError(f"result {result_index} is not an object")
            queries = result.get("queries", []) or []
            if not isinstance(queries, list):
                raise SamplingDataError(f"result {result_index} queries is not a list")
            for query_index, query_item in enumerate(queries):
                yield from _pairs_for_query(
                    query_item,
                    dataset=dataset,
                    result_index=result_index,
                    query_index=query_index,
                )


def _pairs_for_query(
    query_item: Any,
    *,
    dataset: str,
    result_index: int,
    query_index: int,
) -> Iterator[dict[str, Any]]:
    if not isinstance(query_item, dict):
        raise SamplingDataError(
            f"result {result_index} query {query_index} is not an object"
        )
    query = query_item.get("query")
    if not isinstance(query, str) or not query.strip():
        raise SamplingDataError(
            f"result {result_index} query {query_index} has empty query text"
        )
    candidates = query_item.get("candidates", []) or []
    annotations = query_item.get("annotations", []) or []
    if not isinstance(candidates, list) or not isinstance(annotations, list):
        raise SamplingDataError(
            f"result {result_index} query {query_index} has invalid lists"
        )
    candidates_by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict) or "doc_id" not in candidate:
            raise SamplingDataError(
                f"result {result_index} query {query_index} has invalid candidate"
            )
        doc_id = str(candidate["doc_id"])
        if doc_id in candidates_by_id:
            raise SamplingDataError(
                f"result {result_index} query {query_index} duplicates doc_id {doc_id}"
            )
        candidates_by_id[doc_id] = candidate

    annotated_ids: set[str] = set()
    for annotation in annotations:
        if not isinstance(annotation, dict) or "doc_id" not in annotation:
            raise SamplingDataError(
                f"result {result_index} query {query_index} has invalid annotation"
            )
        doc_id = str(annotation["doc_id"])
        if doc_id in annotated_ids:
            raise SamplingDataError(
                f"result {result_index} query {query_index} annotates {doc_id} twice"
            )
        annotated_ids.add(doc_id)
        candidate = candidates_by_id.get(doc_id)
        if candidate is None:
            raise SamplingDataError(
                f"result {result_index} query {query_index} annotation {doc_id} "
                "has no candidate"
            )
        document = candidate.get("doc")
        score = annotation.get("score")
        if not isinstance(document, str) or not document.strip():
            raise SamplingDataError(
                f"result {result_index} query {query_index} candidate {doc_id} "
                "has empty document"
            )
        if type(score) is not int or score not in {0, 1, 2, 3}:
            raise SamplingDataError(
                f"result {result_index} query {query_index} annotation {doc_id} "
                "has invalid score"
            )
        yield {
            "dataset": dataset,
            "query": query,
            "document": document,
            "source_doc_id": doc_id,
            "llm_score": score,
            "llm_classification": annotation.get("classification", ""),
            "llm_reasoning": annotation.get("reasoning", ""),
            "retrieval_score": candidate.get("score"),
            "retrieval_rank": candidate.get("rank"),
            "is_original": candidate.get("is_original"),
            "source_result_index": result_index,
            "source_query_index": query_index,
        }


def blind_samples(
    selected_by_dataset: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Create a public four-field sample list and a private answer-key list."""
    public: list[dict[str, str]] = []
    private: list[dict[str, Any]] = []
    for dataset, records in selected_by_dataset.items():
        for index, record in enumerate(records, start=1):
            sample_id = f"{dataset}-{index:03d}"
            public.append(
                {
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "query": record["query"],
                    "document": record["document"],
                }
            )
            private.append({"sample_id": sample_id, **record})
    return public, private
