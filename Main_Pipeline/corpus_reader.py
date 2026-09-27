"""
Unified corpus reader for the Synthetic Data Research pipeline.

Supports multiple corpus formats and normalises every corpus to a
(doc_id, title, text) schema so downstream code never needs to care
about the original storage format.

Supported formats
-----------------
- **tsv / csv** – with or without a header row (auto-detected).
- **jsonl** – one JSON object per line.
- **parquet** – single Parquet file.
- **hf_dataset** – HuggingFace Datasets directory (Arrow cache).

Column mapping
--------------
Columns are auto-detected by matching against common name conventions
(e.g. ``_id``, ``pid`` for IDs; ``text``, ``passage`` for body text;
``title`` for titles).  All three standard names can also be set
explicitly via :class:`CorpusConfig`.
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Candidate column names used by auto-detection (checked in order).
_ID_CANDIDATES = ("_id", "id", "pid", "doc_id", "docid", "document_id")
_TEXT_CANDIDATES = ("text", "passage", "body", "content", "document")
_TITLE_CANDIDATES = ("title", "doc_title")


# --------------------------------------------------------------------------- #
#  Config                                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class CorpusConfig:
    """Where the corpus lives and how to interpret it."""

    corpus_path: str
    corpus_format: str = "auto"  # auto | tsv | csv | jsonl | parquet | hf_dataset
    id_column: Optional[str] = None  # None → auto-detect
    text_column: Optional[str] = None  # None → auto-detect
    title_column: Optional[str] = None  # None → auto-detect;  "" → disable title

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "CorpusConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _match_column(candidates: tuple, available: List[str]) -> Optional[str]:
    """Return the first *candidate* that exists in *available* (case-insensitive)."""
    lookup = {c.lower(): c for c in available}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return None


# --------------------------------------------------------------------------- #
#  Reader                                                                      #
# --------------------------------------------------------------------------- #


class CorpusReader:
    """Read corpora in various formats and yield standardised batches.

    Each batch item is a dict ``{"doc_id": str, "title": str, "text": str}``.

    Iteration via :meth:`iter_batches` yields ``(batch, raw_row_count)``
    tuples so callers can track resume offsets accurately (``raw_row_count``
    includes rows that were filtered out, e.g. because of NaN text).
    """

    def __init__(self, config: CorpusConfig):
        self.config = config
        self._format: str = self._detect_format()
        self._hf_dataset = None
        self._col_map: Optional[Dict[str, Optional[str]]] = None
        self._has_header: Optional[bool] = None
        self._raw_columns: Optional[List[str]] = None
        self._length: Optional[int] = None
        logger.info(
            "CorpusReader: path=%s  format=%s", config.corpus_path, self._format
        )

    # -- public properties -------------------------------------------------- #

    @property
    def format(self) -> str:
        return self._format

    @property
    def column_map(self) -> Dict[str, Optional[str]]:
        """Standardised→actual column mapping (lazy-initialised)."""
        self._ensure_columns()
        return dict(self._col_map)

    # ------------------------------------------------------------------ #
    #  Format detection                                                    #
    # ------------------------------------------------------------------ #

    _EXT_MAP = {
        ".tsv": "tsv",
        ".csv": "csv",
        ".jsonl": "jsonl",
        ".ndjson": "jsonl",
        ".parquet": "parquet",
    }

    def _detect_format(self) -> str:
        if self.config.corpus_format != "auto":
            return self.config.corpus_format
        ext = os.path.splitext(self.config.corpus_path)[1].lower()
        if ext in self._EXT_MAP:
            return self._EXT_MAP[ext]
        if os.path.isdir(self.config.corpus_path):
            return "hf_dataset"
        raise ValueError(f"Cannot auto-detect corpus format for: {self.config.corpus_path}")

    # ------------------------------------------------------------------ #
    #  Column detection                                                    #
    # ------------------------------------------------------------------ #

    def _ensure_columns(self):
        if self._col_map is not None:
            return
        {
            "tsv": self._peek_tabular,
            "csv": self._peek_tabular,
            "jsonl": self._peek_jsonl,
            "parquet": self._peek_parquet,
            "hf_dataset": self._load_hf,
        }[self._format]()

    def _build_col_map(self, available: List[str]):
        m: Dict[str, Optional[str]] = {}

        # --- doc_id ---
        if self.config.id_column:
            m["doc_id"] = self.config.id_column
        else:
            found = _match_column(_ID_CANDIDATES, available)
            if found is None:
                raise ValueError(
                    f"Cannot auto-detect ID column from {available}. "
                    "Use --id-column to specify it."
                )
            m["doc_id"] = found

        # --- text ---
        if self.config.text_column:
            m["text"] = self.config.text_column
        else:
            found = _match_column(_TEXT_CANDIDATES, available)
            if found is None:
                raise ValueError(
                    f"Cannot auto-detect text column from {available}. "
                    "Use --text-column to specify it."
                )
            m["text"] = found

        # --- title (optional) ---
        if self.config.title_column == "":
            m["title"] = None
        elif self.config.title_column:
            m["title"] = self.config.title_column
        else:
            m["title"] = _match_column(_TITLE_CANDIDATES, available)

        self._col_map = m
        self._raw_columns = available
        logger.info("Column mapping: %s  (available: %s)", m, available)

    # -- format-specific peek --

    def _peek_tabular(self):
        import pandas as pd

        sep = "\t" if self._format == "tsv" else ","
        raw = pd.read_csv(
            self.config.corpus_path, sep=sep, header=None, nrows=2, quoting=3,
        )
        first_vals = [str(v).strip().lower() for v in raw.iloc[0]]
        known = {c.lower() for c in _ID_CANDIDATES + _TEXT_CANDIDATES + _TITLE_CANDIDATES}
        self._has_header = sum(1 for v in first_vals if v in known) >= 2

        if self._has_header:
            cols = [str(v).strip() for v in raw.iloc[0]]
        else:
            ncols = len(raw.columns)
            if ncols == 2:
                cols = ["pid", "passage"]
            elif ncols >= 3:
                cols = ["pid", "title", "passage"] + [
                    f"_col{i}" for i in range(3, ncols)
                ]
            else:
                raise ValueError(
                    f"Tabular corpus has {ncols} column(s); expected at least 2."
                )

        self._build_col_map(cols)

    def _peek_jsonl(self):
        with open(self.config.corpus_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    first = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"First non-empty line of JSONL file is not valid JSON: {exc}"
                    ) from exc
                self._build_col_map(list(first.keys()))
                return
        raise ValueError(f"JSONL file appears to be empty: {self.config.corpus_path}")

    def _peek_parquet(self):
        import pyarrow.parquet as pq

        schema = pq.read_schema(self.config.corpus_path)
        self._build_col_map(schema.names)

    def _load_hf(self):
        if self._hf_dataset is not None:
            return
        from datasets import Dataset, concatenate_datasets, load_from_disk

        path = self.config.corpus_path
        # 1) Try load_from_disk (works if saved via save_to_disk)
        try:
            ds = load_from_disk(path)
            if hasattr(ds, "column_names"):
                self._hf_dataset = ds
                self._build_col_map(ds.column_names)
                return
        except Exception:
            pass

        # 2) HuggingFace cache layout: one Arrow file per shard (e.g. *-00000-of-00007.arrow).
        #    Load and merge all shards — using only the first file under-counts the corpus.
        arrow_files: List[str] = []
        for root, _, files in os.walk(path):
            for fname in sorted(files):
                if fname.endswith(".arrow") and not fname.startswith("cache"):
                    arrow_files.append(os.path.join(root, fname))
        arrow_files.sort()

        if not arrow_files:
            raise ValueError(f"No loadable HuggingFace dataset found under: {path}")

        try:
            if len(arrow_files) == 1:
                ds = Dataset.from_file(arrow_files[0])
                logger.info(
                    "Loaded Arrow file: %s (%d rows)", arrow_files[0], len(ds)
                )
            else:
                parts = []
                for fp in arrow_files:
                    part = Dataset.from_file(fp)
                    parts.append(part)
                    logger.info("Loaded Arrow shard: %s (%d rows)", fp, len(part))
                ds = concatenate_datasets(parts)
                logger.info(
                    "Merged %d Arrow shards → %d total rows",
                    len(arrow_files),
                    len(ds),
                )
        except Exception as exc:
            raise ValueError(
                f"Failed to load Arrow dataset(s) under: {path}"
            ) from exc

        self._hf_dataset = ds
        self._build_col_map(ds.column_names)

    # ------------------------------------------------------------------ #
    #  Length                                                               #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        if self._length is not None:
            return self._length

        self._ensure_columns()

        if self._format in ("tsv", "csv", "jsonl"):
            n = int(
                subprocess.run(
                    ["wc", "-l", self.config.corpus_path],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip().split()[0]
            )
            if self._format in ("tsv", "csv") and self._has_header:
                n -= 1
            self._length = n
        elif self._format == "parquet":
            import pyarrow.parquet as pq

            self._length = pq.read_metadata(self.config.corpus_path).num_rows
        elif self._format == "hf_dataset":
            self._load_hf()
            self._length = len(self._hf_dataset)

        return self._length

    # ------------------------------------------------------------------ #
    #  Row normalisation                                                   #
    # ------------------------------------------------------------------ #

    def _normalize(self, row: dict) -> Optional[Dict[str, str]]:
        raw_text = row.get(self._col_map["text"])
        # NaN check: float NaN is the only value where x != x
        if raw_text is None or (isinstance(raw_text, float) and raw_text != raw_text):
            return None
        text = str(raw_text).strip()
        if not text:
            return None

        doc_id = str(row[self._col_map["doc_id"]])

        title_col = self._col_map.get("title")
        if title_col and title_col in row:
            raw_t = row[title_col]
            title = (
                ""
                if raw_t is None
                or (isinstance(raw_t, float) and raw_t != raw_t)
                else str(raw_t)
            )
        else:
            title = ""

        return {"doc_id": doc_id, "title": title, "text": text}

    # ------------------------------------------------------------------ #
    #  Batch iteration                                                     #
    # ------------------------------------------------------------------ #

    def iter_batches(
        self,
        batch_size: int,
        skip_rows: int = 0,
        max_rows: Optional[int] = None,
    ) -> Iterator[Tuple[List[Dict[str, str]], int]]:
        """Yield ``(batch, raw_row_count)`` tuples.

        *raw_row_count* is the number of positional rows consumed by this
        chunk **including** rows filtered out (e.g. NaN text).
        """
        self._ensure_columns()
        handler = {
            "tsv": self._iter_tabular,
            "csv": self._iter_tabular,
            "jsonl": self._iter_jsonl,
            "parquet": self._iter_parquet,
            "hf_dataset": self._iter_hf,
        }[self._format]
        yield from handler(batch_size, skip_rows, max_rows)

    # -- format-specific iterators --

    def _iter_tabular(self, batch_size, skip_rows, max_rows):
        import pandas as pd

        sep = "\t" if self._format == "tsv" else ","
        # Always read with header=None + explicit names so that integer
        # skiprows works uniformly for both headerless and header files.
        skip_lines = (skip_rows + 1) if self._has_header else skip_rows

        reader = pd.read_csv(
            self.config.corpus_path,
            sep=sep,
            header=None,
            names=self._raw_columns,
            skiprows=skip_lines or None,
            nrows=max_rows,
            chunksize=batch_size,
            quoting=3,
        )

        text_col = self._col_map["text"]
        for chunk in reader:
            raw_count = len(chunk)
            chunk = chunk.dropna(subset=[text_col])
            batch = [
                r
                for row in chunk.to_dict("records")
                if (r := self._normalize(row)) is not None
            ]
            yield batch, raw_count

    def _iter_jsonl(self, batch_size, skip_rows, max_rows):
        rows_seen = 0
        rows_read = 0
        batch: List[Dict[str, str]] = []
        raw_count = 0

        with open(self.config.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if rows_seen < skip_rows:
                    rows_seen += 1
                    continue
                if max_rows is not None and rows_read >= max_rows:
                    break

                line = line.strip()
                if not line:
                    rows_seen += 1
                    rows_read += 1
                    raw_count += 1
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON at file row %d", rows_seen)
                    rows_seen += 1
                    rows_read += 1
                    raw_count += 1
                    continue
                norm = self._normalize(obj)
                if norm:
                    batch.append(norm)
                rows_seen += 1
                rows_read += 1
                raw_count += 1

                if raw_count >= batch_size:
                    yield batch, raw_count
                    batch = []
                    raw_count = 0

        if raw_count > 0:
            yield batch, raw_count

    def _iter_parquet(self, batch_size, skip_rows, max_rows):
        import pyarrow.parquet as pq

        table = pq.read_table(self.config.corpus_path)
        total = table.num_rows
        end = total if max_rows is None else min(skip_rows + max_rows, total)

        for s in range(skip_rows, end, batch_size):
            n = min(batch_size, end - s)
            chunk = table.slice(s, n).to_pydict()
            keys = list(chunk.keys())
            batch = [
                r
                for i in range(n)
                if (r := self._normalize({k: chunk[k][i] for k in keys})) is not None
            ]
            yield batch, n

    def _iter_hf(self, batch_size, skip_rows, max_rows):
        self._load_hf()
        ds = self._hf_dataset
        total = len(ds)
        end = total if max_rows is None else min(skip_rows + max_rows, total)

        for s in range(skip_rows, end, batch_size):
            n = min(batch_size, end - s)
            raw = ds[s : s + n]  # dict of lists
            keys = list(raw.keys())
            batch = [
                r
                for i in range(n)
                if (r := self._normalize({k: raw[k][i] for k in keys})) is not None
            ]
            yield batch, n
