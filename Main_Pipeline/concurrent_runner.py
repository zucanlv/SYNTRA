"""Concurrent LLM call utilities for the synthetic data pipeline.

Wraps qwen-agent's ``parallel_exec`` with order-preserving semantics and a
serial fallback so that every module can toggle concurrency via a single flag.
"""

import threading
from typing import Any, Callable, List, Optional

from qwen_agent.utils.parallel_executor import parallel_exec, serial_exec


def run_concurrent(
    fn: Callable,
    kwargs_list: List[dict],
    enabled: bool = True,
    max_workers: Optional[int] = None,
    jitter: float = 0.0,
    *,
    progress_logger: Optional[Any] = None,
    progress_label: str = "",
    progress_interval_sec: float = 0.0,
) -> List[Any]:
    """Call *fn* for every kwargs dict in *kwargs_list*, returning results in
    the **same order as the input**.

    Parameters
    ----------
    fn:
        Function to call.  Must accept ``**kwargs`` matching the dicts in
        *kwargs_list*.  Do **not** include an ``_idx`` key in those dicts —
        it is injected internally.
    kwargs_list:
        List of keyword-argument dicts, one per call.
    enabled:
        When *False* (or when the list has ≤ 1 item), falls back to serial
        execution — useful for debugging and for deterministic test runs.
    max_workers:
        Maximum number of worker threads passed to ``ThreadPoolExecutor``.
        ``None`` lets Python choose ``min(32, os.cpu_count() + 4)``.
    jitter:
        Seconds of random sleep injected between task submissions to avoid
        hitting API rate limits (``parallel_exec`` uses ``jitter * random()``).
    progress_logger:
        Optional logger for periodic ``[Progress]`` lines during the run.
    progress_label:
        Label prefix for progress messages (default: ``"concurrent"``).
    progress_interval_sec:
        If > 0 and *progress_logger* is set, log completed/total every this many seconds.

    Returns
    -------
    list
        Results in the **same order** as *kwargs_list*.
    """
    if not kwargs_list:
        return []

    total = len(kwargs_list)
    label = progress_label or "concurrent"
    done_lock = threading.Lock()
    done_count = [0]

    # Wrap fn so that we can sort results by submission index afterwards.
    def _indexed(_idx: int, **kw: Any):
        try:
            return (_idx, fn(**kw))
        finally:
            with done_lock:
                done_count[0] += 1

    indexed_kwargs = [{"_idx": i, **kw} for i, kw in enumerate(kwargs_list)]

    stop_hb: Optional[threading.Event] = None
    hb_thread: Optional[threading.Thread] = None
    if progress_logger is not None and progress_interval_sec and progress_interval_sec > 0:

        def _heartbeat():
            while stop_hb is not None and not stop_hb.is_set():
                if stop_hb.wait(timeout=progress_interval_sec):
                    break
                with done_lock:
                    d = done_count[0]
                progress_logger.info(
                    "[Progress] %s: %d/%d completed (%.1f%%)",
                    label,
                    d,
                    total,
                    100.0 * d / total if total else 0.0,
                )

        stop_hb = threading.Event()
        progress_logger.info("[Progress] %s: starting %d tasks", label, total)
        hb_thread = threading.Thread(target=_heartbeat, daemon=True, name="run_concurrent_hb")
        hb_thread.start()

    try:
        if not enabled or len(kwargs_list) <= 1:
            raw = serial_exec(_indexed, indexed_kwargs)
        else:
            raw = parallel_exec(
                _indexed,
                indexed_kwargs,
                max_workers=max_workers,
                jitter=jitter,
            )
    finally:
        if stop_hb is not None:
            stop_hb.set()
            if hb_thread is not None:
                hb_thread.join(timeout=2.0)
            if progress_logger is not None:
                with done_lock:
                    d = done_count[0]
                progress_logger.info("[Progress] %s: finished %d/%d", label, d, total)

    raw.sort(key=lambda x: x[0])
    return [r[1] for r in raw]
