"""
HumanReview — Interactive Human-in-the-Loop hooks for SelfRefineLoop.
======================================================================
Provides ``HumanReviewer``, which can be plugged into ``SelfRefineLoop``
to insert a human decision point after each evaluate step:

1. **After evaluate**: display the current artifact, score, and feedback;
   let the human override pass/fail (10-second timeout falls back to the
   LLM's decision) and optionally edit the feedback text.
2. **Before refine LLM call**: display the full refine prompt and let the
   human edit it (opens ``$EDITOR`` or reads from stdin).

Usage::

    from HumanReview import HumanReviewer

    reviewer = HumanReviewer(enabled=True, timeout=10)
    # Pass to SelfRefineLoop(..., human_reviewer=reviewer)
    # In each _run_*_self_refine, pass
    #   prompt_hook=reviewer.review_refine_prompt  to the refine_fn wrapper.

Setting ``enabled=False`` (or running without a TTY) makes every method a
no-op that returns its inputs unchanged.
"""

import os
import select
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional, Tuple

_WIDTH = 80
_SEP_HEAVY = "=" * _WIDTH
_SEP_LIGHT = "-" * _WIDTH


def _print_section(title: str) -> None:
    print(_SEP_HEAVY)
    print(f"  {title}")
    print(_SEP_HEAVY)


def _indent(text: str, prefix: str = "    ") -> str:
    """Prefix every line of *text* with *prefix*."""
    return "\n".join(prefix + line for line in text.splitlines())


def _which(name: str) -> bool:
    """Return True if *name* is available on PATH."""
    return (
        subprocess.run(["which", name], capture_output=True, check=False).returncode
        == 0
    )


class HumanReviewer:
    """Interactive human reviewer for self-refine calibration loops.

    Parameters
    ----------
    enabled : bool
        When ``False`` all public methods are no-ops that return their
        inputs unchanged.  Automatically forced to ``False`` when stdin
        is not a TTY (non-interactive execution).
    timeout : int
        Seconds to wait on the PASS override prompt before falling back
        to the LLM's own evaluation decision.
    use_editor : bool
        When ``True`` (default), multi-line text edits open the system
        editor (``$EDITOR`` env var; falls back to nano / vim / vi).
        When ``False``, text is collected line-by-line from stdin (a
        blank line signals end-of-input).
    """

    def __init__(
        self,
        enabled: bool = True,
        timeout: int = 10,
        use_editor: bool = True,
    ) -> None:
        if enabled and not sys.stdin.isatty():
            enabled = False
        self.enabled = enabled
        self.timeout = timeout
        self.use_editor = use_editor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def review_after_evaluate(
        self,
        label: str,
        attempt: int,
        max_attempts: int,
        artifact: Any,
        passed: bool,
        feedback: str,
        eval_details: Optional[Dict] = None,
    ) -> Tuple[bool, str]:
        """Display evaluation results and let the human decide what to do next.

        The human can:
        * Override the result as **PASS** (ends the self-refine loop).
        * Accept the **FAIL** decision (proceed to refine), optionally
          editing the feedback text that will be fed into the refine step.

        A 10-second timeout (configurable via ``self.timeout``) falls back
        to the LLM's original pass/fail decision.

        Parameters
        ----------
        label : str
            Short name for the self-refine stage, e.g. ``"IdAttr"``.
        attempt, max_attempts : int
            Current and maximum attempt numbers (1-based).
        artifact : Any
            The artifact (guidance/instruction string) used this round.
        passed, feedback : bool, str
            LLM evaluation outcome.
        eval_details : dict, optional
            Raw parsed evaluation fields (``score``, ``feedback``, issue
            sub-fields, ``suggestions``).  Used for richer display.

        Returns
        -------
        (final_passed, final_feedback) : Tuple[bool, str]
            The (possibly human-overridden) pass decision and feedback.
        """
        if not self.enabled:
            return passed, feedback

        eval_details = eval_details or {}

        print()
        _print_section(f"[Human Review]  {label}  —  Attempt {attempt}/{max_attempts}")
        print()

        # ---- Current artifact ----
        print("CURRENT ARTIFACT:")
        artifact_str = str(artifact) if artifact else ""
        if artifact_str.strip():
            print(_indent(artifact_str))
        else:
            print("    (empty — no prior guidance)")
        print()

        # ---- Evaluation result ----
        print("EVALUATION RESULT:")
        score = eval_details.get("score")
        if score is not None:
            print(f"  Score   : {score} / 4")
        print(f"  Passed  : {'YES' if passed else 'NO'}")
        print()

        # Detail sub-fields (skip "none" / empty / meta keys)
        _META_KEYS = {"score", "feedback", "passed"}
        detail_items = [
            (k, v)
            for k, v in eval_details.items()
            if k not in _META_KEYS
            and isinstance(v, str)
            and v.strip().lower() not in ("", "none")
        ]
        if detail_items:
            print("  Details:")
            for k, v in detail_items:
                field_label = k.replace("_", " ").title()
                print(f"    [{field_label}]")
                print(_indent(v, "      "))
                print()

        print("  Feedback:")
        if feedback.strip():
            print(_indent(feedback, "    "))
        else:
            print("    (no feedback)")
        print()
        print(_SEP_LIGHT)

        # ---- Timed PASS override ----
        default_str = "YES" if passed else "NO"
        override = self._timed_yes_no(
            f"Override as PASS? [y/n]  "
            f"(timeout={self.timeout}s → LLM default={default_str}): ",
            self.timeout,
        )

        if override is True:
            print("  → Human override: PASS")
            print()
            return True, feedback

        if override is None:
            print(f"\n  → Timeout: using LLM decision ({default_str})")
            print()
            return passed, feedback

        # override is False: user explicitly chose NOT to pass
        print("  → Continuing with FAIL ...")
        print()

        # ---- Optional feedback edit ----
        if self._ask_yes_no("Modify feedback? [y/n]: "):
            hint = "(editor will open)" if self.use_editor else "(blank line to finish)"
            print(f"  Enter new feedback {hint}:")
            new_feedback = self._get_multiline_input("feedback", initial=feedback)
            if new_feedback.strip():
                feedback = new_feedback.strip()
                print(f"  → Feedback updated ({len(feedback)} chars).")
            else:
                print("  → Feedback unchanged (empty input).")
        print()

        return False, feedback

    def review_refine_prompt(self, prompt: str) -> str:
        """Display the full refine prompt and let the human edit it.

        Called inside each ``refine_*`` function via the ``prompt_hook``
        parameter, just after the prompt is built but before the LLM call.

        Parameters
        ----------
        prompt : str
            The refine prompt as constructed by ``build_*_refine_prompt``.

        Returns
        -------
        str
            The (possibly modified) prompt to send to the LLM.
        """
        if not self.enabled:
            return prompt

        print()
        _print_section("[Human Review]  Refine Prompt Preview")
        print()
        print(prompt)
        print()
        print(_SEP_LIGHT)

        if self._ask_yes_no("Modify refine prompt? [y/n]: "):
            hint = "(editor will open)" if self.use_editor else "(blank line to finish)"
            print(f"  Enter modified prompt {hint}:")
            new_prompt = self._get_multiline_input("refine_prompt", initial=prompt)
            if new_prompt.strip():
                prompt = new_prompt.strip()
                print(f"  → Refine prompt updated ({len(prompt)} chars).")
            else:
                print("  → Refine prompt unchanged (empty input).")
        print()

        return prompt

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _timed_yes_no(self, prompt_text: str, timeout: int) -> Optional[bool]:
        """Print *prompt_text* and wait up to *timeout* seconds for y / n.

        Returns
        -------
        True   — user typed 'y' or 'yes'
        False  — user typed 'n' or 'no'
        None   — timeout elapsed or unrecognised input
        """
        sys.stdout.write(prompt_text)
        sys.stdout.flush()
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if not ready:
            return None
        line = sys.stdin.readline().strip().lower()
        if line in ("y", "yes"):
            return True
        if line in ("n", "no"):
            return False
        return None  # unrecognised → treat as timeout

    def _ask_yes_no(self, prompt_text: str) -> bool:
        """Blocking y/n prompt.  Returns ``True`` for 'y', ``False`` for 'n'."""
        while True:
            sys.stdout.write(prompt_text)
            sys.stdout.flush()
            line = sys.stdin.readline().strip().lower()
            if line in ("y", "yes"):
                return True
            if line in ("n", "no"):
                return False
            print("  Please enter 'y' or 'n'.")

    def _get_multiline_input(self, label: str, initial: str = "") -> str:
        """Return multi-line text from the user.

        Uses ``$EDITOR`` when ``self.use_editor=True`` and an editor is
        available; otherwise falls back to line-by-line stdin collection.
        """
        if self.use_editor:
            return self._open_editor(initial)
        return self._multiline_stdin(initial)

    def _open_editor(self, initial: str) -> str:
        """Write *initial* to a temp file, open ``$EDITOR``, return result."""
        editor = os.environ.get("EDITOR", "")
        if not editor:
            for candidate in ("nano", "vim", "vi"):
                if _which(candidate):
                    editor = candidate
                    break
        if not editor:
            print("  No suitable editor found — falling back to line-by-line input.")
            return self._multiline_stdin(initial)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(initial)
            fname = f.name

        try:
            subprocess.run([editor, fname], check=False)
            with open(fname, "r", encoding="utf-8") as f:
                return f.read()
        finally:
            try:
                os.unlink(fname)
            except OSError:
                pass

    def _multiline_stdin(self, initial: str = "") -> str:
        """Collect lines from stdin until a blank line or EOF."""
        if initial.strip():
            print(
                "  (current content shown above — "
                "type replacement text, blank line to finish)"
            )
        else:
            print("  (type text, blank line to finish)")
        lines: list = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line == "":
                break
            lines.append(line)
        return "\n".join(lines)
