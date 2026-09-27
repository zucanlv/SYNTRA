"""
Generic Self-Refine Loop
=========================
Iteratively improve an *artifact* (e.g. a prompt addendum, a config, a set of
instructions) by running a generate-evaluate-refine cycle up to a fixed number
of attempts.

The module is deliberately task-agnostic — callers supply three callables that
define the domain-specific behaviour:

* **generate_fn(artifact)** — produce outputs using the current artifact.
* **evaluate_fn(outputs)** — judge quality; return ``(passed, feedback)``.
* **refine_fn(artifact, outputs, feedback)** — incorporate feedback into a
  better artifact.

Typical usage (IdAttr prompt calibration)::

    from SelfRefine import SelfRefineLoop

    loop = SelfRefineLoop(max_attempts=3)
    result = loop.run(
        initial_artifact="",
        generate_fn=my_generate,
        evaluate_fn=my_evaluate,
        refine_fn=my_refine,
    )
    if result.passed:
        print("Calibration succeeded on attempt", result.attempt)
    final_guidance = result.artifact
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class RefineAttempt:
    """Record for a single generate-evaluate iteration."""
    attempt: int                          # 1-based attempt number
    artifact: Any                         # artifact used in this attempt
    outputs: Any                          # raw outputs from generate_fn
    passed: bool                          # whether evaluate_fn said "pass"
    feedback: str                         # textual feedback from evaluate_fn
    eval_details: Dict = field(default_factory=dict)  # raw LLM eval fields (score, sub-issues, …)


@dataclass
class RefineResult:
    """Outcome of a full self-refine run."""
    artifact: Any           # the final (best) artifact
    passed: bool            # whether the final evaluation passed
    attempt: int            # which attempt produced the final artifact (1-based)
    history: List[RefineAttempt] = field(default_factory=list)


GenerateFn = Callable[[Any], Any]
EvaluateFn = Callable[[Any], Tuple]   # returns (bool, str) or (bool, str, dict)
RefineFn = Callable[[Any, Any, str], Any]


class SelfRefineLoop:
    """Run a generate → evaluate → refine loop up to *max_attempts* times.

    Parameters
    ----------
    max_attempts : int
        Total attempts including the initial one (so ``max_attempts=3`` means
        one initial try plus at most two refinement rounds).
    logger : logging.Logger, optional
        Logger instance; defaults to module-level logger.
    human_reviewer : HumanReviewer, optional
        When provided, a human decision point is inserted after each evaluate
        step: the human can override the pass/fail result and edit the
        feedback before the refine step proceeds.
    label : str, optional
        Short display name for the self-refine stage shown in the human
        review prompt (e.g. ``"IdAttr"``, ``"QueryInstr"``).
    """

    def __init__(
        self,
        max_attempts: int = 3,
        logger: Optional[logging.Logger] = None,
        human_reviewer: Optional[Any] = None,
        label: str = "",
    ):
        self.max_attempts = max(1, max_attempts)
        self.log = logger or logging.getLogger(__name__)
        self.human_reviewer = human_reviewer
        self.label = label

    def run(
        self,
        initial_artifact: Any,
        generate_fn: GenerateFn,
        evaluate_fn: EvaluateFn,
        refine_fn: RefineFn,
    ) -> RefineResult:
        """Execute the self-refine loop.

        Parameters
        ----------
        initial_artifact
            Starting artifact (e.g. empty string for first-time calibration).
        generate_fn
            ``(artifact) -> outputs``.  Run the generation step using the
            current artifact and return the outputs.
        evaluate_fn
            ``(outputs) -> (passed, feedback)``.  Evaluate the outputs and
            return a boolean pass/fail plus textual feedback.
        refine_fn
            ``(artifact, outputs, feedback) -> new_artifact``.  Incorporate
            the evaluation feedback into the artifact to produce an improved
            version.

        Returns
        -------
        RefineResult
        """
        artifact = initial_artifact
        history: List[RefineAttempt] = []

        for attempt_num in range(1, self.max_attempts + 1):
            self.log.info(
                "[SelfRefine] Attempt %d/%d — generating outputs ...",
                attempt_num, self.max_attempts,
            )
            outputs = generate_fn(artifact)

            self.log.info(
                "[SelfRefine] Attempt %d/%d — evaluating outputs ...",
                attempt_num, self.max_attempts,
            )
            eval_result = evaluate_fn(outputs)
            # Support both 2-tuple (passed, feedback) and 3-tuple (passed, feedback, eval_details)
            if len(eval_result) == 3:
                passed, feedback, eval_details = eval_result
            else:
                passed, feedback = eval_result
                eval_details = {}

            history.append(RefineAttempt(
                attempt=attempt_num,
                artifact=artifact,
                outputs=outputs,
                passed=passed,
                feedback=feedback,
                eval_details=eval_details,
            ))

            # Human review hook: lets the operator override pass/fail and feedback
            if self.human_reviewer is not None:
                passed, feedback = self.human_reviewer.review_after_evaluate(
                    label=self.label,
                    attempt=attempt_num,
                    max_attempts=self.max_attempts,
                    artifact=artifact,
                    passed=passed,
                    feedback=feedback,
                    eval_details=eval_details,
                )

            if passed:
                self.log.info(
                    "[SelfRefine] Attempt %d/%d — PASSED.",
                    attempt_num, self.max_attempts,
                )
                return RefineResult(
                    artifact=artifact,
                    passed=True,
                    attempt=attempt_num,
                    history=history,
                )

            self.log.info(
                "[SelfRefine] Attempt %d/%d — FAILED. Feedback: %s",
                attempt_num, self.max_attempts, feedback,
            )

            if attempt_num < self.max_attempts:
                self.log.info(
                    "[SelfRefine] Refining artifact for next attempt ..."
                )
                artifact = refine_fn(artifact, outputs, feedback)

        self.log.warning(
            "[SelfRefine] Exhausted %d attempts without passing.",
            self.max_attempts,
        )
        return RefineResult(
            artifact=artifact,
            passed=False,
            attempt=self.max_attempts,
            history=history,
        )
