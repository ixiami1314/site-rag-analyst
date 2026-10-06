"""Retrieval-quality evaluation: a small labeled set, honest metrics.

The question this package answers: *is retrieval actually good enough to
ground an analyst?* Embedding demos usually assert it; we measure it —
hit@k, MRR and a strict snippet-hit metric over a hand-labeled query set
against the bundled demo corpus. CI runs the same harness, so a change that
silently degrades retrieval fails the build instead of shipping.
"""

from site_rag_analyst.eval.runner import EvalCase, load_eval_set, main, run_evaluation

__all__ = ["EvalCase", "load_eval_set", "main", "run_evaluation"]
