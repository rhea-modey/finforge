"""Recursive Harness Improvement (RHI) optimizer layer (DESIGN section 9).

Exposes:
    propose_revision -- meta-prompt an optimizer LLM into a new, validated
                        harness spec. Never sees gold, rubrics, held-out
                        data, or gold scores.
"""

from rhi.optimizer import propose_revision

__all__ = ["propose_revision"]
