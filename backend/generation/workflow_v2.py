"""Compatibility import for callers that still reference the Phase 2 module.

The production graph lives in :mod:`generation.workflow`; keeping one graph
prevents the API and tests from drifting onto different orchestration paths.
"""

from .workflow import create_workflow, run_workflow, workflow

__all__ = ["create_workflow", "run_workflow", "workflow"]
