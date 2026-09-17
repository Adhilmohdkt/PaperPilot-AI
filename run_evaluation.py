#!/usr/bin/env python
"""
Phase 4 Evaluation Runner - PaperPilot

Runs structural and quality evaluation and produces a report.
Sets up environment and runs all evaluation components.
"""

import os
import sys
import json
import asyncio

# Set environment variables BEFORE any imports
os.environ['GROQ_API_KEY'] = 'REDACTED_GROQ_KEY'
os.environ['GEMINI_API_KEY'] = 'AIzaSyCyXIg5jnowM6HMLcff400feIwUc5gCzIs'

sys.path.insert(0, r'C:\Users\ASUS\Desktop\PaperPilot AI\backend')

def run_structural_evaluation():
    """Run the 60 structural checks across 10 golden cases."""
    print("=" * 70)
    print("PHASE 4: STRUCTURAL EVALUATION")
    print("=" * 70)
    
    # Import golden cases
    from tests.evaluation_dataset.golden_cases import golden_cases
    
    # Import structural check functions
    from tests.structural.test_structural import TestStructuralEvaluation
    
    # Create test instance and run
    test_obj = TestStructuralEvaluation()
    
    # We need to run the async test
    async def run_all():
        # The test method runs all cases and checks internally
        # We'll capture the results
        from unittest import result
        # Just run the test and capture output
        return await test_obj.test_structural_checks_across_cases()
    
    # Run via asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        results = loop.run_until_complete(run_all())
    finally:
        loop.close()
    
    # The test already prints its own summary, so we just return
    # The structural evaluation is embedded in the test
    return "structural_eval_complete"


def run_quality_evaluation():
    """Run DeepEval quality evaluation."""
    print("\n" + "=" * 70)
    print("PHASE 4: QUALITY EVALUATION (DeepEval)")
    print("=" * 70)
    
    from tests.quality.test_quality import run_deepeval_quality_evaluation, QUALITY_TEST_CASES
    
    results = run_deepeval_quality_evaluation(QUALITY_TEST_CASES)
    
    # Summarize
    total_metrics = 0
    total_passed = 0
    
    for case_name, case_results in results.items():
        metric_scores = case_results["metric_scores"]
        for metric_name, score in metric_scores.items():
            total_metrics += 1
            if score >= 0.5:
                total_passed += 1
    
    score_pct = (total_passed / total_metrics * 100) if total_metrics > 0 else 0
    
    print(f"\nQuality Score: {total_passed}/{total_metrics} metrics passed ({score_pct:.1f}%)")
    return score_pct


def run_citation_evaluation():
    """Run deterministic citation evaluation."""
    print("\n" + "=" * 70)
    print("PHASE 4: CITATION EVALUATION")
    print("=" * 70)
    
    from tests.quality.test_quality import run_citation_evaluation, QUALITY_TEST_CASES
    
    results = run_citation_evaluation(QUALITY_TEST_CASES)
    
    total_checks = 0
    passed_checks = 0
    
    for case_name, case_results in results.items():
        cr = case_results["citation_results"]
        total_checks += 1
        if cr["all_citations_have_source"]:
            passed_checks += 1
    
    score_pct = (passed_checks / total_checks * 100) if total_checks > 0 else 0
    
    print(f"\nCitation Score: {passed_checks}/{total_checks} checks passed ({score_pct:.1f}%)")
    return score_pct


def main():
    """Run all evaluation phases and produce the final report."""
    print("PHASE 4: EVALUATION SYSTEM FOR PAPERPILOT")
    print("=" * 70)
    
    # 1. Structural Evaluation
    structural_result = run_structural_evaluation()
    
    # 2. Quality Evaluation
    quality_score = run_quality_evaluation()
    
    # 3. Citation Evaluation
    citation_score = run_citation_evaluation()
    
    # 4. Produce Final Report
    print("\n" + "=" * 70)
    print("PHASE 4: FINAL EVALUATION REPORT")
    print("=" * 70)
    
    print("\nSTRUCTURAL EVALUATION")
    print(f"  Cases: 10 golden test cases")
    print(f"  Checks: 60 deterministic checks (10 cases x 6 checks)")
    print(f"  Status: Complete (detailed results in test output above)")
    
    print("\nQUALITY EVALUATION")
    print(f"  Answer Relevancy: measured via DeepEval")
    print(f"  Faithfulness: measured via DeepEval")
    print(f"  Contextual Relevancy: measured via DeepEval")
    print(f"  Quality Score: {quality_score*100:.1f}%")
    
    print("\nCITATION EVALUATION")
    print(f"  Deterministic citation checks: run")
    print(f"  Citation Quality Score: {citation_score*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("KEY FINDING: Structural and quality scores remain SEPARATE")
    print("=" * 70)
    print("Structural: 'Did the agent behave correctly?'")
    print("Quality:    'Was the resulting answer good?'")
    print("=" * 70)
    
    # Cleanup temp files
    import glob
    for f in glob.glob("test_output.txt") + glob.glob("run_tests.py"):
        try:
            os.remove(f)
        except:
            pass


if __name__ == "__main__":
    main()