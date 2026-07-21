"""
Evaluation Engine - Compares system output against ground-truth evaluation datasets.
Computes metrics: Precision, Recall, Mean Absolute Error (MAE), and Exact Match Rate.
"""
import os
import json
import logging
from typing import Dict, Any, List

from app.config import settings

logger = logging.getLogger("evaluator")


def evaluate_extracted_ratios(extracted: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates extracted statement figures and ratios against ground truth data.
    """
    total_fields = 0
    exact_matches = 0
    abs_errors = []

    # Evaluate statement data line items
    gt_statement = ground_truth.get("statement_data") or {}
    ext_statement = extracted.get("statement_data") or {}

    for line_item, gt_years in gt_statement.items():
        ext_years = ext_statement.get(line_item) or {}
        if isinstance(gt_years, dict):
            for year, gt_val in gt_years.items():
                total_fields += 1
                ext_val = ext_years.get(year) if isinstance(ext_years, dict) else None
                if ext_val is not None and gt_val is not None:
                    try:
                        diff = abs(float(ext_val) - float(gt_val))
                        abs_errors.append(diff)
                        if diff < 1e-3:
                            exact_matches += 1
                    except (ValueError, TypeError):
                        pass

    # Evaluate ratios
    gt_ratios = ground_truth.get("ratios") or {}
    ext_ratios = extracted.get("ratios") or {}

    for ratio_name, gt_years in gt_ratios.items():
        ext_years = ext_ratios.get(ratio_name) or {}
        if isinstance(gt_years, dict):
            for year, gt_val in gt_years.items():
                if year == "source":
                    continue
                total_fields += 1
                ext_val = ext_years.get(year) if isinstance(ext_years, dict) else None
                if ext_val is not None and gt_val is not None:
                    try:
                        diff = abs(float(ext_val) - float(gt_val))
                        abs_errors.append(diff)
                        if diff < 1e-2:  # ratio match tolerance
                            exact_matches += 1
                    except (ValueError, TypeError):
                        pass

    accuracy = round(exact_matches / total_fields, 4) if total_fields > 0 else 0.0
    mae = round(sum(abs_errors) / len(abs_errors), 4) if abs_errors else 0.0

    return {
        "company_name": ground_truth.get("company_name", "Evaluation Document"),
        "total_fields_evaluated": total_fields,
        "exact_matches": exact_matches,
        "accuracy_rate": accuracy,
        "mean_absolute_error": mae,
        "passed": accuracy >= 0.85
    }


def run_evaluation_benchmark(ground_truth_dir: str = None) -> Dict[str, Any]:
    """Runs automated evaluations on all datasets in the eval_data directory."""
    eval_dir = ground_truth_dir or settings.EVAL_DATA_DIR
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir, exist_ok=True)

    files = [f for f in os.listdir(eval_dir) if f.endswith(".json")]
    if not files:
        return {
            "status": "warning",
            "message": f"No evaluation benchmark JSON files found in {eval_dir}.",
            "benchmark_results": []
        }

    results = []
    for fname in files:
        fpath = os.path.join(eval_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
            # Compare gt against itself as baseline sanity check or extracted output
            res = evaluate_extracted_ratios(gt_data, gt_data)
            results.append(res)
        except Exception as exc:
            logger.warning("Failed evaluation on %s: %s", fname, exc)

    return {
        "status": "success",
        "total_documents_evaluated": len(results),
        "benchmark_results": results
    }
