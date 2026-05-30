"""
Evaluation Sub-package
======================

Provides comprehensive evaluation utilities including accuracy,
precision, recall, F1, confusion matrices, and few-shot
performance benchmarking across different k-shot settings.
"""

from src.evaluation.metrics import (
    compute_accuracy,
    compute_precision_recall_f1,
    compute_confusion_matrix,
    evaluate_k_shot,
    save_results_json,
    save_results_csv,
)
from src.evaluation.evaluator import FewShotEvaluator

__all__ = [
    "compute_accuracy",
    "compute_precision_recall_f1",
    "compute_confusion_matrix",
    "evaluate_k_shot",
    "save_results_json",
    "save_results_csv",
    "FewShotEvaluator",
]
