#!/usr/bin/env python3
"""
Top-K Metrics for Compound Prioritization

This script demonstrates the evaluation metrics used in the MD-enhanced
compound prioritization feasibility study.

Key Metrics:
- Median Potency @K: Median binding affinity (nM) of top-K compounds
- Regret @K: Worst selected compound vs. best available
- Enrichment Factor @K: Hit rate vs. random selection

Author: [Anonymized]
Date: February 2026
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import argparse


def compute_topk_metrics(
    y_true_potency: np.ndarray,
    y_pred_score: np.ndarray,
    train_potency: np.ndarray,
    k: int = 5,
    hit_quantile: float = 0.25,
) -> Dict[str, float]:
    """
    Compute Top-K decision quality metrics for compound prioritization.

    Parameters
    ----------
    y_true_potency : np.ndarray
        True binding potencies (nM) for test compounds
    y_pred_score : np.ndarray
        Predicted scores (lower = better predicted affinity)
    train_potency : np.ndarray
        True potencies from training set (for computing hit threshold)
    k : int
        Number of top compounds to select
    hit_quantile : float
        Quantile for defining "hit" compounds (default: 0.25 = top 25%)

    Returns
    -------
    metrics : Dict[str, float]
        Dictionary containing:
        - median_potency_k: Median potency of top-K compounds (nM)
        - regret_k: Regret at K (nM)
        - ef_k: Enrichment factor at K
        - precision_k: Precision at K
        - hit_present_k: Whether any hit is in top-K (0 or 1)
    """

    n_test = len(y_true_potency)
    k = min(k, n_test)  # Cap K at test set size

    # Compute hit threshold from TRAIN set only (avoid data leakage)
    hit_threshold = np.quantile(train_potency, hit_quantile)

    # Identify hits in test set
    is_hit = y_true_potency <= hit_threshold
    n_hits = is_hit.sum()

    # Rank compounds by predicted score (lower = better)
    topk_idx = np.argsort(y_pred_score)[:k]
    topk_potencies = y_true_potency[topk_idx]

    # Median Potency @K: Quality of selected compounds
    median_potency_k = float(np.median(topk_potencies))

    # Regret @K: Worst selected vs. best available
    worst_selected = topk_potencies.max()
    best_available = y_true_potency.min()
    regret_k = float(worst_selected - best_available)

    # Enrichment Factor @K
    if n_hits > 0:
        n_hits_selected = is_hit[topk_idx].sum()
        hit_rate_topk = n_hits_selected / k
        hit_rate_random = n_hits / n_test
        ef_k = hit_rate_topk / hit_rate_random if hit_rate_random > 0 else 0.0
        precision_k = hit_rate_topk
        hit_present_k = float(n_hits_selected > 0)
    else:
        ef_k = 0.0
        precision_k = 0.0
        hit_present_k = 0.0

    return {
        f'median_potency@{k}': median_potency_k,
        f'regret@{k}': regret_k,
        f'ef@{k}': ef_k,
        f'precision@{k}': precision_k,
        f'hit_present@{k}': hit_present_k,
    }


def example_usage():
    """
    Example usage with synthetic data.

    In practice, you would load:
    - y_true_potency: Experimental binding affinities from your test set
    - y_pred_score: Model predictions (e.g., from Extra Trees regressor)
    - train_potency: Experimental affinities from training set
    """

    print("=" * 70)
    print("Top-K Metrics: Example Usage")
    print("=" * 70)

    # Synthetic test data (20 compounds)
    np.random.seed(42)
    y_true_potency = np.random.lognormal(mean=6, sigma=1.5, size=20) * 1000  # nM

    # Synthetic training data (100 compounds)
    train_potency = np.random.lognormal(mean=6, sigma=1.5, size=100) * 1000  # nM

    # Scenario 1: Perfect predictions (score = true potency)
    print("\nScenario 1: Perfect Ranking")
    print("-" * 70)
    y_pred_perfect = y_true_potency.copy()
    metrics_perfect = compute_topk_metrics(
        y_true_potency, y_pred_perfect, train_potency, k=5, hit_quantile=0.25
    )
    for metric, value in metrics_perfect.items():
        print(f"  {metric:25s}: {value:>10.2f}")

    # Scenario 2: Random predictions
    print("\nScenario 2: Random Ranking")
    print("-" * 70)
    y_pred_random = np.random.rand(20)
    metrics_random = compute_topk_metrics(
        y_true_potency, y_pred_random, train_potency, k=5, hit_quantile=0.25
    )
    for metric, value in metrics_random.items():
        print(f"  {metric:25s}: {value:>10.2f}")

    # Scenario 3: Noisy but informative predictions
    print("\nScenario 3: Noisy Ranking (correlation ~ 0.7)")
    print("-" * 70)
    noise = np.random.randn(20) * y_true_potency.std() * 0.5
    y_pred_noisy = y_true_potency + noise
    metrics_noisy = compute_topk_metrics(
        y_true_potency, y_pred_noisy, train_potency, k=5, hit_quantile=0.25
    )
    for metric, value in metrics_noisy.items():
        print(f"  {metric:25s}: {value:>10.2f}")

    print("\n" + "=" * 70)
    print("Interpretation:")
    print("=" * 70)
    print("- Median Potency @5: Lower is better (more potent compounds)")
    print("- Regret @5: Lower is better (fewer dead-end compounds)")
    print("- EF @5: Higher is better (enrichment above random)")
    print("- Precision @5: Fraction of top-5 that are hits")
    print("- Hit Present @5: 1.0 if at least one hit in top-5, else 0.0")
    print("=" * 70)


def load_and_evaluate(predictions_csv: str, output_dir: str = "./results"):
    """
    Load predictions from CSV and compute Top-K metrics.

    Expected CSV format:
    - compound_id: Unique identifier
    - y_true_potency: Experimental potency (nM)
    - y_pred_score: Model prediction
    - fold: Cross-validation fold ID
    - split: 'train' or 'test'

    Parameters
    ----------
    predictions_csv : str
        Path to predictions CSV file
    output_dir : str
        Directory to save results
    """
    import os

    # Load predictions
    print(f"Loading predictions from: {predictions_csv}")
    df = pd.read_csv(predictions_csv)

    # Validate columns
    required_cols = ['y_true_potency', 'y_pred_score', 'fold', 'split']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Compute metrics for each fold
    results = []
    folds = sorted(df['fold'].unique())

    for fold_id in folds:
        train_df = df[(df['fold'] == fold_id) & (df['split'] == 'train')]
        test_df = df[(df['fold'] == fold_id) & (df['split'] == 'test')]

        if len(test_df) < 5:
            print(f"Skipping fold {fold_id}: test set too small (n={len(test_df)})")
            continue

        train_potency = train_df['y_true_potency'].values
        y_true = test_df['y_true_potency'].values
        y_pred = test_df['y_pred_score'].values

        # Compute metrics for K=3,5
        for k in [3, 5]:
            metrics = compute_topk_metrics(
                y_true, y_pred, train_potency, k=k, hit_quantile=0.25
            )
            metrics['fold'] = fold_id
            metrics['k'] = k
            metrics['n_test'] = len(test_df)
            results.append(metrics)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(results)

    output_path = os.path.join(output_dir, 'topk_metrics.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Summary statistics
    summary = results_df.groupby('k').agg({
        'median_potency@3': ['mean', 'std'],
        'median_potency@5': ['mean', 'std'],
        'regret@3': ['mean', 'std'],
        'regret@5': ['mean', 'std'],
        'ef@3': ['mean', 'std'],
        'ef@5': ['mean', 'std'],
    })

    summary_path = os.path.join(output_dir, 'topk_summary.csv')
    summary.to_csv(summary_path)
    print(f"Summary saved to: {summary_path}")

    return results_df


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Compute Top-K metrics for compound prioritization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run example with synthetic data
  python compute_topk_metrics.py --example

  # Evaluate predictions from CSV
  python compute_topk_metrics.py --predictions my_predictions.csv --output results/
        """
    )

    parser.add_argument(
        '--example',
        action='store_true',
        help='Run example with synthetic data'
    )

    parser.add_argument(
        '--predictions',
        type=str,
        help='Path to predictions CSV file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./results',
        help='Output directory for results (default: ./results)'
    )

    args = parser.parse_args()

    if args.example:
        example_usage()
    elif args.predictions:
        load_and_evaluate(args.predictions, args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
