"""shared/analysis.py

Analysis functions for the Phase 2 hierarchical inference benchmark.
Takes CSV result files produced by pipeline/benchmark.py and extracts
scientific conclusions: Pareto front, escalation rates, communication
cost, statistical comparison, and error breakdown.

All functions are pure (no side effects) except load_results, which
reads from disk. No global state, no hardcoded paths or thresholds.

CSV column schema (from benchmark.py):
    sample_idx  : int   -- index of sample in the test set
    tier_used   : int   -- 1, 2, or 3
    prediction  : int   -- predicted class index
    gold_label  : int   -- ground-truth class index
    correct     : bool  -- prediction == gold_label
    confidence  : float -- max probability from the final tier
    latency_ms  : float -- wall-clock latency for this sample
    bytes_sent  : int   -- UTF-8 size of the input utterance
"""

import glob
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SNIPS_INTENTS = [
    'AddToPlaylist',
    'BookRestaurant',
    'GetWeather',
    'PlayMusic',
    'RateBook',
    'SearchCreativeWork',
    'SearchScreeningEvent',
]

# Regex for parsing theta values from pipeline CSV filenames.
_PIPELINE_FNAME_RE = re.compile(
    r'pipeline_theta1_(\d+\.\d+)_theta2_(\d+\.\d+)\.csv$'
)


# ---------------------------------------------------------------------------
# 1. load_results
# ---------------------------------------------------------------------------

def load_results(results_dir: str, pattern: str) -> pd.DataFrame:
    """Load and concatenate all benchmark CSV files matching a glob pattern.

    Each file is read into a DataFrame. A 'config' column is added whose
    value is the filename stem (e.g. 'pipeline_theta1_0.70_theta2_0.70').
    For pipeline files, 'theta1' and 'theta2' columns are also parsed from
    the filename and added. For ablation files these columns are set to NaN.
    All files are concatenated into a single DataFrame.

    Args:
        results_dir (str): Path to the directory containing result CSVs.
                           CONFIGURABLE: pass str(RESULTS_DIR) from notebook.
        pattern (str): Glob pattern to match files, e.g. 'pipeline_*.csv'
                       or 'ablation_*.csv' or '*.csv'.

    Returns:
        pd.DataFrame: Concatenated results with one row per sample per config.
                      Columns: sample_idx, tier_used, prediction, gold_label,
                      correct, confidence, latency_ms, bytes_sent, config,
                      theta1 (float or NaN), theta2 (float or NaN).

    Raises:
        FileNotFoundError: If no files match the pattern in results_dir.
    """
    search_path = os.path.join(results_dir, pattern)
    filepaths = sorted(glob.glob(search_path))

    if not filepaths:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in '{results_dir}'."
        )

    frames = []
    for fp in filepaths:
        df = pd.read_csv(fp)

        # Add config name from filename stem
        stem = Path(fp).stem
        df['config'] = stem

        # Parse theta values from pipeline filenames
        match = _PIPELINE_FNAME_RE.search(Path(fp).name)
        if match:
            df['theta1'] = float(match.group(1))
            df['theta2'] = float(match.group(2))
        else:
            df['theta1'] = float('nan')
            df['theta2'] = float('nan')

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Ensure correct dtypes after concat
    combined['tier_used']  = combined['tier_used'].astype(int)
    combined['prediction'] = combined['prediction'].astype(int)
    combined['gold_label'] = combined['gold_label'].astype(int)
    combined['correct']    = combined['correct'].astype(bool)

    return combined


# ---------------------------------------------------------------------------
# 2. compute_pareto_front
# ---------------------------------------------------------------------------

def compute_pareto_front(results: list[dict],
                         x_metric: str,
                         y_metric: str,
                         x_minimize: bool = True,
                         y_maximize: bool = True) -> list[dict]:
    """Compute the Pareto-optimal frontier from a list of configuration results.

    A point P dominates point Q if P is at least as good as Q on both
    metrics and strictly better on at least one. A point is Pareto-optimal
    if no other point dominates it. The function returns all non-dominated
    points sorted by x_metric ascending.

    For this project the typical call is:
        compute_pareto_front(summaries, 'mean_lat_ms', 'accuracy')
    where lower latency is better (x_minimize=True) and higher accuracy
    is better (y_maximize=True).

    Args:
        results (list[dict]): List of per-configuration summary dicts.
                              Each dict must contain x_metric and y_metric.
        x_metric (str): Metric on the x-axis (e.g. 'mean_lat_ms').
        y_metric (str): Metric on the y-axis (e.g. 'accuracy').
        x_minimize (bool): If True, lower x is better. Default True.
        y_maximize (bool): If True, higher y is better. Default True.

    Returns:
        list[dict]: Non-dominated points sorted by x_metric ascending.

    Raises:
        ValueError: If results is empty.
        KeyError: If x_metric or y_metric is absent from any result dict.
    """
    if not results:
        raise ValueError("results list is empty; cannot compute Pareto front.")

    # Validate keys on the first element
    for metric in [x_metric, y_metric]:
        if metric not in results[0]:
            raise KeyError(
                f"Metric '{metric}' not found in result dicts. "
                f"Available keys: {list(results[0].keys())}"
            )

    # Direction multipliers: flip sign so that "better" always means larger.
    # After multiplying, a larger value is always better.
    x_sign = -1.0 if x_minimize else 1.0
    y_sign =  1.0 if y_maximize else -1.0

    pareto = []
    for candidate in results:
        cx = x_sign * candidate[x_metric]
        cy = y_sign * candidate[y_metric]

        dominated = False
        for other in results:
            if other is candidate:
                continue
            ox = x_sign * other[x_metric]
            oy = y_sign * other[y_metric]
            # Other dominates candidate if it is >= on both and > on at least one
            if ox >= cx and oy >= cy and (ox > cx or oy > cy):
                dominated = True
                break

        if not dominated:
            pareto.append(candidate)

    # Sort by x_metric ascending (raw value)
    pareto.sort(key=lambda d: d[x_metric])
    return pareto


# ---------------------------------------------------------------------------
# 3. compute_escalation_rates
# ---------------------------------------------------------------------------

def compute_escalation_rates(all_results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the fraction of samples routed to each tier per configuration.

    Groups the concatenated results DataFrame by 'config'. For each config,
    counts the fraction of 700 samples handled at Tier 1, Tier 2, and Tier 3.
    Also returns accuracy and mean latency per config for convenience.

    For pipeline configs, theta1 and theta2 are included. For ablation configs
    these columns contain NaN.

    Args:
        all_results_df (pd.DataFrame): Concatenated results from load_results.
                                       Must contain 'config', 'tier_used',
                                       'correct', 'latency_ms', 'theta1',
                                       'theta2' columns.

    Returns:
        pd.DataFrame: One row per config with columns:
                      config, theta1, theta2, accuracy, mean_latency_ms,
                      rate_tier1, rate_tier2, rate_tier3,
                      n_tier1, n_tier2, n_tier3.

    Raises:
        KeyError: If required columns are missing from all_results_df.
    """
    required = {'config', 'tier_used', 'correct', 'latency_ms'}
    missing = required - set(all_results_df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    rows = []
    for config, group in all_results_df.groupby('config'):
        n_total = len(group)
        n1 = int((group['tier_used'] == 1).sum())
        n2 = int((group['tier_used'] == 2).sum())
        n3 = int((group['tier_used'] == 3).sum())

        # theta values (same for all rows in this config)
        theta1 = group['theta1'].iloc[0]
        theta2 = group['theta2'].iloc[0]

        rows.append({
            'config':         config,
            'theta1':         theta1,
            'theta2':         theta2,
            'accuracy':       float(group['correct'].mean()),
            'mean_latency_ms': float(group['latency_ms'].mean()),
            'rate_tier1':     n1 / n_total,
            'rate_tier2':     n2 / n_total,
            'rate_tier3':     n3 / n_total,
            'n_tier1':        n1,
            'n_tier2':        n2,
            'n_tier3':        n3,
        })

    result_df = pd.DataFrame(rows).sort_values('config').reset_index(drop=True)
    return result_df


# ---------------------------------------------------------------------------
# 4. compute_comms_cost_per_correct
# ---------------------------------------------------------------------------

def compute_comms_cost_per_correct(results_df: pd.DataFrame) -> float:
    """Compute total bytes transmitted divided by number of correct predictions.

    This metric captures communication efficiency: how many bytes does the
    system transmit in order to produce one correct prediction? Lower is
    better. Bytes sent is the UTF-8 size of each input utterance, which is
    the minimum message size for transmitting a sample between tiers.

    Args:
        results_df (pd.DataFrame): Per-sample results for one configuration.
                                   Must contain 'bytes_sent' and 'correct'.

    Returns:
        float: Total bytes sent / number of correct predictions.
               Units: bytes per correct prediction.

    Raises:
        ValueError: If results_df is empty or contains no correct predictions.
        KeyError: If required columns are missing.
    """
    required = {'bytes_sent', 'correct'}
    missing = required - set(results_df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if results_df.empty:
        raise ValueError("results_df is empty.")

    n_correct = int(results_df['correct'].sum())
    if n_correct == 0:
        raise ValueError(
            "No correct predictions in results_df; "
            "cannot compute cost per correct prediction."
        )

    total_bytes = int(results_df['bytes_sent'].sum())
    return total_bytes / n_correct


# ---------------------------------------------------------------------------
# 5. run_mcnemar_test
# ---------------------------------------------------------------------------

def run_mcnemar_test(predictions_a: list[int],
                     predictions_b: list[int],
                     gold_labels: list[int]) -> dict:
    """Run McNemar's test to statistically compare two prediction sets.

    McNemar's test asks: do configurations A and B disagree on predictions
    in a statistically significant way, or could the difference be due to
    chance? It operates on the 2x2 contingency table of per-sample outcomes:
        n_both_correct: both A and B correct
        n_a_only:       A correct, B wrong
        n_b_only:       A wrong, B correct
        n_both_wrong:   both wrong

    The test statistic with continuity correction (Yates) is:
        chi2 = (|n_a_only - n_b_only| - 1)^2 / (n_a_only + n_b_only)

    Under the null hypothesis of no difference, chi2 follows a chi-squared
    distribution with 1 degree of freedom. This implementation does not
    require statsmodels; the p-value is computed from the chi-squared CDF
    using the regularised incomplete gamma function (math.gammainc).

    Args:
        predictions_a (list[int]): Predicted class indices from config A.
        predictions_b (list[int]): Predicted class indices from config B.
        gold_labels (list[int]): Ground-truth class indices.

    Returns:
        dict: {
          'n_both_correct': int,
          'n_a_only':       int,   # A correct, B wrong
          'n_b_only':       int,   # B correct, A wrong
          'n_both_wrong':   int,
          'statistic':      float, # chi-squared value (with continuity correction)
          'p_value':        float,
          'significant':    bool,  # True if p_value < alpha
          'alpha':          float, # 0.05
        }

    Raises:
        ValueError: If input lists have different lengths or are empty.
    """
    if not (len(predictions_a) == len(predictions_b) == len(gold_labels)):
        raise ValueError(
            f"All input lists must have the same length. "
            f"Got: {len(predictions_a)}, {len(predictions_b)}, {len(gold_labels)}"
        )

    if len(predictions_a) == 0:
        raise ValueError("Input lists are empty.")

    # Build contingency table
    n_both_correct = 0
    n_a_only       = 0
    n_b_only       = 0
    n_both_wrong   = 0

    for pred_a, pred_b, gold in zip(predictions_a, predictions_b, gold_labels):
        a_correct = (pred_a == gold)
        b_correct = (pred_b == gold)

        if a_correct and b_correct:
            n_both_correct += 1
        elif a_correct and not b_correct:
            n_a_only += 1
        elif not a_correct and b_correct:
            n_b_only += 1
        else:
            n_both_wrong += 1

    # McNemar's test with continuity correction (Yates).
    # The discordant cells are b = n_a_only and c = n_b_only.
    b = n_a_only
    c = n_b_only

    if b + c == 0:
        # No disagreements: the two configurations are identical. p-value = 1.0.
        return {
            'n_both_correct': n_both_correct,
            'n_a_only':       b,
            'n_b_only':       c,
            'n_both_wrong':   n_both_wrong,
            'statistic':      0.0,
            'p_value':        1.0,
            'significant':    False,
            'alpha':          0.05,
        }

    # Chi-squared with continuity correction
    numerator = (abs(b - c) - 1.0) ** 2
    statistic = numerator / (b + c)

    # p-value from chi-squared CDF with 1 degree of freedom.
    # P(X > statistic) = 1 - CDF(statistic) = regularised upper incomplete gamma.
    # chi2 CDF(x, k=1) = regularised lower incomplete gamma(k/2, x/2)
    #                   = math.gammainc(0.5, statistic/2)  (Python 3.11+)
    # For older Python, implement using the error function:
    # chi2 CDF with 1 df = erf(sqrt(x/2))
    p_value = 1.0 - math.erf(math.sqrt(statistic / 2.0))

    alpha = 0.05

    return {
        'n_both_correct': n_both_correct,
        'n_a_only':       b,
        'n_b_only':       c,
        'n_both_wrong':   n_both_wrong,
        'statistic':      round(statistic, 6),
        'p_value':        round(p_value, 6),
        'significant':    p_value < alpha,
        'alpha':          alpha,
    }


# ---------------------------------------------------------------------------
# 6. error_analysis
# ---------------------------------------------------------------------------

def error_analysis(results_df: pd.DataFrame,
                   label_names: list[str]) -> dict:
    """Analyse prediction errors broken down by tier and intent class.

    Computes per-tier accuracy and error counts, per-class error breakdown,
    and an escalation effectiveness metric: among all samples that were
    escalated (tier_used > 1), what fraction ended up being predicted
    correctly? This tells you whether escalation is actually correcting
    uncertain samples or simply adding latency.

    Note: This function works entirely from the final-tier predictions stored
    in the CSV. It does not re-run Tier 1 inference on escalated samples,
    so the escalation_corrective_rate reflects the accuracy of the tier that
    handled each sample, not a counterfactual comparison against Tier 1.

    Args:
        results_df (pd.DataFrame): Per-sample results for one configuration.
                                   Must contain 'tier_used', 'prediction',
                                   'gold_label', 'correct', 'sample_idx'.
        label_names (list[str]): Intent class name strings in class-index order.
                                 For this project: SNIPS_INTENTS.

    Returns:
        dict: {
          'by_tier': {
            1: {'n': int, 'correct': int, 'wrong': int, 'accuracy': float},
            2: {'n': int, 'correct': int, 'wrong': int, 'accuracy': float},
            3: {'n': int, 'correct': int, 'wrong': int, 'accuracy': float},
          },
          'errors_by_class': pd.DataFrame,
              columns: ['class', 'n_errors', 'pct_of_all_errors']
              rows sorted by n_errors descending.
          'escalation_corrective_rate': float,
              Among all escalated samples (tier_used > 1), fraction that
              were predicted correctly. NaN if no samples were escalated.
          'n_escalated': int,
              Total number of samples that reached Tier 2 or Tier 3.
          'n_total_errors': int,
              Total number of incorrect predictions across all tiers.
        }

    Raises:
        KeyError: If required columns are missing from results_df.
        ValueError: If results_df is empty.
    """
    required = {'tier_used', 'prediction', 'gold_label', 'correct', 'sample_idx'}
    missing = required - set(results_df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if results_df.empty:
        raise ValueError("results_df is empty.")

    # --- Per-tier breakdown ---
    by_tier = {}
    for tier in [1, 2, 3]:
        subset = results_df[results_df['tier_used'] == tier]
        n = len(subset)
        n_correct = int(subset['correct'].sum())
        n_wrong = n - n_correct
        by_tier[tier] = {
            'n':        n,
            'correct':  n_correct,
            'wrong':    n_wrong,
            'accuracy': float(n_correct / n) if n > 0 else float('nan'),
        }

    # --- Per-class error breakdown (across all tiers) ---
    errors_df = results_df[~results_df['correct']].copy()
    n_total_errors = len(errors_df)

    if n_total_errors > 0:
        # Map gold_label index to class name
        errors_df['class'] = errors_df['gold_label'].map(
            lambda idx: label_names[idx] if 0 <= idx < len(label_names) else 'unknown'
        )
        class_error_counts = (
            errors_df.groupby('class')
            .size()
            .reset_index(name='n_errors')
            .sort_values('n_errors', ascending=False)
            .reset_index(drop=True)
        )
        class_error_counts['pct_of_all_errors'] = (
            class_error_counts['n_errors'] / n_total_errors * 100
        ).round(1)
    else:
        class_error_counts = pd.DataFrame(
            columns=['class', 'n_errors', 'pct_of_all_errors']
        )

    # --- Escalation effectiveness ---
    escalated = results_df[results_df['tier_used'] > 1]
    n_escalated = len(escalated)

    if n_escalated > 0:
        escalation_corrective_rate = float(escalated['correct'].mean())
    else:
        escalation_corrective_rate = float('nan')

    return {
        'by_tier':                    by_tier,
        'errors_by_class':            class_error_counts,
        'escalation_corrective_rate': escalation_corrective_rate,
        'n_escalated':                n_escalated,
        'n_total_errors':             n_total_errors,
    }
