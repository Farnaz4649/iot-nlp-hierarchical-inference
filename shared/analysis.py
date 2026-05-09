def load_results(results_dir: str, pattern: str) -> pd.DataFrame:
    """
    Load and concatenate all benchmark CSV files matching a glob pattern.
    
    Process: Search results_dir for files matching pattern, read each into
    a DataFrame, add a 'config' column from the filename, and concatenate
    all into one DataFrame for analysis.
    
    Args:
        results_dir (str): Path to the directory containing result CSVs.
                           CONFIGURABLE: set in notebook as str(RESULTS_DIR).
        pattern (str): Glob pattern to match files, e.g. 'pipeline_*.csv'
                       or 'ablation_*.csv'.
    
    Returns:
        pd.DataFrame: Concatenated results with one row per sample per config.
                      Contains all columns from the CSV plus a 'config' column.
    
    Raises:
        FileNotFoundError: If no files match the pattern in results_dir.
    """
    pass


def compute_pareto_front(results: list[dict],
                         x_metric: str,
                         y_metric: str,
                         x_minimize: bool = True,
                         y_maximize: bool = True) -> list[dict]:
    """
    Compute the Pareto-optimal frontier from a list of configuration results.
    
    Process: A configuration is Pareto-optimal if no other configuration
    simultaneously achieves a better x_metric AND a better y_metric. For
    each point, check whether any other point dominates it: lower x AND
    higher y. Return only the non-dominated points, sorted by x_metric.
    
    Args:
        results (list[dict]): List of per-configuration summary dicts.
                              Each dict must contain x_metric and y_metric keys.
        x_metric (str): Name of the metric to minimize (e.g. 'mean_lat_ms').
        y_metric (str): Name of the metric to maximize (e.g. 'accuracy').
        x_minimize (bool): If True, lower x is better. Default True.
        y_maximize (bool): If True, higher y is better. Default True.
    
    Returns:
        list[dict]: Subset of results that are Pareto-optimal, sorted by
                    x_metric ascending. Each dict is a copy of the input dict
                    with no added fields.
    
    Raises:
        KeyError: If x_metric or y_metric is not present in any result dict.
        ValueError: If results is empty.
    """
    pass


def compute_escalation_rates(all_results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the fraction of samples escalated to each tier per configuration.
    
    Process: Group the concatenated results DataFrame by 'config'. For each
    config, count the fraction of samples with tier_used == 1, 2, and 3.
    Return one row per config with escalation rates as fractions (0.0 to 1.0).
    
    Args:
        all_results_df (pd.DataFrame): Concatenated results from load_results.
                                       Must contain 'config' and 'tier_used' columns.
    
    Returns:
        pd.DataFrame: One row per config with columns:
                      ['config', 'rate_tier1', 'rate_tier2', 'rate_tier3',
                       'theta1', 'theta2', 'accuracy', 'mean_latency_ms']
    
    Raises:
        KeyError: If required columns are missing from all_results_df.
    """
    pass


def compute_comms_cost_per_correct(results_df: pd.DataFrame) -> float:
    """
    Compute total bytes transmitted divided by number of correct predictions.
    
    Process: Sum the 'bytes_sent' column for all samples. Count the number
    of rows where 'correct' is True. Return the ratio. This metric captures
    communication efficiency: how many bytes does it cost to get one correct
    prediction?
    
    Args:
        results_df (pd.DataFrame): Per-sample results for a single configuration.
                                   Must contain 'bytes_sent' and 'correct' columns.
    
    Returns:
        float: Total bytes sent divided by number of correct predictions.
               Lower is better. Units: bytes per correct prediction.
    
    Raises:
        ValueError: If results_df is empty or has no correct predictions.
        KeyError: If required columns are missing.
    """
    pass


def run_mcnemar_test(predictions_a: list[int],
                     predictions_b: list[int],
                     gold_labels: list[int]) -> dict:
    """
    Run McNemar's test to compare two sets of predictions statistically.
    
    Process: Build a 2x2 contingency table counting four outcome combinations:
    both correct, A correct and B wrong, A wrong and B correct, both wrong.
    Apply McNemar's test with continuity correction (statsmodels).
    Report whether the difference is statistically significant at alpha=0.05.
    
    Args:
        predictions_a (list[int]): Predicted class indices from configuration A.
                                   Length must match predictions_b and gold_labels.
        predictions_b (list[int]): Predicted class indices from configuration B.
        gold_labels (list[int]): Ground-truth class indices.
    
    Returns:
        dict: {
          'n_both_correct': int,
          'n_a_only': int,       # A correct, B wrong
          'n_b_only': int,       # B correct, A wrong
          'n_both_wrong': int,
          'statistic': float,
          'p_value': float,
          'significant': bool,   # True if p_value < 0.05
          'alpha': float,        # 0.05
        }
    
    Raises:
        ValueError: If input lists have different lengths.
    
    Note: McNemar's test requires the statsmodels library. If not installed,
          raise ImportError with a clear installation message.
    """
    pass


def error_analysis(results_df: pd.DataFrame,
                   label_names: list[str]) -> dict:
    """
    Analyse prediction errors broken down by tier and intent class.
    
    Process: For each tier (1, 2, 3), compute: how many samples were handled,
    how many were correct, how many were wrong, and which intent classes had
    the most errors. For escalated samples (tier_used > 1), compare against
    what Tier 1 would have predicted to quantify how many escalations were
    corrective (Tier 1 was wrong, higher tier was right).
    
    Args:
        results_df (pd.DataFrame): Per-sample results for a single configuration.
                                   Must contain 'tier_used', 'prediction',
                                   'gold_label', 'correct', 'sample_idx' columns.
        label_names (list[str]): List of intent class name strings, ordered
                                 by integer class index.
    
    Returns:
        dict: {
          'by_tier': {
            1: {'n': int, 'correct': int, 'wrong': int, 'error_rate': float},
            2: {'n': int, 'correct': int, 'wrong': int, 'error_rate': float},
            3: {'n': int, 'correct': int, 'wrong': int, 'error_rate': float},
          },
          'errors_by_class': pd.DataFrame,  # columns: class, n_errors, tier
          'escalation_corrective_rate': float,  # fraction of escalations that
                                                # corrected a Tier 1 error
        }
    
    Raises:
        KeyError: If required columns are missing from results_df.
    """
    pass
