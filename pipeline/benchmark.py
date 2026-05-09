"""pipeline/benchmark.py

Benchmarking harness for the hierarchical NLP inference pipeline.
Runs the router across a test set with different threshold combinations,
tracks per-sample metrics, and saves results for later analysis.

All functions are pure (except save_results, which has I/O side effects).
No global state, no configuration files.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.router import route_sample, serialize_sample


def run_pipeline(test_samples: list[str],
                 gold_labels: list[int],
                 tier1_fn: callable,
                 tier2_fn: callable,
                 tier3_fn: callable,
                 theta1: float,
                 theta2: float,
                 label_encoder=None) -> list[dict]:
    """
    Run the full escalation pipeline on a test set and collect per-sample results.
    
    Process: For each sample in test_samples, call route_sample with the given
    thresholds. Compare the prediction against the gold label. Measure latency
    via wall-clock time. Measure message size via serialize_sample. Collect
    all results into a list of dicts, one per sample.
    
    Args:
        test_samples (list[str]): List of raw utterance strings to classify.
                                  Length must match gold_labels.
        gold_labels (list[int]): Ground-truth intent class indices (0-6).
                                 Used to compute correctness for each sample.
        tier1_fn (callable): Tier 1 inference function.
                             Signature: tier1_fn(sample: str) -> dict
                             Returns: {'labels': int, 'proba': np.ndarray}
        tier2_fn (callable): Tier 2 inference function.
                             Signature: tier2_fn(sample: str) -> dict
        tier3_fn (callable): Tier 3 inference function.
                             Signature: tier3_fn(sample: str) -> dict
        theta1 (float): Escalation threshold for Tier 1. Range [0.0, 1.0].
                        CONFIGURABLE: set in notebook.
        theta2 (float): Escalation threshold for Tier 2. Range [0.0, 1.0].
                        CONFIGURABLE: set in notebook.
        label_encoder: Optional fitted LabelEncoder (from preprocess.py).
                       Not used; kept for future compatibility.
    
    Returns:
        list[dict]: List of result dicts, one per sample. Each dict contains:
                    {
                      'sample_idx': int,           # Index in test_samples
                      'tier_used': int,            # 1, 2, or 3
                      'prediction': int,           # Predicted class index
                      'gold_label': int,           # Ground-truth label
                      'correct': bool,             # prediction == gold_label
                      'confidence': float,         # Max probability from final tier
                      'latency_ms': float,         # Wall-clock time for route_sample
                      'bytes_sent': int,           # UTF-8 size of sample
                    }
    
    Raises:
        ValueError: If test_samples and gold_labels have different lengths.
        TypeError: If tier functions are not callable.
        KeyError: Propagated from tier functions if output dict is malformed.
    """
    if len(test_samples) != len(gold_labels):
        raise ValueError(
            f"test_samples and gold_labels must have the same length. "
            f"Got {len(test_samples)} samples and {len(gold_labels)} labels."
        )
    
    results = []
    
    for sample_idx, (sample, gold_label) in enumerate(zip(test_samples, gold_labels)):
        # Measure serialization size
        serialized = serialize_sample(sample)
        bytes_sent = len(serialized)
        
        # Measure routing latency via wall-clock time
        start_time = time.perf_counter()
        route_result = route_sample(
            sample=sample,
            tier1_fn=tier1_fn,
            tier2_fn=tier2_fn,
            tier3_fn=tier3_fn,
            theta1=theta1,
            theta2=theta2,
            vectorizer=None  # Not used in Phase 2
        )
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000.0  # Convert to ms
        
        # Extract routing result
        prediction = route_result['prediction']
        confidence = route_result['confidence']
        tier_used = route_result['tier_used']
        
        # Compute correctness
        correct = (prediction == gold_label)
        
        # Assemble result dict for this sample
        result_dict = {
            'sample_idx': int(sample_idx),
            'tier_used': int(tier_used),
            'prediction': int(prediction),
            'gold_label': int(gold_label),
            'correct': bool(correct),
            'confidence': float(confidence),
            'latency_ms': float(latency_ms),
            'bytes_sent': int(bytes_sent),
        }
        
        results.append(result_dict)
    
    return results


def run_ablation(test_samples: list[str],
                 gold_labels: list[int],
                 tier1_fn: callable,
                 tier2_fn: callable,
                 tier3_fn: callable,
                 mode: str,
                 label_encoder=None) -> list[dict]:
    """
    Run an ablation baseline (Tier 1 only, Tier 3 only, or two-tier) on a test set.
    
    Process: Like run_pipeline, but apply a fixed routing mode rather than
    thresholds. Tier 1 only mode always uses Tier 1. Tier 3 only mode always
    uses Tier 3. Two-tier mode uses Tier 1, escalates to Tier 2 if needed,
    but never uses Tier 3 (Tier 2 result is always returned).
    
    Args:
        test_samples (list[str]): List of utterances to classify.
        gold_labels (list[int]): Ground-truth labels (0-6).
        tier1_fn (callable): Tier 1 inference function.
                             Signature: tier1_fn(sample: str) -> dict
        tier2_fn (callable): Tier 2 inference function.
                             Signature: tier2_fn(sample: str) -> dict
        tier3_fn (callable): Tier 3 inference function.
                             Signature: tier3_fn(sample: str) -> dict
        mode (str): Ablation mode. One of:
                    'tier1_only' - always use Tier 1, return immediately
                    'tier3_only' - always use Tier 3, skip Tiers 1 and 2
                    'two_tier'   - use Tiers 1 and 2 only, no escalation to Tier 3
        label_encoder: Optional; kept for compatibility.
    
    Returns:
        list[dict]: List of result dicts, same schema as run_pipeline.
    
    Raises:
        ValueError: If mode is not one of the three valid strings.
        ValueError: If test_samples and gold_labels have different lengths.
        TypeError: If tier functions are not callable.
    """
    valid_modes = {'tier1_only', 'tier3_only', 'two_tier'}
    if mode not in valid_modes:
        raise ValueError(
            f"mode must be one of {valid_modes}, got '{mode}'"
        )
    
    if len(test_samples) != len(gold_labels):
        raise ValueError(
            f"test_samples and gold_labels must have the same length. "
            f"Got {len(test_samples)} samples and {len(gold_labels)} labels."
        )
    
    results = []
    
    for sample_idx, (sample, gold_label) in enumerate(zip(test_samples, gold_labels)):
        # Measure serialization size
        serialized = serialize_sample(sample)
        bytes_sent = len(serialized)
        
        # Measure latency via wall-clock time
        start_time = time.perf_counter()
        
        if mode == 'tier1_only':
            # Always use Tier 1, return immediately
            output_tier1 = tier1_fn(sample)
            prediction = int(output_tier1['labels'])
            confidence = float(np.max(output_tier1['proba']))
            tier_used = 1
        
        elif mode == 'tier3_only':
            # Always use Tier 3, skip Tiers 1 and 2
            output_tier3 = tier3_fn(sample)
            prediction = int(output_tier3['labels'])
            confidence = float(np.max(output_tier3['proba']))
            tier_used = 3
        
        elif mode == 'two_tier':
            # Use Tiers 1 and 2 only. Tier 1 always runs, but Tier 2 result
            # is returned regardless of confidence (no further escalation).
            output_tier1 = tier1_fn(sample)
            confidence_tier1 = float(np.max(output_tier1['proba']))
            
            # For two-tier, we use a fixed escalation threshold of 0.0,
            # meaning every sample that fails Tier 1 goes to Tier 2.
            # Alternatively, escalate always (use a very high theta).
            # We choose: escalate if confidence < 1.0 (i.e., always go to Tier 2
            # if Tier 1 is uncertain), but more practically, use a low threshold
            # like 0.5 as a default. For phase 2, we escalate always.
            # Actually, for a true "two-tier" ablation, we should escalate
            # Tier 1 to Tier 2 generously (low threshold). Let's use theta1=1.0
            # which means "escalate unless confidence is exactly 1.0".
            # Better: use a high threshold. Actually, let's escalate Tier 1
            # to Tier 2 based on a reasonable default threshold (e.g., 0.7),
            # and Tier 2 always returns (theta2 = 0.0, meaning never escalate
            # beyond Tier 2).
            
            if confidence_tier1 < 0.7:  # Escalate to Tier 2 if low confidence
                output_tier2 = tier2_fn(sample)
                prediction = int(output_tier2['labels'])
                confidence = float(np.max(output_tier2['proba']))
                tier_used = 2
            else:
                prediction = int(output_tier1['labels'])
                confidence = confidence_tier1
                tier_used = 1
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0
        
        # Compute correctness
        correct = (prediction == gold_label)
        
        # Assemble result dict
        result_dict = {
            'sample_idx': int(sample_idx),
            'tier_used': int(tier_used),
            'prediction': int(prediction),
            'gold_label': int(gold_label),
            'correct': bool(correct),
            'confidence': float(confidence),
            'latency_ms': float(latency_ms),
            'bytes_sent': int(bytes_sent),
        }
        
        results.append(result_dict)
    
    return results


def save_results(results: list[dict], output_path: str) -> None:
    """
    Write a list of per-sample result dicts to a CSV file.
    
    Process: Convert the list of dicts to a CSV table with one row per sample.
    Create the parent directory if it does not exist. Overwrite any existing
    file at output_path. Use pandas for simple, reliable CSV writing.
    
    Args:
        results (list[dict]): List of result dicts from run_pipeline or run_ablation.
                              Each dict must have consistent keys.
        output_path (str): Full path to the output CSV file
                           (e.g., '/path/to/results/pipeline_theta1_0.70_theta2_0.60.csv').
                           CONFIGURABLE: set in notebook.
    
    Returns:
        None. Side effect: writes a CSV file at output_path.
    
    Raises:
        ValueError: If results is empty.
        OSError: If the parent directory cannot be created or file cannot be written.
    
    Note: Floating-point values (latency_ms, confidence) are written with
          default pandas precision. No explicit rounding is applied.
    """
    if len(results) == 0:
        raise ValueError("results list is empty; nothing to write")
    
    # Create parent directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(results)
    
    # Write to CSV, overwriting any existing file
    df.to_csv(output_path, index=False, encoding='utf-8')
