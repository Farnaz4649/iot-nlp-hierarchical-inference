"""pipeline/router.py

Routing orchestration for hierarchical NLP inference.
Routes individual samples through the three-tier hierarchy using confidence-based
escalation thresholds. Tracks which tier handled each sample and logs decisions
for post-hoc analysis.

All functions are pure (except log_decision, which has I/O side effects).
No global state, no configuration files.
"""

import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np


def get_confidence(model_output: dict) -> float:
    """
    Extract the maximum class probability (confidence) from a tier's output.
    
    Process: All three tiers return a dict with keys 'labels' (class indices)
    and 'proba' (probability distribution, shape (n_classes,)). The confidence
    is the maximum value in the proba array.
    
    Args:
        model_output (dict): Output dict from any tier with keys:
                             {'labels': int or array, 'proba': np.ndarray}
    
    Returns:
        float: Maximum probability value in model_output['proba'].
               Range [0.0, 1.0]. Represents how confident the model is
               in its top-1 prediction.
    
    Raises:
        KeyError: If 'proba' key is missing from model_output.
        ValueError: If model_output['proba'] is empty.
    """
    if 'proba' not in model_output:
        raise KeyError(
            f"model_output missing 'proba' key. Keys present: {model_output.keys()}"
        )
    
    proba = model_output['proba']
    
    if len(proba) == 0:
        raise ValueError("model_output['proba'] is empty (length 0)")
    
    confidence = float(np.max(proba))
    return confidence


def should_escalate(confidence: float, threshold: float) -> bool:
    """
    Determine if a sample should be escalated to the next tier.
    
    Process: Compare the model's confidence against the tier threshold.
    If confidence is strictly below the threshold, escalation is needed.
    
    Args:
        confidence (float): Max class probability from the current tier.
                            Range [0.0, 1.0].
        threshold (float): Escalation threshold for the current tier
                           (theta_1 or theta_2). Range [0.0, 1.0].
    
    Returns:
        bool: True if confidence < threshold (escalate),
              False if confidence >= threshold (return result).
    
    Raises:
        ValueError: If confidence or threshold is not in [0.0, 1.0].
    """
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(
            f"confidence must be in [0.0, 1.0], got {confidence}"
        )
    
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"threshold must be in [0.0, 1.0], got {threshold}"
        )
    
    return confidence < threshold


def route_sample(sample: str,
                 tier1_fn: callable,
                 tier2_fn: callable,
                 tier3_fn: callable,
                 theta1: float,
                 theta2: float,
                 vectorizer=None) -> dict:
    """
    Route a single sample through the three-tier hierarchy and return result.
    
    Process: Invoke Tier 1 inference. If confidence >= theta_1, return
    immediately with tier_used=1. Otherwise, invoke Tier 2. If confidence
    >= theta_2, return with tier_used=2. Otherwise, invoke Tier 3 and
    return its result with tier_used=3 (Tier 3 is terminal).
    
    Args:
        sample (str): The input utterance (raw text string).
        tier1_fn (callable): Inference function for Tier 1.
                             Signature: tier1_fn(sample: str) -> dict
                             Returns: {'labels': int/array, 'proba': np.ndarray}
        tier2_fn (callable): Inference function for Tier 2.
                             Signature: tier2_fn(sample: str) -> dict
                             Returns: {'labels': int/array, 'proba': np.ndarray}
        tier3_fn (callable): Inference function for Tier 3.
                             Signature: tier3_fn(sample: str) -> dict
                             Returns: {'labels': int/array, 'proba': np.ndarray}
        theta1 (float): Confidence threshold for Tier 1 escalation.
                        Range [0.0, 1.0]. CONFIGURABLE: set in notebook.
        theta2 (float): Confidence threshold for Tier 2 escalation.
                        Range [0.0, 1.0]. CONFIGURABLE: set in notebook.
        vectorizer: Optional fitted TfidfVectorizer (from preprocess.py).
                    Currently unused but kept for future use in input_length_strategy.
    
    Returns:
        dict: Result dict with keys:
              {
                'prediction': int or str,  # Class index or label name
                'confidence': float,        # Max probability from final tier
                'tier_used': int,          # 1, 2, or 3
              }
    
    Raises:
        TypeError: If any tier function is not callable.
        KeyError: Propagated from tier functions if output dict is malformed.
        ValueError: Propagated from should_escalate if thresholds are invalid.
    
    Note: This function does not measure latency; the calling benchmark loop
          should wrap this call with time.perf_counter() if latency tracking
          is needed. This keeps route_sample pure (no timing side effects).
    """
    if not callable(tier1_fn):
        raise TypeError(f"tier1_fn must be callable, got {type(tier1_fn).__name__}")
    if not callable(tier2_fn):
        raise TypeError(f"tier2_fn must be callable, got {type(tier2_fn).__name__}")
    if not callable(tier3_fn):
        raise TypeError(f"tier3_fn must be callable, got {type(tier3_fn).__name__}")
    
    # Tier 1: always start here
    output_tier1 = tier1_fn(sample)
    confidence_tier1 = get_confidence(output_tier1)
    
    if not should_escalate(confidence_tier1, theta1):
        # Confidence is high enough; return Tier 1 result
        return {
            'prediction': int(output_tier1['labels']),
            'confidence': confidence_tier1,
            'tier_used': 1,
        }
    
    # Escalate to Tier 2
    output_tier2 = tier2_fn(sample)
    confidence_tier2 = get_confidence(output_tier2)
    
    if not should_escalate(confidence_tier2, theta2):
        # Confidence is high enough; return Tier 2 result
        return {
            'prediction': int(output_tier2['labels']),
            'confidence': confidence_tier2,
            'tier_used': 2,
        }
    
    # Escalate to Tier 3 (terminal; always return)
    output_tier3 = tier3_fn(sample)
    confidence_tier3 = get_confidence(output_tier3)
    
    return {
        'prediction': int(output_tier3['labels']),
        'confidence': confidence_tier3,
        'tier_used': 3,
    }


def serialize_sample(sample: str) -> bytes:
    """
    Serialize a text sample for network transmission.
    
    Process: Encode the UTF-8 string to bytes. This represents the minimum
    message size if the sample were transmitted between tiers.
    
    Args:
        sample (str): The input utterance (raw text string).
    
    Returns:
        bytes: UTF-8 encoded bytes representation of the sample.
               Size is the number of bytes that would be transmitted.
    
    Raises:
        UnicodeEncodeError: If the sample contains characters that cannot
                            be encoded as UTF-8 (rare, UTF-8 is universal).
    """
    return sample.encode('utf-8')


def log_decision(log_path: str,
                 sample_id: str,
                 tier_used: int,
                 correct: bool,
                 latency_ms: float,
                 bytes_sent: int) -> None:
    """
    Append a routing decision record to a log CSV file.
    
    Process: Create or append to a CSV. If the file does not exist, write
    a header row first. Then append one row with all metrics. This allows
    post-hoc analysis of which tiers handled which samples and whether
    escalation was correct.
    
    Args:
        log_path (str): Full path to the log CSV file (e.g.,
                        '/path/to/logs/routing_log.csv').
                        CONFIGURABLE: set in notebook as LOGS_DIR / filename.
        sample_id (str): Unique identifier for the sample (e.g., index or
                         hash of the utterance).
        tier_used (int): Which tier made the final prediction (1, 2, or 3).
        correct (bool): Whether the prediction matched the gold label.
        latency_ms (float): Total end-to-end latency in milliseconds.
        bytes_sent (int): Number of bytes transmitted (from serialize_sample).
    
    Returns:
        None. Side effect: appends one row to the CSV at log_path.
    
    Note: If the file does not exist, create it with a header row.
          If it exists, append without re-writing the header.
          Thread-safe for single-process, single-thread access.
          For concurrent writes, use file locking or a queue.
    """
    # Ensure the parent directory exists
    log_dir = Path(log_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists to decide whether to write header
    file_exists = os.path.isfile(log_path)
    
    fieldnames = [
        'timestamp',
        'sample_id',
        'tier_used',
        'correct',
        'latency_ms',
        'bytes_sent',
    ]
    
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the decision record
        writer.writerow({
            'timestamp': datetime.now().isoformat(),
            'sample_id': str(sample_id),
            'tier_used': int(tier_used),
            'correct': bool(correct),
            'latency_ms': float(latency_ms),
            'bytes_sent': int(bytes_sent),
        })
