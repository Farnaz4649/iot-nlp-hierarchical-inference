"""pipeline/offloading_strategies.py

Offloading decision strategies for hierarchical NLP inference.
Each strategy defines a rule for when to escalate a sample to the next tier.
Used by the router to decide escalation based on confidence, input length,
or other criteria.

All functions are pure and accept strategy parameters as arguments
(no global state, no configuration files).
"""

import numpy as np


def confidence_threshold_strategy(output: np.ndarray, theta: float) -> bool:
    """
    Determine if a sample should be escalated based on output confidence.
    
    Strategy: compare the maximum class probability against a threshold.
    If confidence is below the threshold, the model is not confident enough
    and the sample should escalate to the next tier.
    
    Args:
        output (np.ndarray): A 1D array of class probabilities (shape (n_classes,)).
                             Assumed to sum to 1.0 (i.e., softmax output).
        theta (float): Confidence threshold. Escalation occurs if max(output) < theta.
                       Typical range: 0.50 to 0.99.
    
    Returns:
        bool: True if the sample should be escalated (confidence too low),
              False if the prediction is confident enough to return.
    
    Raises:
        ValueError: If theta is not between 0.0 and 1.0.
    """
    if not (0.0 <= theta <= 1.0):
        raise ValueError(f"theta must be between 0.0 and 1.0, got {theta}")
    
    max_confidence = np.max(output)
    return max_confidence < theta


def input_length_strategy(text: str, 
                         length_limit: int, 
                         vectorizer=None) -> bool:
    """
    Determine if a sample should be escalated based on input length.
    
    Strategy: compare the token count of the input text against a limit.
    Shorter inputs are easier for lightweight models; longer inputs may
    benefit from more capable models. If token count exceeds the limit,
    the sample is routed to a higher tier.
    
    Args:
        text (str): The input utterance (raw text string).
        length_limit (int): Maximum token count before escalation.
                            Typical range: 20 to 128 tokens.
        vectorizer: An optional fitted TfidfVectorizer instance (from preprocess.py).
                    If provided, its tokenizer is used for accuracy. If None,
                    falls back to simple whitespace-based splitting. The 
                    vectorizer should be the same one fitted in preprocess.py 
                    so token counts are consistent with Tier 1 encoding.
    
    Returns:
        bool: True if the sample should be escalated (too long),
              False if the sample can be handled at the current tier.
    
    Raises:
        ValueError: If length_limit is negative.
    """
    if length_limit < 0:
        raise ValueError(f"length_limit must be non-negative, got {length_limit}")
    
    # Count tokens using the vectorizer's tokenizer if available,
    # otherwise fall back to whitespace-based splitting.
    if vectorizer is not None:
        # Use the vectorizer's built-in analyzer to tokenize consistently.
        analyzer = vectorizer.build_analyzer()
        tokens = analyzer(text)
        token_count = len(tokens)
    else:
        # Fallback: simple whitespace-based split.
        tokens = text.split()
        token_count = len(tokens)
    
    return token_count > length_limit


def apply_strategy(sample: str, 
                   model_output: np.ndarray,
                   strategy_fn: callable,
                   **kwargs) -> bool:
    """
    Apply a single offloading strategy to a sample.
    
    This is a wrapper function that takes a sample, its model output,
    and a strategy function, then applies the strategy to decide
    escalation. It allows the router to call any strategy function
    using a uniform interface.
    
    Args:
        sample (str): The input utterance (raw text string).
        model_output (np.ndarray): Output from the current tier
                                  (shape (n_classes,), values sum to 1.0).
        strategy_fn (callable): A strategy function (e.g.,
                               confidence_threshold_strategy or
                               input_length_strategy). The function
                               must accept **kwargs.
        **kwargs: Additional arguments passed to strategy_fn.
                  For confidence_threshold_strategy: theta (float)
                  For input_length_strategy: length_limit (int), 
                                            vectorizer (optional)
    
    Returns:
        bool: True if the sample should be escalated according to
              strategy_fn, False otherwise.
    
    Raises:
        TypeError: If strategy_fn is not callable.
        ValueError: Propagated from strategy_fn if validation fails.
    
    Example:
        # Escalate if confidence < 0.70
        should_escalate = apply_strategy(
            sample="find me a song",
            model_output=np.array([0.15, 0.65, 0.20]),
            strategy_fn=confidence_threshold_strategy,
            theta=0.70
        )
        # should_escalate returns True (max prob 0.65 < 0.70)
        
        # Escalate if input length > 20 tokens (using vectorizer)
        should_escalate = apply_strategy(
            sample="find me a creative work that is a song or movie",
            model_output=np.array([0.5, 0.3, 0.2]),
            strategy_fn=input_length_strategy,
            length_limit=20,
            vectorizer=fitted_vectorizer
        )
        # should_escalate returns True if token count > 20
    """
    if not callable(strategy_fn):
        raise TypeError(
            f"strategy_fn must be callable, got {type(strategy_fn).__name__}"
        )
    
    # For confidence-based strategies, pass model_output.
    # For input-based strategies, pass sample.
    # The strategy function decides which to use based on its signature.
    if strategy_fn == confidence_threshold_strategy:
        return strategy_fn(model_output, **kwargs)
    else:
        # Assume input_length_strategy or similar input-based strategy.
        return strategy_fn(sample, **kwargs)
