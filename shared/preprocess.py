"""shared/preprocess.py

Shared preprocessing functions for the IoT NLP hierarchical inference project.
Used by all three tiers. Contains only pure functions with no side effects.

All functions that fit a transformer (tokenize_tfidf, encode_labels) return
both the transformed data and the fitted object, so the same fitted instance
can be applied at inference time without re-fitting on new data.
"""

import json
import os
import warnings

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# SNIPS NLU dataset constants
# CONFIGURABLE: update if a different dataset or file naming pattern is used.
# ---------------------------------------------------------------------------

SNIPS_INTENTS = [
    "AddToPlaylist",
    "BookRestaurant",
    "GetWeather",
    "PlayMusic",
    "RateBook",
    "SearchCreativeWork",
    "SearchScreeningEvent",
]

TRAIN_FILE_PATTERN    = "train_{intent}_full.json"
VALIDATE_FILE_PATTERN = "validate_{intent}.json"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten_utterance(utterance: dict) -> str:
    """Join all token text fields in a single SNIPS utterance dict.

    SNIPS stores each utterance as a list of token dicts under the key 'data',
    where each dict has a 'text' key and an optional 'entity' key. Tokens
    already include their own surrounding whitespace, so they are concatenated
    with no added separator and the result is stripped.

    Args:
        utterance: A dict of the form {"data": [{"text": "...", ...}, ...]}.

    Returns:
        A single stripped string for the utterance.
    """
    return "".join(token["text"] for token in utterance.get("data", [])).strip()


def _load_split_from_files(data_dir: str, file_pattern: str) -> tuple[list, list]:
    """Load one split (train or validate) from all SNIPS intent files.

    The SNIPS JSON format uses the intent name string as the top-level key,
    not the word 'data'. Each value is a list of utterance dicts, where each
    utterance dict contains a 'data' key holding a list of token dicts.
    Files must be opened with latin-1 encoding as the SNIPS corpus contains
    Latin-1 encoded characters.

    Args:
        data_dir: Path to the directory containing the SNIPS JSON files.
        file_pattern: Filename pattern with '{intent}' as placeholder,
            e.g. 'train_{intent}_full.json'.

    Returns:
        A tuple (texts, intents) where both are flat lists of strings.

    Raises:
        FileNotFoundError: If no files matching the pattern are found in data_dir.
    """
    texts, intents = [], []
    found_any = False

    for intent in SNIPS_INTENTS:
        filename = file_pattern.format(intent=intent)
        filepath = os.path.join(data_dir, filename)

        if not os.path.isfile(filepath):
            warnings.warn(f"Expected file not found, skipping: {filepath}")
            continue

        found_any = True
        # latin-1 encoding is required: SNIPS files contain Latin-1 characters
        with open(filepath, "r", encoding="latin-1") as f:
            raw = json.load(f)

        # Top-level key is the intent name, not 'data'
        for utterance in raw[intent]:
            texts.append(_flatten_utterance(utterance))
            intents.append(intent)

    if not found_any:
        raise FileNotFoundError(
            f"No SNIPS data files found in '{data_dir}' "
            f"matching pattern '{file_pattern}'. "
            "Check that DATA_DIR points to the correct folder."
        )

    return texts, intents


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> dict:
    """Load the SNIPS NLU dataset from a directory of JSON files.

    Reads the train and validate splits. Each split is stored as a sub-dict
    with keys 'text' (list of utterance strings) and 'intent' (list of
    intent label strings). The validate split is exposed under the key
    'validate' and is intended to be used as the held-out test set via
    split_data(), since SNIPS does not ship a separate test JSON.

    Args:
        path: Path to the directory containing the SNIPS JSON files.
            Expected to contain files named 'train_{Intent}_full.json'
            and 'validate_{Intent}.json' for each of the 7 SNIPS intents.

    Returns:
        A dict with keys 'train' and 'validate', each mapping to a sub-dict:
            {
                'text':   list[str],   # raw utterance strings
                'intent': list[str],   # corresponding intent label strings
            }

    Raises:
        FileNotFoundError: If no matching files are found under path.
    """
    train_texts, train_intents       = _load_split_from_files(path, TRAIN_FILE_PATTERN)
    validate_texts, validate_intents = _load_split_from_files(path, VALIDATE_FILE_PATTERN)

    return {
        "train": {
            "text":   train_texts,
            "intent": train_intents,
        },
        "validate": {
            "text":   validate_texts,
            "intent": validate_intents,
        },
    }


def tokenize_tfidf(
    texts: list,
    vocab_size: int,
    fitted_vectorizer=None,
) -> tuple:
    """Convert a list of text strings to a dense TF-IDF float32 matrix.

    If fitted_vectorizer is None, a new TfidfVectorizer is fitted on texts
    and returned alongside the matrix. If a fitted_vectorizer is provided,
    it is used to transform texts without re-fitting (inference mode).

    The output matrix is always dense float32, which is the required input
    dtype for the ONNX classifier exported by export_to_onnx().

    Args:
        texts: List of raw utterance strings to vectorize.
        vocab_size: Maximum number of TF-IDF features (vocabulary size).
            CONFIGURABLE: increase if accuracy is limited by vocabulary size.
        fitted_vectorizer: An already-fitted TfidfVectorizer instance, or
            None to fit a new one on texts.

    Returns:
        A tuple (X, vectorizer) where:
            X:          np.ndarray of shape [n_samples, vocab_size], dtype float32.
            vectorizer: The fitted TfidfVectorizer instance.
    """
    if fitted_vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=vocab_size,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",
            sublinear_tf=True,
        )
        X = vectorizer.fit_transform(texts)
    else:
        vectorizer = fitted_vectorizer
        X = vectorizer.transform(texts)

    return X.toarray().astype(np.float32), vectorizer


def encode_labels(
    labels: list,
    fitted_encoder=None,
) -> tuple:
    """Encode a list of intent label strings to integer class indices.

    If fitted_encoder is None, a new LabelEncoder is fitted on labels and
    returned. If a fitted_encoder is provided, it is used to transform
    labels without re-fitting (inference mode).

    Args:
        labels: List of intent label strings, e.g. ['GetWeather', 'PlayMusic'].
        fitted_encoder: An already-fitted LabelEncoder instance, or None
            to fit a new one on labels.

    Returns:
        A tuple (y, encoder) where:
            y:       np.ndarray of shape [n_samples,], dtype int, class indices.
            encoder: The fitted LabelEncoder instance.
    """
    if fitted_encoder is None:
        encoder = LabelEncoder()
        y = encoder.fit_transform(labels)
    else:
        encoder = fitted_encoder
        y = encoder.transform(labels)

    return y.astype(np.int64), encoder


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    ratio: float,
    random_state: int,
) -> tuple:
    """Perform a stratified train/test split on feature matrix X and labels y.

    Stratification ensures each intent class appears in both splits in
    proportion to its frequency in the full dataset, which is important
    for the 7-class SNIPS dataset where class sizes are unequal.

    Args:
        X: Feature matrix of shape [n_samples, n_features], dtype float32.
        y: Label array of shape [n_samples,], dtype int.
        ratio: Fraction of samples to assign to the test split.
            CONFIGURABLE: set via TEST_SPLIT_RATIO in the notebook config cell.
        random_state: Random seed for reproducibility.
            CONFIGURABLE: set via RANDOM_STATE in the notebook config cell.

    Returns:
        A tuple (X_train, X_test, y_train, y_test) where each element is a
        np.ndarray with the appropriate shape.

    Raises:
        ValueError: Propagated from sklearn if the test split would be too
            small to contain at least one sample per class.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=ratio,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
