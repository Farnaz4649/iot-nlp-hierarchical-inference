"""tier1/train_tier1.py

Training, evaluation, and export functions for the Tier 1 model:
Logistic Regression trained on TF-IDF features.

Design note on ONNX export:
    The TfidfVectorizer is intentionally kept outside the ONNX graph.
    Embedding it inside the ONNX graph causes a locale initialisation error
    in onnxruntime on Linux when the StringNormalizer op is used. Instead,
    the vectorizer is saved separately via joblib and applied in Python before
    passing the float32 dense array to the ONNX session. This is also the
    correct approach for hardware portability: the vectorizer is loaded once
    and reused across many inference calls without re-instantiating it.

    Saved artefacts (all written by functions in this file):
        tier1_logreg.joblib       -- the fitted LogisticRegression classifier
        tier1_tfidf.joblib        -- the fitted TfidfVectorizer
        tier1_label_encoder.joblib -- the fitted LabelEncoder
        tier1_logreg.onnx         -- the classifier only (float32 input)
"""

import os
import time

import joblib
import numpy as np
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> LogisticRegression:
    """Fit a Logistic Regression classifier on TF-IDF features.

    Uses the lbfgs solver with multinomial cross-entropy loss, which is
    appropriate for the 7-class SNIPS intent classification task. The
    regularisation parameter C is set to 1.0 as a starting point; it
    should be tuned via cross-validation if accuracy is unsatisfactory.

    Args:
        X_train: Dense float32 matrix of shape [n_train, n_features],
            produced by preprocess.tokenize_tfidf().
        y_train: Integer label array of shape [n_train,],
            produced by preprocess.encode_labels().
        random_state: Random seed for reproducibility.
            CONFIGURABLE: set via RANDOM_STATE in the notebook config cell.

    Returns:
        A fitted sklearn LogisticRegression instance.
    """
    clf = LogisticRegression(
        C=1.0,             # CONFIGURABLE: regularisation strength
        max_iter=1000,
        solver="lbfgs",    # lbfgs handles multiclass natively in all sklearn versions
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    return clf


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list,
) -> dict:
    """Evaluate a fitted classifier on the test split.

    Computes accuracy, macro-averaged F1 score, and a full per-class
    classification report. The macro F1 score is the primary metric for
    comparison across tiers because it treats all intent classes equally
    regardless of their frequency in the test split.

    Args:
        model: A fitted LogisticRegression instance returned by
            train_logistic_regression().
        X_test: Dense float32 matrix of shape [n_test, n_features].
        y_test: Integer label array of shape [n_test,].
        label_names: List of human-readable intent name strings in the same
            order as the LabelEncoder's classes_ attribute. Used to make the
            classification report readable.

    Returns:
        A dict with keys:
            'accuracy':  float, overall accuracy on the test split.
            'f1_macro':  float, macro-averaged F1 score.
            'report':    str, full sklearn classification report.
            'y_pred':    np.ndarray of shape [n_test,], predicted class indices.
            'y_proba':   np.ndarray of shape [n_test, n_classes], class
                         probabilities for each test sample. Used by
                         shared/profile.py and the offloading router.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report   = classification_report(
        y_test, y_pred, target_names=label_names, zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1),
        "report":   report,
        "y_pred":   y_pred,
        "y_proba":  y_proba.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_to_onnx(
    model: LogisticRegression,
    n_features: int,
    path: str,
) -> None:
    """Export the fitted LogisticRegression classifier to ONNX format.

    Only the classifier is exported, not the TfidfVectorizer. The vectorizer
    must be applied in Python before passing data to the ONNX session. See
    the module docstring for the reasoning behind this design.

    The ONNX session returns two outputs:
        'output_label':       np.ndarray of shape [n_samples,], predicted
                              class indices (int64).
        'output_probability': list of dicts mapping class index to float
                              probability. Convert to np.ndarray using:
                              np.array([[d[i] for i in range(n_classes)]
                                        for d in output_probability],
                                       dtype=np.float32)

    Args:
        model: A fitted LogisticRegression instance.
        n_features: Number of TF-IDF features (vocabulary size). Must match
            the second dimension of the arrays passed to the ONNX session.
        path: File path where the .onnx file will be written.

    Returns:
        None
    """
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model   = convert_sklearn(model, initial_types=initial_type, target_opset=15)

    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def export_to_joblib(
    model: LogisticRegression,
    vectorizer,
    label_encoder,
    model_path: str,
    vectorizer_path: str,
    label_encoder_path: str,
) -> None:
    """Serialise the classifier, vectorizer, and label encoder to joblib files.

    All three artefacts are saved together because inference requires all
    three: the vectorizer transforms text to float32 arrays, the classifier
    predicts class indices, and the label encoder maps indices back to intent
    name strings.

    Args:
        model: A fitted LogisticRegression instance.
        vectorizer: The fitted TfidfVectorizer instance from
            preprocess.tokenize_tfidf().
        label_encoder: The fitted LabelEncoder instance from
            preprocess.encode_labels().
        model_path: File path for the classifier joblib file.
        vectorizer_path: File path for the vectorizer joblib file.
        label_encoder_path: File path for the label encoder joblib file.

    Returns:
        None
    """
    for path in [model_path, vectorizer_path, label_encoder_path]:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    joblib.dump(model,         model_path)
    joblib.dump(vectorizer,    vectorizer_path)
    joblib.dump(label_encoder, label_encoder_path)


# ---------------------------------------------------------------------------
# Inference helpers (used by profile.py and the offloading router)
# ---------------------------------------------------------------------------

def load_artefacts_joblib(
    model_path: str,
    vectorizer_path: str,
    label_encoder_path: str,
) -> tuple:
    """Load all three joblib artefacts from disk.

    This is the load_fn passed to profile.measure_load_time().

    Args:
        model_path: Path to the classifier joblib file.
        vectorizer_path: Path to the vectorizer joblib file.
        label_encoder_path: Path to the label encoder joblib file.

    Returns:
        A tuple (model, vectorizer, label_encoder) of fitted sklearn objects.
    """
    model         = joblib.load(model_path)
    vectorizer    = joblib.load(vectorizer_path)
    label_encoder = joblib.load(label_encoder_path)
    return model, vectorizer, label_encoder


def load_artefacts_onnx(
    onnx_path: str,
    vectorizer_path: str,
    label_encoder_path: str,
) -> tuple:
    """Load the ONNX session and supporting artefacts from disk.

    This is the load_fn passed to profile.measure_load_time() for the
    ONNX inference path.

    Args:
        onnx_path: Path to the .onnx file.
        vectorizer_path: Path to the vectorizer joblib file.
        label_encoder_path: Path to the label encoder joblib file.

    Returns:
        A tuple (session, vectorizer, label_encoder) where session is an
        onnxruntime.InferenceSession.
    """
    session       = rt.InferenceSession(onnx_path)
    vectorizer    = joblib.load(vectorizer_path)
    label_encoder = joblib.load(label_encoder_path)
    return session, vectorizer, label_encoder


def infer_joblib(
    model: LogisticRegression,
    vectorizer,
    X: np.ndarray,
) -> dict:
    """Run inference using the joblib-loaded sklearn classifier.

    This function is the inference_fn passed to profile.measure_latency()
    and profile.measure_peak_ram() for the joblib inference path.

    Args:
        model: A fitted LogisticRegression instance.
        vectorizer: The fitted TfidfVectorizer (already applied; X is the
            pre-vectorized dense float32 matrix).
        X: Dense float32 matrix of shape [n_samples, n_features].

    Returns:
        A dict with keys:
            'labels': np.ndarray of shape [n_samples,], predicted class indices.
            'proba':  np.ndarray of shape [n_samples, n_classes], class proba.
    """
    labels = model.predict(X)
    proba  = model.predict_proba(X).astype(np.float32)
    return {"labels": labels, "proba": proba}


def infer_onnx(
    session: rt.InferenceSession,
    X: np.ndarray,
) -> dict:
    """Run inference using the ONNX runtime session.

    This function is the inference_fn passed to profile.measure_latency()
    and profile.measure_peak_ram() for the ONNX inference path.

    The ONNX output_probability field is a list of dicts; this function
    converts it to a float32 numpy array for consistency with infer_joblib().

    Args:
        session: A loaded onnxruntime.InferenceSession.
        X: Dense float32 matrix of shape [n_samples, n_features]. Must be
            pre-vectorized using the same TfidfVectorizer used at export time.

    Returns:
        A dict with keys:
            'labels': np.ndarray of shape [n_samples,], predicted class indices.
            'proba':  np.ndarray of shape [n_samples, n_classes], class proba.
    """
    inp_name = session.get_inputs()[0].name
    outputs  = session.run(None, {inp_name: X})

    labels    = outputs[0]
    raw_proba = outputs[1]  # list of dicts: [{class_idx: prob, ...}, ...]
    n_classes = len(raw_proba[0])
    proba     = np.array(
        [[d[i] for i in range(n_classes)] for d in raw_proba],
        dtype=np.float32,
    )
    return {"labels": labels, "proba": proba}
