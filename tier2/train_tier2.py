"""tier2/train_tier2.py

Training, evaluation, and export functions for the Tier 2 model:
DistilBERT (distilbert/distilbert-base-uncased) fine-tuned for 7-class intent
classification on the SNIPS NLU dataset.

Design notes
------------
Tokenizer vs. vectorizer:
    Unlike Tier 1, there is no separate TF-IDF vectorizer to manage. The
    HuggingFace AutoTokenizer handles all text preprocessing. At ONNX export
    time, optimum embeds the tokenizer configuration in the ONNX graph, so a
    single directory contains everything needed for inference.

Inference interface contract:
    infer_hf() and infer_onnx() return the same dict schema as
    train_tier1.infer_joblib() and train_tier1.infer_onnx():
        {'labels': np.ndarray, 'proba': np.ndarray}
    This allows shared/profiler.run_full_profile() to call them without any
    modification.

Lazy imports:
    optimum and onnxruntime are imported lazily inside the functions that need
    them (export_to_onnx, load_onnx_session, infer_onnx). This ensures that
    training and evaluation work on machines where those packages are not yet
    installed (e.g. before the ONNX export step).

Saved artefacts (all written by functions in this file):
    MODELS_DIR/tier2_distilbert/          HuggingFace native format (weights +
                                           tokenizer + config)
    MODELS_DIR/tier2_distilbert_onnx/     ONNX export produced by optimum;
                                           contains model.onnx and tokenizer
                                           files

Package requirements (pin in the Tier 2 notebook installation cell):
    torch>=2.0.0
    transformers>=4.40.0
    datasets>=2.19.0
    optimum[onnxruntime-gpu]>=1.19.0   (or optimum[onnxruntime] on CPU)
    onnxruntime-gpu>=1.17.0            (or onnxruntime on CPU)
    scikit-learn>=1.4.0
    numpy>=1.24.0
"""

import os

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# optimum and onnxruntime are imported lazily inside export_to_onnx(),
# load_onnx_session(), and infer_onnx() to avoid import crashes during
# training on machines where those packages are not installed.


# ---------------------------------------------------------------------------
# Model and tokenizer loading
# ---------------------------------------------------------------------------

def load_pretrained_model(
    model_name: str,
    num_labels: int,
    label2id: dict,
    id2label: dict,
) -> tuple:
    """Load a pre-trained DistilBERT tokenizer and sequence-classification model.

    Attaches a randomly initialised classification head with num_labels output
    neurons. The label2id and id2label dicts are stored in the model config so
    that HuggingFace's Trainer and ONNX export tools can resolve class names
    without referring to an external LabelEncoder.

    Args:
        model_name: HuggingFace model identifier string.
            CONFIGURABLE: set via TIER2_MODEL_NAME in the notebook config cell.
            Confirmed value: 'distilbert/distilbert-base-uncased'.
        num_labels: Number of intent classes. For SNIPS this is 7.
        label2id: Dict mapping intent name strings to integer class indices,
            e.g. {'AddToPlaylist': 0, 'BookRestaurant': 1, ...}.
            Derive with:
                {name: int(idx) for idx, name in enumerate(label_encoder.classes_)}
        id2label: Inverse of label2id, mapping int -> intent name string,
            e.g. {0: 'AddToPlaylist', 1: 'BookRestaurant', ...}.
            Derive with:
                {int(idx): name for idx, name in enumerate(label_encoder.classes_)}

    Returns:
        A tuple (model, tokenizer) where:
            model:     AutoModelForSequenceClassification instance with a
                       freshly initialised classification head. Not yet trained.
            tokenizer: AutoTokenizer matched to model_name, with fast tokenizer
                       enabled for maximum throughput.

    Raises:
        OSError: If model_name is not found on the HuggingFace Hub and is not
            a valid local path. Check TIER2_MODEL_NAME in the config cell.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,  # classification head is new; suppress warning
    )

    return model, tokenizer


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def tokenize_dataset(
    texts: list,
    labels: list,
    tokenizer,
    max_len: int,
) -> Dataset:
    """Tokenize a list of utterance strings and wrap them in a HuggingFace Dataset.

    Applies the DistilBERT tokenizer with padding and truncation, then attaches
    integer labels as a column named 'labels' (the exact column name that
    HuggingFace Trainer expects). Sets the tensor format to PyTorch so that the
    returned Dataset can be passed directly to Trainer without any further
    conversion.

    Args:
        texts: List of raw utterance strings, e.g. from
            preprocess.load_dataset()['train']['text'].
        labels: List of integer class indices of the same length as texts.
            Produced by preprocess.encode_labels() as a np.ndarray; convert
            with labels.tolist() before passing here if necessary.
        tokenizer: A fitted AutoTokenizer instance from load_pretrained_model().
        max_len: Maximum token sequence length (padding and truncation target).
            CONFIGURABLE: set via MAX_SEQ_LEN in the notebook config cell.
            Recommended value: 64. SNIPS utterances average ~9 words; 64 tokens
            covers the entire validation set with zero truncation. Using 128
            doubles the memory cost per batch with no benefit on this dataset.

    Returns:
        A HuggingFace Dataset with three columns, tensor format 'pt':
            'input_ids':      LongTensor of shape [max_len].
            'attention_mask': LongTensor of shape [max_len].
            'labels':         LongTensor scalar, integer class index.
        The Dataset is ready to be passed directly to Trainer as train_dataset
        or eval_dataset.

    Raises:
        ValueError: If len(texts) != len(labels).
    """
    if len(texts) != len(labels):
        raise ValueError(
            f"texts and labels must have the same length; "
            f"got {len(texts)} texts and {len(labels)} labels."
        )

    # Convert labels to plain Python ints; HuggingFace Dataset requires that
    # the list contain only Python scalars, not numpy int64 values.
    labels_list = [int(lbl) for lbl in labels]

    encodings = tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors=None,  # return plain lists; Dataset handles tensor conversion
    )

    raw = {
        "input_ids":      encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels":         labels_list,
    }

    dataset = Dataset.from_dict(raw)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return dataset


# ---------------------------------------------------------------------------
# Trainer configuration
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred: tuple) -> dict:
    """Compute accuracy and macro F1 from Trainer's EvalPrediction object.

    This function is passed to Trainer as the compute_metrics callback.
    Trainer calls it after each evaluation epoch, supplying logits and
    ground-truth label indices. The function argmaxes the logits to get
    predicted class indices and then computes the two metrics.

    Args:
        eval_pred: A named tuple (or equivalent) with two fields:
            predictions: np.ndarray of shape [n_eval_samples, num_labels],
                         raw logits (not softmax probabilities).
            label_ids:   np.ndarray of shape [n_eval_samples,], integer
                         ground-truth class indices.

    Returns:
        A dict with keys:
            'accuracy': float, overall accuracy on the evaluation set.
            'f1_macro': float, macro-averaged F1 score. Used as the primary
                        metric for early stopping (metric_for_best_model in
                        TrainingArguments).
    """
    logits, label_ids = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = float(accuracy_score(label_ids, predictions))
    f1       = float(f1_score(label_ids, predictions, average="macro", zero_division=0))

    return {"accuracy": accuracy, "f1_macro": f1}


def build_training_args(
    output_dir: str,
    epochs: int,
    lr: float,
    train_batch_size: int,
    eval_batch_size: int,
    warmup_ratio: float,
    weight_decay: float,
    fp16: bool,
) -> TrainingArguments:
    """Construct a HuggingFace TrainingArguments object from explicit parameters.

    All hyperparameters are passed as function arguments rather than hardcoded
    so that the notebook config cell remains the single source of truth and
    ablation runs require only config-cell changes.

    Early stopping is handled by an EarlyStoppingCallback added inside
    fine_tune(), not here. load_best_model_at_end is set to True so that
    Trainer automatically restores the best checkpoint at the end of training,
    regardless of whether early stopping fired.

    The evaluation strategy is set to 'epoch' to align evaluation and save
    events. Saving more frequently than evaluation would cause Trainer to
    raise a ValueError.

    Args:
        output_dir: Directory where Trainer writes checkpoints and evaluation
            results. CONFIGURABLE: set to str(MODELS_DIR / 'tier2_checkpoints')
            in the notebook config cell.
        epochs: Maximum number of full training passes over the dataset.
            CONFIGURABLE: set via TIER2_EPOCHS in the config cell.
            Recommended starting value: 5. Early stopping will terminate
            training before this limit if the evaluation F1 plateaus.
        lr: Peak learning rate for the AdamW optimizer.
            CONFIGURABLE: set via TIER2_LR. Recommended: 2e-5 for DistilBERT
            fine-tuning on SNIPS. Values above 5e-5 typically cause instability.
        train_batch_size: Per-device batch size during training.
            CONFIGURABLE: set via TIER2_TRAIN_BATCH. Recommended: 32 on T4.
            Reduce to 16 if CUDA out-of-memory errors occur.
        eval_batch_size: Per-device batch size during evaluation.
            CONFIGURABLE: set via TIER2_EVAL_BATCH. Recommended: 64.
            Evaluation does not compute gradients so a larger batch is safe.
        warmup_ratio: Fraction of total training steps used for linear
            learning-rate warm-up before the cosine/linear decay begins.
            CONFIGURABLE: recommended 0.1 (10% of steps).
        weight_decay: L2 regularisation coefficient applied to all non-bias
            and non-LayerNorm parameters.
            CONFIGURABLE: recommended 0.01.
        fp16: Whether to enable mixed-precision (FP16) training.
            CONFIGURABLE: set True on T4 GPU, False on CPU-only Colab.
            Has no effect if CUDA is unavailable; Trainer ignores it safely.

    Returns:
        A TrainingArguments instance ready to pass to fine_tune().
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        fp16=fp16,
        eval_strategy="epoch",        # evaluate at the end of every epoch
        save_strategy="epoch",              # must match evaluation_strategy
        load_best_model_at_end=True,        # restore best checkpoint automatically
        metric_for_best_model="f1_macro",   # matches key returned by compute_metrics
        greater_is_better=True,
        logging_strategy="epoch",           # one log line per epoch
        report_to="none",                   # disable wandb/tensorboard
        save_total_limit=2,                 # keep only the 2 most recent checkpoints
        dataloader_num_workers=0,           # 0 is safest on Colab; avoids fork issues
        seed=42,                            # CONFIGURABLE: set via RANDOM_STATE
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def fine_tune(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    training_args: TrainingArguments,
    early_stopping_patience: int,
) -> tuple:
    """Fine-tune the DistilBERT model using HuggingFace Trainer.

    Constructs a Trainer with the provided datasets and training arguments,
    attaches an EarlyStoppingCallback, runs training to completion (or until
    early stopping fires), and returns the best model state. Because
    load_best_model_at_end is True in training_args, the returned model
    already holds the weights from the best evaluation checkpoint.

    Args:
        model: An AutoModelForSequenceClassification instance from
            load_pretrained_model(). Will be mutated in place during training.
        tokenizer: The matching AutoTokenizer from load_pretrained_model().
            Passed to Trainer for correct data collation (padding).
        train_dataset: HuggingFace Dataset from tokenize_dataset() for the
            training split.
        eval_dataset: HuggingFace Dataset from tokenize_dataset() for the
            validation/test split (used as the evaluation set during training).
        training_args: A TrainingArguments instance from build_training_args().
        early_stopping_patience: Number of evaluation epochs with no improvement
            in eval F1 before training stops early.
            CONFIGURABLE: set via EARLY_STOPPING_PATIENCE in the config cell.
            Recommended default: 2.

    Returns:
        A tuple (model, trainer) where:
            model:   The fine-tuned model holding the best checkpoint weights,
                     ready to pass to evaluate_model() or save_hf_model().
            trainer: The fitted Trainer instance. Access training logs with
                     trainer.state.log_history.
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=0.0,   # any improvement counts
            )
        ],
    )

    trainer.train()

    # trainer.model holds the best checkpoint because load_best_model_at_end=True
    return trainer.model, trainer


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    tokenizer,
    eval_dataset: Dataset,
    label_names: list,
    batch_size: int,
    device: str,
) -> dict:
    """Run inference over the evaluation set and compute classification metrics.

    Performs a manual batched forward pass rather than using Trainer.predict(),
    so that the same function can be called after ONNX export to verify the
    exported model produces numerically consistent predictions. Returns the
    same dict schema as train_tier1.evaluate_model() for consistency across
    tiers.

    The model is set to eval mode and all gradient computation is disabled
    during inference. The model is moved to the specified device if it is not
    already there.

    Args:
        model: A fine-tuned AutoModelForSequenceClassification, either on CPU
            or GPU. The function moves it to device if necessary.
        tokenizer: The matching AutoTokenizer. Not used during the forward pass
            itself; included for interface symmetry with load_hf_model().
        eval_dataset: HuggingFace Dataset with columns 'input_ids',
            'attention_mask', and 'labels', in torch tensor format.
            Produced by tokenize_dataset().
        label_names: List of intent name strings in class-index order.
            Obtain with label_encoder.classes_.tolist() from
            preprocess.encode_labels().
        batch_size: Number of samples per forward pass.
            CONFIGURABLE: set via TIER2_EVAL_BATCH in the config cell.
        device: PyTorch device string, e.g. 'cuda' or 'cpu'.
            CONFIGURABLE: auto-detected in the notebook config cell as:
                'cuda' if torch.cuda.is_available() else 'cpu'

    Returns:
        A dict with keys:
            'accuracy':  float, overall accuracy on the evaluation set.
            'f1_macro':  float, macro-averaged F1 score.
            'report':    str, full sklearn classification report with per-class
                         precision, recall, and F1.
            'y_pred':    np.ndarray of shape [n_eval_samples,], predicted class
                         indices (int64).
            'y_proba':   np.ndarray of shape [n_eval_samples, num_labels],
                         softmax probabilities (float32). Used by
                         shared/profiler.py and the Phase 2 offloading router
                         to compute confidence scores.
    """
    model = model.to(device)
    model.eval()

    all_logits = []
    all_labels = []

    n_samples = len(eval_dataset)
    for start in range(0, n_samples, batch_size):
        end   = min(start + batch_size, n_samples)
        batch = eval_dataset[start:end]

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"]         # keep on CPU for collection

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        all_logits.append(outputs.logits.cpu().numpy())
        all_labels.append(
            labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
        )

    logits_np = np.concatenate(all_logits, axis=0)   # [n_samples, num_labels]
    labels_np = np.concatenate(all_labels, axis=0)   # [n_samples,]

    # Softmax over logits for probability output
    exp_logits = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
    proba      = (exp_logits / exp_logits.sum(axis=1, keepdims=True)).astype(np.float32)

    y_pred = logits_np.argmax(axis=-1).astype(np.int64)

    accuracy = float(accuracy_score(labels_np, y_pred))
    f1       = float(f1_score(labels_np, y_pred, average="macro", zero_division=0))
    report   = classification_report(
        labels_np, y_pred, target_names=label_names, zero_division=0
    )

    return {
        "accuracy": accuracy,
        "f1_macro": f1,
        "report":   report,
        "y_pred":   y_pred,
        "y_proba":  proba,
    }


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_hf_model(
    model,
    tokenizer,
    output_dir: str,
) -> None:
    """Save the fine-tuned model and tokenizer in HuggingFace native format.

    Writes model weights, configuration, and tokenizer vocabulary to
    output_dir. The saved directory can be reloaded on any machine with:
        AutoModelForSequenceClassification.from_pretrained(output_dir)
        AutoTokenizer.from_pretrained(output_dir)

    This native format is the primary artefact used during Tier 2 profiling.
    It is also the input required by export_to_onnx().

    Args:
        model: The fine-tuned AutoModelForSequenceClassification returned by
            fine_tune(). Should be on CPU before saving to avoid any device
            mismatch on reload; the function moves it to CPU automatically.
        tokenizer: The matching AutoTokenizer from load_pretrained_model().
        output_dir: Directory path where the model folder will be written.
            CONFIGURABLE: set to str(MODELS_DIR / 'tier2_distilbert') in the
            notebook config cell. Created automatically if it does not exist.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    model.cpu().save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_to_onnx(
    hf_model_dir: str,
    onnx_output_dir: str,
    device: str,
) -> None:
    """Export the fine-tuned DistilBERT model to ONNX using optimum.

    Uses optimum.exporters.onnx.main_export() to produce a self-contained
    ONNX export directory. Unlike Tier 1, there is no separate vectorizer to
    manage: optimum embeds the tokenizer configuration alongside the ONNX graph
    in the output directory.

    The resulting model.onnx can be loaded by onnxruntime with
    CUDAExecutionProvider on T4 or CPUExecutionProvider on any device.

    Note on lazy import: optimum is imported inside this function to avoid
    import crashes on machines where it is not installed (e.g. the Raspberry
    Pi, which never runs Tier 2 ONNX export).

    Args:
        hf_model_dir: Path to the directory written by save_hf_model().
            Must contain config.json, pytorch_model.bin (or model.safetensors),
            and tokenizer files.
        onnx_output_dir: Directory where optimum writes the ONNX artefacts,
            including model.onnx and tokenizer files.
            CONFIGURABLE: set to str(MODELS_DIR / 'tier2_distilbert_onnx') in
            the notebook config cell.
        device: 'cpu' or 'cuda'. Passed to the optimum exporter to select
            the appropriate operator set.
            CONFIGURABLE: auto-detected in the notebook config cell.

    Returns:
        None

    Raises:
        ImportError: If optimum is not installed in the current environment.
            Install with: pip install optimum[onnxruntime-gpu]
        OSError: If hf_model_dir does not exist or is missing required files.
    """
    from optimum.exporters.onnx import main_export  # lazy import

    os.makedirs(onnx_output_dir, exist_ok=True)

    main_export(
        model_name_or_path=hf_model_dir,
        output=onnx_output_dir,
        task="text-classification",
        device=device,
        opset=15,                  # matches Tier 1 ONNX opset for consistency
    )


# ---------------------------------------------------------------------------
# Loading (load_fn interface for profiler.py)
# ---------------------------------------------------------------------------

def load_hf_model(
    model_dir: str,
    device: str,
) -> tuple:
    """Load a saved HuggingFace DistilBERT model and tokenizer from disk.

    This function is designed to be passed as the load_fn argument to
    profiler.measure_load_time() and profiler.measure_model_ram_footprint()
    for the HuggingFace inference path. Wrap it in a lambda to fix the
    arguments:
        load_fn = lambda: load_hf_model(str(MODELS_DIR / 'tier2_distilbert'), device)

    Args:
        model_dir: Path to the directory written by save_hf_model().
        device: PyTorch device string, e.g. 'cuda' or 'cpu'.
            CONFIGURABLE: auto-detected in the notebook config cell.

    Returns:
        A tuple (model, tokenizer) where:
            model:     AutoModelForSequenceClassification loaded from model_dir,
                       moved to device, set to eval mode.
            tokenizer: AutoTokenizer loaded from model_dir.

    Raises:
        OSError: If model_dir does not exist or is missing required files.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model     = model.to(device)
    model.eval()
    return model, tokenizer


def load_onnx_session(
    onnx_dir: str,
    use_gpu: bool,
) -> object:
    """Load an ONNX inference session for the Tier 2 DistilBERT model.

    Looks for a file named 'model.onnx' inside onnx_dir, which is the
    standard filename written by optimum. Requests CUDAExecutionProvider when
    use_gpu is True; onnxruntime falls back to CPUExecutionProvider
    automatically if CUDA is unavailable.

    This function is designed to be passed as the load_fn argument to
    profiler.measure_load_time() and profiler.measure_model_ram_footprint()
    for the ONNX inference path:
        load_fn = lambda: load_onnx_session(str(MODELS_DIR / 'tier2_distilbert_onnx'), use_gpu=True)

    Note on lazy import: onnxruntime is imported inside this function to avoid
    import crashes on machines where it is not installed.

    Args:
        onnx_dir: Path to the directory written by export_to_onnx(). Must
            contain a file named 'model.onnx'.
        use_gpu: If True, requests CUDAExecutionProvider as the first provider.
            CONFIGURABLE: set via USE_GPU in the notebook config cell as:
                USE_GPU = torch.cuda.is_available()

    Returns:
        An onnxruntime.InferenceSession ready to use with infer_onnx().

    Raises:
        FileNotFoundError: If model.onnx is not found inside onnx_dir.
        ImportError: If onnxruntime is not installed.
    """
    import onnxruntime as ort  # lazy import

    onnx_path = os.path.join(onnx_dir, "model.onnx")
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(
            f"model.onnx not found in {onnx_dir}. "
            "Run export_to_onnx() first."
        )

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_gpu
        else ["CPUExecutionProvider"]
    )

    session = ort.InferenceSession(onnx_path, providers=providers)
    return session


# ---------------------------------------------------------------------------
# Inference helpers (inference_fn interface for profiler.py)
# ---------------------------------------------------------------------------

def infer_hf(
    model,
    tokenizer,
    texts: list,
    device: str,
    max_len: int,
) -> dict:
    """Run inference on a list of raw text strings using the HuggingFace model.

    Tokenizes the input texts, runs a single forward pass through the fine-tuned
    DistilBERT model, and returns predicted class indices and softmax
    probabilities. This function is the inference_fn passed to
    profiler.measure_latency() and profiler.measure_inference_peak_ram() for
    the HuggingFace inference path.

    Usage with the profiler:
        inference_fn = lambda inputs: infer_hf(model, tokenizer, inputs, device, MAX_SEQ_LEN)
        latency_stats = profiler.measure_latency(inference_fn, test_texts[:1], N_LATENCY_RUNS)

    The model must already be in eval mode (set by load_hf_model()). For
    latency profiling, pass a single-item list representing one query at a
    time to measure per-query latency rather than batch throughput.

    Args:
        model: A fine-tuned AutoModelForSequenceClassification on device,
            in eval mode.
        tokenizer: The matching AutoTokenizer.
        texts: List of raw utterance strings. Pass a single-item list for
            latency profiling; pass the full eval list for batch evaluation.
        device: PyTorch device string, e.g. 'cuda' or 'cpu'.
        max_len: Maximum token sequence length. Must match the value used
            in tokenize_dataset() during training to avoid distribution shift
            at inference time.
            CONFIGURABLE: set via MAX_SEQ_LEN in the notebook config cell.

    Returns:
        A dict with keys:
            'labels': np.ndarray of shape [n_texts,] and dtype int64,
                      predicted class indices.
            'proba':  np.ndarray of shape [n_texts, num_labels] and dtype
                      float32, softmax class probabilities. The max value
                      across the num_labels axis is the confidence score used
                      by the Phase 2 offloading router.
    """
    encoding = tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits_np = outputs.logits.cpu().numpy()

    # Numerically stable softmax
    exp_logits = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
    proba      = (exp_logits / exp_logits.sum(axis=1, keepdims=True)).astype(np.float32)

    labels = logits_np.argmax(axis=-1).astype(np.int64)

    return {"labels": labels, "proba": proba}


def infer_onnx(
    session,
    tokenizer,
    texts: list,
    max_len: int,
) -> dict:
    """Run inference on a list of raw text strings using the ONNX session.

    Tokenizes with the HuggingFace tokenizer (Python-side tokenization is
    still required because onnxruntime does not execute the Python tokenizer
    even when optimum embeds the tokenizer config in the ONNX directory),
    feeds the numpy arrays to the ONNX session, and returns predictions and
    softmax probabilities. The output schema is identical to infer_hf() so
    that the profiler and the Phase 2 router can call either interchangeably.

    Usage with the profiler:
        inference_fn = lambda inputs: infer_onnx(session, tokenizer, inputs, MAX_SEQ_LEN)
        latency_stats = profiler.measure_latency(inference_fn, test_texts[:1], N_LATENCY_RUNS)

    Note on lazy import: onnxruntime is imported inside load_onnx_session();
    this function receives the already-loaded session object and does not need
    to import onnxruntime directly.

    Args:
        session: An onnxruntime.InferenceSession from load_onnx_session().
        tokenizer: The matching AutoTokenizer. Load it from the ONNX export
            directory with:
                AutoTokenizer.from_pretrained(str(MODELS_DIR / 'tier2_distilbert_onnx'))
        texts: List of raw utterance strings. Pass a single-item list for
            latency profiling.
        max_len: Maximum token sequence length. Must match the value used
            in tokenize_dataset() during training.

    Returns:
        A dict with keys:
            'labels': np.ndarray of shape [n_texts,] and dtype int64,
                      predicted class indices.
            'proba':  np.ndarray of shape [n_texts, num_labels] and dtype
                      float32, softmax class probabilities.
    """
    encoding = tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="np",  # onnxruntime expects numpy arrays, not torch tensors
    )

    # Retrieve the session input names dynamically; do not hardcode 'input_ids'
    # because optimum may produce sessions with slightly different input names
    # depending on the model architecture and export version.
    input_names = [inp.name for inp in session.get_inputs()]

    feed = {}
    for name in input_names:
        if name in encoding:
            feed[name] = encoding[name].astype(np.int64)

    outputs = session.run(None, feed)

    # optimum text-classification ONNX output: index 0 is logits [n, num_labels]
    logits_np = outputs[0].astype(np.float32)

    # Numerically stable softmax
    exp_logits = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
    proba      = (exp_logits / exp_logits.sum(axis=1, keepdims=True)).astype(np.float32)

    labels = logits_np.argmax(axis=-1).astype(np.int64)

    return {"labels": labels, "proba": proba}
