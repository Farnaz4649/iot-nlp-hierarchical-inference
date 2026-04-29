"""tier3/tier3_llm.py

Inference and evaluation functions for the Tier 3 model:
meta-llama/Llama-3.2-3B-Instruct used as a zero-shot or few-shot intent
classifier on the SNIPS NLU dataset.

Design notes
------------
Role in the project:
    Tier 3 is the cloud / data-centre tier. The model is not fine-tuned on
    SNIPS; it is prompted to choose one of the 7 SNIPS intents from a list.
    This makes Tier 3 a generalist fall-back for samples that Tier 1 and
    Tier 2 are not confident on, which is the configuration the Phase 2
    router relies on.

Inference interface contract:
    predict_batch() returns the same dict schema as Tier 1 and Tier 2:
        {'labels': np.ndarray of int64, 'proba': np.ndarray of float32}
    Labels are integer indices into preprocess.SNIPS_INTENTS so that the
    router compares predictions across tiers using the same label space.
    Unparseable model outputs map to label -1 and a uniform proba row.

Zero-shot vs few-shot:
    Both modes share the same chat-template structure. Zero-shot passes
    examples=[]; few-shot passes a list of (text, label) tuples that are
    rendered as alternating user/assistant turns before the final user
    turn. The mode is therefore controlled entirely by the contents of
    examples, not by branching prompt logic. The notebook config sets
    FEW_SHOT_K, and build_few_shot_pool produces the example list.

Pseudo-probabilities:
    A causal LLM does not emit a calibrated probability over the label
    set. We construct a pseudo-proba vector that is one-hot at the parsed
    label index and uniform (1/N) when the output cannot be parsed. This
    keeps the schema compatible with confidence-based routing in Phase 2,
    while making it explicit that confidence here is a parse-success
    signal rather than a calibrated probability.

LLaMA-3 chat template:
    We rely on tokenizer.apply_chat_template(..., add_generation_prompt=True)
    to render the LLaMA-3 native chat format. Manually constructing the
    template tokens is fragile across transformers versions; the tokenizer
    is the source of truth.

Hardware portability:
    load_llm() supports two paths:
        - bfloat16 with no quantization (default on A100, the project target)
        - 4-bit via BitsAndBytesConfig (fallback for consumer GPUs)
    The choice is controlled by the use_4bit flag, set from the notebook
    config cell. The function never reads any environment variable directly.

Saved artefacts:
    No artefacts are saved by this file. The HuggingFace cache holds the
    model weights; profiler results are written by the notebook profiling
    cell, not from inside any function here.

Package requirements (pin in the Tier 3 notebook installation cell):
    torch>=2.3.0
    transformers>=4.43.0   (required for LLaMA-3.2 chat template support)
    accelerate>=0.30.0
    bitsandbytes>=0.43.0   (only needed when use_4bit=True)
    scikit-learn>=1.4.0
    numpy>=1.24.0
"""

import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from preprocess import SNIPS_INTENTS


# ---------------------------------------------------------------------------
# Default prompt strings
# CONFIGURABLE: override task_description from the notebook to experiment
# with alternative phrasings. The default is intentionally short and direct,
# since LLaMA-3.2-3B follows brief instructions reliably.
# ---------------------------------------------------------------------------

DEFAULT_TASK_DESCRIPTION = (
    "You are an intent classifier for a smart-home and media voice assistant. "
    "Given a single user utterance, decide which intent label from the "
    "provided list best matches it. Reply with exactly one label from the "
    "list, copied verbatim, and nothing else."
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_llm(
    model_id: str,
    dtype: 'torch.dtype',
    use_4bit: bool,
) -> tuple:
    """Load a causal LLM and its tokenizer for Tier 3 inference.

    Loads the model with device_map='auto' so HuggingFace places it on the
    available GPU automatically. On A100 80 GB with the LLaMA-3.2-3B
    weights, bfloat16 fits comfortably and is the project default. The
    use_4bit path is provided for portability to smaller GPUs and uses
    BitsAndBytesConfig with NF4 quantization.

    The model is set to eval mode before being returned. The tokenizer's
    pad_token is set to its eos_token if it is missing, since LLaMA-3
    ships without an explicit pad token and HuggingFace generate() warns
    in that case.

    Args:
        model_id: HuggingFace model identifier, for example
            'meta-llama/Llama-3.2-3B-Instruct'.
            CONFIGURABLE: set via LLM_MODEL_ID in the notebook config cell.
        dtype: Torch dtype for model weights. Use torch.bfloat16 on A100/H100.
            Ignored when use_4bit is True (bitsandbytes manages its own dtype).
            CONFIGURABLE: set via LLM_DTYPE in the notebook config cell.
        use_4bit: If True, load the model in 4-bit NF4 quantization.
            Set False on A100 (default); set True on consumer GPUs (e.g. T4).
            CONFIGURABLE: set via USE_4BIT_QUANT in the notebook config cell.

    Returns:
        A tuple (model, tokenizer):
            model:     PreTrainedModel placed on device by accelerate, eval mode.
            tokenizer: PreTrainedTokenizer with pad_token set.

    Raises:
        OSError: Propagated from HuggingFace if the model is gated and the
            HF token has not been registered, or if the model id is wrong.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
        )

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Few-shot example pool construction
# ---------------------------------------------------------------------------

def build_few_shot_pool(
    train_texts: list,
    train_labels: list,
    label_set: list,
    k: int,
    random_state: int,
) -> list:
    """Sample k examples per class from the train split for few-shot prompting.

    For each label in label_set, collects all matching training utterances
    and samples k of them without replacement. The combined list is then
    shuffled so that the order of labels in the prompt does not bias the
    model toward the last example. When k is 0, returns an empty list,
    which yields zero-shot prompts downstream.

    Sampling uses a local random.Random instance seeded by random_state, so
    this function does not perturb the global RNG state.

    Args:
        train_texts: Training utterances (list of strings).
        train_labels: Intent labels aligned with train_texts (list of strings).
        label_set: Allowed intent labels in canonical order. Examples are
            sampled per label in this list.
        k: Examples per class. 0 returns []; positive integers return
            k * len(label_set) examples total. If a class has fewer than k
            samples, all of its samples are taken.
        random_state: Seed for reproducible sampling.

    Returns:
        A list of (text, label) tuples in shuffled order. Empty list when k=0.

    Raises:
        ValueError: If train_texts and train_labels have different lengths.
    """
    if len(train_texts) != len(train_labels):
        raise ValueError(
            f"train_texts ({len(train_texts)}) and train_labels "
            f"({len(train_labels)}) must have the same length."
        )

    if k <= 0:
        return []

    rng = random.Random(random_state)
    pool = []

    for label in label_set:
        candidates = [t for t, lab in zip(train_texts, train_labels) if lab == label]
        chosen     = rng.sample(candidates, min(k, len(candidates)))
        pool.extend((text, label) for text in chosen)

    rng.shuffle(pool)
    return pool


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_chat_messages(
    sample: str,
    task_description: str,
    label_set: list,
    examples: list,
) -> list:
    """Build the message list for tokenizer.apply_chat_template.

    The system message contains the task description and the bulleted
    label list. Each few-shot example becomes a user/assistant turn pair.
    The final user turn contains the utterance to be classified.

    Args:
        sample: The utterance to classify.
        task_description: One-paragraph framing of the task.
            Pass DEFAULT_TASK_DESCRIPTION for the project default.
        label_set: Allowed intent labels in the order they should appear
            in the system message (canonical SNIPS_INTENTS order).
        examples: List of (text, label) tuples to render as few-shot turns.
            Pass [] for zero-shot.

    Returns:
        A list of chat-template message dicts of the form
            [{'role': 'system', 'content': ...},
             {'role': 'user', 'content': example_text}, ...
             {'role': 'assistant', 'content': example_label}, ...
             {'role': 'user', 'content': sample}]
    """
    label_list_str = "\n".join(f"- {label}" for label in label_set)
    system_content = f"{task_description}\n\nAllowed labels:\n{label_list_str}"

    messages = [{"role": "system", "content": system_content}]

    for example_text, example_label in examples:
        messages.append({"role": "user", "content": example_text})
        messages.append({"role": "assistant", "content": example_label})

    messages.append({"role": "user", "content": sample})
    return messages


def build_prompt(
    sample: str,
    task_description: str,
    label_set: list,
    examples: list,
    tokenizer,
) -> str:
    """Render a chat-template prompt string for one utterance.

    Delegates message construction to build_chat_messages, then renders
    the messages through the tokenizer's chat template with
    add_generation_prompt=True so the LLM continues from the assistant
    role marker.

    Args:
        sample: The utterance to classify.
        task_description: One-paragraph framing of the task.
        label_set: Allowed intent labels in canonical order.
        examples: List of (text, label) tuples; [] for zero-shot.
        tokenizer: PreTrainedTokenizer with a chat template configured
            (LLaMA-3.2-Instruct ships with one).

    Returns:
        The rendered prompt as a string, ready to be passed to the
        tokenizer for ID conversion.
    """
    messages = build_chat_messages(sample, task_description, label_set, examples)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int,
) -> str:
    """Run a single greedy generation pass and return the generated suffix.

    Uses do_sample=False for deterministic outputs, which is appropriate
    for a classification task where we want the same input to produce the
    same label every time. Only the newly generated tokens are decoded;
    the prompt is sliced off using the input length, so the returned
    string does not contain the prompt or any chat-template control tokens.

    Args:
        prompt: Full prompt string from build_prompt.
        model: Loaded causal LM in eval mode.
        tokenizer: Matching PreTrainedTokenizer.
        max_new_tokens: Cap on generated tokens. The model only needs to
            emit a label string, so 16-20 tokens is typically enough.
            CONFIGURABLE: set via MAX_NEW_TOKENS in the notebook config cell.

    Returns:
        The decoded generated text, stripped, with special tokens removed.
    """
    inputs    = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[0, input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_output(raw_output: str, label_set: list) -> str:
    """Map the raw LLM output string to one of the allowed labels.

    The matching strategy is:
        1. Exact (case-insensitive) match against the full output.
        2. Case-insensitive substring search; if exactly one label appears
           in the output, return that label.
        3. Otherwise return 'unknown' so the caller can flag the parse
           as a failure (used by predict_batch to assign label index -1).

    A more aggressive fuzzy match (e.g. edit distance) is intentionally
    avoided here: silent corrections would mask prompt-quality problems
    that should surface in the error analysis for D2.

    Args:
        raw_output: Generated text from run_inference.
        label_set: Allowed intent labels (canonical SNIPS_INTENTS order).

    Returns:
        One of the strings in label_set, or 'unknown' if no clean match.
    """
    if raw_output is None:
        return "unknown"

    cleaned = raw_output.strip()
    cleaned_lower = cleaned.lower()

    # Pass 1: exact case-insensitive match.
    for label in label_set:
        if cleaned_lower == label.lower():
            return label

    # Pass 2: substring search; only accept when exactly one label appears.
    hits = [label for label in label_set if label.lower() in cleaned_lower]
    if len(hits) == 1:
        return hits[0]

    return "unknown"


# ---------------------------------------------------------------------------
# Batch prediction with router-compatible schema
# ---------------------------------------------------------------------------

def predict_batch(
    samples: list,
    model,
    tokenizer,
    label_set: list,
    task_description: str,
    examples: list,
    max_new_tokens: int,
) -> dict:
    """Run Tier 3 inference and return the standard router-compatible dict.

    For each sample, builds the prompt, runs greedy generation, parses the
    output, and produces a one-hot pseudo-probability row at the predicted
    label index. Unparseable outputs map to label index -1 and a uniform
    proba row of 1/N, where N is the size of label_set. This pseudo-proba
    convention is described in the module docstring; the Phase 2 router is
    aware that Tier 3 confidence is a parse-success signal, not a
    calibrated posterior.

    The returned 'labels' array uses the same int64 dtype and the same
    label space (indices into label_set, expected to be SNIPS_INTENTS) as
    Tier 1 and Tier 2, so the router can compare predictions across tiers
    without rescaling.

    Args:
        samples: Utterances to classify (list of strings).
        model: Loaded causal LM in eval mode.
        tokenizer: Matching PreTrainedTokenizer.
        label_set: Allowed intent labels in canonical order. The order
            defines column ordering of the proba array.
        task_description: One-paragraph task framing.
        examples: Few-shot example list; [] for zero-shot.
        max_new_tokens: Generation cap, forwarded to run_inference.

    Returns:
        A dict with:
            'labels':      np.ndarray, shape (N,), dtype int64. Indices into
                           label_set; -1 for unparseable outputs.
            'proba':       np.ndarray, shape (N, len(label_set)), dtype float32.
                           One-hot at the predicted index, or uniform 1/N row
                           when the prediction is -1.
            'raw_outputs': list[str] of length N, the generated suffixes.
                           Useful for error analysis in D2.
            'pred_strings': list[str] of length N, the parsed label strings
                           or 'unknown'. Provided alongside the int labels
                           because the parser is the natural debugging surface.
    """
    n_labels   = len(label_set)
    n_samples  = len(samples)
    label_to_i = {label: i for i, label in enumerate(label_set)}

    labels_int   = np.full(n_samples, -1, dtype=np.int64)
    proba        = np.full((n_samples, n_labels), 1.0 / n_labels, dtype=np.float32)
    raw_outputs  = []
    pred_strings = []

    for i, sample in enumerate(samples):
        prompt = build_prompt(sample, task_description, label_set, examples, tokenizer)
        raw    = run_inference(prompt, model, tokenizer, max_new_tokens)
        parsed = parse_output(raw, label_set)

        raw_outputs.append(raw)
        pred_strings.append(parsed)

        if parsed != "unknown":
            idx              = label_to_i[parsed]
            labels_int[i]    = idx
            proba[i]         = 0.0
            proba[i, idx]    = 1.0

    return {
        "labels":       labels_int,
        "proba":        proba,
        "raw_outputs":  raw_outputs,
        "pred_strings": pred_strings,
    }


# ---------------------------------------------------------------------------
# Evaluation wrapper
# ---------------------------------------------------------------------------

def evaluate_llm(
    samples: list,
    gold_labels: list,
    model,
    tokenizer,
    label_set: list,
    task_description: str,
    examples: list,
    max_new_tokens: int,
) -> dict:
    """Evaluate Tier 3 predictions against gold labels.

    Calls predict_batch and computes accuracy and macro F1 against the
    string-form gold labels. Unparseable predictions ('unknown') are
    counted as incorrect, since they do not match any gold label and
    cannot be silently dropped without inflating the score.

    Args:
        samples: Utterances to evaluate (list of strings).
        gold_labels: Ground-truth intent strings aligned with samples.
        model: Loaded causal LM in eval mode.
        tokenizer: Matching PreTrainedTokenizer.
        label_set: Allowed intent labels in canonical order.
        task_description: One-paragraph task framing.
        examples: Few-shot example list; [] for zero-shot.
        max_new_tokens: Generation cap, forwarded to predict_batch.

    Returns:
        A dict with:
            'accuracy':    float, top-1 accuracy.
            'f1_macro':    float, macro-averaged F1.
            'report':      str, sklearn classification_report.
            'predictions': np.ndarray of int64 label indices (with -1 for
                           unparseable).
            'pred_strings': list[str] of length len(samples) with parsed
                           label strings or 'unknown'. Used for D2 error
                           analysis.
            'raw_outputs': list[str], the raw generated suffixes.

    Raises:
        ValueError: If samples and gold_labels have different lengths.
    """
    if len(samples) != len(gold_labels):
        raise ValueError(
            f"samples ({len(samples)}) and gold_labels ({len(gold_labels)}) "
            "must have the same length."
        )

    result      = predict_batch(
        samples=samples,
        model=model,
        tokenizer=tokenizer,
        label_set=label_set,
        task_description=task_description,
        examples=examples,
        max_new_tokens=max_new_tokens,
    )
    pred_strings = result["pred_strings"]

    accuracy = accuracy_score(gold_labels, pred_strings)
    f1_macro = f1_score(
        gold_labels,
        pred_strings,
        average="macro",
        labels=label_set,
        zero_division=0,
    )
    report = classification_report(
        gold_labels,
        pred_strings,
        labels=label_set,
        zero_division=0,
    )

    return {
        "accuracy":     float(accuracy),
        "f1_macro":     float(f1_macro),
        "report":       report,
        "predictions":  result["labels"],
        "pred_strings": pred_strings,
        "raw_outputs":  result["raw_outputs"],
    }


# ---------------------------------------------------------------------------
# Profiler-compatible inference closure
# ---------------------------------------------------------------------------

def make_inference_fn(
    model,
    tokenizer,
    label_set: list,
    task_description: str,
    examples: list,
    max_new_tokens: int,
):
    """Return a single-argument inference closure for the profiler.

    shared/profiler.py calls inference_fn(inputs) where inputs is a list
    of utterance strings. This helper bakes the model, tokenizer, label
    set, prompt config, and few-shot examples into a closure so that
    measure_latency() and measure_inference_peak_ram() can be reused
    unchanged across all three tiers.

    Args:
        model: Loaded causal LM in eval mode.
        tokenizer: Matching PreTrainedTokenizer.
        label_set: Allowed intent labels in canonical order.
        task_description: One-paragraph task framing.
        examples: Few-shot example list; [] for zero-shot.
        max_new_tokens: Generation cap.

    Returns:
        A callable inference_fn(inputs: list[str]) -> dict that calls
        predict_batch and returns the standard dict schema.
    """

    def inference_fn(inputs: list) -> dict:
        return predict_batch(
            samples=inputs,
            model=model,
            tokenizer=tokenizer,
            label_set=label_set,
            task_description=task_description,
            examples=examples,
            max_new_tokens=max_new_tokens,
        )

    return inference_fn
