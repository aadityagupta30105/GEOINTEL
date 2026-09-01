"""
Event Classification Model
==========================
Fine-tunes and serves a DistilBERT sequence classifier over geopolitical event
text.

Categories
----------
0 Military/Conflict
1 Trade/Economic
2 Diplomatic
3 Humanitarian/Aid
4 Political

Import contract
---------------
This module imports without PyTorch or Transformers present. Heavy
dependencies are resolved on first use so that the dashboard, the test suite
and the graph-only pipeline paths remain importable on machines without a
deep-learning stack. When those dependencies are unavailable, inference falls
back to a deterministic keyword classifier.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TypedDict

import numpy as np
import pandas as pd

from utils.logging_config import ERROR, OK, WARN, get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

__all__ = [
    "LABEL_NAMES",
    "NUM_LABELS",
    "Prediction",
    "TrainingConfig",
    "create_synthetic_training_data",
    "GeopoliticalEventClassifier",
    "build_training_dataset_from_gdelt",
]

_log = get_logger(__name__)

LABEL_NAMES: Final[tuple[str, ...]] = (
    "Military/Conflict",
    "Trade/Economic",
    "Diplomatic",
    "Humanitarian/Aid",
    "Political",
)
NUM_LABELS: Final[int] = len(LABEL_NAMES)

# Label assigned when the keyword fallback finds no evidence for any class.
_DEFAULT_FALLBACK_LABEL: Final[int] = 4  # Political
_INFERENCE_BATCH_SIZE: Final[int] = 32


class Prediction(TypedDict):
    """Single-text classification result.

    Attributes
    ----------
    label_id : int
        Index into :data:`LABEL_NAMES`.
    label_name : str
        Human-readable category.
    confidence : float
        Probability assigned to ``label_id`` in ``[0, 1]``.
    all_scores : dict
        Full probability distribution keyed by category name.
    method : str
        ``"transformer"`` or ``"rule_based"``.
    """

    label_id: int
    label_name: str
    confidence: float
    all_scores: dict[str, float]
    method: str


@dataclass(slots=True)
class TrainingConfig:
    """Hyper-parameters and paths for the classification model.

    Attributes
    ----------
    model_name : str
        HuggingFace checkpoint used when no local model exists.
    max_length : int
        Token truncation length.
    batch_size : int
        Per-device batch size for training and evaluation.
    num_epochs : int
        Training epochs.
    learning_rate : float
        Peak learning rate.
    warmup_ratio : float
        Fraction of steps spent warming up the schedule.
    weight_decay : float
        Decoupled weight decay.
    save_path : str
        Directory holding the fine-tuned artefacts.
    eval_split : float
        Fraction of the training frame held out when no eval set is supplied.
    early_stopping_patience : int
        Evaluations without improvement before training stops.
    """

    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    batch_size: int = 16
    num_epochs: int = 3
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    save_path: str = "models/artifacts/event_classifier"
    eval_split: float = 0.15
    early_stopping_patience: int = 2


@lru_cache(maxsize=1)
def resolve_device() -> "torch.device":
    """Resolve and memoise the inference device.

    Resolution is deferred to first use so that importing this module does not
    initialise CUDA or emit console output.

    Returns
    -------
    torch.device
        ``cuda`` when available, otherwise ``cpu``.

    Raises
    ------
    ImportError
        When PyTorch is not installed.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log.info("%s Inference device resolved: %s", OK, device)
    return device


# --- Synthetic training corpus ---------------------------------------------

_TEMPLATES: Final[dict[int, tuple[str, ...]]] = {
    0: (  # Military/Conflict
        "{c1} launched military strikes against {c2} forces near the border",
        "{c1} troops clashed with {c2} soldiers in disputed territory",
        "{c1} deployed naval vessels in response to {c2} provocations",
        "Armed conflict escalates between {c1} and {c2} in the northern region",
        "{c1} conducted airstrikes targeting {c2} infrastructure",
        "{c1} imposed a naval blockade on {c2} shipping lanes",
        "Casualties reported as {c1} and {c2} exchange fire along the frontier",
    ),
    1: (  # Trade/Economic
        "{c1} and {c2} signed a comprehensive free trade agreement",
        "{c1} imposed tariffs on imports from {c2} citing dumping concerns",
        "{c2} became the largest trading partner of {c1} for the third year",
        "Investment flows between {c1} and {c2} reached a record high",
        "{c1} announced sanctions targeting the energy sector of {c2}",
        "Trade negotiations between {c1} and {c2} stalled over agriculture",
        "{c1} and {c2} agreed on joint infrastructure development projects",
    ),
    2: (  # Diplomatic
        "The {c1} ambassador was summoned to the {c2} foreign ministry",
        "{c1} and {c2} leaders held a bilateral summit in Geneva",
        "{c1} recognized the territorial claims of {c2} in the disputed region",
        "Diplomatic ties between {c1} and {c2} were fully restored",
        "{c1} recalled its ambassador to {c2} amid the crisis",
        "Foreign ministers of {c1} and {c2} signed a cooperation treaty",
        "{c1} formally protested the {c2} decision at the UN Security Council",
    ),
    3: (  # Humanitarian/Aid
        "{c1} pledged emergency humanitarian aid to disaster-stricken {c2}",
        "Relief convoys from {c1} crossed into {c2} to assist refugees",
        "{c1} and {c2} launched a joint vaccination campaign",
        "{c1} donated medical supplies and food aid to {c2}",
        "Search and rescue teams from {c1} arrived to support {c2}",
        "{c1} opened its borders to refugees fleeing violence in {c2}",
        "Development assistance from {c1} will build infrastructure in {c2}",
    ),
    4: (  # Political
        "{c1} condemned the recent {c2} election results as fraudulent",
        "{c1} expressed concern over democratic backsliding in {c2}",
        "Political pressure from {c1} forced {c2} to reconsider its stance",
        "{c1} supported the {c2} bid for UN Security Council membership",
        "{c1} and {c2} coordinated positions ahead of the G20 summit",
        "Opposition leaders from {c2} sought political asylum in {c1}",
        "{c1} and {c2} jointly proposed a resolution at the UN General Assembly",
    ),
}

_TEMPLATE_COUNTRIES: Final[tuple[str, ...]] = (
    "USA", "China", "Russia", "Germany", "UK", "France", "India",
    "Brazil", "Japan", "South Korea", "Iran", "Saudi Arabia",
    "Turkey", "Pakistan", "Israel", "Australia", "Canada",
)

# Keyword evidence used by the deterministic fallback classifier.
_KEYWORDS: Final[dict[int, tuple[str, ...]]] = {
    0: ("attack", "military", "strike", "war", "conflict", "troops", "weapon",
        "bomb", "missile", "blockade", "invasion", "casualties", "fight", "assault"),
    1: ("trade", "tariff", "import", "export", "economic", "investment",
        "gdp", "market", "commerce", "deal", "agreement", "material cooperation"),
    2: ("diplomat", "summit", "ambassador", "treaty", "foreign minister",
        "negotiate", "recognize", "bilateral", "relations", "consult"),
    3: ("humanitarian", "aid", "refugee", "disaster", "relief", "food",
        "medical", "asylum", "donate", "provide aid"),
    4: ("election", "democracy", "political", "resolution", "vote",
        "government", "protest", "sanction", "opposition", "statement", "demand"),
}


def create_synthetic_training_data(n: int = 2000, seed: int | None = None) -> pd.DataFrame:
    """Generate a balanced synthetic corpus for the classifier.

    Intended for bootstrapping and tests. Production training should use
    labelled GDELT or news data.

    Parameters
    ----------
    n : int, optional
        Number of examples to generate.
    seed : int or None, optional
        Seed for the local random source.

    Returns
    -------
    pandas.DataFrame
        Frame with ``text`` and integer ``label`` columns.
    """
    rng = random.Random(seed)
    labels = list(_TEMPLATES)

    records: list[dict[str, str | int]] = []
    for _ in range(n):
        label = rng.choice(labels)
        country_a, country_b = rng.sample(_TEMPLATE_COUNTRIES, 2)
        records.append({
            "text": rng.choice(_TEMPLATES[label]).format(c1=country_a, c2=country_b),
            "label": label,
        })

    return pd.DataFrame(records)


# --- Classifier -------------------------------------------------------------

class GeopoliticalEventClassifier:
    """DistilBERT event-type classifier with a deterministic fallback.

    The transformer stack is loaded lazily. If it cannot be loaded, or if no
    fine-tuned checkpoint exists, :meth:`predict` degrades to keyword
    classification rather than raising.

    Parameters
    ----------
    config : TrainingConfig or None, optional
        Hyper-parameters and paths. Defaults are used when omitted.
    """

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config: TrainingConfig = config or TrainingConfig()
        self.tokenizer: Any = None
        self.model: Any = None
        self._loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """Whether a transformer model is resident in memory."""
        return self._loaded

    @property
    def checkpoint_path(self) -> Path:
        """Filesystem location of the fine-tuned artefacts."""
        return Path(self.config.save_path)

    def load_tokenizer_model(self, from_checkpoint: str | Path | None = None) -> bool:
        """Load the tokenizer and model, from a checkpoint or the base model.

        Parameters
        ----------
        from_checkpoint : str, pathlib.Path or None, optional
            Local checkpoint directory. Falls back to
            ``TrainingConfig.model_name`` when omitted.

        Returns
        -------
        bool
            ``True`` when the model is resident, ``False`` when the transformer
            stack is unavailable.
        """
        try:
            from transformers import (
                DistilBertForSequenceClassification,
                DistilBertTokenizerFast,
            )
        except ImportError:
            _log.warning(
                "%s transformers is not installed; keyword classification only", WARN
            )
            return False

        source = str(from_checkpoint or self.config.model_name)
        try:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(source)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                source,
                num_labels=NUM_LABELS,
                ignore_mismatched_sizes=True,
            ).to(resolve_device())
        except (OSError, ValueError, ImportError) as exc:
            _log.error("%s Failed to load model from %s: %s", ERROR, source, exc)
            return False

        self._loaded = True
        _log.info("%s Model loaded: %s", OK, source)
        return True

    def prepare_dataset(self, df: pd.DataFrame) -> Any:
        """Tokenise a labelled frame into a HuggingFace dataset.

        Parameters
        ----------
        df : pandas.DataFrame
            Frame with ``text`` and ``label`` columns.

        Returns
        -------
        datasets.Dataset
            Tokenised dataset formatted for PyTorch.
        """
        from datasets import Dataset

        def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
            return self.tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=self.config.max_length,
            )

        dataset = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
        dataset = dataset.map(tokenize, batched=True)
        dataset = dataset.rename_column("label", "labels")
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        return dataset

    def train(
        self,
        df: pd.DataFrame | None = None,
        eval_df: pd.DataFrame | None = None,
    ) -> Any:
        """Fine-tune the classifier and persist the resulting artefacts.

        Parameters
        ----------
        df : pandas.DataFrame or None, optional
            Training frame with ``text`` and ``label``. A synthetic corpus is
            generated when omitted.
        eval_df : pandas.DataFrame or None, optional
            Evaluation frame. Split from ``df`` when omitted.

        Returns
        -------
        transformers.Trainer
            The trainer instance after ``train`` has completed.

        Raises
        ------
        RuntimeError
            When the transformer stack cannot be loaded.
        """
        from sklearn.metrics import accuracy_score, f1_score
        from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

        if df is None:
            _log.info("No training frame supplied; generating a synthetic corpus")
            df = create_synthetic_training_data(2000)

        if not self._loaded and not self.load_tokenizer_model():
            raise RuntimeError(
                "Transformer stack unavailable. Install torch and transformers "
                "to train the event classifier."
            )

        if eval_df is None:
            split_at = int(len(df) * (1 - self.config.eval_split))
            eval_df = df.iloc[split_at:].reset_index(drop=True)
            df = df.iloc[:split_at].reset_index(drop=True)

        train_dataset = self.prepare_dataset(df)
        eval_dataset = self.prepare_dataset(eval_df)

        def compute_metrics(prediction: Any) -> dict[str, float]:
            predicted = np.argmax(prediction.predictions, axis=1)
            return {
                "accuracy": float(accuracy_score(prediction.label_ids, predicted)),
                "f1": float(
                    f1_score(prediction.label_ids, predicted, average="weighted")
                ),
            }

        training_args = TrainingArguments(
            output_dir=self.config.save_path,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=50,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience
                )
            ],
        )

        _log.info("Training started: %d examples, %d epochs",
                  len(df), self.config.num_epochs)
        trainer.train()

        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.checkpoint_path)
        self.tokenizer.save_pretrained(self.checkpoint_path)

        label_map_path = self.checkpoint_path / "label_map.json"
        with label_map_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {str(index): name for index, name in enumerate(LABEL_NAMES)},
                handle,
                indent=2,
            )

        _log.info("%s Model artefacts written: %s", OK, self.checkpoint_path)
        return trainer

    def predict(self, texts: list[str]) -> list[Prediction]:
        """Classify a list of event texts.

        Loads the fine-tuned checkpoint on first call. When neither a
        checkpoint nor the transformer stack is available, every text is
        classified by the keyword fallback.

        Parameters
        ----------
        texts : list of str
            Event texts to classify.

        Returns
        -------
        list of Prediction
            One result per input, in input order.
        """
        if not texts:
            return []

        if not self._loaded:
            has_checkpoint = self.checkpoint_path.exists()
            if not has_checkpoint or not self.load_tokenizer_model(self.checkpoint_path):
                if not has_checkpoint:
                    _log.warning(
                        "%s No fine-tuned checkpoint at %s; using keyword "
                        "classification", WARN, self.checkpoint_path,
                    )
                return [self._rule_based_classify(text) for text in texts]

        import torch

        device = resolve_device()
        self.model.eval()
        results: list[Prediction] = []

        for start in range(0, len(texts), _INFERENCE_BATCH_SIZE):
            batch = texts[start:start + _INFERENCE_BATCH_SIZE]
            encodings = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                logits = self.model(**encodings).logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

            for row in probabilities:
                predicted = int(np.argmax(row))
                results.append({
                    "label_id": predicted,
                    "label_name": LABEL_NAMES[predicted],
                    "confidence": round(float(row[predicted]), 4),
                    "all_scores": {
                        name: round(float(row[index]), 4)
                        for index, name in enumerate(LABEL_NAMES)
                    },
                    "method": "transformer",
                })

        return results

    @staticmethod
    def _rule_based_classify(text: str) -> Prediction:
        """Classify one text by keyword evidence.

        Used whenever the neural model is unavailable. Texts matching no
        keyword are assigned the Political class with zero confidence, which
        keeps unmatched input out of the Military/Conflict bucket.

        Parameters
        ----------
        text : str
            Event text.

        Returns
        -------
        Prediction
            Classification result with ``method`` set to ``rule_based``.
        """
        lowered = text.lower()
        scores = {
            label: sum(1 for keyword in keywords if keyword in lowered)
            for label, keywords in _KEYWORDS.items()
        }
        total = sum(scores.values())

        if total == 0:
            return {
                "label_id": _DEFAULT_FALLBACK_LABEL,
                "label_name": LABEL_NAMES[_DEFAULT_FALLBACK_LABEL],
                "confidence": 0.0,
                "all_scores": dict.fromkeys(LABEL_NAMES, 0.0),
                "method": "rule_based",
            }

        best = min(scores.items(), key=lambda item: (-item[1], item[0]))[0]
        return {
            "label_id": best,
            "label_name": LABEL_NAMES[best],
            "confidence": round(scores[best] / total, 4),
            "all_scores": {
                LABEL_NAMES[label]: round(count / total, 4)
                for label, count in scores.items()
            },
            "method": "rule_based",
        }

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str | None = None,
    ) -> pd.DataFrame:
        """Append predicted event-type columns to an event frame.

        Parameters
        ----------
        df : pandas.DataFrame
            Preprocessed event frame.
        text_col : str or None, optional
            Column holding the text to classify. When omitted, text is
            synthesised as ``"<actor1> <event_label> <actor2>"``, matching the
            distribution produced by
            :func:`build_training_dataset_from_gdelt`.

        Returns
        -------
        pandas.DataFrame
            Copy of ``df`` with ``ml_event_type``, ``ml_confidence`` and
            ``ml_method`` columns appended.
        """
        result = df.copy()

        if result.empty:
            result["ml_event_type"] = pd.Series(dtype="object")
            result["ml_confidence"] = pd.Series(dtype="float64")
            result["ml_method"] = pd.Series(dtype="object")
            return result

        if text_col is not None and text_col in result.columns:
            texts = result[text_col].fillna("").astype(str).tolist()
        else:
            if text_col is not None:
                _log.warning(
                    "%s Column %s not present; synthesising classifier input",
                    WARN, text_col,
                )
            texts = _synthesise_event_text(result).tolist()

        predictions = self.predict(texts)

        result["ml_event_type"] = [item["label_name"] for item in predictions]
        result["ml_confidence"] = [item["confidence"] for item in predictions]
        result["ml_method"] = [item["method"] for item in predictions]
        return result


# --- Corpus construction from GDELT ----------------------------------------

# CAMEO root code to classifier label.
_CAMEO_TO_LABEL: Final[dict[str, int]] = {
    "01": 4, "02": 4, "03": 2, "04": 2, "05": 2,
    "06": 1, "07": 3, "08": 4, "09": 4, "10": 4,
    "11": 4, "12": 4, "13": 0, "14": 4, "15": 0,
    "16": 2, "17": 0, "18": 0, "19": 0, "20": 0,
}


def _synthesise_event_text(df: pd.DataFrame) -> pd.Series:
    """Build classifier input text from actor codes and the event label.

    Parameters
    ----------
    df : pandas.DataFrame
        Preprocessed event frame.

    Returns
    -------
    pandas.Series
        One text per row, aligned to ``df.index``.
    """
    actor1 = df.get("Actor1CountryCode", pd.Series("", index=df.index)).astype(str)
    actor2 = df.get("Actor2CountryCode", pd.Series("", index=df.index)).astype(str)
    label = df.get("event_label", pd.Series("", index=df.index)).astype(str)
    return (actor1 + " " + label + " " + actor2).str.strip()


def build_training_dataset_from_gdelt(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a labelled corpus from GDELT event codes.

    CAMEO root codes provide distant supervision: each code maps to one
    classifier label, and the text is synthesised from the actor pair and the
    event description.

    Parameters
    ----------
    df : pandas.DataFrame
        Preprocessed event frame carrying ``EventRootCode``.

    Returns
    -------
    pandas.DataFrame
        Frame with ``text`` and integer ``label`` columns.
    """
    if df.empty:
        return pd.DataFrame({"text": pd.Series(dtype="object"),
                             "label": pd.Series(dtype="int64")})

    codes = df["EventRootCode"].astype(str).str[:2]
    labels = codes.map(_CAMEO_TO_LABEL).fillna(_DEFAULT_FALLBACK_LABEL).astype("int64")

    return pd.DataFrame({
        "text": _synthesise_event_text(df),
        "label": labels,
    }).reset_index(drop=True)
