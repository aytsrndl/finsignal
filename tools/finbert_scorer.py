"""Lazy CPU FinBERT inference; no heuristic substitute when weights are absent."""
import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SentimentScore(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    positive: float = Field(ge=0, le=1)
    negative: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)
    truncated: bool

    @model_validator(mode='after')
    def probabilities_sum_to_one(self):
        if not math.isclose(self.positive + self.negative + self.neutral, 1, abs_tol=1e-4):
            raise ValueError('Sentiment probabilities must sum to one.')
        return self

    @property
    def directional_score(self):
        return self.positive - self.negative

    @property
    def label(self):
        return max(('positive', 'negative', 'neutral'), key=lambda name: getattr(self, name))


class FinBertScorer:
    model_name = 'ProsusAI/finbert'

    def __init__(self, *, revision: str = 'main', local_files_only: bool = False):
        self.revision = revision
        self.resolved_revision = None
        self.local_files_only = local_files_only
        self._classifier = None

    def _load(self):
        if self._classifier is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
            cache = Path(__file__).resolve().parents[1] / 'data' / 'cache' / 'huggingface'
            cache.mkdir(parents=True, exist_ok=True)
            options = dict(revision=self.revision, cache_dir=str(cache),
                           local_files_only=self.local_files_only, trust_remote_code=False)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, **options, weights_only=True,
            )
            self.resolved_revision = getattr(model.config, '_commit_hash', None)
            tokenizer_options = options | {'revision': self.resolved_revision or self.revision}
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_options)
            model.eval()
            self._classifier = pipeline('text-classification', model=model, tokenizer=tokenizer,
                                        device=-1, framework='pt')
        except ImportError as error:
            raise ValueError('Install requirements-sentiment.txt to enable CPU FinBERT scoring.') from error
        except Exception as error:
            raise ValueError(f'FinBERT weights/tokenizer could not be loaded ({type(error).__name__}); allow Hugging Face access or populate the model cache. No fallback scores were produced.') from error

    def score(self, texts: list[str]) -> list[SentimentScore]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError('Cannot score blank article text.')
        self._load()
        lengths = [len(self._classifier.tokenizer(text, truncation=False, verbose=False)['input_ids']) for text in texts]
        results = self._classifier(texts, top_k=None, truncation=True, max_length=512, batch_size=16)
        if len(results) != len(texts):
            raise ValueError('FinBERT returned a different number of results than articles.')
        scores = []
        for result, length in zip(results, lengths):
            labels = {item['label'].lower(): item['score'] for item in result}
            if len(result) != 3 or set(labels) != {'positive', 'negative', 'neutral'}:
                raise ValueError('FinBERT model must expose positive, negative, and neutral labels.')
            scores.append(SentimentScore(**labels, truncated=length > 512))
        return scores
