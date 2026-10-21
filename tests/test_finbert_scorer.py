import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from tools.finbert_scorer import FinBertScorer


class FinBertScorerTests(unittest.TestCase):
    def classifier(self, output, length=12):
        classifier = Mock(return_value=output)
        classifier.tokenizer.return_value = {'input_ids': [1] * length}
        scorer = FinBertScorer()
        scorer._classifier = classifier
        return scorer, classifier

    def test_label_names_not_positions_determine_scores(self):
        scorer, classifier = self.classifier([[{'label': 'negative', 'score': .1},
                                             {'label': 'neutral', 'score': .2}, {'label': 'positive', 'score': .7}]])
        result = scorer.score(['Company grows revenue'])[0]
        self.assertAlmostEqual(result.directional_score, .6)
        self.assertEqual(result.label, 'positive')
        self.assertFalse(result.truncated)
        self.assertTrue(classifier.call_args.kwargs['truncation'])
        self.assertEqual(classifier.call_args.kwargs['max_length'], 512)

    def test_long_input_is_flagged(self):
        scorer, _ = self.classifier([[{'label': 'negative', 'score': 0}, {'label': 'neutral', 'score': 1}, {'label': 'positive', 'score': 0}]], length=600)
        self.assertTrue(scorer.score(['Long text'])[0].truncated)

    def test_invalid_labels_probabilities_and_result_counts_fail(self):
        cases = [[], [[{'label': 'LABEL_0', 'score': 1}]],
                 [[{'label': 'positive', 'score': .9}, {'label': 'negative', 'score': .9}, {'label': 'neutral', 'score': .9}]]]
        for output in cases:
            scorer, _ = self.classifier(output)
            with self.subTest(output=output), self.assertRaises(ValueError):
                scorer.score(['Example'])

    def test_empty_input_does_not_load_weights(self):
        scorer = FinBertScorer()
        scorer._load = Mock()
        self.assertEqual(scorer.score([]), [])
        scorer._load.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'), 'Optional CPU model dependencies not installed')
    def test_actual_cpu_inference_with_local_tiny_model(self):
        # Verify the installed inference stack without a network call or claiming
        # random test weights are a trained sentiment classifier.
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
        with tempfile.TemporaryDirectory() as directory:
            vocab = Path(directory) / 'vocab.txt'
            vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ncompany\ngrows\nrevenue\n', encoding='utf-8')
            BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(directory)
            config = BertConfig(vocab_size=8, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                intermediate_size=32, num_labels=3,
                                id2label={0: 'positive', 1: 'negative', 2: 'neutral'})
            BertForSequenceClassification(config).save_pretrained(directory, safe_serialization=True)
            scorer = FinBertScorer(local_files_only=True)
            scorer.model_name = directory
            result = scorer.score(['company grows revenue'])
            self.assertEqual(len(result), 1)
            self.assertAlmostEqual(result[0].positive + result[0].negative + result[0].neutral, 1, places=5)
