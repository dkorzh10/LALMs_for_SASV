"""Tests for Model.get_tokenizer() and count_tokens() API, and DummyTokenizer."""
import unittest
from src.models.dummy.model import DummyModel
from src.models.dummy.tokenizer import DummyTokenizer


class TestDummyTokenizer(unittest.TestCase):
    def test_encode_returns_list_of_ids(self):
        tok = DummyTokenizer(chars_per_token=4)
        ids = tok.encode("hello world", add_special_tokens=False)
        self.assertIsInstance(ids, list)
        self.assertEqual(len(ids), 2)  # 11 chars // 4 = 2

    def test_encode_empty_string(self):
        tok = DummyTokenizer(chars_per_token=4)
        ids = tok.encode("", add_special_tokens=False)
        self.assertEqual(ids, [])

    def test_encode_custom_chars_per_token(self):
        tok = DummyTokenizer(chars_per_token=5)
        ids = tok.encode("abcd" * 5, add_special_tokens=False)
        self.assertEqual(len(ids), 4)  # 20 chars // 5 = 4

    def test_pad_token_id(self):
        tok = DummyTokenizer()
        self.assertEqual(tok.pad_token_id, 0)

    def test_decode(self):
        tok = DummyTokenizer()
        out = tok.decode([0, 1, 2])
        self.assertIn("decoded", out)
        self.assertIn("3", out)


class TestTokenizerAPI(unittest.TestCase):
    def test_dummy_model_get_tokenizer_returns_dummy_tokenizer(self):
        m = DummyModel({"model_name": "dummy"})
        tok = m.get_tokenizer()
        self.assertIsNotNone(tok)
        self.assertIsInstance(tok, DummyTokenizer)

    def test_dummy_model_count_tokens_uses_tokenizer(self):
        m = DummyModel({"model_name": "dummy"})
        self.assertEqual(m.count_tokens(""), 0)
        self.assertEqual(m.count_tokens("abcd"), 1)  # 4//4
        self.assertEqual(m.count_tokens("a" * 100), 25)

    def test_dummy_model_count_tokens_via_count_tokens(self):
        m = DummyModel({"model_name": "dummy"})
        self.assertGreaterEqual(m.count_tokens("hello world"), 0)
