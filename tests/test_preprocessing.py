"""Unit tests for the preprocessing module."""

from __future__ import annotations

import re

import pandas as pd
import pytest

from idtweets import preprocessing as pp
from idtweets.slang import SLANG_MAP


class TestScalarSteps:
    def test_strip_mentions_removes_all(self):
        out = pp.strip_mentions("@a @b @c hi")
        assert out.split() == ["hi"]
        assert "@" not in out

    def test_strip_mentions_keeps_hashtags_and_words(self):
        assert pp.strip_mentions("@user #tag word").split() == ["#tag", "word"]

    def test_strip_urls(self):
        assert pp.strip_urls("lihat https://t.co/abcDEF ya") == "lihat   ya"

    def test_strip_urls_handles_www(self):
        assert "www" not in pp.strip_urls("go www.example.com now")

    def test_hashtag_keeps_word_by_default(self):
        assert pp.strip_hashtags("#RekayasaDidzolimi kabar") == "RekayasaDidzolimi kabar"

    def test_hashtag_drop_word(self):
        assert pp.strip_hashtags("#tag kabar", keep_word=False).strip() == "kabar"

    def test_retweet_prefix(self):
        assert pp.strip_retweet_marker("RT @a: hi").strip() == "@a: hi"

    def test_strip_numbers(self):
        assert pp.strip_numbers("angka 12345 di sini").split() == ["angka", "di", "sini"]

    def test_strip_punctuation(self):
        out = pp.strip_punctuation("halo, dunia! (tes)")
        assert not re.search(r"[^\w\s]", out)

    def test_collapse_whitespace(self):
        assert pp.collapse_whitespace("  a   b \n c  ") == "a b c"

    def test_fix_mojibake_curly_quotes(self):
        out = pp.fix_mojibake("governance\u00e2\u20ac\ufffd modi\u00e2\u20ac\u2122s")
        assert "\u00e2" not in out
        assert "government" in out.lower() or "governance" in out.lower()

    def test_fix_mojibake_real_row(self):
        raw = "modi promised \u00e2\u20ac\u0153minimum government maximum governance\u00e2\u20ac\u009d"
        out = pp.fix_mojibake(raw)
        assert "minimum government maximum governance" in out
        assert "\u00e2" not in out


class TestSlang:
    def test_slang_map_has_no_identity_mappings(self):
        assert all(k != v for k, v in SLANG_MAP.items()), (
            "identity mappings are noise; remove them"
        )

    def test_normalize_token(self):
        assert pp.normalize_slang("yg gak tau") == "yang tidak tahu"

    def test_normalize_does_not_touch_unknown(self):
        assert pp.normalize_slang("novel baswedan") == "novel baswedan"

    def test_normalize_word_level_not_substring(self):
        # "gak" inside "gakarta" must NOT be rewritten.
        assert pp.normalize_slang("gakarta") == "gakarta"


class TestTokenUtils:
    def test_tokenize_lowercases(self):
        assert pp.tokenize("HALO Dunia") == ["halo", "dunia"]

    def test_tokenizer_strips_handles(self):
        assert "@User" not in pp.tokenize("@User halo")

    def test_filter_tokens_drops_stopwords(self):
        out = pp.filter_tokens(["ini", "novel", "baswedan"], {"ini"})
        assert out == ["novel", "baswedan"]

    def test_filter_tokens_drops_punct_and_digits(self):
        out = pp.filter_tokens(["!!!", "123", "novel"], set())
        assert out == ["novel"]

    def test_stemming_reduces_inflections(self):
        assert pp.stem_tokens(["menyiram", "penyiraman"]) != [
            "menyiram",
            "penyiraman",
        ]


class TestPreprocessor:
    def test_full_pipeline_matches_documented_example(self):
        p = pp.Preprocessor()
        out = p("RT @a @b @c: Ini KASUS Novel Baswedan! https://t.co/x #Rekayasa")
        assert out == "novel baswedan rekayasa"

    def test_all_mentions_gone(self):
        p = pp.Preprocessor(stemming=False)
        out = p("@x @y @z novel")
        assert "@" not in out
        assert "novel" in out

    def test_handles_none_and_nan(self):
        p = pp.Preprocessor()
        assert p(None) == ""
        assert p(float("nan")) == ""
        assert p("") == ""

    def test_cache_returns_same_object_value(self):
        p = pp.Preprocessor()
        assert p("Novel Baswedan") == p("Novel Baswedan")

    def test_tokens_helper(self):
        p = pp.Preprocessor()
        assert p.tokens("Novel Baswedan") == ["novel", "baswedan"]

    def test_transform_series_dtype_and_length(self):
        p = pp.Preprocessor()
        s = pd.Series(["RT @a: Novel Baswedan!!", None, "kasus   Novel"])
        out = p.transform_series(s)
        assert len(out) == 3
        assert all(isinstance(x, str) for x in out)

    def test_stemming_can_be_disabled(self):
        p = pp.Preprocessor(stemming=False, stopwords=False)
        assert "menyiram" in p("menyiram")

    def test_stopwords_can_be_disabled(self):
        p = pp.Preprocessor(stopwords=False, stemming=False)
        assert "yang" in p("yang mana")

    def test_extra_stopwords(self):
        p = pp.Preprocessor(stemming=False, extra_stopwords=["novel"])
        assert "novel" not in p("novel baswedan")


class TestRegexOnlyPipeline:
    def test_matches_callable_pipeline_on_mentions(self):
        raw = pd.Series(["RT @a @b: Novel Baswedan https://t.co/x #Tag!!"])
        vec = pp.preprocess_regex_only(raw)
        callable_out = pp.Preprocessor(stemming=False, stopwords=False)(raw.iloc[0])
        assert "@" not in vec.iloc[0]
        assert "novel" in vec.iloc[0]
        assert "baswedan" in callable_out

    def test_series_in_series_out(self):
        out = pp.preprocess_regex_only(pd.Series(["a", "b"]))
        assert isinstance(out, pd.Series)
        assert len(out) == 2
