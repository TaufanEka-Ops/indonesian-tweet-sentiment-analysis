"""Tests for contact-PII detection (governs which rows get dropped)."""

from __future__ import annotations

import pandas as pd

from idtweets import pii


class TestEmail:
    def test_detects_simple_email(self):
        assert pii.find_emails("reach me at a.b@example.co.id please") == [
            "a.b@example.co.id"
        ]

    def test_has_contact_pii_true_for_email(self):
        assert pii.has_contact_pii("mail: someone@mail.com")


class TestPhone:
    def test_detects_indian_mobile(self):
        assert pii.find_phones("call 9426216831 zahra")

    def test_detects_whatsapp_with_country_code(self):
        assert pii.find_phones("whatsapp number 916385859493")

    def test_detects_anchored_short_number(self):
        assert pii.find_phones("mbl 9646741311 blood serve")

    def test_ignores_years(self):
        assert pii.find_phones("from 2002 2019 the congress") == []

    def test_ignores_large_sums(self):
        assert pii.find_phones("scheme 72000 per month for 6000") == []

    def test_ignores_long_reference_numbers(self):
        # 16-digit application reference must not be treated as a phone.
        assert pii.find_phones("application reference 7287789622478624 dated") == []

    def test_ignores_random_indices(self):
        assert pii.find_phones("117 118 122 133 140 out total 153") == []


class TestFlagSeries:
    def test_flag_pii_series(self):
        s = pd.Series(["hello modi", "call 9426216831 now", "2002 2019 history"])
        mask = pii.flag_pii(s)
        assert mask.tolist() == [False, True, False]

    def test_find_pii_hits_frame(self):
        s = pd.Series(["nothing", "email x@y.io and call 9426216831"])
        hits = pii.find_pii_hits(s)
        assert len(hits) == 1
        assert hits.iloc[0]["row_index"] == 1
        assert "x@y.io" in hits.iloc[0]["emails"]


class TestRealDatasets:
    def test_modi_dataset_has_few_pii_rows(self):
        from idtweets.io import load_modi

        df = load_modi()
        mask = pii.flag_pii(df["text"])
        # We expect a handful, not hundreds: precision matters more than recall.
        assert 0 < int(mask.sum()) < 30

    def test_novel_dataset_has_no_contact_pii(self):
        from idtweets.io import load_novel

        df = load_novel()
        assert int(pii.flag_pii(df["text"]).sum()) == 0
