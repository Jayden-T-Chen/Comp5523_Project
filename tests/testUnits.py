"""Unit tests for the parser, the metric, the fusion arithmetic and the cache.

Run with `python main.py test`. Nothing here loads CLIP or touches the network.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evaluation.tools.runEvaluation import average_precision, summarize
from retrieval import queryParse
from retrieval.queryParse import parse_query


class ParserTest(unittest.TestCase):
    def setUp(self):
        queryParse.apply_preset("shipped")

    def tearDown(self):
        queryParse.apply_preset("shipped")

    def test_long_phrase_masks_the_short_keys_inside_it(self):
        parsed = parse_query("sports sandals")
        self.assertEqual(parsed["type"], ["Sports Sandals"])
        self.assertNotIn("Sports Shoes", parsed["type"])
        self.assertNotIn("Sandals", parsed["type"])

    def test_legacy_preset_lets_both_short_keys_fire(self):
        queryParse.apply_preset("legacy")
        parsed = parse_query("sports sandals")
        self.assertIn("Sports Shoes", parsed["type"])
        self.assertIn("Sandals", parsed["type"])

    def test_colour_family_values_carry_partial_weight(self):
        parsed = parse_query("grey shoes")
        self.assertEqual(parsed["color"][0], "Grey")
        self.assertEqual(parsed["weights"]["color"]["Grey"], 1.0)
        self.assertEqual(parsed["weights"]["color"]["Silver"],
                         queryParse.COLOR_FAMILY_WEIGHT)

    def test_type_synonyms_keep_full_weight(self):
        parsed = parse_query("white sneakers")
        self.assertEqual(parsed["weights"]["type"]["Sports Shoes"], 1.0)
        self.assertEqual(parsed["weights"]["type"]["Casual Shoes"], 1.0)

    def test_gender_slot(self):
        self.assertEqual(parse_query("casual shoes for men")["gender"], "Men")
        self.assertIsNone(parse_query("casual shoes")["gender"])
        self.assertIsNone(parse_query("shoes for men and women")["gender"])

    def test_vocabulary_comes_from_the_catalogue(self):
        import csv

        from retrieval.clipCore import META_PATH

        with META_PATH.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        colours = {r["color"] for r in rows}
        types = {r["articleType"] for r in rows}
        for values in queryParse.COLOR_ALIASES.values():
            self.assertTrue(set(values) <= colours, values)
        for values in queryParse.TYPE_ALIASES.values():
            self.assertTrue(set(values) <= types, values)

    def test_repeated_whitespace_and_case_do_not_matter(self):
        self.assertEqual(parse_query("  BLACK   Shoes ")["color"],
                         parse_query("black shoes")["color"])


class MetricTest(unittest.TestCase):
    def test_perfect_ranking_scores_one(self):
        self.assertEqual(average_precision(["a", "b"], {"a", "b"}, 50), 1.0)

    def test_empty_relevant_set_scores_zero(self):
        self.assertEqual(average_precision(["a"], set(), 50), 0.0)

    def test_single_hit_at_rank_two(self):
        self.assertAlmostEqual(average_precision(["x", "a"], {"a"}, 50), 0.5)

    def test_denominator_is_capped_by_the_depth(self):
        relevant = {f"r{i}" for i in range(80)}
        ranked = [f"r{i}" for i in range(80)]
        self.assertAlmostEqual(average_precision(ranked, relevant, 50), 1.0)

    def test_items_past_the_depth_are_ignored(self):
        ranked = ["x"] * 50 + ["a"]
        self.assertEqual(average_precision(ranked, {"a"}, 50), 0.0)

    def test_summarize_groups(self):
        rows = [{"qid": "1", "group": "color", "ap50": 0.2, "p10": 0.5},
                {"qid": "2", "group": "color", "ap50": 0.4, "p10": 0.7},
                {"qid": "3", "group": "type", "ap50": 0.6, "p10": 0.9}]
        out = summarize(rows, ("ap50", "p10"))
        self.assertEqual(out["n_queries"], 3)
        self.assertEqual(out["overall"]["ap50"], 0.4)
        self.assertEqual(out["by_group"]["color"]["ap50"], 0.3)
        self.assertEqual(out["by_group"]["type"]["n"], 1)


class FusionTest(unittest.TestCase):
    def test_no_named_attribute_means_no_reranking(self):
        import torch

        from retrieval.rerank import attribute_agreement

        tables = {"values": {"color": ["Black", "White"]},
                  "color": torch.tensor([[0.9, 0.1], [0.2, 0.8]])}
        parsed = parse_query("comfortable shoes")
        self.assertIsNone(attribute_agreement(parsed, tables))

    def test_family_weight_lowers_the_agreement_of_a_neighbour(self):
        import torch

        from retrieval.rerank import attribute_agreement

        tables = {"values": {"color": ["Grey", "Silver"]},
                  "color": torch.tensor([[0.2, 0.8], [0.7, 0.3]])}
        literal = attribute_agreement({"color": ["Grey"],
                                       "weights": {"color": {"Grey": 1.0}}},
                                      tables)
        with_family = attribute_agreement(
            {"color": ["Grey", "Silver"],
             "weights": {"color": {"Grey": 1.0, "Silver": 0.5}}}, tables)
        self.assertAlmostEqual(literal[0].item(), 0.2)
        self.assertAlmostEqual(with_family[0].item(), 0.2 + 0.5 * 0.8)
        self.assertGreater(with_family[1].item(), literal[1].item())

    def test_two_named_attributes_multiply(self):
        import torch

        from retrieval.rerank import attribute_agreement

        tables = {"values": {"color": ["Black"], "type": ["Heels"]},
                  "color": torch.tensor([[0.5]]),
                  "type": torch.tensor([[0.4]])}
        parsed = {"color": ["Black"], "type": ["Heels"], "weights": {}}
        self.assertAlmostEqual(attribute_agreement(parsed, tables)[0].item(),
                               0.2)


class CacheTest(unittest.TestCase):
    def test_signature_moves_when_a_file_changes(self):
        from retrieval.clipCore import gallery_signature

        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.jpg"
            second = Path(tmp) / "b.jpg"
            first.write_bytes(b"x" * 10)
            second.write_bytes(b"y" * 20)
            before = gallery_signature([first, second])
            self.assertEqual(before[0], 2)
            self.assertEqual(before[2], 30)
            second.write_bytes(b"y" * 40)
            after = gallery_signature([first, second])
            self.assertNotEqual(before, after)
