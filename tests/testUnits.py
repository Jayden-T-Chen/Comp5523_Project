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


class WebLayoutTest(unittest.TestCase):
    """The page's card markup: shipped order, why it scored that, on demand."""

    POOL = [
        {"id": "a", "rank": 1, "sim": 0.30, "agree": 0.90, "title": "A shoe",
         "type": "Sports Shoes", "color": "Silver", "gender": "Men",
         "path": "/x/a.jpg"},
        {"id": "b", "rank": 2, "sim": 0.32, "agree": 0.50, "title": "B shoe",
         "type": "Sports Shoes", "color": "White", "gender": "Women",
         "path": "/x/b.jpg"},
        {"id": "c", "rank": 3, "sim": 0.34, "agree": None, "title": "C shoe",
         "type": "Casual Shoes", "color": "Silver", "gender": "Men",
         "path": "/x/c.jpg"},
        {"id": "d", "rank": 4, "sim": 0.29, "agree": 0.80, "title": "D shoe",
         "type": "Heels", "color": "Black", "gender": "Women",
         "path": "/x/d.jpg"},
        {"id": "e", "rank": 5, "sim": 0.31, "agree": 0.70, "title": "E shoe",
         "type": "Sports Shoes", "color": "Grey", "gender": "Men",
         "path": "/x/e.jpg"},
    ]

    def test_the_list_keeps_the_shipped_order_and_knows_the_cosine_rank(self):
        from retrieval import webStyle as style

        cards = style.ranking(self.POOL, 3)
        self.assertEqual([c["id"] for c in cards], ["a", "b", "c"])
        self.assertEqual([c["rank"] for c in cards], [1, 2, 3])
        self.assertEqual([c["cos_rank"] for c in cards], [4, 2, 1])

    def test_the_tags_separate_a_named_value_from_a_family_one(self):
        from retrieval import webStyle as style

        parsed = parse_query("grey sports shoes")
        cards = style.ranking(self.POOL, 5)
        by_id = {c["id"]: style.card(c, parsed) for c in cards}
        self.assertIn("<span class='tag named'>Grey</span>", by_id["e"])
        self.assertIn("<span class='tag family'>Silver</span>", by_id["a"])
        self.assertIn("<span class='tag'>Heels</span>", by_id["d"])
        self.assertIn("<span class='tag'>Black</span>", by_id["d"])
        self.assertIn("<span class='tag'>Men</span>", by_id["e"])

    def test_the_score_parts_are_in_the_markup_and_shown_on_request(self):
        from retrieval import webStyle as style

        parsed = parse_query("grey sports shoes")
        cards = style.ranking(self.POOL, 3)
        folded = style.pane(cards, "Ranked list", "", parsed)
        open_it = style.pane(cards, "Ranked list", "", parsed, raw=True)
        self.assertIn("cos 0.300 · attr 0.90", folded)
        self.assertNotIn("showraw", folded)
        self.assertIn("showraw", open_it)

    def test_the_cosine_rank_badge_appears_only_when_asked(self):
        from retrieval import webStyle as style

        parsed = parse_query("grey sports shoes")
        cards = style.ranking(self.POOL, 3)
        plain = style.pane(cards, "Ranked list", "", parsed)
        ranked = style.pane(cards, "Ranked list", "", parsed, show_rank=True)
        self.assertNotIn("cos #", plain)
        self.assertIn("cos #4", ranked)
        self.assertIn("cos #n", style.tag_legend(True))
        self.assertNotIn("cos #n", style.tag_legend(False))

    def test_the_status_line_names_the_score_and_no_comparison(self):
        from retrieval import webStyle as style

        named = style.status("grey sports shoes", 12.0, "mps", True, True)
        self.assertIn("feature cache hit", named)
        self.assertNotIn("head changed", named)
        self.assertIn("cosine ranks shown",
                      style.status("grey sports shoes", 12.0, "mps", True,
                                   True, True))
        self.assertIn("no attribute named",
                      style.status("shoes", 12.0, "mps", True, False))

    def test_slots_list_named_and_family_values_apart(self):
        from retrieval import webStyle as style

        parsed = parse_query("silver sports shoes")
        html = style.slots(parsed)
        self.assertIn("color <b>Silver</b>", html)
        self.assertIn("also accepted <b>Metallic</b>", html)
        self.assertNotIn("Silver, Metallic</b>", html)
        self.assertEqual(style.accepted(parsed)["color"], {"Silver", "Metallic"})

    def test_every_sample_chip_names_an_attribute(self):
        """A chip that parses to nothing would score cosine alone."""
        from retrieval import webStyle as style
        from retrieval.searchWeb import SAMPLES

        self.assertEqual(len(SAMPLES), len(set(SAMPLES)))
        for query in SAMPLES:
            with self.subTest(query=query):
                parsed = parse_query(query)
                hits = style.accepted(parsed)
                self.assertTrue(hits["color"] or hits["type"]
                                or parsed["gender"])
