"""Unit tests for the parts that are easy to get silently wrong.

Run with `python main.py test`. Nothing here touches the network or the CLIP
weights: the parser, the metric, the fusion arithmetic and the cache signature
are all testable on their own.
"""
