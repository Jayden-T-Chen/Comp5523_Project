"""Retrieval side of the project: gallery loading, CLIP core, query parsing,
attribute re-ranking and the two interfaces.

Importable as a package (`from retrieval import rerank`) so that the evaluation
tools and `main.py` share one copy of the ranking code. Nothing in here reads the
evaluation sets.
"""
