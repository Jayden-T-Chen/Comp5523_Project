#!/usr/bin/env python3
"""One entry point for the repository.

    python main.py <command> [flags]

    search         rank gallery images for a text query on the command line
    web            serve the Gradio page used in the demonstration
    eval           score one method against one query set, write results/*.json
    compare        paired bootstrap between two results files
    ablate-parser  four parser configurations over the 60 main queries
    ablate-fusion  candidate pool and fusion weight sweep
    probe          zero-shot attribute accuracy and query latency
    encoder-check  the same queries under both CLIP text activations
    figure         failure-case montage for the report
    label-sheet    click-through sheet for the open human judgements
    test           run the unit tests

Each command lives in its own module and is imported only when it is used, so
`compare` and `test` never pay for loading torch.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMMANDS: dict[str, tuple[str, str]] = {
    "search": ("retrieval.searchCli",
               "rank gallery images for a text query"),
    "web": ("retrieval.searchWeb",
            "serve the Gradio page (demonstration and recording)"),
    "eval": ("evaluation.tools.runEvaluation",
             "score a method against a query set"),
    "compare": ("evaluation.tools.compareResults",
                "paired bootstrap between two results files"),
    "ablate-parser": ("evaluation.tools.ablateParser",
                      "parser configurations over the 60 main queries"),
    "ablate-fusion": ("evaluation.tools.ablateFusion",
                      "candidate pool and fusion weight sweep"),
    "probe": ("evaluation.tools.probeAttributes",
              "attribute table accuracy, fused-term scales and query cost"),
    "encoder-check": ("evaluation.tools.encoderCheck",
                      "score the queries under both CLIP text activations"),
    "figure": ("evaluation.tools.renderFailureCases",
               "failure-case montage for the report"),
    "label-sheet": ("evaluation.tools.labelSheet",
                    "judge the head cells nobody has looked at yet"),
    "test": ("tests.testUnits", "run the unit tests"),
}


def usage() -> str:
    lines = ["usage: python main.py <command> [flags]", "",
             "commands:"]
    width = max(len(name) for name in COMMANDS)
    for name, (_module, help_text) in COMMANDS.items():
        lines.append(f"  {name:<{width}}  {help_text}")
    lines += ["", "Run `python main.py <command> --help` for the flags of one "
              "command.", "The command table and the numbers behind it are in "
              "README.md."]
    return "\n".join(lines)


def run_tests(_args) -> None:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(usage())
        raise SystemExit(0 if len(sys.argv) < 2 else 2)

    name = sys.argv[1]
    parser = argparse.ArgumentParser(prog=f"main.py {name}",
                                     description=COMMANDS[name][1])
    if name == "test":
        runner = run_tests
    else:
        module = importlib.import_module(COMMANDS[name][0])
        module.add_args(parser)
        runner = module.main

    args = parser.parse_args(sys.argv[2:])
    args.command = name
    runner(args)


if __name__ == "__main__":
    main()
