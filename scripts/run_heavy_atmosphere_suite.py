#!/usr/bin/env python3
"""Run only the heavy-atmosphere evaluation suite."""

import sys

from alchemi.cli import evaluate


def main() -> None:
    argv = list(sys.argv[1:]) + ["--only", "heavy_atmosphere"]
    evaluate.main(argv)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
