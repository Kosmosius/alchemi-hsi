#!/usr/bin/env python3
"""Convenience wrapper to build the lab embedding index."""

import sys

from alchemi.cli import preprocess


def main() -> None:
    argv = list(sys.argv[1:])
    if "--build-lab-index" not in argv:
        argv.append("--build-lab-index")
    preprocess.main(argv)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
