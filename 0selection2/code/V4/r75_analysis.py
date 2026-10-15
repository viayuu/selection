"""R75 reporting entrypoint, called only after the locked TEST completes."""

import argparse
from pathlib import Path


def analyze(root):
    from .r75_delivery_analysis import analyze as build_report

    return build_report(Path(root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    analyze(parser.parse_args().root)


if __name__ == '__main__':
    main()
