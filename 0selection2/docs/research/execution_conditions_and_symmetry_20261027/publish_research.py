"""Publish only this completed scientific report, not unfinished R119 work."""

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'post_r116_major_direction_20261027/publish_research.py'
FILES = ('REQUEST.md', 'ROUTING.json', 'REPORT.md', 'sources.csv', 'directions.csv',
         'REPORT.pdf', 'render_notes.json', 'DELIVERY.md', 'run_astra.sh',
         'cli_started.json', 'cli_exit.json', 'CLI_FINAL.txt', 'render_pdf.py', 'publish_research.py')


def main():
    spec = importlib.util.spec_from_file_location('execution_conditions_publisher', SOURCE)
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    publisher.ROOT, publisher.FILES = ROOT, FILES
    original = publisher.git

    def git(*args):
        if args and args[0] == 'commit':
            args = list(args)
            args[args.index('-m') + 1] = 'Publish Astra execution-conditions and symmetry scientific follow-up'
        return original(*args)

    publisher.git = git
    print(json.dumps(publisher.publish(), ensure_ascii=False))


if __name__ == '__main__':
    main()
