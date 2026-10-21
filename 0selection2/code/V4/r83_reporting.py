"""R83 same-lock TEST/LIB PDFs via the R80 reporter; never trains or selects."""

import argparse
import json
import sys
from pathlib import Path

from . import r80_reporting as engine


DEFAULT_ROOT = Path(__file__).resolve().parent / 'runs/R83_model_only_source_embedding'
GROUP = 'model_only_seed2'
BUDGETS = {GROUP: 49680}
FACTORY_MODULE = 'code.V4.r83_experiment'


def locked_inputs(root, arm=GROUP):
    return engine.locked_inputs(root, arm, experiment='R83', arms=BUDGETS)


def status_snapshot(root):
    """One read-only snapshot, including an honest not-started state."""
    run = root.expanduser().resolve() / GROUP
    result = dict(experiment='R83', group=GROUP, root=str(run), budget=BUDGETS[GROUP],
        status=engine._optional(run / 'status.json'),
        reporting=engine._optional(run / 'postprocess_status.json'))
    print(json.dumps(result, indent=2))
    return result


def report(args):
    return engine.report(args, experiment='R83', arms=BUDGETS, factory_module=FACTORY_MODULE)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--group', '--arm', dest='arm', choices=(GROUP,), default=GROUP)
    result.add_argument('--status', action='store_true')
    result.add_argument('--device', default='cpu')
    result.add_argument('--lib-batch-size', type=int, choices=(1, 2), default=2)
    result.add_argument('--nss-root', type=Path)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.status:
        status_snapshot(args.root)
        return 0
    try:
        report(args)
        return 0
    except Exception as error:
        print(f'R83 reporting failed: {error}. Completed training/TEST are retained.', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
