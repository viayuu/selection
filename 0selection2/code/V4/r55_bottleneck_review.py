"""Read-only post-hoc interpretation of the locked R55 edge predictions."""

import argparse
import json
from pathlib import Path

import numpy as np

from .multitask_probe import dump, file_hash
from .r48_common import write_csv
from .r53_data import MVRP
from .r55_experiment import ARM
from .r55_targets import ROOT


def review(root=ROOT):
    directory = root / ARM
    result = json.loads((directory / 'result.json').read_text())
    lock = json.loads((root / 'locked_checkpoint.json').read_text())
    rows = []
    for p in MVRP:
        with np.load(directory / 'bottleneck_predictions' / 'val' / (p + '.npz'), allow_pickle=False) as saved:
            pred, truth, valid = saved['edge_difference'], saved['target'], saved['edge_mask']
            count = valid.sum(-1)
            groups = [valid & (truth > 0), valid & (truth < 0)]
            means = []
            for group in groups:
                active = group.sum(-1) > 0
                means.append(float(((pred * group).sum(-1) / np.maximum(group.sum(-1), 1))[active].mean()))
            rows.append(dict(problem=p, n=len(pred),
                predicted_abs_mean=float(((np.abs(pred) * valid).sum(-1) / count).mean()),
                predicted_abs_max=float(np.abs(pred[valid]).max()),
                prediction_abs_ge_half_pct=float((((np.abs(pred) >= .5) & valid).sum(-1) / count).mean() * 100),
                true_nonzero_pct=float(((np.count_nonzero(truth, axis=-1)) / count).mean() * 100),
                prediction_on_true_positive_mean=means[0], prediction_on_true_negative_mean=means[1],
                edge_mse=result['best_structure_val']['per_problem'][p]['edge_loss'],
                zero_edge_reference_mse=result['best_structure_val']['per_problem'][p]['zero_prediction_edge_loss']))
    write_csv(root / 'bottleneck_behavior.csv', rows)
    macro = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ('problem', 'n', 'predicted_abs_max')}
    macro['predicted_abs_max'] = max(r['predicted_abs_max'] for r in rows)
    dump(root / 'bottleneck_behavior.json', dict(macro=macro, per_problem=rows,
        diagnostic_only=True, checkpoint_sha256=lock['sha256'], analysis_source_sha256=file_hash(Path(__file__)),
        boundary='fixed 960 validation route diagnostics, after selection/test; no fitting, new evaluation or decision changes',
        threshold='0.5 is the nearest-integer count boundary, not a tuned deployment threshold'))
    text = ['## Post-hoc Bottleneck Interpretation',
        'This read-only diagnosis uses the locked checkpoint outputs on the existing fixed 960 validation route instances. It does not select a new checkpoint, change inference, or start another experiment.',
        f"Best epoch {result['best_epoch']}, final epoch {result['final_epoch']}: checkpoint saving uses any strict new validation minimum, while early stopping requires an improvement of at least 0.001 percentage points. A smaller strict minimum need not reset the plateau counter.",
        f"Mean absolute predicted billed-count difference: {macro['predicted_abs_mean']:.9g}; maximum absolute prediction: {macro['predicted_abs_max']:.9g}. Predicted absolute counts at least 0.5: {macro['prediction_abs_ge_half_pct']:.6f}% of valid edges; true nonzero edges: {macro['true_nonzero_pct']:.4f}%.",
        f"On true positive/negative difference edges, mean predictions are {macro['prediction_on_true_positive_mean']:.9g}/{macro['prediction_on_true_negative_mean']:.9g}. Group-balanced edge MSE is {macro['edge_mse']:.9f}, versus {macro['zero_edge_reference_mse']:.9f} for all-zero predictions."]
    for split in ('train', 'val'):
        pair = result['best_pair_' + split]['macro']
        text.append(f"- {split} pair accuracy: {pair['accuracy']*100:.4f}%; always-MTL reference: {100-pair['moel_win_pct']:.4f}%; model selects MOEL on {pair['moel_prediction_pct']:.4f}% of instances.")
    if macro['predicted_abs_max'] < .5:
        text.append('Every predicted difference on this diagnostic subset would round to zero as an integer count. The nonzero final scalar therefore comes from summing small continuous edge estimates, not from recovering the actual discrete route-count differences. The structural readout is connected, but this run did not learn a faithful difference-edge bottleneck. It does not establish whether accurate difference reconstruction would improve selection.')
    else:
        text.append('Some predictions reach an integer-count decision boundary. Their existence alone does not show accurate structural reconstruction; grouped edge losses and final selection results remain the relevant checks.')
    text += ['The fixed readout also uses observed input distances. This is a task-specific supervised structural bottleneck, not a strict concept-only replica of the cited paper. Loss-gradient scaling, decoder capacity and conditional predictability are not separated by this single configuration; no unique cause is established.', '']
    report = root / 'RESULTS.md'
    body = report.read_text().split('## Post-hoc Bottleneck Interpretation')[0].rstrip()
    report.write_text(body + '\n\n' + '\n'.join(text))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    review(parser.parse_args().root)
