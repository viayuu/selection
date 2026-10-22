"""CPU-only follow-up feasibility, not an R101 model or training dependency.

No dataset, labels, routes, decoder, CUDA context or package change. MatNet's
three exact native backbone definitions are isolated from an unrelated broken
optional PyG import; their numerical implementations are not rewritten.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
from unittest.mock import patch

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '2'

import torch
from torch import nn
from torch.nn import functional as F


HOME = Path('/public/home/shiys')
ELG = HOME / 'methods\u8bba\u6587_mineru_output/source_code/ELG/TSP'
EASY = HOME / 'EasyNCO'
PINS = {
    'ELG_TSP': '94448cce3ba413388286449886eae05f27c03f94996ff45daea37ebdce88fcdf',
    'MatNet_ATSP': 'a86d1ab1d8db62088e6a9d54118db92ebe8472c316dc046fcf755ca61280b5ca',
}


def sha(path):
    with Path(path).open('rb') as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(2**20), b''):
            result.update(block)
    return result.hexdigest()


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encoder_state(path, key, identity):
    if sha(path) != PINS[identity]:
        raise ValueError('Available native checkpoint changed: ' + identity)
    state = torch.load(path, map_location='cpu', weights_only=False)[key]
    return {name[8:]: value for name, value in state.items() if name.startswith('encoder.')}


def elg_probe():
    source, weight = ELG / 'models.py', ELG / 'weights/ELG.pt'
    native = load_file(source, '_coverage_elg_native')
    model = native.TSP_Encoder(embedding_dim=128, encoder_layer_num=6,
                              head_num=8, qkv_dim=16, ff_hidden_dim=512)
    model.load_state_dict(encoder_state(weight, 'model_state_dict', 'ELG_TSP'), strict=True)
    model.eval().requires_grad_(False)
    inputs = torch.rand(2, 7, 2, device='cpu')
    with torch.inference_mode():
        first, repeat = model(inputs), model(inputs)
    return dict(source=str(source), source_sha256=sha(source),
        checkpoint=str(weight), checkpoint_sha256=PINS['ELG_TSP'],
        parameters=sum(p.numel() for p in model.parameters()),
        shape=list(first.shape), finite=bool(first.isfinite().all()),
        same_input_exact_repeat=bool(torch.equal(first, repeat)),
        encoder_only=True, historical_cost_recipe_certified=False)


def isolated_matnet():
    source = EASY / 'neural_solvers/methods/matnet/matnet_encoder.py'
    backbone = EASY / 'neural_solvers/backbones/Transformer/attr_component.py'
    names = {'Normalization', 'FeedForward', 'reshape_by_heads'}
    body = [node for node in ast.parse(backbone.read_text()).body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in body} != names:
        raise ValueError('Native MatNet backbone definitions changed')
    scope = dict(torch=torch, nn=nn, F=F)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(backbone), 'exec'), scope)
    stub = types.ModuleType('EasyNCO.neural_solvers.backbones')
    for name in names:
        setattr(stub, name, scope[name])
    with patch.dict(sys.modules, {'EasyNCO.neural_solvers.backbones': stub}):
        native = load_file(source, '_coverage_matnet_native')
    return native.MatNetEncoder, source, backbone


def matnet_probe():
    factory, source, backbone = isolated_matnet()
    weight = EASY / 'pretrained/matnet_pretrain/matnet_atsp100.ckpt'
    model = factory()
    model.load_state_dict(encoder_state(weight, 'state_dict', 'MatNet_ATSP'), strict=True)
    model.eval().requires_grad_(False)
    distance = torch.rand(2, 7, 7, device='cpu')
    distance.diagonal(dim1=-2, dim2=-1).zero_()
    row = torch.zeros(2, 7, 256, device='cpu')
    col = torch.zeros_like(row)
    col[:, :, :7] = torch.eye(7, device='cpu')
    with torch.inference_mode():
        first = model(row, col, distance)
        repeat = model(row, col, distance)
        single = model(row[:1], col[:1], distance[:1])
    return dict(source=str(source), source_sha256=sha(source),
        backbone=str(backbone), backbone_sha256=sha(backbone),
        checkpoint=str(weight), checkpoint_sha256=PINS['MatNet_ATSP'],
        parameters=sum(p.numel() for p in model.parameters()),
        row_shape=list(first[0].shape), col_shape=list(first[1].shape),
        finite=all(bool(value.isfinite().all()) for value in first),
        same_fixed_role_exact_repeat=all(torch.equal(a, b) for a, b in zip(first, repeat)),
        single_batch_max_abs_error=max(float((a[:1] - b).abs().max())
                                       for a, b in zip(first, single)),
        roles='Identity onehot for synthetic CPU probe ONLY; no extraction recipe locked',
        encoder_only=True, original_package_optional_PyG_import_bypassed=True,
        native_definitions_rewritten=False, historical_cost_recipe_certified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        raise FileExistsError('Preserve the existing feasibility receipt')
    torch.set_num_threads(2)
    torch.manual_seed(2)
    with torch.device('cpu'):
        result = dict(ELG_TSP=elg_probe(), MatNet_ATSP=matnet_probe(),
            scope='Two synthetic instances of seven nodes; CPU static interface only',
            no_dataset_or_labels=True, no_decoder_or_routes=True,
            no_GPU_or_package_change=True, extraction_protocol_locked=False,
            scientific_experiment_started=False, goal_complete=False)
    if not result['ELG_TSP']['finite'] or not result['MatNet_ATSP']['finite']:
        raise FloatingPointError('Nonfinite synthetic source encoding')
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
