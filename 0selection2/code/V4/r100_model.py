"""Independent frozen binary TabICL contexts, then fixed mean-win aggregation.

Pair win probabilities are conditional on a strict TRAIN relation. They are
not full-pool winner probabilities. No coupling or cost-sensitive objective.
"""

from __future__ import annotations

import gc
import importlib
import time
from types import MethodType

import numpy as np

from .r97_data import _source_ids
from .r97_model import expand_probabilities, public_classifier
from .r100_data import legal_pairs, pair_support, validate_relation_binding


INFERENCE_RULE = 'argmax mean opponent win probability (expected-rank/Borda); ascending global-ID exact ties'
PROBABILITY_SEMANTICS = 'P[i,j]=P(i beats j given strict TRAIN relation); P[j,i]=1-P[i,j]; not full-pool winner probability'


def guard_gpu_managers(classifier, inference):
    """Instance-local guards only; never edit public packages or shared classes."""
    count = 0
    for component in classifier.model_.modules():
        manager = getattr(component, 'inference_mgr', None)
        if manager is None:
            continue
        original = manager._resolve_offload_mode

        def resolve(self, *args, _original=original, **kwargs):
            mode, reason = _original(*args, **kwargs)
            if mode != inference.OffloadMode.GPU:
                raise RuntimeError('R100 CPU/disk fallback rejected: ' + str(reason))
            return mode, reason

        def allocate(self, mode, shape, dtype):
            if mode != inference.OffloadMode.GPU or self.exe_device.type != 'cuda':
                raise RuntimeError('R100 output allocation must stay on assigned GPU')
            output = inference.torch.empty(shape, dtype=dtype, device=self.exe_device)
            return output, dict(mode=mode.name, shape=shape, dtype=str(dtype))

        manager._resolve_offload_mode = MethodType(resolve, manager)
        manager._allocate_output_buffer = MethodType(allocate, manager)
        count += 1
    if count != 3:
        raise RuntimeError('Pinned TabICL must expose exactly three inference managers')


def public_pair_classifier(config, device):
    classifier = public_classifier(config, device)
    inference = importlib.import_module('tabicl.model.inference')

    class BinaryClassifier:
        @property
        def classes_(self):
            return classifier.classes_

        def fit(self, x, y, *, kv_cache=False):
            if kv_cache is not False:
                raise ValueError('R100 requires kv_cache=False')
            classifier.fit(x, y, kv_cache=False)
            guard_gpu_managers(classifier, inference)
            return self

        def predict_proba(self, x):
            return classifier.predict_proba(x)

    return BinaryClassifier()


def mean_opponent_win(pair_probabilities):
    p = np.asarray(pair_probabilities)
    if (p.dtype != np.float64 or p.ndim != 3 or not len(p) or p.shape[1] < 2
            or p.shape[1] != p.shape[2] or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p > 1)
            or not np.array_equal(p + p.transpose(0, 2, 1), np.ones_like(p))
            or not np.all(np.diagonal(p, axis1=1, axis2=2) == .5)):
        raise ValueError('Require complementary FP64 pair probabilities with neutral .5 diagonal')
    # Exclude self-comparisons rather than treating the neutral diagonal as an opponent.
    mask = ~np.eye(p.shape[1], dtype=bool)
    return p[:, mask].reshape(len(p), p.shape[1], p.shape[1]-1).mean(axis=2)


def stable_mean_win_rank(mean_win, pool_ids):
    scores, ids = np.asarray(mean_win), np.asarray(pool_ids)
    legal_pairs(ids)
    if (scores.dtype != np.float64 or scores.ndim != 2 or not len(scores)
            or scores.shape[1] != len(ids) or not np.isfinite(scores).all()
            or np.any(scores < 0) or np.any(scores > 1)):
        raise ValueError('Need finite FP64 mean opponent win scores')
    order = np.argsort(ids, kind='stable')
    return order[np.argsort(-scores[:, order], axis=1, kind='stable')]


def formal_score_view(mean_win):
    """Formal CE/report log-softmax only; NEVER a winner-probability claim."""
    scores = np.asarray(mean_win)
    if (scores.dtype != np.float64 or scores.ndim != 2 or not len(scores)
            or scores.shape[1] < 2 or not np.isfinite(scores).all()):
        raise ValueError('Need a finite mean-win score matrix')
    shifted = scores - scores.max(axis=1, keepdims=True)
    logits = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return logits, np.exp(logits)


class FrozenPairwiseContext:
    def __init__(self, context, relations, classifier_factory, *, cleanup=None, call=None):
        validate_relation_binding(context, relations.metadata)
        self.context, self.relations = context, relations
        self.classifier_factory, self.cleanup, self.call = classifier_factory, cleanup, call
        self.pairs = legal_pairs(context.pool_ids)
        if (context.x.ndim != 2 or context.x.dtype != np.float32
                or context.x.shape != (len(context.source_ids), len(context.kept_columns))
                or not np.isfinite(context.x).all()
                or relations.labels.shape != (len(context.x), len(self.pairs))
                or pair_support(relations.labels, context.pool, context.pool_ids) != relations.metadata['pair_support']):
            raise ValueError('Context X and strict TRAIN labels/support disagree')
        _source_ids(context.source_ids)
        self.call_records = []

    def _call(self, function, stage, support):
        if self.call is None:
            started = time.monotonic()
            result = function()
            timing = dict(seconds=time.monotonic()-started)
        else:
            result, timing = self.call(function)
        self.call_records.append(dict(timing, stage=stage, pair_slot=support['pair_slot'],
            train_rows=len(self.context.x), strict_rows=support['strict_rows'],
            tie_rows=support['tie_rows'], label0_rows=support['label0_rows'], label1_rows=support['label1_rows']))
        return result

    def predict_features(self, query_features, query_source_ids):
        """Only reduced input features and base IDs; no query outcome API."""
        x = np.asarray(query_features)
        ids = _source_ids(query_source_ids)
        if (x.ndim != 2 or x.dtype.kind not in 'fiu' or not len(x)
                or x.shape != (len(ids), len(self.context.kept_columns)) or not np.isfinite(x).all()):
            raise ValueError('Query inputs must match the immutable R97 feature transform')
        # This MUST precede pair filtering, including all-tie and constant shortcuts.
        if np.intersect1d(ids, self.context.source_ids).size:
            raise ValueError('Query base ID would occur in its own context (full TRAIN ID check)')
        with np.errstate(over='ignore', invalid='ignore'):
            x = np.asarray(x, dtype=np.float32, order='C')
        if not np.isfinite(x).all():
            raise ValueError('Query features overflowed FP32')
        k = len(self.context.pool)
        probabilities = np.full((len(x), k, k), .5, dtype=np.float64)
        self.call_records = []
        for support in self.relations.metadata['pair_support']:
            slot, i, j = (support[key] for key in ('pair_slot', 'first_slot', 'second_slot'))
            constant = support['constant_probability_first']
            if constant is not None:
                probabilities[:, i, j], probabilities[:, j, i] = constant, 1.-constant
                self.call_records.append(dict(stage='constant_pair', seconds=0., **support))
                continue
            labels = self.relations.labels[:, slot]
            valid = labels >= 0
            classifier = None
            try:
                classifier = self.classifier_factory()
                self._call(lambda: classifier.fit(np.array(self.context.x[valid], copy=True),
                    np.array(labels[valid], dtype=np.int64, copy=True), kv_cache=False), 'fit_pair', support)
                result = self._call(lambda: classifier.predict_proba(np.array(x, copy=True)), 'predict_pair', support)
                classes = np.asarray(classifier.classes_)
                if set(classes.tolist()) != {0, 1}:
                    raise ValueError('Nonconstant pair must retain both strict binary classes')
                result = expand_probabilities(result, classes, 2)
                if result.shape != (len(x), 2) or np.any(result > 1):
                    raise ValueError('Binary pair output must retain every query row')
                probabilities[:, i, j] = result[:, 0]
                probabilities[:, j, i] = 1.-result[:, 0]
            finally:
                del classifier
                gc.collect()
                if self.cleanup is not None:
                    self.cleanup()
        mean_opponent_win(probabilities)
        return probabilities

    def predict(self, query_features, query_source_ids):
        scores = mean_opponent_win(self.predict_features(query_features, query_source_ids))
        return stable_mean_win_rank(scores, self.context.pool_ids)[:, 0]
