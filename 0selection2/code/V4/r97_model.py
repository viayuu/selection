"""Frozen tabular ICL adapter with explicit historical-context isolation.

Only ``fit`` consumes historical TRAIN labels. ``predict_features`` accepts
input statistics and base-instance identities, never query supervision.
"""

import importlib
from pathlib import Path
import sys

import numpy as np


def expand_probabilities(probabilities, classes, candidate_count):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    classes = np.asarray(classes)
    if (probabilities.ndim != 2 or classes.ndim != 1
            or probabilities.shape[1] != len(classes)
            or not np.issubdtype(classes.dtype, np.integer)
            or len(set(classes.tolist())) != len(classes)
            or np.any(classes < 0) or np.any(classes >= candidate_count)
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(1), 1., rtol=1e-5, atol=1e-7)):
        raise ValueError('Classifier probabilities/classes do not match legal local IDs')
    result = np.zeros((len(probabilities), candidate_count), dtype=np.float64)
    result[:, classes] = probabilities
    return result


def public_classifier(config, device):
    """Late import from the isolated, pinned runtime, not a package mutation."""
    directory = Path(config['library_root']).resolve()
    if not directory.is_dir():
        raise FileNotFoundError('Prepare the isolated TabICL runtime first')
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    module = importlib.import_module('tabicl')
    if not Path(module.__file__).resolve().is_relative_to(directory):
        raise RuntimeError('TabICL was imported outside the locked runtime')
    parameters = dict(config['classifier_parameters'])
    if parameters.get('device') not in (None, device):
        raise ValueError('Classifier must use the launcher-assigned device')
    parameters['device'] = device
    return module.TabICLClassifier(**parameters)


class FrozenContextClassifier:
    def __init__(self, context, classifier_factory):
        self.context = context
        self.classifier_factory = classifier_factory
        self.classifier = None
        self.constant_class = None
        if (len(context.pool) != len(context.pool_ids)
                or list(context.pool_ids) != sorted(set(context.pool_ids))
                or len(context.x) != len(context.y)
                or len(context.y) != len(context.source_ids)
                or context.x.ndim != 2 or context.y.ndim != 1
                or not np.issubdtype(context.y.dtype, np.integer)
                or not np.isfinite(context.x).all()
                or np.any(context.y < 0) or np.any(context.y >= len(context.pool))):
            raise ValueError('Historical context and legal candidate order disagree')

    def fit(self):
        classes = np.unique(self.context.y)
        if len(classes) == 0:
            raise ValueError('Historical TRAIN context is empty')
        if len(classes) == 1:
            self.constant_class = int(classes[0])
        else:
            self.classifier = self.classifier_factory()
            self.classifier.fit(np.array(self.context.x, copy=True), np.array(self.context.y, copy=True))
        return self

    def predict_features(self, query_features, query_source_ids):
        if self.classifier is None and self.constant_class is None:
            raise RuntimeError('Fit historical TRAIN context before predicting')
        query_features = np.asarray(query_features)
        query_source_ids = np.asarray(query_source_ids)
        if (query_features.ndim != 2 or query_features.shape[1] != len(self.context.kept_columns)
                or query_source_ids.shape != (len(query_features),)
                or not np.isfinite(query_features).all()):
            raise ValueError('Query inputs must match the TRAIN-fitted column transform')
        if np.intersect1d(query_source_ids, self.context.source_ids).size:
            raise ValueError('Query/base-instance label would be present in its own context')
        if self.constant_class is not None:
            result = np.zeros((len(query_features), len(self.context.pool)), dtype=np.float64)
            result[:, self.constant_class] = 1.
            return result
        return expand_probabilities(self.classifier.predict_proba(query_features),
                                    self.classifier.classes_, len(self.context.pool))

    def predict(self, query_features, query_source_ids):
        # Columns follow increasing global IDs, so numpy's first tie is fixed.
        return self.predict_features(query_features, query_source_ids).argmax(1)
