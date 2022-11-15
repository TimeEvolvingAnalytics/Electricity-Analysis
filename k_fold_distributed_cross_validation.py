from random import randint
import copy
import numpy as np
import pandas as pd
from timeit import default_timer as timer
import tracemalloc
from adaptive_xgboost import AdaptiveXGBoostClassifier
from holt_winters import HoltWinters
from river.time_series import Forecaster


class KFoldDistributedCrossValidation:
    """ K Fold Distributed Cross Validation
        Class for implementing k fold distributed cross validation, where k models are trained and tested in parallel.
        In the k-fold distributed cross-validation, each instance is used for testing one randomly selected model and
        for training all the others. By applying the prequential evaluation mode, all models are tested
        before the training phase.
    """

    def __init__(self, metrics, model, ensemble_size=10):

        self.base_learners = {}
        self.ensemble_size = ensemble_size
        for id in range(ensemble_size):
            if isinstance(model, HoltWinters):
                b_l = HoltWinters(alpha=model.alpha)
            else:
                b_l = copy.deepcopy(model)
            self.base_learners[id] = {
                'model': b_l,
                'metric1': copy.deepcopy(metrics[0]),
                'metric2': copy.deepcopy(metrics[1]),
                'memory': [],
                'processing_time': 0.0
            }
        self._random_state = np.random.RandomState(42)
        self._peak = 0

    def reset_metrics(self,metrics):
        for key, b_l in self.base_learners.items():
            b_l['metric1'] = copy.deepcopy(metrics[0])
            b_l['metric2'] = copy.deepcopy(metrics[1])


    def evaluate(self, stream):
        tracemalloc.start()
        for x, y in stream:
            self._evaluate(x, y)
        _, self._peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    def _evaluate(self, X, Y):
        id_not_train = randint(0, self.ensemble_size - 1)
        for key, b_l in self.base_learners.items():
            # test each instance
            t_1 = timer()
            if isinstance(b_l['model'], AdaptiveXGBoostClassifier):
                val_x = pd.DataFrame([X]).to_numpy()
                prediction = b_l['model'].predict(val_x)[0]
            elif isinstance(b_l['model'], Forecaster):
                try:
                    prediction = b_l['model'].forecast(horizon=1)[0]
                    if prediction >= 0.5:
                        prediction = 1
                    else:
                        prediction = 0
                except IndexError:
                    prediction = 0
            else:
                prediction = b_l['model'].predict_one(X)
            t_2 = timer()
            b_l['processing_time'] += t_2 - t_1
            if prediction is not None:
                b_l['metric1'].update(Y, prediction)
                b_l['metric2'].update(Y, prediction)
            
            # cross-validation: train all instances but one
            if key != id_not_train or self.ensemble_size == 1:
                t_1 = timer()
                if isinstance(b_l['model'], AdaptiveXGBoostClassifier):
                    val_x = pd.DataFrame([X]).to_numpy()
                    val_y = np.array([Y])
                    b_l['model'].partial_fit(val_x, val_y)
                elif isinstance(b_l['model'], Forecaster):
                    b_l['model'].learn_one(x=X, y=Y)
                else:
                    b_l['model'].learn_one(X, Y)
                t_2 = timer()
            b_l['processing_time'] += t_2 - t_1

    def get_measurements(self):
        metrics1 = []
        metrics2 = []
        memory = self._peak / self.ensemble_size
        time = []
        for _, b_l in self.base_learners.items():
            metrics1.append(b_l['metric1'])
            metrics2.append(b_l['metric2'])
            time.append(np.sum(b_l['processing_time']))
        return metrics1, metrics2, memory, time
