import abc
import typing

from river import base

class PredictedTemporallyAugmentedClassifier(base.Classifier):
    """Temporally augmented classifier adding also predicting labels."""

    def __init__(
        self, 
        model: base.Classifier, 
        n_old_labels: int = -1,
        n_old_predicted_labels: int = -1
    ):
        self.n_old_labels = n_old_labels
        self.n_old_predicted_labels = n_old_predicted_labels
        self.model = model
        self.old_labels = []
        self.old_predicted_labels = []

    def add_features(self, x: dict):
        if self.n_old_labels != -1:
            num_labels = len(self.old_labels)
            last_label = 0
            for i in range(num_labels):
                x['label_' + str(i)] = self.old_labels[i]
                last_label = i + 1
            if num_labels < self.n_old_labels:
                for i in range(self.n_old_labels-num_labels):
                    x['label_' + str(i+last_label)] = 0 
        if self.n_old_predicted_labels != -1:
            num_predicted_labels = len(self.old_predicted_labels)
            last_predicted_label = 0
            for i in range(num_predicted_labels):
                x['predicted_label_' + str(i)] = self.old_predicted_labels[i]
                last_predicted_label = i + 1
            if num_predicted_labels < self.n_old_predicted_labels:
                for i in range(self.n_old_predicted_labels-num_predicted_labels):
                    x['predicted_label_' + str(i+last_predicted_label)] = 0  
      
        return x

    def add_labels(self, y: base.typing.ClfTarget, y_pred: base.typing.ClfTarget):
        if self.n_old_labels != -1:
            num_labels = len(self.old_labels)
            if num_labels == self.n_old_labels:
                self.old_labels.pop(0)
            self.old_labels.append(y)
        if self.n_old_predicted_labels != -1:
            num_predicted_labels = len(self.old_predicted_labels)
            if num_predicted_labels == self.n_old_predicted_labels:
                self.old_predicted_labels.pop(0)
            self.old_predicted_labels.append(y_pred)

            
    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> "Classifier":
        """Update the model with a set of features `x` and a label `y`.

        Parameters
        ----------
        x
            A dictionary of features.
        y
            A label.
        kwargs
            Some models might allow/require providing extra parameters, such as sample weights.

        Returns
        -------
        self

        """
        augmented_x = self.add_features(x)
        self.model.learn_one(augmented_x,y)
        self.add_labels(y,self.model.predict_one(augmented_x))

        return self

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        """Predict the probability of each label for a dictionary of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        A dictionary that associates a probability which each label.

        """

        # Some classifiers don't have the ability to output probabilities, and instead only
        # predict labels directly. Therefore, we cannot impose predict_proba_one as an abstract
        # method that each classifier has to implement. Instead, we raise an exception to indicate
        # that a classifier does not support predict_proba_one.
        raise NotImplementedError

    def predict_one(self, x: dict) -> base.typing.ClfTarget:
        """Predict the label of a set of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        The predicted label.

        """

        augmented_x = self.add_features(x)
        y_pred = self.model.predict_proba_one(augmented_x)
        if y_pred:
            return max(y_pred, key=y_pred.get)
        return None

    @property
    def _multiclass(self):
        return False

    @property
    def _supervised(self):
        return True