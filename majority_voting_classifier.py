import abc
import typing
from collections import Counter

from river import base


class MajorityVotingClassifier(base.Classifier):
    """Majority voting classifier."""

    def __init__(
        self,         
        n_old_labels: int = 2        
    ):
        self.n_old_labels = n_old_labels        
        self.old_labels = []  
        self.classes = set()  

    def add_label(self, y: base.typing.ClfTarget):
        num_labels = len(self.old_labels)
        if num_labels == self.n_old_labels:
            self.old_labels.pop(0)
        self.old_labels.append(y)

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
        
        self.add_label(y)

        return self

    def most_frequent(self):
        if len(self.old_labels) == 0:
            most_frequent = 0
        else:
            counter=Counter(self.old_labels)
            commons = counter.most_common()        
            if len(commons) > 1:
                if commons[0][1] == commons[1][1]:
                    most_frequent = commons[1][0]
                else:
                    most_frequent = commons[0][0]
            else: 
                most_frequent = commons[0][0]

        return most_frequent

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
        probas = {c: 0 for c in self.classes}
        probas[self.most_frequent()] = 1
        
        return probas



    def predict_one(self, x: dict) -> base.typing.ClfTarget:
        """Predict the label of a set of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        The most frequent old label.
        In case of a tie, return the last saved

        """

        return self.most_frequent()

    @property
    def _multiclass(self):
        return False

    @property
    def _supervised(self):
        return True