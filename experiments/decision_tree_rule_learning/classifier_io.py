import gzip
import pickle
from typing import Union

import jsonpickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from mdrsl.rule_models.ids.ids_classifier import IDSClassifier
from experiments.file_naming.single_target_classifier_indicator import SingleTargetClassifierIndicator

from pyarc.algorithms.classifier import Classifier

ClassifierT = Union[Classifier, IDSClassifier, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]


def store_classifier(classifier_indicator: SingleTargetClassifierIndicator, abs_file_name: str,
                     classifier: ClassifierT) -> None:
    if (classifier_indicator == SingleTargetClassifierIndicator.cba
            or classifier_indicator == SingleTargetClassifierIndicator.ids):
        frozen = jsonpickle.encode(classifier)
        with gzip.open(abs_file_name, 'wt') as ofile:
            ofile.write(frozen)
    elif (classifier_indicator == SingleTargetClassifierIndicator.logistic_regression
          or classifier_indicator == SingleTargetClassifierIndicator.decision_tree
          or classifier_indicator == SingleTargetClassifierIndicator.random_forest):
        with open(abs_file_name, 'wb') as ofile:
            pickle.dump(classifier, ofile)
    else:
        raise Exception(f"SingleTargetClassifierIndicator {classifier_indicator} is unaccounted for")


def load_classifier(classifier_indicator: SingleTargetClassifierIndicator, abs_file_name: str) -> ClassifierT:
    if (classifier_indicator == SingleTargetClassifierIndicator.cba
            or classifier_indicator == SingleTargetClassifierIndicator.ids):
        classifier: Union[Classifier, IDSClassifier]
        with gzip.open(abs_file_name, 'rt') as ifile:
            file_contents = ifile.read()
            classifier = jsonpickle.decode(file_contents)
        return classifier
    elif (classifier_indicator == SingleTargetClassifierIndicator.logistic_regression
          or classifier_indicator == SingleTargetClassifierIndicator.decision_tree
          or classifier_indicator == SingleTargetClassifierIndicator.random_forest):
        classifier: Union[LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]
        with open(abs_file_name, 'rb') as ifile:
            classifier = pickle.load(ifile)
        return classifier
    else:
        raise Exception(f"SingleTargetClassifierIndicator {classifier_indicator} is unaccounted for")
