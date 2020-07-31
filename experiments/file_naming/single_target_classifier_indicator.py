from enum import Enum


class SingleTargetClassifierIndicator(Enum):
    ids = 'ids'
    cba = 'cba'
    logistic_regression = 'logistic_regression'
    decision_tree = 'decision_tree'
    random_forest = 'random_forest'