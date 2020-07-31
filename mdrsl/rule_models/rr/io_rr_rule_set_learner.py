import gzip

import jsonpickle

from rule_models.rr.rr_rule_set_learner import GreedyRoundRobinTargetRuleClassifier


def store_greedy_naive_classifier(greedy_naive_classifier_abs_file_name: str,
                                  greedy_naive_clf: GreedyRoundRobinTargetRuleClassifier) -> None:
    frozen = jsonpickle.encode(greedy_naive_clf)
    with gzip.open(greedy_naive_classifier_abs_file_name, 'wt') as ofile:
        ofile.write(frozen)


def load_greedy_naive_classifier(greedy_naive_classifier_abs_file_name: str) -> GreedyRoundRobinTargetRuleClassifier:
    greedy_naive_clf: GreedyRoundRobinTargetRuleClassifier
    with gzip.open(greedy_naive_classifier_abs_file_name, 'rt') as ifile:
        file_contents = ifile.read()
        greedy_naive_clf = jsonpickle.decode(file_contents)
    return greedy_naive_clf
