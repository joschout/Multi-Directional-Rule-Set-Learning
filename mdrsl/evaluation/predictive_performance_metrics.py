import functools
from typing import Collection

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

micro_avged_f1_score = functools.partial(f1_score, average='micro')


class ScoreInfo:
    metrics = [accuracy_score, balanced_accuracy_score, micro_avged_f1_score]

    def __init__(self, acc: float, balanced_acc: float, micro_avged_f1: float):
        self.acc: float = acc
        self.balanced_acc: float = balanced_acc
        self.micro_avged_f1: float = micro_avged_f1

    @staticmethod
    def score(y_true, y_predicted) -> 'ScoreInfo':
        try:
            acc: float = accuracy_score(y_true, y_predicted)
        except ValueError as err:
            print(f"{len(y_true)} vs {len(y_predicted)}")
            print(y_predicted)
            print(y_true.values)
            raise err
        balanced_acc: float = balanced_accuracy_score(y_true, y_predicted)
        micro_avged_f1_score_value: float = f1_score(y_true, y_predicted, average='micro')
        return ScoreInfo(acc=acc, balanced_acc=balanced_acc, micro_avged_f1=micro_avged_f1_score_value)

    def to_str(self, indentation: str = "") -> str:
        output_str = (
                indentation + "acc: " + str(self.acc) + "\n"
                + indentation + "balanced acc: " + str(self.balanced_acc) + "\n"
                + indentation + "micro averaged f1: " + str(self.micro_avged_f1) + "\n"
        )
        return output_str

    def relative_to(self, score_info_to_compare_with: 'ScoreInfo'):
        relative_acc = score_info_to_compare_with.acc - self.acc
        relative_balanced_acc = score_info_to_compare_with.balanced_acc - self.balanced_acc
        relative_micro_avged_f1 = score_info_to_compare_with.micro_avged_f1 - self.micro_avged_f1
        return ScoreInfo(acc=relative_acc, balanced_acc=relative_balanced_acc, micro_avged_f1=relative_micro_avged_f1)


def average_score_info(score_infos: Collection[ScoreInfo]) -> ScoreInfo:
    n_score_infos = len(score_infos)
    avg_acc: float = 0.0
    avg_balanced_acc: float = 0.0
    avg_micro_avged_f1: float = 0.0
    for score_info in score_infos:
        avg_acc += score_info.acc
        avg_balanced_acc += score_info.balanced_acc
        avg_micro_avged_f1 += score_info.micro_avged_f1
    avg_acc = avg_acc / n_score_infos
    avg_balanced_acc = avg_balanced_acc / n_score_infos
    avg_micro_avged_f1 = avg_micro_avged_f1 / n_score_infos
    return ScoreInfo(acc=avg_acc, balanced_acc=avg_balanced_acc, micro_avged_f1=avg_micro_avged_f1)
