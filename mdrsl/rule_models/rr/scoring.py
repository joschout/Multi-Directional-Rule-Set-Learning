from typing import Dict, List

import pandas as pd

from mdrsl.data_handling.nan_data_filtering import remove_instances_with_nans_in_column

from mdrsl.evaluation.predictive_performance_metrics import ScoreInfo

from rule_models.rr.rr_rule_set_learner import GreedyRoundRobinTargetRuleClassifier

TargetAttr = str


def score_mt_clf_on_its_targets_without_nans(
        mt_clf: GreedyRoundRobinTargetRuleClassifier,
        test_dataframe: pd.DataFrame,
        filter_nans: bool
) -> Dict[TargetAttr, ScoreInfo]:
    target_to_acc: Dict[TargetAttr, ScoreInfo] = {}

    target_attrs: List[str] = mt_clf.target_attributes
    if target_attrs is None:
        raise Exception("No target attributes were saved during rule selection")

    target_attribute: TargetAttr
    for target_attribute in target_attrs:
        if filter_nans:
            filtered_test_dataframe = remove_instances_with_nans_in_column(test_dataframe, target_attribute)
        else:
            filtered_test_dataframe = test_dataframe
        predicted_values = mt_clf.predict(filtered_test_dataframe, target_attribute)
        actual_values = filtered_test_dataframe[target_attribute].values
        score_info = ScoreInfo.score(y_true=actual_values, y_predicted=predicted_values)
        target_to_acc[target_attribute] = score_info
    return target_to_acc
