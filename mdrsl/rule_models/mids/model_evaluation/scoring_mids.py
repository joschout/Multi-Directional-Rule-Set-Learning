from typing import Dict, List

import pandas as pd

from mdrsl.data_handling.nan_data_filtering import remove_instances_with_nans_in_column

from mdrsl.evaluation.predictive_performance_metrics import ScoreInfo

from rule_models.mids.model_fitting.mids_abstract_base import MIDSAbstractBase

TargetAttr = str


def score_MIDS_on_its_targets_without_nans(
        mids: MIDSAbstractBase,
        test_dataframe: pd.DataFrame,
        filter_nans: bool
) -> Dict[TargetAttr, ScoreInfo]:
    target_to_acc: Dict[TargetAttr, ScoreInfo] = {}

    target_attrs: List[str] = mids.classifier.target_attrs
    if target_attrs is None:
        raise Exception("No target attributes were saved during MIDS optimization")

    target_attribute: TargetAttr
    for target_attribute in target_attrs:
        if filter_nans:
            filtered_test_dataframe = remove_instances_with_nans_in_column(test_dataframe, target_attribute)
        else:
            filtered_test_dataframe = test_dataframe
        predicted_values = mids.predict(filtered_test_dataframe, target_attribute)
        actual_values = filtered_test_dataframe[target_attribute].values
        score_info = ScoreInfo.score(y_true=actual_values, y_predicted=predicted_values)
        target_to_acc[target_attribute] = score_info
    return target_to_acc

