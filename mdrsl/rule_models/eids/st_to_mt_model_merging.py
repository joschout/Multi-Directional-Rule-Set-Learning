from typing import Dict, List, Set, Optional, KeysView

import pandas as pd

from mdrsl.data_handling.nan_data_filtering import remove_instances_with_nans_in_column

from mdrsl.evaluation.predictive_performance_metrics import ScoreInfo, average_score_info

from mdrsl.rule_models.mids.model_evaluation.mids_interpretability_metrics import MIDSInterpretabilityStatistics, \
    MIDSInterpretabilityStatisticsCalculator

from mdrsl.rule_models.mids.mids_classifier import MIDSClassifier
from mdrsl.rule_models.mids.mids_rule import MIDSRule
from mdrsl.rule_models.mids.mids_ruleset import MIDSRuleSet

TargetAttr = str


class AggregateStatistics:
    def __init__(self):
        self.total_ground_set_size: int = 0
        self.total_n_rules: int = 0
        self.total_run_time: float = 0.0
        self.total_n_f_calls: int = 0

    def update_statistics(self,
                          st_ground_set_size: int,
                          st_n_rules: int,
                          st_run_time: float,
                          st_n_f_calls: int
                          ) -> None:
        self.total_ground_set_size += st_ground_set_size
        self.total_n_rules += st_n_rules
        self.total_run_time += st_run_time
        self.total_n_f_calls += st_n_f_calls


class MergedSTMIDSClassifier:

    def __init__(self):
        self.target_attribute_to_mids_classifier: Dict[TargetAttr, MIDSClassifier] = {}

        self._aggregate_statistics = AggregateStatistics()
        # self.are_statistics_finalized = False
        self._interpretability_statistics: Optional[MIDSInterpretabilityStatistics] = None
        self._target_to_score_info_map: Optional[Dict[TargetAttr, ScoreInfo]] = None

        self._total_rule_generation_time: float = 0

        self.avg_score_info: Optional[ScoreInfo] = None

    def get_target_attributes(self):
        return self.target_attribute_to_mids_classifier.keys()

    def add_single_target_model(self, mids_classifier: MIDSClassifier):
        st_target_attributes: List[TargetAttr] = mids_classifier.target_attrs
        n_targets = len(st_target_attributes)
        if n_targets != 1:
            raise Exception(
                f"Expected to receive only single target models, but received model predicting {n_targets} targets")
        target_attr = st_target_attributes[0]

        # check if the target is already predicted
        if self.target_attribute_to_mids_classifier.get(target_attr, None) is None:
            self.target_attribute_to_mids_classifier[target_attr] = mids_classifier
            self._aggregate_statistics.update_statistics(
                st_ground_set_size=mids_classifier.optimization_meta_data.ground_set_size,
                st_n_rules=len(mids_classifier.rules),
                st_run_time=mids_classifier.optimization_meta_data.run_time_collector.sum,
                st_n_f_calls=mids_classifier.optimization_meta_data.run_time_collector.count
            )
        else:
            raise Exception(
                f'Received single target model predicting {target_attr}, for which their already is a model')

    def add_rule_generation_time(self, rule_generation_time: float):
        if rule_generation_time < 0:
            raise  Exception("rule generation time cannot be negative")
        self._total_rule_generation_time += rule_generation_time

    def get_total_rule_generation_time(self)-> float:
        if self._total_rule_generation_time <= 0.0:
            raise Exception("Rule generation time unknown")
        else:
            return self._total_rule_generation_time

    def predict(self, dataframe: pd.DataFrame, target_attribute: str):
        return self.target_attribute_to_mids_classifier[target_attribute].predict(dataframe, target_attribute)

    def get_total_rule_set(self) -> MIDSRuleSet:

        total_rule_set: Set[MIDSRule] = set()
        for target_attribute in self.target_attribute_to_mids_classifier.keys():
            for rule in self.target_attribute_to_mids_classifier[target_attribute].rules:
                total_rule_set.add(rule)

        total_mids_rule_set = MIDSRuleSet(total_rule_set)
        return total_mids_rule_set

    def get_total_ground_set_size(self):
        return self._aggregate_statistics.total_ground_set_size

    def get_total_nb_of_rules(self):
        return self._aggregate_statistics.total_n_rules

    def get_total_nb_of_f_calls(self):
        return self._aggregate_statistics.total_n_f_calls

    def get_total_run_time(self):
        return self._aggregate_statistics.total_run_time

    def calculate_ruleset_interpretability_statistics(self,
                                                test_dataframe: pd.DataFrame,
                                                target_attributes: List[TargetAttr]
                                                ) -> MIDSInterpretabilityStatistics:
        total_mids_rule_set: MIDSRuleSet = self.get_total_rule_set()

        self._interpretability_statistics = MIDSInterpretabilityStatisticsCalculator.calculate_ruleset_statistics(
            total_mids_rule_set, test_dataframe, target_attributes
        )
        return self._interpretability_statistics

    def get_interpretability_statistics(self) -> MIDSInterpretabilityStatistics:
        if self._interpretability_statistics is None:
            raise Exception("InterpretabilityStatistics not initialized")
        else:
            return self._interpretability_statistics

    def calculate_score_info(self, test_dataframe: pd.DataFrame, filter_nans: bool = True) -> Dict[TargetAttr, ScoreInfo]:
        self._target_to_score_info_map: Dict[TargetAttr, ScoreInfo] = score_MergedSTMIDS_on_its_targets_without_nans(
            self, test_dataframe=test_dataframe, filter_nans=filter_nans
        )
        self.avg_score_info = average_score_info(list(self._target_to_score_info_map.values()))
        return self._target_to_score_info_map

    def get_target_to_score_info_map(self) -> Dict[TargetAttr, ScoreInfo]:
        if self._target_to_score_info_map is None:
            raise Exception()
        else:
            return self._target_to_score_info_map


def score_MergedSTMIDS_on_its_targets_without_nans(
        merged_st_clf: MergedSTMIDSClassifier,
        test_dataframe: pd.DataFrame,
        filter_nans: bool
) -> Dict[TargetAttr, ScoreInfo]:
    target_to_acc: Dict[TargetAttr, ScoreInfo] = {}

    target_attrs: KeysView[str] = merged_st_clf.get_target_attributes()
    if target_attrs is None:
        raise Exception("No target attributes were saved during MIDS optimization")

    target_attribute: TargetAttr
    for target_attribute in target_attrs:
        if filter_nans:
            filtered_test_dataframe = remove_instances_with_nans_in_column(test_dataframe, target_attribute)
        else:
            filtered_test_dataframe = test_dataframe
        predicted_values = merged_st_clf.predict(filtered_test_dataframe, target_attribute)
        actual_values = filtered_test_dataframe[target_attribute].values
        score_info = ScoreInfo.score(y_true=actual_values, y_predicted=predicted_values)
        target_to_acc[target_attribute] = score_info
    return target_to_acc
