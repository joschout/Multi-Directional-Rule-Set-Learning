import math
import statistics
import warnings
from collections import Counter
from enum import Enum, auto
from typing import List, Iterable, Dict, Optional, Set, Tuple

import pandas as pd

from mdrsl.data_handling.type_checking_dataframe import type_check_dataframe
from mdrsl.rule_models.mids.mids_rule import does_rule_fire_for_instance, MIDSRule

AttrValue = object


class RuleCombinator:
    def predict(self, rules: Iterable[MIDSRule], dataframe: pd.DataFrame, target_attribute: str, default_value=None) -> List[object]:
        """

        Predict a value for the given target attribute for each row,
        """
        type_check_dataframe(dataframe)

        predicted_class_per_row = []
        for _, instance_row in dataframe.iterrows():
            predicted_class = self.get_combined_prediction(rules, instance_row, target_attribute, default_value)
            if predicted_class is not None:
                predicted_class_per_row.append(predicted_class)
            else:
                predicted_class_per_row.append(default_value)

        return predicted_class_per_row

    def get_combined_prediction(self, rules: Iterable[MIDSRule], instance_row, target_attribute, default_value):
        """
        Returns the prediction for a single instance,
            by combining the predictions of the different rules.

        :param rules:
        :param instance_row:
        :param target_attribute:
        :param default_value:
        :return:
        """
        raise NotImplementedError("abstract method")


class F1BasedRuleCombinator(RuleCombinator):

    def get_combined_prediction(self, rules, instance_row, target_attribute, default_value):
        """
        Returns the prediction for a single instance,
            by combining the predictions of the different rules.

        :param rules:
        :param instance_row:
        :param target_attribute:
        :param default_value:
        :return:
        """
        warnings.warn(
            "F1 score is probably incorrect, as its current calculation still assumes a predetermined single target "
            "attribute.")

        rule_list = list(rules)
        rule: MIDSRule
        f1_sorted_rules = sorted(rule_list, key=lambda rule: rule.calc_f1())

        for rule in f1_sorted_rules:
            if rule.head_contains_target_attribute(target_attribute):
                if does_rule_fire_for_instance(rule, instance_row):
                    predicted_value = rule.get_predicted_value_for(target_attribute)
                    # note: only uses the first rule it can find,
                    # as it is assumed the rules are sorted by f1 score
                    return predicted_value
        return default_value


class MajorityVotingRuleCombinator(RuleCombinator):
    def get_combined_prediction(self, rules, instance_row, target_attribute, default_value):
        """
        Returns the prediction for a single instance,
            by combining the predictions of the different rules.

        Here, each rule that fires for an instance votes for the value that the rule predicts.
        The value with the most votes is used as the predicted value.

        :param rules:
        :param instance_row:
        :param target_attribute:
        :param default_value:
        :return:
        """
        possible_values = []
        for rule in rules:
            if rule.head_contains_target_attribute(target_attribute):
                if does_rule_fire_for_instance(rule, instance_row):
                    predicted_value = rule.get_predicted_value_for(target_attribute)
                    possible_values.append(predicted_value)

        if len(possible_values) > 0:
            try:
                return statistics.mode(possible_values)
            except statistics.StatisticsError as err:
                # multiple modes
                value_counter = Counter(possible_values)
                # warnings.warn(str(err) + "\nWe pick one of them. Should be done more explicitly")
                one_of_the_modes = max(value_counter, key=value_counter.get)
                return one_of_the_modes
        else:
            return default_value


class WeightedVotingRuleCombinator(RuleCombinator):

    def get_combined_prediction(self, rules, instance_row, target_attribute, default_value):
        """
        NOTE: REQUIRES THE CARS OF MIDS RULES TO HAVE A CONFIDENCE.

        Returns the prediction for a single instance,
            by combining the predictions of the different rules.

        Here, each rule that fires for an instance votes for the value that the rule predicts.
        Instead of letting each vote count as 1, each rule has as weight is confidence in the training set.

        Value that gets the highest sum-of-confidences from the voting rules, is returned as predicted value.

        :param rules:
        :param instance_row:
        :param target_attribute:
        :param default_value:
        :return:
        """
        possible_values: Set[AttrValue] = set()
        rule_confidences: List[Tuple[AttrValue, float]] = []

        for rule in rules:
            if rule.head_contains_target_attribute(target_attribute):
                if does_rule_fire_for_instance(rule, instance_row):

                    rule_confidence: float = rule.car.confidence
                    if math.isnan(rule_confidence):
                        rule_confidence = 0
                    if not (0 <= rule_confidence <= 1.0):
                        raise Exception(f"A {self.__class__} can only be used if all rules have a confidence in [0, 1]."
                                        f"There is a rule with confidence {rule_confidence}: {rule}")
                    else:
                        predicted_value = rule.get_predicted_value_for(target_attribute)
                        possible_values.add(predicted_value)
                        rule_confidences.append((predicted_value, rule_confidence))

        if len(possible_values) > 0:
            value_to_sum_of_confidences_map: Dict[AttrValue, float] = {}
            for value, confidence in rule_confidences:
                opt_tmp_confidence: Optional[float] = value_to_sum_of_confidences_map.get(value, None)
                if opt_tmp_confidence is None:
                    value_to_sum_of_confidences_map[value] = confidence
                else:
                    value_to_sum_of_confidences_map[value] = opt_tmp_confidence + confidence
            value_with_max_confidence_vote = max(value_to_sum_of_confidences_map.keys(),
                                                 key=(lambda k: value_to_sum_of_confidences_map[k]))
            return value_with_max_confidence_vote
        else:
            return default_value


class RuleCombiningStrategy(Enum):
    """
     How to combine the rules for prediction.
        a. Sort the rules according to their f1-score with respect to the target attribute,
           and use the first rule predicting the target that matches the test instance
        b. use a majority vote for the rules that match the instance
    """
    F1_SCORE = auto()
    MAJORITY_VOTE = auto()
    WEIGHTED_VOTE = auto()

    def create(self) -> 'RuleCombinator':
        if self == RuleCombiningStrategy.F1_SCORE:
            return F1BasedRuleCombinator()
        elif self == RuleCombiningStrategy.MAJORITY_VOTE:
            return MajorityVotingRuleCombinator()
        elif self == RuleCombiningStrategy.WEIGHTED_VOTE:
            return WeightedVotingRuleCombinator()
        else:
            raise Exception("SHOULD NEVER REACH THIS")
