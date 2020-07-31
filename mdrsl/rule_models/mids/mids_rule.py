import warnings
from typing import Set, List, Dict, KeysView

import pandas as pd
from scipy import stats as st

from mdrsl.data_structures.rules.multi_target_class_association_rule import MCAR
from mdrsl.data_structures.rules.rule_part import Antecedent, Consequent


class MIDSRule:

    # uncached_cover_checker = CoverChecker()

    def __init__(self, class_association_rule: MCAR):
        self.car = class_association_rule  # type: MCAR
        # self.cover_cache = None  # type: Optional[RuleCoverCache]

    def calc_f1(self):
        """
        F1 score is the harmonic mean of the precision and recall.
        :return:
        """
        if self.car.support == 0 or self.car.confidence == 0:
            return 0
        warnings.warn("possibly not correct for multi-target prediction")

        return st.hmean([self.car.support, self.car.confidence])

    def __repr__(self):
        f1 = self.calc_f1()
        return "MIDS-" + repr(self.car) + " f1: {}".format(f1)

    def __len__(self):
        return len(self.car)

    def __hash__(self):
        return hash(self.car)

    def __eq__(self, other: 'MIDSRule'):
        return (isinstance(other, type(self))
                and self.car == other.car)

    def get_rule_id(self) -> int:
        return self.car.id

    def head_contains_target_attribute(self, target_attribute: str):
        return target_attribute in self.get_target_attributes()

    def body_contains_descriptive_attribute(self, descriptive_attribute: str):
        return descriptive_attribute in self.get_descriptive_attributes()

    def get_predicted_value_for(self, target_attribute):
        """
        WARNING: unchecked!
        """
        return self.get_consequent().get_predicted_value(target_attribute)

    def __gt__(self, other):
        """
        precedence operator. Determines if this rule
        has higher precedence. Rules are sorted according
        to their f1 score.
        """

        f1_score_self = self.calc_f1()
        f1_score_other = other.calc_f1()
        return f1_score_self > f1_score_other

    def __lt__(self, other):
        """
        rule precedence operator
        """
        return not self > other

    def get_antecedent(self) -> Antecedent:
        return self.car.antecedent

    def get_consequent(self) -> Consequent:
        return self.car.consequent

    def get_target_attributes(self) -> KeysView[str]:
        return self.get_consequent().get_attributes()

    def get_descriptive_attributes(self) -> KeysView[str]:
        return self.get_antecedent().get_attributes()


def does_rule_fire_for_instance(rule: MIDSRule, instance: pd.Series) -> bool:
    """
    Return True if rule condition holds for an instance represented as a pandas Series, False otherwise.

    :param rule:
    :param instance:
    :return:
    """
    antecedent: Antecedent = rule.get_antecedent()

    for literal in antecedent.get_literals():
        does_literal_hold = literal.does_literal_hold_for_instance(instance)
        if not does_literal_hold:
            return False
    return True


def get_rules_per_target_attribute(ruleset: Set[MIDSRule], target_attributes) -> Dict[str, List[MIDSRule]]:
    target_attr_to_list_of_rules_map = {}

    for target_attribute in target_attributes:
        rules_predicting_target = []
        for rule in ruleset:
            if rule.head_contains_target_attribute(target_attribute):
                rules_predicting_target.append(rule)
        target_attr_to_list_of_rules_map[target_attribute] = rules_predicting_target
    return target_attr_to_list_of_rules_map


def get_rules_per_descriptive_attribute(ruleset: Set[MIDSRule], descriptive_attributes) -> Dict[str, List[MIDSRule]]:
    descriptive_attr_to_list_of_rules_map = {}

    for descriptive_attribute in descriptive_attributes:
        rules_with_descriptive_attr = []
        for rule in ruleset:
            if rule.body_contains_descriptive_attribute(descriptive_attribute):
                rules_with_descriptive_attr.append(rule)
        descriptive_attr_to_list_of_rules_map[descriptive_attribute] = rules_with_descriptive_attr
    return descriptive_attr_to_list_of_rules_map


def get_values_per_target_attribute(ruleset: Set[MIDSRule], target_attributes) -> Dict[str, Set[object]]:
    target_attr_to_set_of_values = {}

    for target_attribute in target_attributes:
        values_for_target_attribute = set()
        for rule in ruleset:
            if rule.head_contains_target_attribute(target_attribute):
                values_for_target_attribute.add(rule.get_predicted_value_for(target_attribute))
        target_attr_to_set_of_values[target_attribute] = values_for_target_attribute
    return target_attr_to_set_of_values


def get_values_per_descriptive_attribute(ruleset: Set[MIDSRule], descriptive_attributes) -> Dict[str, Set[object]]:
    descriptive_attr_to_set_of_values = {}

    for descriptive_attribute in descriptive_attributes:
        values_for_descriptive_attribute = set()
        for rule in ruleset:
            if rule.body_contains_descriptive_attribute(descriptive_attribute):
                descriptive_attr_value = rule.get_antecedent().get_literal(descriptive_attribute).get_value()
                values_for_descriptive_attribute.add(descriptive_attr_value)
        descriptive_attr_to_set_of_values[descriptive_attribute] = values_for_descriptive_attribute
    return descriptive_attr_to_set_of_values
