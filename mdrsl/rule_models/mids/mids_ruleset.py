from typing import Iterable, Set, Dict, Optional, KeysView

from scipy import stats

from mdrsl.data_structures.rules.rule_part import Consequent
from mdrsl.data_structures.rules.multi_target_class_association_rule import MCAR
from mdrsl.rule_models.mids.mids_rule import MIDSRule

TargetAttr = str
DescrAttr = str
TargetVal = str


class MIDSRuleSet:

    def __init__(self, rules: Optional[Iterable[MIDSRule]] = None):
        self.ruleset = set(rules)  # type: Set[MIDSRule]

    def __len__(self):
        return len(self.ruleset)

    def sum_rule_length(self):
        rule_lens = []

        for rule in self.ruleset:  # type: MIDSRule
            rule_lens.append(len(rule))

        return sum(rule_lens)

    def max_rule_length(self):
        rule_lengths = []

        for rule in self.ruleset:
            rule_lengths.append(len(rule))

        if not rule_lengths:
            return 0

        return max(rule_lengths)

    def get_target_attributes(self) -> Set[TargetAttr]:

        all_target_attributes: Set[TargetAttr] = set()
        rule: MIDSRule
        for rule in self.ruleset:
            target_attrs: KeysView[TargetAttr] = rule.get_target_attributes()
            all_target_attributes.update(target_attrs)
        return all_target_attributes

    def get_nb_of_rules_predicting_each_attribute(self) -> Dict[TargetAttr, int]:
        """
        :return: a dictionary mapping each attribute to the number of rules predicting a value for it
        """

        attribute_to_nb_of_rules_map: Dict[TargetAttr, int] = {}

        rule: MIDSRule
        for rule in self.ruleset:
            target_attrs = rule.get_target_attributes()
            for attr in target_attrs:
                attribute_to_nb_of_rules_map[attr] = attribute_to_nb_of_rules_map.get(attr, 0) + 1
        return attribute_to_nb_of_rules_map

    def get_nb_of_rules_using_each_attribute(self) -> Dict[DescrAttr, int]:
        attribute_to_nb_of_rules_map: Dict[DescrAttr, int] = {}

        rule: MIDSRule
        for rule in self.ruleset:
            descriptive_attrs = rule.get_descriptive_attributes()
            for attr in descriptive_attrs:
                attribute_to_nb_of_rules_map[attr] = attribute_to_nb_of_rules_map.get(attr, 0) + 1
        return attribute_to_nb_of_rules_map

    @staticmethod
    def from_CAR_rules(car_rules: Iterable[MCAR]):
        ids_rules = list(map(MIDSRule, car_rules))
        ids_ruleset = MIDSRuleSet(ids_rules)

        return ids_ruleset

    def get_predicted_values_per_predicted_attribute(self) \
            -> Dict[TargetAttr, Set[TargetVal]]:

        target_attr_to_value_set_dict: Dict[TargetAttr, Set[TargetVal]] = {}

        rule: MIDSRule
        for rule in self.ruleset:
            cons: Consequent = rule.car.consequent
            for literal in cons.get_literals():
                target_attr: TargetAttr = literal.get_attribute()
                target_val: TargetVal = literal.get_value()
                if target_attr in target_attr_to_value_set_dict:
                    target_attr_to_value_set_dict[target_attr].add(target_val)
                else:
                    target_attr_to_value_set_dict[target_attr] = set([target_val])
        return target_attr_to_value_set_dict

    def count_value_occurrences_per_target_attribute(self) \
            -> Dict[TargetAttr, Dict[TargetVal, int]]:

        target_attr_to_value_counts_map: Dict[TargetAttr, Dict[TargetVal, int]] = {}

        rule: MIDSRule
        for rule in self.ruleset:
            cons: Consequent = rule.get_consequent()
            for literal in cons.get_literals():
                target_attr: TargetAttr = literal.get_attribute()
                target_val: TargetVal = literal.get_value()
                value_count_map: Optional[Dict[TargetVal, int]] =\
                    target_attr_to_value_counts_map.get(target_attr, None)
                if value_count_map is None:
                    new_dict: Dict[TargetVal, int] = {target_val: 1}
                    target_attr_to_value_counts_map[target_attr] = new_dict
                else:
                    value_count: Optional[int] = value_count_map.get(target_val, None)
                    if value_count is None:
                        value_count_map[target_val] = 1
                    else:
                        value_count_map[target_val] += 1

        return target_attr_to_value_counts_map


def get_statistics_on_nb_of_literal_in_head(rule_set: MIDSRuleSet):
    nb_of_literals_per_rule_conseq = []
    for rule in rule_set.ruleset:
        nb_of_literals_per_rule_conseq.append(len(rule.get_consequent()))
    return stats.describe(nb_of_literals_per_rule_conseq)
