from typing import List, Optional, Dict, Iterable, Set

import pandas as pd

from mdrsl.rule_models.mids.mids_ruleset import MIDSRuleSet

TargetAttr = str


class ObjectiveFunctionParameters:

    use_ids_bounds = True

    def __init__(self, all_rules: MIDSRuleSet,
                 quant_dataframe: pd.DataFrame,
                 target_attributes: List[TargetAttr],
                 lambda_array: Optional[List[float]] = None):

        self.all_rules: MIDSRuleSet = all_rules
        self.ground_set_size: int = len(all_rules)

        self.target_attrs: List[TargetAttr] = target_attributes
        self.target_attr_set: Set[TargetAttr] = set(target_attributes)
        # self.target_attrs: List[TargetAttr] = list(quant_dataframe.columns.values)
        self.nb_of_target_attrs: int = len(self.target_attrs)
        self.nb_of_training_examples: int = quant_dataframe.shape[0]

        if ObjectiveFunctionParameters.use_ids_bounds:
            self.f1_upper_bound_nb_literals: int = self._f1_upper_bound_single_target_ids(all_rules=all_rules)
        else:
            self.f1_upper_bound_nb_literals: int = self._f1_upper_modified_bound(all_rules=all_rules)

        self.f2_f3_target_attr_to_upper_bound_map: Dict[TargetAttr, int] = self._f2_f3_target_attr_to_upper_bound_map(
            all_rules, self.nb_of_training_examples, self.target_attrs)

        self.f4_target_attr_to_dom_size_map: Dict[TargetAttr, int] = self._f4_target_attr_to_dom_size_map(
            quant_dataframe, self.target_attrs)
        self.f5_upper_bound = self._f5_upper_bound(nb_of_training_examples=self.nb_of_training_examples,
                                                   nb_of_ground_rules=self.ground_set_size)

        lambda_array: List[float]
        if lambda_array is None:
            self.lambda_array = 7 * [1]
        else:
            self.lambda_array = lambda_array

        self.quant_dataframe = quant_dataframe  # type: pd.DataFrame

    @staticmethod
    def _f1_upper_modified_bound(all_rules: MIDSRuleSet) -> int:
        n_literals_in_ground_set: int = all_rules.sum_rule_length()
        return n_literals_in_ground_set

    @staticmethod
    def _f1_upper_bound_single_target_ids(all_rules: MIDSRuleSet) -> int:
        nb_of_ground_rules: int = len(all_rules)
        L_max: int = all_rules.max_rule_length()
        return L_max * nb_of_ground_rules


    @staticmethod
    def _f2_f3_target_attr_to_upper_bound_map(all_rules: MIDSRuleSet,
                                              nb_of_training_examples: int,
                                              target_attrs: Iterable[TargetAttr]
                                              ) \
            -> Dict[TargetAttr, int]:
        """
        F2 and f3 =  low overlap of rules for a given target (avged over the different targets).

        How do we define the upper bound for a given target?
        --> the max overlap for that target

        R_{init, X_j} = the number of rules in R_init predicting target X_j
        N = the number of training examples

        For any two rules ri, rj, max(overlap(ri, rj)) == N.

        THUS a given target Xj:
            upper bound = N * |R_{init, X_j}|^2

        (Note: can we divide this by two? Otherwise, we count double!)

        :param all_rules:
        :param nb_of_training_examples:
        :param target_attrs:
        :return:
        """

        # Xj --> |R_{init, X_j}|
        target_attr_to_nb_of_predicting_rules_map: Dict[TargetAttr, int] \
            = all_rules.get_nb_of_rules_predicting_each_attribute()

        f2_f3_target_attr_to_upper_bound_map: Dict[TargetAttr, int] = {}
        for target_attr in target_attrs:
            n_ground_set_rules_predicting_target: int = target_attr_to_nb_of_predicting_rules_map.get(target_attr, 0)
            f3_upper_bound_for_target: int = nb_of_training_examples * n_ground_set_rules_predicting_target ** 2
            f2_f3_target_attr_to_upper_bound_map[target_attr] = f3_upper_bound_for_target

        target_attr_set = set(target_attrs)
        for target_attr in f2_f3_target_attr_to_upper_bound_map.keys():
            if target_attr not in target_attr_set:
                raise Exception(f"ILLEGAL TARGET ATTRIBUTE: {target_attr}")

        # print(f2_f3_target_attr_to_upper_bound_map)
        return f2_f3_target_attr_to_upper_bound_map

    @staticmethod
    def _f4_target_attr_to_dom_size_map(quant_dataframe: pd.DataFrame,
                                        target_attrs: Iterable[TargetAttr]
                                        ) -> Dict[TargetAttr, int]:
        f4_target_attr_to_dom_size_map: Dict[TargetAttr, int] = {}
        unique_values: pd.Series = quant_dataframe.nunique()
        # for target_attr in unique_values.index:
        for target_attr in target_attrs:
            f4_target_attr_to_dom_size_map[target_attr] = unique_values[target_attr]
        return f4_target_attr_to_dom_size_map

    @staticmethod
    def _f5_upper_bound(nb_of_training_examples: int, nb_of_ground_rules: int) -> int:
        return nb_of_training_examples * nb_of_ground_rules
