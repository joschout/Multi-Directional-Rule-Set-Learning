import copy
import time
from typing import Optional, Iterable, Dict, Tuple, List

import numpy as np

from mdrsl.data_structures.rules.rule_part import Consequent
from mdrsl.rule_models.mids.cover.cover_checker import CoverChecker
from mdrsl.rule_models.mids.cover.cover_metric import get_avg_incorrect_cover_size
from mdrsl.rule_models.mids.objective_function.mids_objective_function_abstract import AbstractMIDSObjectiveFunction
from mdrsl.rule_models.mids.cover.overlap_cacher import OverlapChecker
from mdrsl.rule_models.mids.objective_function.f2_f3_cacher import f2_f3_value_reuse_minimize_overlap_caching
from mdrsl.rule_models.mids.objective_function.mids_objective_function_parameters import ObjectiveFunctionParameters
from mdrsl.rule_models.mids.mids_rule import MIDSRule
from submodmax.value_reuse.abstract_optimizer import AbstractSubmodularFunctionValueReuse, FuncInfo
from submodmax.value_reuse.set_info import SetInfo

TargetAttr = str
TargetVal = object


class MIDSFuncInfo(FuncInfo):
    def __init__(self, func_value: float,
                 f0_nb_of_rules_minimization: float,
                 f1_nb_of_literals_minimization: float,
                 f2_same_value_overlap_minimization: float,
                 f3_different_value_overlap_minimization: float,
                 f4_nb_of_distinct_values_covered: float,
                 f4_rule_counter_per_attr_value_dict: Dict[TargetAttr, Dict[TargetVal, int]],
                 f5_incorrect_cover_minimization: float,
                 f6_cover_each_example,
                 f6_target_attr_to_count_nb_of_rules_covering_each_example: Dict[TargetAttr, np.ndarray]
                 ):
        super().__init__(func_value)
        self.f0_nb_of_rules_minimization: float = f0_nb_of_rules_minimization
        self.f1_nb_of_literals_minimization: float = f1_nb_of_literals_minimization
        self.f2_same_value_overlap_minimization: float = f2_same_value_overlap_minimization
        self.f3_different_value_overlap_minimization: float = f3_different_value_overlap_minimization
        self.f4_nb_of_distinct_values_covered: float = f4_nb_of_distinct_values_covered
        self.f4_rule_counter_per_value_dict: Dict[TargetAttr, Dict[TargetVal, int]]\
            = f4_rule_counter_per_attr_value_dict
        self.f5_incorrect_cover_minimization: float = f5_incorrect_cover_minimization
        self.f6_cover_each_example = f6_cover_each_example
        self.f6_target_attr_to_count_nb_of_rules_covering_each_example: Dict[TargetAttr, np.ndarray] \
            = f6_target_attr_to_count_nb_of_rules_covering_each_example

    @staticmethod
    def get_initial(f1_upper_bound: int,
                    f2_f3_target_attr_to_upper_bound_map: Dict[str, int],
                    f5_upper_bound,
                    target_attrs: Iterable[TargetAttr],
                    nb_of_training_examples: int,
                    normalized: bool):

        f6_target_attr_to_counts: Dict[TargetAttr, np.ndarray] = {}
        for target_attr in target_attrs:
            f6_target_attr_to_counts[target_attr] = np.zeros(nb_of_training_examples)

        if normalized:
            f1_upper_bound = 1
            f2_f3_upper_bound = 1
            f5_upper_bound = 1
        else:
            f2_f3_upper_bound = 0
            nb_of_target_attrs = len(f2_f3_target_attr_to_upper_bound_map)
            for target_attr, val in f2_f3_target_attr_to_upper_bound_map.items():
                f2_f3_upper_bound += val
            f2_f3_upper_bound = f2_f3_upper_bound / nb_of_target_attrs

        return MIDSFuncInfo(
            func_value=0,
            f0_nb_of_rules_minimization=None,
            f1_nb_of_literals_minimization=f1_upper_bound,
            f2_same_value_overlap_minimization=f2_f3_upper_bound,
            f3_different_value_overlap_minimization=f2_f3_upper_bound,
            f4_nb_of_distinct_values_covered=0,
            f4_rule_counter_per_attr_value_dict={},
            f5_incorrect_cover_minimization=f5_upper_bound,
            f6_cover_each_example=None,
            f6_target_attr_to_count_nb_of_rules_covering_each_example=f6_target_attr_to_counts
        )


class MIDSObjectiveFunctionValueReuse(AbstractSubmodularFunctionValueReuse, AbstractMIDSObjectiveFunction):

    def __init__(self, objective_func_params: ObjectiveFunctionParameters,
                 cover_checker: CoverChecker,
                 overlap_checker: OverlapChecker,
                 scale_factor: float = 1.0):
        AbstractMIDSObjectiveFunction.__init__(self, objective_func_params,
                                               cover_checker=cover_checker,
                                               overlap_checker=overlap_checker,
                                               scale_factor=scale_factor)
        self.empty_set = {}

        nb_of_training_examples = self.objective_func_params.nb_of_training_examples
        target_attrs = self.objective_func_params.target_attrs

        self.init_objective_function_value_info = MIDSFuncInfo.get_initial(
            f1_upper_bound=self.objective_func_params.f1_upper_bound_nb_literals,
            f2_f3_target_attr_to_upper_bound_map=self.objective_func_params.f2_f3_target_attr_to_upper_bound_map,
            f5_upper_bound=0,
            target_attrs=target_attrs,
            nb_of_training_examples=nb_of_training_examples,
            normalized=self.normalize
        )

    def f1_minimize_total_nb_of_literals(self, f1_previous: float,
                                         added_rules: Iterable[MIDSRule],
                                         deleted_rules: Iterable[MIDSRule]) -> float:
        nb_lits_rules_to_add = 0
        for rule in added_rules:
            nb_lits_rules_to_add += len(rule)
        nb_lits_rules_to_delete = 0
        for rule in deleted_rules:
            nb_lits_rules_to_delete += len(rule)

        f1_upper_bound_nb_of_literals = self.objective_func_params.f1_upper_bound_nb_literals
        if self.normalize and f1_upper_bound_nb_of_literals > 0:
            f1 = f1_previous + (nb_lits_rules_to_delete - nb_lits_rules_to_add) / f1_upper_bound_nb_of_literals
        else:
            f1 = f1_previous + nb_lits_rules_to_delete - nb_lits_rules_to_add
        self._normalized_boundary_check(f1, 'f1')
        return f1

    def f2_f3_minimize_overlap(self, f2_previous: float, f3_previous: float,
                               rules_intersection_previous_and_current: Iterable[MIDSRule],
                               added_rules: Iterable[MIDSRule],
                               deleted_rules: Iterable[MIDSRule])\
            -> Tuple[float, float]:

        f2_added_rules_target_attr_to_overlap_sum_map: Dict[TargetAttr, int]
        f3_added_rules_target_attr_to_overlap_sum_map: Dict[TargetAttr, int]

        f2_deleted_rules_target_attr_to_overlap_sum_map: Dict[TargetAttr, int]
        f3_deleted_rules_target_attr_to_overlap_sum_map: Dict[TargetAttr, int]

        f2_added_rules_target_attr_to_overlap_sum_map, f3_added_rules_target_attr_to_overlap_sum_map \
            = self._f2_f3_get_overlap_sum_maps(rules_intersection_previous_and_current, added_rules)

        f2_deleted_rules_target_attr_to_overlap_sum_map, f3_deleted_rules_target_attr_to_overlap_sum_map \
            = self._f2_f3_get_overlap_sum_maps(rules_intersection_previous_and_current, deleted_rules)

        f2: float = self._calc_f2_f3_from_map(f2_previous, f2_added_rules_target_attr_to_overlap_sum_map,
                                              f2_deleted_rules_target_attr_to_overlap_sum_map)

        f3: float = self._calc_f2_f3_from_map(f3_previous, f3_added_rules_target_attr_to_overlap_sum_map,
                                              f3_deleted_rules_target_attr_to_overlap_sum_map)

        if f2 == 0 or f3 == 0:
            raise Exception()
        self._normalized_boundary_check(f2, 'f2')
        self._normalized_boundary_check(f3, 'f3')

        return f2, f3

    def _calc_f2_f3_from_map(self, f_val_previous: float,
                             added_rules_target_attr_to_overlap_sum_map: Dict[TargetAttr, int],
                             deleted_rules_target_attr_to_overlap_sum_map: Dict[TargetAttr, int]) -> float:
        nb_of_target_attributes = self.objective_func_params.nb_of_target_attrs

        f_val: float = f_val_previous
        for target_attr in self.objective_func_params.f2_f3_target_attr_to_upper_bound_map.keys():
            f2_target_attr_upper_bound: float = self.objective_func_params.f2_f3_target_attr_to_upper_bound_map[
                target_attr]
            if nb_of_target_attributes == 0:
                raise Exception()

            if f2_target_attr_upper_bound == 0:
                pass
            else:
                normalizer: float = 1.0 / (nb_of_target_attributes * f2_target_attr_upper_bound)
                f_overlap_added_rules = added_rules_target_attr_to_overlap_sum_map.get(target_attr, 0)
                f_overlap_deleted_rules = deleted_rules_target_attr_to_overlap_sum_map.get(target_attr, 0)

                f_val = f_val + normalizer * (f_overlap_deleted_rules - f_overlap_added_rules)
        return f_val

    def _f2_f3_get_overlap_sum_maps(self, rules_intersection_previous_and_current: Iterable[MIDSRule],
                                    added_or_deleted_rules: Optional[Iterable[MIDSRule]] = None) \
            -> Tuple[Dict[TargetAttr, int], Dict[TargetAttr, int]]:

        quant_dataframe = self.objective_func_params.quant_dataframe

        f2_target_attr_to_intra_class_overlap_sum_map: Dict[TargetAttr, int] = {}
        f3_target_attr_to_inter_class_overlap_sum_map: Dict[TargetAttr, int] = {}

        if added_or_deleted_rules is not None:
            rule_i: MIDSRule
            for rule_i in rules_intersection_previous_and_current:
                rule_a: MIDSRule
                for rule_a in added_or_deleted_rules:
                    target_attrs_rule_i = rule_i.get_target_attributes()
                    target_attrs_rule_a = rule_a.get_target_attributes()
                    shared_attributes = target_attrs_rule_i & target_attrs_rule_a

                    # if both rules have at least one target attribute in common
                    if len(shared_attributes) > 0:
                        overlap_count = self.overlap_checker.get_pure_overlap_count(rule_i, rule_a, quant_dataframe)

                        for target_attr in shared_attributes:
                            # check whether the rules predict the same value for the target attribute
                            target_value_rule_i = rule_i.get_predicted_value_for(target_attr)
                            target_value_rule_a = rule_a.get_predicted_value_for(target_attr)

                            if target_value_rule_i == target_value_rule_a:
                                f2_target_attr_to_intra_class_overlap_sum_map[target_attr] = \
                                    f2_target_attr_to_intra_class_overlap_sum_map.get(target_attr, 0) + overlap_count
                            else:
                                f3_target_attr_to_inter_class_overlap_sum_map[target_attr] = \
                                    f3_target_attr_to_inter_class_overlap_sum_map.get(target_attr, 0) + overlap_count

            rule_i: MIDSRule
            rule_j: MIDSRule
            for i, rule_i in enumerate(added_or_deleted_rules):
                for j, rule_j in enumerate(added_or_deleted_rules):
                    if i >= j:
                        continue
                    #
                    # for i in range(0, len(rules_to_add_or_delete)):
                    #     for j in range(i + 1, len(rules_to_add_or_delete)):
                    #         rule_i = rules_to_add_or_delete[i]
                    #         rule_j = rules_to_add_or_delete[j]

                    target_attrs_rule_i = rule_i.get_target_attributes()
                    target_attrs_rule_j = rule_j.get_target_attributes()
                    shared_attributes = target_attrs_rule_i & target_attrs_rule_j

                    # if both rules have at least one target attribute in common
                    if len(shared_attributes) > 0:
                        overlap_count = self.overlap_checker.get_pure_overlap_count(rule_i, rule_j, quant_dataframe)

                        for target_attr in shared_attributes:
                            # check whether the rules predict the same value for the target attribute
                            target_value_rule_i = rule_i.get_predicted_value_for(target_attr)
                            target_value_rule_j = rule_j.get_predicted_value_for(target_attr)

                            if target_value_rule_i == target_value_rule_j:
                                f2_target_attr_to_intra_class_overlap_sum_map[target_attr] = \
                                    f2_target_attr_to_intra_class_overlap_sum_map.get(target_attr, 0) + overlap_count
                            else:
                                f3_target_attr_to_inter_class_overlap_sum_map[target_attr] = \
                                    f3_target_attr_to_inter_class_overlap_sum_map.get(target_attr, 0) + overlap_count

        return f2_target_attr_to_intra_class_overlap_sum_map, f3_target_attr_to_inter_class_overlap_sum_map

    def f2_f3_minimize_overlap_using_cache(self,
                                           f2_previous, f3_previous,
                                           rules_intersection_previous_and_current: Iterable[MIDSRule],
                                           added_rules: Iterable[MIDSRule],
                                           deleted_rules: Iterable[MIDSRule]) -> Tuple[float, float]:
        f2, f3 = f2_f3_value_reuse_minimize_overlap_caching(
            self.f2_f3_cache,
            f2_previous,
            f3_previous,
            rules_intersection_previous_and_current,
            added_rules,
            deleted_rules)
        self._normalized_boundary_check(f2, 'f2')
        self._normalized_boundary_check(f3, 'f3')
        return f2, f3

    def f4_one_rule_per_value(self, previous_f4_target_attr_val_rule_counts_map: Dict[TargetAttr, Dict[TargetVal, int]],
                              previous_f4_nb_of_values_covered: float,
                              added_rules: Iterable[MIDSRule],
                              deleted_rules: Iterable[MIDSRule]
                              ) -> Tuple[float, Dict[TargetAttr, Dict[TargetVal, int]]]:
        """
            Each target value should be predicted by at least 1 rule.
        """

        f4_target_attr_val_rule_counts_map: Dict[TargetAttr, Dict[TargetVal, int]] = copy.deepcopy(
            previous_f4_target_attr_val_rule_counts_map)

        weighted_total_nb_of_vals_covered: float = previous_f4_nb_of_values_covered

        f4_target_attr_to_dom_size_map: Dict[TargetAttr, int] \
            = self.objective_func_params.f4_target_attr_to_dom_size_map
        nb_of_target_attrs: int = self.objective_func_params.nb_of_target_attrs

        rule_a: MIDSRule
        for rule_a in added_rules:
            cons: Consequent = rule_a.get_consequent()

            # for target_attr, target_val in cons.itemset.items():
            for literal in cons.get_literals():
                target_attr: TargetAttr = literal.get_attribute()
                target_val: TargetVal = literal.get_value()

                value_to_count_map:  Optional[Dict[TargetVal, int]] \
                    = f4_target_attr_val_rule_counts_map.get(target_attr, None)
                if value_to_count_map is None:
                    # no dictionary yet for the current target attribute
                    new_dict: Dict[TargetVal, int] = {target_val: 1}
                    f4_target_attr_val_rule_counts_map[target_attr] = new_dict
                    weighted_total_nb_of_vals_covered += \
                        1.0 / (nb_of_target_attrs * f4_target_attr_to_dom_size_map[target_attr])
                else:
                    value_count: Optional[int] = value_to_count_map.get(target_val, None)
                    if value_count is None:
                        value_to_count_map[target_val] = 1
                        weighted_total_nb_of_vals_covered += \
                            1.0 / (nb_of_target_attrs * f4_target_attr_to_dom_size_map[target_attr])
                    else:
                        value_to_count_map[target_val] += 1

        rule_d: MIDSRule
        for rule_d in deleted_rules:
            cons: Consequent = rule_d.get_consequent()

            target_att: TargetAttr
            target_val: TargetVal
            # for target_attr, target_val in cons.itemset.items():
            for literal in cons.get_literals():
                target_attr: TargetAttr = literal.get_attribute()
                target_val: TargetVal = literal.get_value()

                value_to_count_map: Optional[Dict[TargetVal, int]] \
                    = f4_target_attr_val_rule_counts_map.get(target_attr, None)
                if value_to_count_map is None:
                    raise Exception("Should still have counts for target " + str(target_attr))
                else:
                    value_count: Optional[int] = value_to_count_map.get(target_val, None)
                    if value_count is None:
                        raise Exception(
                            "Should still have counts for target " + str(target_attr) + " and value " + str(target_val))
                    else:
                        if value_count > 1:
                            value_to_count_map[target_val] = value_count - 1
                        else:
                            # value count should be 1 and become zero
                            del value_to_count_map[target_val]
                            if len(value_to_count_map) == 0:
                                del f4_target_attr_val_rule_counts_map[target_attr]
                            weighted_total_nb_of_vals_covered -= 1.0 / (
                                    nb_of_target_attrs * f4_target_attr_to_dom_size_map[target_attr])
        f4 = weighted_total_nb_of_vals_covered
        self._normalized_boundary_check(f4, 'f4')
        return f4, f4_target_attr_val_rule_counts_map

    def f5_incorrect_cover_minimization(self, f5_previous: float,
                                        added_rules: Iterable[MIDSRule],
                                        deleted_rules: Iterable[MIDSRule]) -> float:

        f5_upper_bound: int = self.objective_func_params.f5_upper_bound
        quant_dataframe = self.objective_func_params.quant_dataframe

        if self.normalize:
            f5 = f5_previous * f5_upper_bound
        else:
            f5 = f5_previous

        rule_a: MIDSRule
        for rule_a in added_rules:
            f5 = f5 - get_avg_incorrect_cover_size(rule_a, quant_dataframe, self.cover_checker)

        rule_d: MIDSRule
        for rule_d in deleted_rules:
            f5 = f5 + get_avg_incorrect_cover_size(rule_d, quant_dataframe, self.cover_checker)

        if self.normalize and f5_upper_bound > 0:
            f5 = f5 / f5_upper_bound

        self._normalized_boundary_check(f5, 'f5')

        return f5

    def f6_cover_each_example(self,
                              previous_target_attr_to_count_nb_of_rules_covering_each_example: Dict[
                                  TargetAttr, np.ndarray],
                              added_rules: Iterable[MIDSRule],
                              deleted_rules: Iterable[MIDSRule]
                              ) -> Tuple[float, Dict[TargetAttr, np.ndarray]]:
        quant_dataframe = self.objective_func_params.quant_dataframe
        target_attrs = self.objective_func_params.target_attrs

        target_attr_to_count_nb_of_rules_covering_each_example: Dict[TargetAttr, np.ndarray] = {}
        for target_attr in target_attrs:
            previous_target_attr_counts: np.ndarray \
                = previous_target_attr_to_count_nb_of_rules_covering_each_example[target_attr]
            new_target_attr_counts = np.copy(previous_target_attr_counts)
            target_attr_to_count_nb_of_rules_covering_each_example[target_attr] = new_target_attr_counts

        rule_a: MIDSRule
        for rule_a in added_rules:
            for target_attr in rule_a.get_target_attributes():
                correct_cover_mask = self.cover_checker.get_correct_cover(
                    rule_a, quant_dataframe, target_attribute=target_attr)
                target_attr_to_count_nb_of_rules_covering_each_example[target_attr] += correct_cover_mask

        rule_d: MIDSRule
        for rule_d in deleted_rules:
            for target_attr in rule_d.get_target_attributes():
                correct_cover_mask = self.cover_checker.get_correct_cover(
                    rule_d, quant_dataframe, target_attribute=target_attr)
                target_attr_to_count_nb_of_rules_covering_each_example[target_attr] -= correct_cover_mask

        sum_correct_cover_sizes_over_all_attributes = 0
        for target_attr in target_attrs:
            sum_correct_cover_sizes_over_all_attributes += np.count_nonzero(
                target_attr_to_count_nb_of_rules_covering_each_example[target_attr])

        nb_of_training_examples = self.objective_func_params.nb_of_training_examples
        nb_of_target_attrs = self.objective_func_params.nb_of_target_attrs

        f6 = sum_correct_cover_sizes_over_all_attributes / (
            nb_of_training_examples * nb_of_target_attrs
        )

        self._normalized_boundary_check(f6, 'f6')
        return f6, target_attr_to_count_nb_of_rules_covering_each_example

    def evaluate(self, current_set_info: SetInfo,
                 previous_func_info: Optional[MIDSFuncInfo]
                 ) -> MIDSFuncInfo:
        self.call_counter += 1
        start_time = time.time()
        if previous_func_info is None:
            previous_func_info = self.init_objective_function_value_info

        ground_set_size = current_set_info.get_ground_set_size()
        current_rule_set_size = current_set_info.get_current_set_size()
        added_rules = current_set_info.get_added_elems()
        deleted_rules = current_set_info.get_deleted_elems()
        intersection_previous_and_current_rules = current_set_info.get_intersection_previous_and_current_elems()

        self.set_size_collector.add_value(current_rule_set_size)

        f0 = self.f0_minimize_rule_set_size(
            ground_set_size,
            current_rule_set_size
        )

        f1 = self.f1_minimize_total_nb_of_literals(
            previous_func_info.f1_nb_of_literals_minimization,
            added_rules=added_rules,
            deleted_rules=deleted_rules
        )

        f2, f3 = self.f2_f3_minimize_overlap(
            previous_func_info.f2_same_value_overlap_minimization,
            previous_func_info.f3_different_value_overlap_minimization,
            intersection_previous_and_current_rules,
            added_rules=added_rules,
            deleted_rules=deleted_rules
        )

        f4, f4_rule_counter_per_value_dict = self.f4_one_rule_per_value(
            previous_func_info.f4_rule_counter_per_value_dict,
            previous_func_info.f4_nb_of_distinct_values_covered,
            added_rules=added_rules,
            deleted_rules=deleted_rules
        )
        f5 = self.f5_incorrect_cover_minimization(
            previous_func_info.f5_incorrect_cover_minimization,
            added_rules=added_rules,
            deleted_rules=deleted_rules
        )
        f6, f6_count_nb_of_rules_covering_each_example = self.f6_cover_each_example(
            previous_func_info.f6_target_attr_to_count_nb_of_rules_covering_each_example,
            added_rules=added_rules,
            deleted_rules=deleted_rules
        )

        self.f0_val = f0
        self.f1_val = f1
        self.f2_val = f2
        self.f3_val = f3
        self.f4_val = f4
        self.f5_val = f5
        self.f6_val = f6

        # l = self.lambda_array
        # print()
        # print(tabulate([['value', f0, f1, f2, f3, f4, f5, f6],
        #                 ['l*val', f0 * l[0], f1 * l[1], f2 * l[2], f3 * l[3], f4 * l[4], f5 * l[5], f6 * l[6]]
        #                 ],
        #                headers=['type', 'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6']))
        # print()

        l: List[float] = self.objective_func_params.lambda_array

        fs = np.array([
            f0, f1, f2, f3, f4, f5, f6
        ]) / self.scale_factor

        objective_function_value = np.dot(l, fs)

        if self.stat_collector is not None:
            self.stat_collector.add_values(f0, f1, f2, f3, f4, f5, f6, objective_function_value)


        objective_function_value_info = MIDSFuncInfo(
            func_value=objective_function_value,
            f0_nb_of_rules_minimization=f0,
            f1_nb_of_literals_minimization=f1,
            f2_same_value_overlap_minimization=f2,
            f3_different_value_overlap_minimization=f3,
            f4_nb_of_distinct_values_covered=f4,
            f4_rule_counter_per_attr_value_dict=f4_rule_counter_per_value_dict,
            f5_incorrect_cover_minimization=f5,
            f6_cover_each_example=f6,
            f6_target_attr_to_count_nb_of_rules_covering_each_example=f6_count_nb_of_rules_covering_each_example
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.run_time_collector.add_value(elapsed_time)

        return objective_function_value_info
