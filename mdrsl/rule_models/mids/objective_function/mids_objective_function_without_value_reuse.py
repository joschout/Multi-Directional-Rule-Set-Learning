from typing import List, Optional, Dict, Set, Tuple
import time

import numpy as np

from mdrsl.rule_models.mids.cover.cover_checker import CoverChecker
from mdrsl.rule_models.mids.cover.cover_metric import get_avg_incorrect_cover_size
from mdrsl.rule_models.mids.objective_function.mids_objective_function_abstract import AbstractMIDSObjectiveFunction
from mdrsl.rule_models.mids.cover.overlap_cacher import OverlapChecker
from mdrsl.rule_models.mids.objective_function.f2_f3_cacher import f2_f3_combo_minimize_overlap_predicting_the_same_and_different_class_caching
from mdrsl.rule_models.mids.mids_ruleset import MIDSRuleSet
from mdrsl.rule_models.mids.objective_function.mids_objective_function_parameters import ObjectiveFunctionParameters

from submodmax.abstract_optimizer import AbstractSubmodularFunction

TargetAttr = str
TargetVal = str


class MIDSObjectiveFunction(AbstractSubmodularFunction, AbstractMIDSObjectiveFunction):

    def __init__(self, objective_func_params: ObjectiveFunctionParameters,
                 cover_checker: CoverChecker,
                 overlap_checker: OverlapChecker,
                 scale_factor: float = 1.0):
        AbstractMIDSObjectiveFunction.__init__(self, objective_func_params,
                                               cover_checker=cover_checker,
                                               overlap_checker=overlap_checker,
                                               scale_factor=scale_factor)

    def f1_minimize_total_nb_of_literals(self, solution_set: MIDSRuleSet):
        """
        Minimize the total number of terms in the rule set

        :param solution_set:
        :return:
        """

        upper_bound_nb_of_literals = self.objective_func_params.f1_upper_bound_nb_literals
        f1_unnormalized = upper_bound_nb_of_literals - solution_set.sum_rule_length()

        if self.normalize:
            f1 = f1_unnormalized / upper_bound_nb_of_literals
        else:
            f1 = f1_unnormalized
        self._normalized_boundary_check(f1, 'f1')
        return f1

    # def f2_minimize_overlap_predicting_the_same_class(self, solution_set: MIDSRuleSet):
    #     """
    #     Minimize the overlap of rules predicting the same class value.
    #
    #     :param solution_set:
    #     :return:
    #     """
    #
    #     quant_dataframe = self.objective_func_params.quant_dataframe
    #     target_attr_to_intraclass_overlap_sum_map: Dict[str, int] = {}
    #
    #     for i, rule_i in enumerate(solution_set.ruleset):
    #         for j, rule_j in enumerate(solution_set.ruleset):
    #             if i >= j:
    #                 continue
    #
    #             target_attr_rule_i = set(rule_i.car.consequent.itemset.keys())
    #             target_attr_rule_j = set(rule_j.car.consequent.itemset.keys())
    #             shared_attributes = target_attr_rule_i.intersection(target_attr_rule_j)
    #
    #             for target_attr in shared_attributes:
    #                 target_value_rule_i = rule_i.car.consequent.itemset[target_attr]
    #                 target_value_rule_j = rule_j.car.consequent.itemset[target_attr]
    #
    #                 if target_value_rule_i == target_value_rule_j:
    #                     overlap_tmp = self.overlap_checker.get_pure_overlap_count(rule_i, rule_j, quant_dataframe)
    #                     target_attr_to_intraclass_overlap_sum_map[
    #                         target_attr] = target_attr_to_intraclass_overlap_sum_map.get(target_attr, 0) + overlap_tmp
    #
    #     f2: float = 0
    #     for target_attr in self.objective_func_params.f2_f3_target_attr_to_upper_bound_map.keys():
    #         f2_target_attr_upper_bound = self.objective_func_params.f2_f3_target_attr_to_upper_bound_map[target_attr]
    #         target_attr_intraclass_overlap_sum = target_attr_to_intraclass_overlap_sum_map[target_attr]
    #         if self.normalize:
    #             f2 = f2 + (f2_target_attr_upper_bound - target_attr_intraclass_overlap_sum) / f2_target_attr_upper_bound
    #         else:
    #             f2 = f2 + (f2_target_attr_upper_bound - target_attr_intraclass_overlap_sum)
    #
    #     nb_of_target_attributes = self.objective_func_params.nb_of_target_attrs
    #
    #     f2 = f2 / nb_of_target_attributes
    #     self._normalized_boundary_check(f2, 'f2')
    #     return f2
    #
    # def f3_minimize_overlap_predicting_different_class(self, solution_set: MIDSRuleSet):
    #     """
    #     Term minimizing the overlap of rules predicting the different class values.
    #
    #     :param solution_set:
    #     :return:
    #     """
    #
    #     quant_dataframe = self.objective_func_params.quant_dataframe
    #     target_attr_to_interclass_overlap_sum_map: Dict[str, int] = {}
    #
    #     for i, rule_i in enumerate(solution_set.ruleset):
    #         for j, rule_j in enumerate(solution_set.ruleset):
    #             if i >= j:
    #                 continue
    #
    #             target_attr_rule_i = set(rule_i.car.consequent.itemset.keys())
    #             target_attr_rule_j = set(rule_j.car.consequent.itemset.keys())
    #             shared_attributes = target_attr_rule_i.intersection(target_attr_rule_j)
    #
    #             for target_attr in shared_attributes:
    #                 target_value_rule_i = rule_i.car.consequent.itemset[target_attr]
    #                 target_value_rule_j = rule_j.car.consequent.itemset[target_attr]
    #
    #                 if target_value_rule_i != target_value_rule_j:
    #                     overlap_tmp = self.overlap_checker.get_pure_overlap_count(rule_i, rule_j, quant_dataframe)
    #                     target_attr_to_interclass_overlap_sum_map[target_attr] = \
    #                         target_attr_to_interclass_overlap_sum_map.get(target_attr, 0) + overlap_tmp
    #     f3: float = 0
    #     for target_attr in self.objective_func_params.f2_f3_target_attr_to_upper_bound_map.keys():
    #         f3_target_attr_upper_bound: int = self.objective_func_params.f2_f3_target_attr_to_upper_bound_map[
    #             target_attr]
    #         target_attr_interclass_overlap_sum: int = target_attr_to_interclass_overlap_sum_map[target_attr]
    #         if self.normalize:
    #             f3 = f3 + (f3_target_attr_upper_bound - target_attr_interclass_overlap_sum)
    #         else:
    #             f3 = f3 + (f3_target_attr_upper_bound - target_attr_interclass_overlap_sum) / f3_target_attr_upper_bound
    #
    #     nb_of_target_attributes = self.objective_func_params.nb_of_target_attrs
    #     f3 = f3 / nb_of_target_attributes
    #
    #     self._normalized_boundary_check(f3, 'f3')
    #     return f3

    def _f2_f3_get_overlap_sum_maps(self,
                                    solution_set: MIDSRuleSet) -> Tuple[Dict[TargetAttr, int], Dict[TargetAttr, int]]:
        quant_dataframe = self.objective_func_params.quant_dataframe

        f2_target_attr_to_intra_class_overlap_sum_map: Dict[TargetAttr, int] = {}
        f3_target_attr_to_inter_class_overlap_sum_map: Dict[TargetAttr, int] = {}

        for i, rule_i in enumerate(solution_set.ruleset):
            for j, rule_j in enumerate(solution_set.ruleset):
                if i >= j:
                    continue

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

    def _calc_f2_f3_from_map(self, target_attr_to_overlap_sum_map: Dict[TargetAttr, int]) -> float:
        nb_of_target_attributes = self.objective_func_params.nb_of_target_attrs

        f_val: float = 0
        for target_attr in self.objective_func_params.f2_f3_target_attr_to_upper_bound_map.keys():
            f2_f3_target_attr_upper_bound = self.objective_func_params.f2_f3_target_attr_to_upper_bound_map[target_attr]
            target_attr_overlap_sum: int = target_attr_to_overlap_sum_map.get(target_attr, 0)
            if f2_f3_target_attr_upper_bound != 0:
                if self.normalize:
                    f_val = f_val + (
                                f2_f3_target_attr_upper_bound - target_attr_overlap_sum) / f2_f3_target_attr_upper_bound
                else:
                    f_val = f_val + (f2_f3_target_attr_upper_bound - target_attr_overlap_sum)
        f_val = f_val / nb_of_target_attributes
        return f_val

    def f2_f3_combo_minimize_overlap_predicting_the_same_and_different_class(self,
                                                                             solution_set: MIDSRuleSet) -> Tuple[float, float]:

        f2_target_attr_to_intra_class_overlap_sum_map: Dict[TargetAttr, int]
        f3_target_attr_to_inter_class_overlap_sum_map: Dict[TargetAttr, int]
        f2_target_attr_to_intra_class_overlap_sum_map, f3_target_attr_to_inter_class_overlap_sum_map = \
            self._f2_f3_get_overlap_sum_maps(solution_set)

        for target_attr in f2_target_attr_to_intra_class_overlap_sum_map.keys():
            if target_attr not in self.objective_func_params.target_attr_set:
                raise Exception(f"Illegal target attr: {target_attr}")
        for target_attr in f3_target_attr_to_inter_class_overlap_sum_map.keys():
            if target_attr not in self.objective_func_params.target_attr_set:
                raise Exception(f"Illegal target attr: {target_attr}")

        f2: float = self._calc_f2_f3_from_map(f2_target_attr_to_intra_class_overlap_sum_map)
        f3: float = self._calc_f2_f3_from_map(f3_target_attr_to_inter_class_overlap_sum_map)

        self._normalized_boundary_check(f2, 'f2')
        self._normalized_boundary_check(f3, 'f3')

        return f2, f3

    def f2_f3_combo_minimize_overlap_predicting_the_same_and_different_class_using_cache(self,
                                                                                         solution_set: MIDSRuleSet) -> \
            Tuple[float, float]:
        return f2_f3_combo_minimize_overlap_predicting_the_same_and_different_class_caching(self.f2_f3_cache,
                                                                                            solution_set
                                                                                            )

    def f4_at_least_one_rule_per_attribute_value_combo(self, solution_set: MIDSRuleSet):
        """
        The requirement to have one rule for each value of each attribute might need to be relaxed,
         as it is no longer guaranteed that each value of each attribute occurs in at least one rule head.


        :param solution_set:
        :return:
        """

        # 1. gather for each attribute the unique values that are predicted
        target_attr_to_val_set_dict: Dict[TargetAttr, Set[TargetVal]] \
            = solution_set.get_predicted_values_per_predicted_attribute()

        # 2. count the total nb of values that are predicted over all attributes
        total_nb_of_attribute_values_covered: int = 0
        for target_attr in self.objective_func_params.f4_target_attr_to_dom_size_map.keys():
            predicted_values: Optional[Set[TargetVal]] = target_attr_to_val_set_dict.get(target_attr, None)
            if predicted_values is None:
                nb_of_predicted_values: int = 0
            else:
                nb_of_predicted_values: int = len(predicted_values)

            if self.normalize:
                target_attr_dom_size: int = self.objective_func_params.f4_target_attr_to_dom_size_map[target_attr]
                total_nb_of_attribute_values_covered += nb_of_predicted_values / target_attr_dom_size
            else:
                total_nb_of_attribute_values_covered += nb_of_predicted_values

        f4: float = total_nb_of_attribute_values_covered / self.objective_func_params.nb_of_target_attrs

        self._normalized_boundary_check(f4, 'f4')
        return f4

    def f5_minimize_incorrect_cover(self, solution_set: MIDSRuleSet):
        """
        Mazimize the precision, or minimize the nb examples that are in the incorrect-cover set the rules
        Parameters
        ----------
        solution_set

        Returns
        -------

        """

        # nb_of_instances = self.objective_func_params.nb_of_training_examples
        # len_all_rules = self.objective_func_params.ground_set_size
        quant_dataframe = self.objective_func_params.quant_dataframe

        sum_incorrect_cover = 0

        for rule in solution_set.ruleset:
            # self.cover_checker.
            sum_incorrect_cover += get_avg_incorrect_cover_size(rule, quant_dataframe, self.cover_checker)

        f5_upper_bound: int = self.objective_func_params.f5_upper_bound

        # print(f"MIDS f5 upper bound: {f5_upper_bound}")
        # print(f"MIDS f5 sum incorrect cover: {sum_incorrect_cover}")
        f5 = f5_upper_bound - sum_incorrect_cover
        if self.normalize:
            f5 = f5 / f5_upper_bound

        self._normalized_boundary_check(f5, 'f5')

        return f5

    def f6_cover_each_example(self, solution_set: MIDSRuleSet):
        """
        Originally:
        Each data point should be covered by at least one rule.
        In other words,
        Each instance should be in the correct cover set (with respect to the target attribute) of at least one rule.


        Extension to multi-target rules:
        Each instance should be in the correct cover set of at least one rule for each of its attributes.

        :param solution_set:
        :return:
        """
        # TODO: this is super expensive

        quant_dataframe = self.objective_func_params.quant_dataframe
        nb_of_training_examples = self.objective_func_params.nb_of_training_examples
        nb_of_target_attrs = self.objective_func_params.nb_of_target_attrs
        target_attrs: List[TargetAttr] = self.objective_func_params.target_attrs

        sum_correct_cover_sizes_over_all_attributes = 0

        for target_attr in target_attrs:
            correctly_covered_instances_for_attribute_by_at_least_one_rule = np.zeros(nb_of_training_examples, dtype=bool)

            for rule in solution_set.ruleset:
                if target_attr in rule.get_target_attributes():
                    correct_cover_mask = self.cover_checker.get_correct_cover(
                        rule, quant_dataframe, target_attribute=target_attr)
                    correctly_covered_instances_for_attribute_by_at_least_one_rule = np.logical_or(
                        correctly_covered_instances_for_attribute_by_at_least_one_rule,
                        correct_cover_mask
                    )

            sum_correct_cover_sizes_over_all_attributes += np.sum(
                correctly_covered_instances_for_attribute_by_at_least_one_rule)

        f6 = sum_correct_cover_sizes_over_all_attributes / (
                    nb_of_training_examples * self.objective_func_params.nb_of_target_attrs)

        self._normalized_boundary_check(f6, 'f6')

        return f6

    def evaluate(self, solution_set: MIDSRuleSet):
        if type(solution_set) == set:
            solution_set = MIDSRuleSet(solution_set)

        if type(solution_set) != MIDSRuleSet:
            raise Exception("Type of solution_set must be MIDSRuleSet")

        self.call_counter += 1
        self.set_size_collector.add_value(len(solution_set))
        start_time = time.time()

        l: List[float] = self.objective_func_params.lambda_array

        ground_set_size = self.objective_func_params.ground_set_size
        current_nb_of_rules = len(solution_set)

        f0 = self.f0_minimize_rule_set_size(ground_set_size, current_nb_of_rules)
        f1 = self.f1_minimize_total_nb_of_literals(solution_set)

        if MIDSObjectiveFunction.should_cache_f2_f3:
            f2, f3 = self.f2_f3_combo_minimize_overlap_predicting_the_same_and_different_class_using_cache(solution_set)
        else:
            f2, f3 = self.f2_f3_combo_minimize_overlap_predicting_the_same_and_different_class(solution_set)
        # f2 = self.f2_minimize_overlap_predicting_the_same_class(solution_set)
        # f3 = self.f3_minimize_overlap_predicting_different_class(solution_set)
        f4 = self.f4_at_least_one_rule_per_attribute_value_combo(solution_set)
        f5 = self.f5_minimize_incorrect_cover(solution_set)
        f6 = self.f6_cover_each_example(solution_set)

        fs = np.array([
            f0, f1, f2, f3, f4, f5, f6
        ]) / self.scale_factor

        result = np.dot(l, fs)

        if self.stat_collector is not None:
            self.stat_collector.add_values(f0, f1, f2, f3, f4, f5, f6, result)


        end_time = time.time()
        elapsed_time = end_time - start_time
        self.run_time_collector.add_value(elapsed_time)

        self.f0_val = f0
        self.f1_val = f1
        self.f2_val = f2
        self.f3_val = f3
        self.f4_val = f4
        self.f5_val = f5
        self.f6_val = f6

        # print(f"MIDS f1:{f1}")

        return result

    def f0(self, solution_set):
        current_nb_of_rules: int = len(solution_set)
        ground_set_size = len(self.objective_func_params.all_rules)
        return self.f0_minimize_rule_set_size(ground_set_size=ground_set_size,
                                              current_nb_of_rules=current_nb_of_rules)

    def f1(self, solution_set):
        return self.f1_minimize_total_nb_of_literals(solution_set)

    def f2(self, solution_set):
        f2, f3 = self.f2_f3_combo_minimize_overlap_predicting_the_same_and_different_class(solution_set)
        return f2

    def f3(self, solution_set):
        f2, f3 = self.f2_f3_combo_minimize_overlap_predicting_the_same_and_different_class(solution_set)
        return f3

    def f4(self, solution_set):
        return self.f4_at_least_one_rule_per_attribute_value_combo(solution_set)

    def f5(self, solution_set):
        return self.f5_minimize_incorrect_cover(solution_set)

    def f6(self, solution_set):
        return self.f6_cover_each_example(solution_set)


if __name__ == '__main__':
    d1 = {'a': 1, 'b': 2}
    d2 = {'a': 3, 'c': 4}
    combo = d1.keys() & d2.keys()
    print(combo)
    print(len(combo))
    for k in combo:
        print(k)
