import copy
import time
from typing import Iterable, Optional, Set, TypeVar, Dict, Tuple

import numpy as np

from pyarc.data_structures import ClassAssocationRule

from pyids.data_structures import IDSRuleSet
from pyids.data_structures.ids_rule import IDSRule

from mdrsl.rule_models.ids.objective_function.ids_objective_function_abstract import AbstractIDSObjectiveFunction
from mdrsl.rule_models.ids.objective_function.ids_objective_function_without_value_reuse import IDSObjectiveFunction

from submodmax.value_reuse.abstract_optimizer import AbstractSubmodularFunctionValueReuse, FuncInfo
from submodmax.value_reuse.set_info import SetInfo

E = TypeVar('E')

TargetVal = object


class IDSFuncInfo(FuncInfo):
    def __init__(self,
                 objective_function_value: float,
                 f0_nb_of_rules_minimization: float,
                 f1_nb_of_literals_minimization: float,
                 f2_same_value_overlap_minimization: float,
                 f3_different_value_overlap_minimization: float,
                 f4_nb_of_distinct_values_covered: float,
                 f4_rule_counter_per_value_dict: Dict[TargetVal, int],
                 f5_incorrect_cover_minimization: float,
                 f6_cover_each_example: float,
                 f6_count_nb_of_rules_covering_each_example: np.ndarray
                 ):
        super().__init__(objective_function_value)
        self.f0_nb_of_rules_minimization: float = f0_nb_of_rules_minimization
        self.f1_nb_of_literals_minimization: float = f1_nb_of_literals_minimization
        self.f2_same_value_overlap_minimization: float = f2_same_value_overlap_minimization
        self.f3_different_value_overlap_minimization: float = f3_different_value_overlap_minimization
        self.f4_nb_of_distinct_values_covered: float = f4_nb_of_distinct_values_covered
        self.f4_rule_counter_per_value_dict: Dict[TargetVal, int] = f4_rule_counter_per_value_dict
        self.f5_incorrect_cover_minimization: float = f5_incorrect_cover_minimization
        self.f6_cover_each_example: float = f6_cover_each_example
        self.f6_count_nb_of_rules_covering_each_example: np.ndarray = f6_count_nb_of_rules_covering_each_example

    def __str__(self):
        ostr = ("obj func value: " + str(self.func_value) + "\n"
                "\tf0_nb_of_rules_minimization: " + str(self.f0_nb_of_rules_minimization) + "\n"
                "\tf1_nb_of_literals_minimization: " + str(self.f1_nb_of_literals_minimization) + "\n"
                "\tf2_same_value_overlap_minimization: " + str(self.f2_same_value_overlap_minimization) + "\n"
                "\tf3_different_value_overlap_minimization: " + str(self.f3_different_value_overlap_minimization) + "\n"
                "\tf4_nb_of_distinct_values_covered: " + str(self.f4_nb_of_distinct_values_covered) + "\n"
                )
        ostr += "\tf4_rule_counter_per_value_dict:\n"
        for value, count in self.f4_rule_counter_per_value_dict.items():
            ostr += "\t\t" + str(value) + ", " + str(count) + "\n"
        ostr += (
                "\tf5_incorrect_cover_minimization: " + str(self.f5_incorrect_cover_minimization) + "\n"
                "\tf6_cover_each_example: " + str(self.f6_cover_each_example) + "\n"
                )
        return ostr

    @staticmethod
    def get_initial(f1_upper_bound: int, f2_f3_upper_bound: int, f5_upper_bound, nb_of_training_examples: int,
                    normalized: bool) -> 'IDSFuncInfo':

        f6_initial_counts = np.zeros(nb_of_training_examples, dtype=int)

        if normalized:
            f1_upper_bound = 1
            f2_f3_upper_bound = 1
            f5_upper_bound = 1

        return IDSFuncInfo(
            objective_function_value=0,
            f0_nb_of_rules_minimization=None,
            f1_nb_of_literals_minimization=f1_upper_bound,
            f2_same_value_overlap_minimization=f2_f3_upper_bound,
            f3_different_value_overlap_minimization=f2_f3_upper_bound,
            f4_nb_of_distinct_values_covered=0,
            f4_rule_counter_per_value_dict={},
            f5_incorrect_cover_minimization=f5_upper_bound,
            f6_cover_each_example=None,
            f6_count_nb_of_rules_covering_each_example=f6_initial_counts
        )


class IDSObjectiveFunctionValueReuse(AbstractSubmodularFunctionValueReuse, AbstractIDSObjectiveFunction):

    def __init__(self, objective_func_params, cacher, scale_factor, normalize=True):
        AbstractIDSObjectiveFunction.__init__(self, objective_func_params=objective_func_params,
                                              cacher=cacher, scale_factor=scale_factor, normalize=normalize)

        self.empty_set = {}
        self.init_objective_function_value_info = IDSFuncInfo.get_initial(self.f1_upper_bound_nb_of_literals,
                                                                          self.f2_f3_upper_bound,
                                                                          self.f5_upper_bound,
                                                                          self.nb_of_training_examples,
                                                                          self.normalize)

        # self.recalculate = True
        # self.obj_func_no_reuse = IDSObjectiveFunction(objective_func_params=self.objective_func_params,
        #                                             cacher=self.cacher,
        #                                             scale_factor=self.scale_factor)

    def f1_minimize_total_nb_of_literals(self, f1_previous: int,
                                         added_rules: Iterable[IDSRule],
                                         deleted_rules: Iterable[IDSRule]):

        nb_lits_rules_to_add = 0
        for rule in added_rules:
            nb_lits_rules_to_add += len(rule)
        nb_lits_rules_to_delete = 0
        for rule in deleted_rules:
            nb_lits_rules_to_delete += len(rule)

        if self.normalize:
            f1 = f1_previous + (nb_lits_rules_to_delete - nb_lits_rules_to_add) / self.f1_upper_bound_nb_of_literals
        else:
            f1 = f1_previous + nb_lits_rules_to_delete - nb_lits_rules_to_add
        self._f1_boundary_check(f1)

        return f1

    def f2_f3_minimize_overlap_predicting_the_same_class(self, f2_previous, f3_previous,
                                                         rules_intersection_previous_and_current: Iterable[IDSRule],
                                                         added_rules: Iterable[IDSRule],
                                                         deleted_rules: Iterable[IDSRule]):

        f2_overlap_added_rules, f3_overlap_added_rules = self._f2_f3_get_overlap_value(
            rules_intersection_previous_and_current, added_rules)
        f2_overlap_deleted_rules, f3_overlap_deleted_rules = self._f2_f3_get_overlap_value(
            rules_intersection_previous_and_current, deleted_rules)

        if self.normalize:
            f2_f3_upper_bound = self.f2_f3_upper_bound
            f2_previous_unnorm = f2_previous * f2_f3_upper_bound
            f3_previous_unnorm = f3_previous * f2_f3_upper_bound

            f2_unnorm = f2_previous_unnorm - f2_overlap_added_rules + f2_overlap_deleted_rules
            f3_unnorm = f3_previous_unnorm - f3_overlap_added_rules + f3_overlap_deleted_rules

            f2 = f2_unnorm / self.f2_f3_upper_bound
            f3 = f3_unnorm / self.f2_f3_upper_bound

            # f2 = f2_previous + (f2_overlap_deleted_rules - f2_overlap_added_rules) / self.f2_f3_upper_bound
            # f3 = f3_previous + (f3_overlap_deleted_rules - f3_overlap_added_rules) / self.f2_f3_upper_bound
        else:
            f2 = f2_previous + f2_overlap_deleted_rules - f2_overlap_added_rules
            f3 = f3_previous + f3_overlap_deleted_rules - f3_overlap_added_rules

        if f2 < 0:
            raise Exception("f2 < 0:", str(f2))
        if self.normalize:
            if f2 > 1:
                obj_func_no_reuse = IDSObjectiveFunction(objective_func_params=self.objective_func_params,
                                                cacher=self.cacher,
                                                scale_factor=self.scale_factor)

                s: Set[IDSRule] = rules_intersection_previous_and_current.copy()
                for rule in added_rules:
                    s.add(rule)
                f2_no_value_reuse = obj_func_no_reuse.f2(IDSRuleSet(s))
                raise Exception("f2 > 1:", str(f2), ", should be:", f2_no_value_reuse)

        self._boundary_check(f2, 'f2')
        self._boundary_check(f3, 'f3')

        # if self.recalculate:
        #
        #     s: Set[IDSRule] = rules_intersection_previous_and_current.copy()
        #     for rule in added_rules:
        #         s.add(rule)
        #     f2_no_value_reuse = self.obj_func_no_reuse.f2(IDSRuleSet(s))
        #     if not math.isclose(f2, f2_no_value_reuse):
        #         raise Exception(
        #             "\nf2 with value reuse: " + str(f2) + "\n" +
        #             "without value reuse: " + str(f2_no_value_reuse))
        #
        #     f3_no_value_reuse = self.obj_func_no_reuse.f3(IDSRuleSet(s))
        #     if not math.isclose(f3, f3_no_value_reuse):
        #         raise Exception("f3 with value reuse:", f3, ", without value reuse:", f3_no_value_reuse)

        return f2, f3

    def _f2_f3_get_overlap_value(self, rules_intersection_previous_and_current: Iterable[IDSRule],
                                 added_or_deleted_rules: Optional[Iterable[IDSRule]] =None):
        f2_overlap_value: int = 0
        f3_overlap_value: int = 0

        if added_or_deleted_rules is not None:
            rule_i: IDSRule
            for rule_i in rules_intersection_previous_and_current:
                rule_a: IDSRule
                for rule_a in added_or_deleted_rules:
                    cached_value: int = self.cacher.overlap(rule_i, rule_a)
                    if rule_i.car.consequent.value == rule_a.car.consequent.value:
                        f2_overlap_value += cached_value
                    else:
                        f3_overlap_value += cached_value

            rule_i: IDSRule
            rule_j: IDSRule
            for i, rule_i in enumerate(added_or_deleted_rules):
                for j, rule_j in enumerate(added_or_deleted_rules):
                    if i >= j:
                        continue
                    #
                    # for i in range(0, len(rules_to_add_or_delete)):
                    #     for j in range(i + 1, len(rules_to_add_or_delete)):
                    #         rule_i = rules_to_add_or_delete[i]
                    #         rule_j = rules_to_add_or_delete[j]

                    cached_value: int = self.cacher.overlap(rule_i, rule_j)
                    if rule_i.car.consequent.value == rule_j.car.consequent.value:
                        f2_overlap_value += cached_value
                    else:
                        f3_overlap_value += cached_value

        return f2_overlap_value, f3_overlap_value

    def f4_one_rule_per_value(self, previous_f4_dict: Dict[TargetVal, int], previous_f4_nb_of_values_covered: int,
                              added_rules: Iterable[IDSRule], deleted_rules: Iterable[IDSRule]):
        tmp_dict: Dict[TargetVal, int] = copy.deepcopy(previous_f4_dict)

        if self.normalize:
            tmp_nb_of_values_covered: int = previous_f4_nb_of_values_covered * self.nb_of_target_values
        else:
            tmp_nb_of_values_covered: int = previous_f4_nb_of_values_covered

        rule_a: IDSRule
        for rule_a in added_rules:
            target_value: TargetVal = rule_a.car.consequent.value
            value_count: int = tmp_dict.get(target_value, None)
            if value_count is None:
                tmp_dict[target_value] = 1
                tmp_nb_of_values_covered += 1
            else:
                tmp_dict[target_value] += 1

        rule_d: IDSRule
        for rule_d in deleted_rules:
            target_value: TargetVal = rule_d.car.consequent.value
            value_count: int = tmp_dict.get(target_value, None)
            if value_count > 1:
                tmp_dict[target_value] = value_count - 1
            else:
                # value count should be 1 and become zero
                del tmp_dict[target_value]
                tmp_nb_of_values_covered -= 1

        if self.normalize:
            f4 = tmp_nb_of_values_covered / self.nb_of_target_values
        else:
            f4 = tmp_nb_of_values_covered

        self._boundary_check(f4, 'f4')

        return f4, tmp_dict

    def f5_incorrect_cover_minimization(self, f5_previous,
                                        added_rules: Iterable[IDSRule], deleted_rules: Iterable[IDSRule]):
        incorrect_cover_count_of_rules_to_add = 0
        incorrect_cover_count_of_rules_to_delete = 0

        quant_dataframe = self.quant_dataframe

        for rule_a in added_rules:
            incorrect_cover_of_rule_a = rule_a.incorrect_cover(quant_dataframe)
            # print("incorrect_cover_of_rule_a", str(incorrect_cover_of_rule_a))
            incorrect_cover_count_of_rules_to_add += np.sum(incorrect_cover_of_rule_a)

        for rule_d in deleted_rules:
            incorrect_cover_count_of_rules_to_delete += np.sum(rule_d.incorrect_cover(quant_dataframe))

        if self.normalize:
            f5 = f5_previous + (
                        incorrect_cover_count_of_rules_to_delete - incorrect_cover_count_of_rules_to_add) / self.f5_upper_bound
        else:
            f5 = f5_previous + incorrect_cover_count_of_rules_to_delete - incorrect_cover_count_of_rules_to_add

        self._boundary_check(f5, 'f5')

        return f5

    def f6_cover_each_example(self, previous_np_counts: np.ndarray,
                              added_rules: Iterable[IDSRule],
                              deleted_rules: Iterable[IDSRule]) -> Tuple[float, np.ndarray]:

        quant_dataframe = self.quant_dataframe

        count_nb_of_rules_covering_each_example: np.ndarray = np.copy(previous_np_counts)

        for rule_a in added_rules:
            correct_cover = rule_a.correct_cover(quant_dataframe)
            count_nb_of_rules_covering_each_example += correct_cover

        for rule_d in deleted_rules:
            correct_cover = rule_d.correct_cover(quant_dataframe)
            count_nb_of_rules_covering_each_example -= correct_cover

        if self.normalize:
            f6 = np.count_nonzero(count_nb_of_rules_covering_each_example) / self.nb_of_training_examples
        else:
            f6 = np.count_nonzero(count_nb_of_rules_covering_each_example)

        self._boundary_check(f6, 'f6')
        return f6, count_nb_of_rules_covering_each_example

    def evaluate(self, current_set_info: SetInfo,
                 previous_func_info: Optional[IDSFuncInfo]
                 ) -> IDSFuncInfo:
        self.call_counter += 1
        start_time = time.time()
        if previous_func_info is None:
            previous_func_info = self.init_objective_function_value_info

        ground_set_size = current_set_info.get_ground_set_size()
        current_rule_set_size = current_set_info.get_current_set_size()
        added_rules = current_set_info.get_added_elems()
        deleted_rules = current_set_info.get_deleted_elems()
        intersection_previous_and_current_rules = current_set_info.get_intersection_previous_and_current_elems()

        self.call_set_sizes.append(current_rule_set_size)

        f0 = self.f0_minimize_rule_set_size(
            ground_set_size,
            current_rule_set_size
        )

        f1 = self.f1_minimize_total_nb_of_literals(
            previous_func_info.f1_nb_of_literals_minimization,
            added_rules=added_rules,
            deleted_rules=deleted_rules
        )

        f2, f3 = self.f2_f3_minimize_overlap_predicting_the_same_class(
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
            previous_func_info.f6_count_nb_of_rules_covering_each_example,
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

        fs = np.array([
            f0, f1, f2, f3, f4, f5, f6
        ]) / self.scale_factor

        objective_function_value = np.dot(self.lambda_array, fs)

        if self.stat_collector is not None:
            self.stat_collector.add_values(f0, f1, f2, f3, f4, f5, f6, objective_function_value)

        objective_function_value_info = IDSFuncInfo(
            objective_function_value=objective_function_value,
            f0_nb_of_rules_minimization=f0,
            f1_nb_of_literals_minimization=f1,
            f2_same_value_overlap_minimization=f2,
            f3_different_value_overlap_minimization=f3,
            f4_nb_of_distinct_values_covered=f4,
            f4_rule_counter_per_value_dict=f4_rule_counter_per_value_dict,
            f5_incorrect_cover_minimization=f5,
            f6_cover_each_example=f6,
            f6_count_nb_of_rules_covering_each_example=f6_count_nb_of_rules_covering_each_example
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.call_run_times.append(elapsed_time)

        return objective_function_value_info


def get_id(rule):
    if isinstance(rule, IDSRule):
        return rule.car.rid
    if isinstance(rule, ClassAssocationRule):
        return rule.id
    raise Exception("unsupported rule type")
