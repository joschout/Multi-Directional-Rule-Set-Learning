import time
from typing import TypeVar, Union, Set

import numpy as np

from rule_models.ids.objective_function.ids_objective_function_abstract import AbstractIDSObjectiveFunction

from pyids.data_structures import IDSRuleSet
from pyids.data_structures.ids_objective_function import ObjectiveFunctionParameters

from submodmax.abstract_optimizer import AbstractSubmodularFunction

E = TypeVar('E')


class IDSObjectiveFunction(AbstractSubmodularFunction, AbstractIDSObjectiveFunction):
    def __init__(self, objective_func_params=ObjectiveFunctionParameters(), cacher=None, scale_factor=1,
                 normalize=True):
        AbstractIDSObjectiveFunction.__init__(self, objective_func_params, cacher=cacher, scale_factor=scale_factor,
                                              normalize=normalize)

    def f1_minimize_total_nb_of_literals(self, solution_set):
        f1_unnormalized = self.f1_upper_bound_nb_of_literals - solution_set.sum_rule_length()
        if self.normalize:
            f1 = f1_unnormalized / self.f1_upper_bound_nb_of_literals
        else:
            f1 = f1_unnormalized
        self._f1_boundary_check(f1)
        return f1

    def f2_minimize_overlap_predicting_the_same_class(self, solution_set):
        overlap_intraclass_sum = 0

        for i, r1 in enumerate(solution_set.ruleset):
            for j, r2 in enumerate(solution_set.ruleset):
                if i >= j:
                    continue

                if r1.car.consequent.value == r2.car.consequent.value:
                    overlap_tmp = self.cacher.overlap(r1, r2)

                    overlap_intraclass_sum += overlap_tmp

        f2_unnormalized = self.f2_f3_upper_bound - overlap_intraclass_sum

        if self.normalize:
            f2 = f2_unnormalized / self.f2_f3_upper_bound
        else:
            f2 = f2_unnormalized

        self._boundary_check(f2, 'f2')
        return f2

    def f3_minimize_overlap_predicting_different_class(self, solution_set):
        overlap_interclass_sum = 0

        for i, r1 in enumerate(solution_set.ruleset):
            for j, r2 in enumerate(solution_set.ruleset):
                if i >= j:
                    continue

                if r1.car.consequent.value != r2.car.consequent.value:
                    overlap_tmp = self.cacher.overlap(r1, r2)

                    overlap_interclass_sum += overlap_tmp

        f3_unnormalized = self.f2_f3_upper_bound - overlap_interclass_sum

        if self.normalize:
            f3 = f3_unnormalized / self.f2_f3_upper_bound
        else:
            f3 = f3_unnormalized
        self._boundary_check(f3, 'f3')
        return f3

    def f4_at_least_one_rule_per_target_value(self, solution_set):
        classes_covered = set()

        for rule in solution_set.ruleset:
            classes_covered.add(rule.car.consequent.value)

        f4_unnormalized = len(classes_covered)

        if self.normalize:
            f4 = f4_unnormalized / self.nb_of_target_values
        else:
            f4 = f4_unnormalized
        self._boundary_check(f4, 'f4')
        return f4

    def f5_minimize_incorrect_cover(self, solution_set):
        sum_incorrect_cover = 0

        for rule in solution_set.ruleset:

            incorrect_cover_size = np.sum(rule._incorrect_cover(self.quant_dataframe))
            # incorrect_cover_size = np.sum(rule.incorrect_cover(self.quant_dataframe))
            # print(f"IDS incorrect cover size: {incorrect_cover_size} for rule {rule}")
            sum_incorrect_cover += incorrect_cover_size

        # print(f"IDS f5 upper bound: {self.f5_upper_bound}")
        # print(f"IDS f5 sum incorrect cover: {sum_incorrect_cover}")
        f5_unnormalized = self.f5_upper_bound - sum_incorrect_cover
        if self.normalize:
            f5 = f5_unnormalized / self.f5_upper_bound
        else:
            f5 = f5_unnormalized
        self._boundary_check(f5, 'f5')
        return f5

    def f6_cover_each_example(self, solution_set):
        correctly_covered = np.zeros(self.nb_of_training_examples).astype(bool)

        for rule in solution_set.ruleset:
            correctly_covered = correctly_covered | rule.correct_cover(self.quant_dataframe)

        f6_unnormalized = np.sum(correctly_covered)
        if self.normalize:
            f6 = f6_unnormalized / self.nb_of_training_examples
        else:
            f6 = f6_unnormalized
        self._boundary_check(f6, 'f6')
        return f6

    def evaluate(self, solution_set: Union[IDSRuleSet, Set[E]]) -> float:
        if type(solution_set) == set:
            solution_set = IDSRuleSet(solution_set)

        if type(solution_set) != IDSRuleSet:
            raise Exception("Type of solution_set must be IDSRuleSet, but is ", type(solution_set))

        self.call_counter += 1
        self.call_set_sizes.append(len(solution_set))
        start_time = time.time()

        l = self.objective_func_params.params["lambda_array"]

        f0 = self.f0_minimize_rule_set_size(self.ground_set_size, len(solution_set)) if l[0] != 0 else 0
        f1 = self.f1_minimize_total_nb_of_literals(solution_set) if l[1] != 0 else 0
        f2 = self.f2_minimize_overlap_predicting_the_same_class(solution_set) if l[2] != 0 else 0
        f3 = self.f3_minimize_overlap_predicting_different_class(solution_set) if l[3] != 0 else 0
        f4 = self.f4_at_least_one_rule_per_target_value(solution_set) if l[4] != 0 else 0
        f5 = self.f5_minimize_incorrect_cover(solution_set) if l[5] != 0 else 0
        f6 = self.f6_cover_each_example(solution_set) if l[6] != 0 else 0

        self.f0_val = f0
        self.f1_val = f1
        self.f2_val = f2
        self.f3_val = f3
        self.f4_val = f4
        self.f5_val = f5
        self.f6_val = f6

        # print(f"IDS f1:{f1}")


        # print()
        # print(tabulate([['value', f0, f1, f2, f3, f4, f5, f6],
        #                 ['l*val', f0 * l[0], f1 * l[1], f2 * l[2], f3 * l[3], f4 * l[4], f5 * l[5], f6 * l[6]]
        #                 ],
        #                headers=['type', 'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6']))
        # print()

        fs = np.array([
            f0, f1, f2, f3, f4, f5, f6
        ]) / self.scale_factor

        result = np.dot(l, fs)

        if self.stat_collector is not None:
            self.stat_collector.add_values(f0, f1, f2, f3, f4, f5, f6, result)

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.call_run_times.append(elapsed_time)

        return result

    def f0(self, solution_set):
        current_nb_of_rules: int = len(solution_set)
        ground_set_size = len(self.objective_func_params.params['all_rules'])
        return self.f0_minimize_rule_set_size(ground_set_size=ground_set_size,
                                              current_nb_of_rules=current_nb_of_rules)

    def f1(self, solution_set):
        return self.f1_minimize_total_nb_of_literals(solution_set)

    def f2(self, solution_set):
        return self.f2_minimize_overlap_predicting_the_same_class(solution_set)

    def f3(self, solution_set):
        return self.f3_minimize_overlap_predicting_different_class(solution_set)

    def f4(self, solution_set):
        return self.f4_at_least_one_rule_per_target_value(solution_set)

    def f5(self, solution_set):
        return self.f5_minimize_incorrect_cover(solution_set)

    def f6(self, solution_set):
        return self.f6_cover_each_example(solution_set)
