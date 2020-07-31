from typing import List, Set

from mdrsl.rule_models.mids.objective_function.mids_objective_function_statistics import MIDSObjectiveFunctionStatistics

from pyids.data_structures.ids_objective_function import ObjectiveFunctionParameters
from pyids.data_structures.ids_ruleset import IDSRuleSet
from pyids.data_structures.ids_rule import IDSRule
from pyids.data_structures import IDSCacher


class AbstractIDSObjectiveFunction:

    valid_sub_func_names = {'f2', 'f3', 'f4', 'f5' , 'f6'}

    def __init__(self, objective_func_params: ObjectiveFunctionParameters,
                 cacher: IDSCacher, scale_factor: float, normalize=True):

        self.normalize = normalize

        self.scale_factor: float = scale_factor
        self.objective_func_params: ObjectiveFunctionParameters = objective_func_params
        self.lambda_array = objective_func_params.params["lambda_array"]

        self.call_counter: int = 0
        self.call_run_times: List[float] = []
        self.call_set_sizes: List[int] = []

        all_rules: IDSRuleSet = self.objective_func_params.params["all_rules"]
        ground_set: Set[IDSRule] = set(all_rules.ruleset)
        self.ground_set_size = len(ground_set)

        self.quant_dataframe = self.objective_func_params.params["quant_dataframe"]
        self.nb_of_training_examples = self.quant_dataframe.dataframe.shape[0]
        self.nb_of_target_values = self.quant_dataframe.dataframe.iloc[:, -1].nunique()

        self.f1_upper_bound_nb_of_literals = self._f1_upper_bound(ground_set)
        self.f2_f3_upper_bound = self._f2_f3_upper_bound(self.nb_of_training_examples, self.ground_set_size)
        self.f5_upper_bound = self._f5_upper_bound(self.nb_of_training_examples, self.ground_set_size)

        self.stat_collector = MIDSObjectiveFunctionStatistics()

        self.cacher: IDSCacher = cacher

    def _boundary_check(self, val, func_name):
        if func_name not in AbstractIDSObjectiveFunction.valid_sub_func_names:
            raise Exception(f"{func_name} does not indicate an IDS sub-function")
        else:
            if self.normalize:
                self.__normalized_boundary_check(val, func_name)
            else:
                if val < 0:
                    raise Exception(f"UN-normalized IDS got negative value {str(val)} for sub-function {func_name}")

    def __normalized_boundary_check(self, val, func_name):
        if val < 0:
            raise Exception(str(func_name) + " < 0:", str(val))
        if self.normalize:
            if val > 1:
                raise Exception(str(func_name) + " > 1:", str(val))

    @staticmethod
    def _f1_upper_bound(ground_set: Set[IDSRule]) -> int:
        lengths_of_rules_in_ground_set: List[int] = [len(rule) for rule in ground_set]
        L_max: int = max(lengths_of_rules_in_ground_set)

        nb_of_ground_rules: int = len(ground_set)

        return L_max * nb_of_ground_rules

    @staticmethod
    def _f2_f3_upper_bound(nb_of_training_examples: int, nb_of_ground_rules: int) -> int:
        return nb_of_training_examples * nb_of_ground_rules ** 2

    @staticmethod
    def _f5_upper_bound(nb_of_training_examples: int, nb_of_ground_rules: int) -> int:
        return nb_of_training_examples * nb_of_ground_rules

    def f0_minimize_rule_set_size(self, ground_set_size: int, current_nb_of_rules: int):
        f0_unnormalized = ground_set_size - current_nb_of_rules

        if self.normalize:
            f0 = f0_unnormalized / ground_set_size
        else:
            f0 = f0_unnormalized

        self._f0_boundary_check(f0)
        return f0

    def _f0_boundary_check(self, val):

        if self.normalize:
            self.__normalized_boundary_check(val, 'f0')
        else:
            if val < 0:
                raise Exception("f0 < 0: " + str(val))
            if val > self.ground_set_size:
                raise Exception("f0 > " + str(self.ground_set_size) + ": " + str(val))

    def _f1_boundary_check(self, val):

        if self.normalize:
            self.__normalized_boundary_check(val, 'f1')
        else:
            if val < 0:
                raise Exception("f1 < 0: " + str(val))
            if val > self.f1_upper_bound_nb_of_literals:
                raise Exception("f1 > " + str(self.f1_upper_bound_nb_of_literals) + ": " + str(val))
