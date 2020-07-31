from typing import List, Dict

from mdrsl.utils.value_collection import ValueCollector

from mdrsl.rule_models.mids.objective_function.mids_objective_function_abstract import AbstractMIDSObjectiveFunction
from mdrsl.rule_models.mids.objective_function.mids_objective_function_parameters import ObjectiveFunctionParameters
from mdrsl.rule_models.mids.objective_function.mids_objective_function_statistics import MIDSObjectiveFunctionStatistics

TargetAttr = str


class MIDSOptimizationMetaData:

    def __init__(self, mids_objective_function: AbstractMIDSObjectiveFunction,
                 optimization_algorithm: str,
                 solution_set_size: int
                 ):

        self.solution_set_size: int = solution_set_size
        self.optimization_algorithm: str = optimization_algorithm

        # -- ObjectiveFunctionPARAMETER info

        mids_objective_function_params: ObjectiveFunctionParameters = mids_objective_function.objective_func_params

        self.ground_set_size: int = mids_objective_function_params.ground_set_size

        self.target_attrs: List[TargetAttr] = mids_objective_function_params.target_attrs
        self.nb_of_training_examples: int = mids_objective_function_params.nb_of_training_examples

        self.f1_upper_bound_nb_literals: int = mids_objective_function_params.f1_upper_bound_nb_literals
        self.f2_f3_target_attr_to_upper_bound_map: Dict[TargetAttr, int] \
            = mids_objective_function_params.f2_f3_target_attr_to_upper_bound_map
        self.f4_target_attr_to_dom_size_map: Dict[TargetAttr, int] \
            = mids_objective_function_params.f4_target_attr_to_dom_size_map
        self.f5_upper_bound: int = mids_objective_function_params.f5_upper_bound

        self.lambda_array: List[float] = mids_objective_function_params.lambda_array

        # -- ObjectiveFunction info

        self.is_normalized: bool = mids_objective_function.normalize
        self.scale_factor: float = mids_objective_function.scale_factor

        self.run_time_collector: ValueCollector = mids_objective_function.run_time_collector
        self.set_size_collector: ValueCollector = mids_objective_function.set_size_collector
        self.objective_function_value_stat_collector: MIDSObjectiveFunctionStatistics = mids_objective_function.stat_collector

        self.evaluation_call_counter: int = mids_objective_function.call_counter

