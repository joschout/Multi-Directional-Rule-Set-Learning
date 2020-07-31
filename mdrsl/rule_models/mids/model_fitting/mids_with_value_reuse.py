from typing import Optional, Dict, Type, Set

from mdrsl.rule_models.mids.model_fitting.mids_abstract_base import MIDSAbstractBase
from mdrsl.rule_models.mids.objective_function.mids_objective_function_parameters import ObjectiveFunctionParameters
from mdrsl.rule_models.mids.objective_function.mids_objective_function_value_reuse import MIDSObjectiveFunctionValueReuse, MIDSFuncInfo
from mdrsl.rule_models.mids.mids_rule import MIDSRule
from submodmax.value_reuse.abstract_optimizer import AbstractOptimizerValueReuse
from submodmax.value_reuse.deterministic_double_greedy_search import DeterministicDoubleGreedySearch
from submodmax.value_reuse.deterministic_local_search_simple_value_reuse import DeterministicLocalSearchValueReuse
from submodmax.value_reuse.randomized_double_greedy_search import RandomizedDoubleGreedySearch
from submodmax.value_reuse.set_info import SetInfo
from submodmax.value_reuse.ground_set_returner import GroundSetReturner


class MIDSValueReuse(MIDSAbstractBase):
    """
    Encapsulates the MIDS algorithm.
    """
    algorithms: Dict[str, Type[AbstractOptimizerValueReuse]] = dict(
        DLS=DeterministicLocalSearchValueReuse,
        DDGS=DeterministicDoubleGreedySearch,
        RDGS=RandomizedDoubleGreedySearch,
        GroundSetReturner=GroundSetReturner
    )

    def __init__(self):
        super().__init__()

        self.rule_set_info: Optional[SetInfo] = None
        self.obj_func_val_info: Optional[MIDSFuncInfo] = None

    def _optimize(self, objective_function_parameters: ObjectiveFunctionParameters, algorithm: str,
                  objective_scale_factor: float, debug: bool) -> Set[MIDSRule]:
        # init objective function
        objective_function = MIDSObjectiveFunctionValueReuse(objective_func_params=objective_function_parameters,
                                                             cover_checker=self.cover_checker,
                                                             overlap_checker=self.overlap_checker,
                                                             scale_factor=objective_scale_factor)
        self.objective_function = objective_function

        optimizer: AbstractOptimizerValueReuse = self.algorithms[algorithm](
            objective_function=objective_function,
            ground_set=objective_function_parameters.all_rules.ruleset,
            debug=debug)

        rule_set_info: SetInfo
        obj_func_val_info: MIDSFuncInfo
        rule_set_info, obj_func_val_info = optimizer.optimize()

        self.rule_set_info = rule_set_info
        self.obj_func_val_info = obj_func_val_info
        solution_set = rule_set_info.current_set

        self.nb_of_objective_function_calls_necessary_for_training = objective_function.call_counter
        return solution_set
