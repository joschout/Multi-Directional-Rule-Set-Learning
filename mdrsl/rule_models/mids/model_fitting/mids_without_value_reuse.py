from typing import Dict, Type, Set, Optional

from mdrsl.rule_models.mids.model_fitting.mids_abstract_base import MIDSAbstractBase
from mdrsl.rule_models.mids.objective_function.mids_objective_function_parameters import ObjectiveFunctionParameters
from mdrsl.rule_models.mids.objective_function.mids_objective_function_without_value_reuse import MIDSObjectiveFunction
from mdrsl.rule_models.mids.mids_rule import MIDSRule
from mdrsl.rule_models.mids.mids_ruleset import MIDSRuleSet
from submodmax.abstract_optimizer import AbstractOptimizer
from submodmax.deterministic_double_greedy_search import DeterministicDoubleGreedySearch
from submodmax.deterministic_local_search import DeterministicLocalSearch
from submodmax.deterministic_local_search_pyids import DeterministicLocalSearchPyIDS
from submodmax.randomized_double_greedy_search import RandomizedDoubleGreedySearch
from submodmax.smooth_local_search import SmoothLocalSearch
from submodmax.smooth_local_search_pyids import SmoothLocalSearchPyIDS
from submodmax.ground_set_returner import GroundSetReturner


class MIDS(MIDSAbstractBase):
    """
    Encapsulates the MIDS algorithm.
    """
    algorithms: Dict[str, Type[AbstractOptimizer]] = dict(
        SLSPyIDS=SmoothLocalSearchPyIDS,
        SLS=SmoothLocalSearch,
        DLSPyIDS=DeterministicLocalSearchPyIDS,
        DLS=DeterministicLocalSearch,
        DDGS=DeterministicDoubleGreedySearch,
        RDGS=RandomizedDoubleGreedySearch,
        GroundSetReturner=GroundSetReturner
    )

    def __init__(self):
        super().__init__()
        self.solution_set: Optional[Set[MIDSRule]] = None
        self.objective_function_value: Optional[float] = None

    def _optimize(self, objective_function_parameters: ObjectiveFunctionParameters, algorithm: str,
                  objective_scale_factor: float, debug: bool) -> Set[MIDSRule]:
        # init objective function
        objective_function = MIDSObjectiveFunction(objective_func_params=objective_function_parameters,
                                                   cover_checker=self.cover_checker,
                                                   overlap_checker=self.overlap_checker,
                                                   scale_factor=objective_scale_factor)
        self.objective_function = objective_function

        optimizer: AbstractOptimizer = self.algorithms[algorithm](
            objective_function=objective_function,
            ground_set=objective_function_parameters.all_rules.ruleset,
            debug=debug)

        solution_set: Set[MIDSRule] = optimizer.optimize()
        objective_function_value: float = objective_function.evaluate(MIDSRuleSet(solution_set))

        self.solution_set = solution_set
        self.objective_function_value = objective_function_value

        self.nb_of_objective_function_calls_necessary_for_training = objective_function.call_counter
        return solution_set
