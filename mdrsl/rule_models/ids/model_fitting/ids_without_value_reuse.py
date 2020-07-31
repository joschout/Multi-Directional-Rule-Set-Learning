from typing import Optional, Set

from pyids.data_structures.ids_objective_function import ObjectiveFunctionParameters
from pyids.data_structures.ids_rule import IDSRule
from pyids.data_structures.ids_ruleset import IDSRuleSet
from pyids.data_structures.ids_cacher import IDSCacher

from mdrsl.rule_models.ids.model_fitting.ids_abstract_base import IDSAbstractBase
from mdrsl.rule_models.ids.objective_function.ids_objective_function_without_value_reuse import IDSObjectiveFunction

from submodmax.abstract_optimizer import AbstractOptimizer
from submodmax.deterministic_double_greedy_search import DeterministicDoubleGreedySearch
from submodmax.deterministic_local_search import DeterministicLocalSearch
from submodmax.deterministic_local_search_pyids import DeterministicLocalSearchPyIDS
from submodmax.randomized_double_greedy_search import RandomizedDoubleGreedySearch
from submodmax.smooth_local_search import SmoothLocalSearch
from submodmax.smooth_local_search_pyids import SmoothLocalSearchPyIDS


class IDS(IDSAbstractBase):

    def __init__(self):
        super().__init__()
        self.algorithms = dict(
            SLS=SmoothLocalSearchPyIDS,
            DLS=DeterministicLocalSearchPyIDS,
            DLSRewrite=DeterministicLocalSearch,
            SLSRewrite=SmoothLocalSearch,
            DDGS=DeterministicDoubleGreedySearch,
            RDGS=RandomizedDoubleGreedySearch
        )
        self.solution_set: Optional[Set[IDSRule]] = None
        self.objective_function_value: Optional[float] = None

    def _optimize(self, params: ObjectiveFunctionParameters, algorithm: str, ground_set: Set[IDSRule],
                  cacher: IDSCacher, objective_scale_factor: float, debug: bool) -> Set[IDSRule]:

        objective_function = IDSObjectiveFunction(
            objective_func_params=params,
            cacher=cacher,
            scale_factor=objective_scale_factor, normalize=self.normalize)
        self.objective_function = objective_function

        optimizer: AbstractOptimizer = self.algorithms[algorithm](
            objective_function=objective_function, ground_set=ground_set, debug=debug)
        solution_set: Set[IDSRule] = optimizer.optimize()

        objective_function_value: float = objective_function.evaluate(IDSRuleSet(solution_set))

        self.solution_set = solution_set
        self.objective_function_value = objective_function_value

        return solution_set
