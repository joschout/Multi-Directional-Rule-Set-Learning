from typing import Optional, Set

from pyids.data_structures.ids_cacher import IDSCacher
from pyids.data_structures.ids_objective_function import ObjectiveFunctionParameters
from pyids.data_structures.ids_rule import IDSRule

from mdrsl.rule_models.ids.model_fitting.ids_abstract_base import IDSAbstractBase
from mdrsl.rule_models.ids.objective_function.ids_objective_function_value_reuse import IDSObjectiveFunctionValueReuse, IDSFuncInfo

from submodmax.value_reuse.abstract_optimizer import AbstractOptimizerValueReuse
from submodmax.value_reuse.deterministic_double_greedy_search import DeterministicDoubleGreedySearch
from submodmax.value_reuse.deterministic_local_search_simple_value_reuse import DeterministicLocalSearchValueReuse
from submodmax.value_reuse.randomized_double_greedy_search import RandomizedDoubleGreedySearch
from submodmax.value_reuse.set_info import SetInfo


class IDSValueReuse(IDSAbstractBase):

    def __init__(self):
        super().__init__()
        self.algorithms = dict(
            DLS=DeterministicLocalSearchValueReuse,
            DDGS=DeterministicDoubleGreedySearch,
            RDGS=RandomizedDoubleGreedySearch
        )
        self.rule_set_info: Optional[SetInfo] = None
        self.obj_func_val_info: Optional[IDSFuncInfo] = None

    def _optimize(self, params: ObjectiveFunctionParameters, algorithm: str, ground_set: Set[IDSRule],
                  cacher: IDSCacher, objective_scale_factor: float, debug: bool) -> Set[IDSRule]:

        objective_function = IDSObjectiveFunctionValueReuse(
            objective_func_params=params,
            cacher=cacher,
            scale_factor=objective_scale_factor,
            normalize=self.normalize
        )
        self.objective_function = objective_function

        optimizer: AbstractOptimizerValueReuse = self.algorithms[algorithm](
            objective_function=objective_function, ground_set=ground_set, debug=debug)

        rule_set_info: SetInfo
        obj_func_val_info: IDSFuncInfo
        rule_set_info, obj_func_val_info = optimizer.optimize()

        self.rule_set_info = rule_set_info
        self.obj_func_val_info = obj_func_val_info

        solution_set = rule_set_info.current_set
        return solution_set
