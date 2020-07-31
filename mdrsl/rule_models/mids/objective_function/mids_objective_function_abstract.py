import sys

from mdrsl.rule_models.mids.cover.cover_checker import CoverChecker
from mdrsl.rule_models.mids.cover.overlap_cacher import OverlapChecker
from mdrsl.rule_models.mids.objective_function.mids_objective_function_parameters import ObjectiveFunctionParameters
from mdrsl.rule_models.mids.objective_function.f2_f3_cacher import create_f2_f3_cache, \
    estimate_upper_bound_cache_size_in_nb_of_integers
from mdrsl.rule_models.mids.objective_function.mids_objective_function_statistics import MIDSObjectiveFunctionStatistics
from mdrsl.utils.value_collection import ValueCollector


class AbstractMIDSObjectiveFunction:

    should_cache_f2_f3 = False

    def __init__(self,
                 objective_func_params: ObjectiveFunctionParameters,
                 cover_checker: CoverChecker,
                 overlap_checker: OverlapChecker,
                 scale_factor=1.0
                 ):
        self.normalize = True

        self.objective_func_params: ObjectiveFunctionParameters = objective_func_params
        self.scale_factor: float = scale_factor

        self.call_counter: int = 0
        self.run_time_collector = ValueCollector()
        self.set_size_collector = ValueCollector()

        # self.call_run_times: List[float] = []
        # self.call_set_sizes: List[int] = []

        self.stat_collector = MIDSObjectiveFunctionStatistics()

        if AbstractMIDSObjectiveFunction.should_cache_f2_f3:
            print("INITIALIZE f2 f3 cache")
            rough_estimate_upperbound_nb_of_integers = estimate_upper_bound_cache_size_in_nb_of_integers(
                len(self.objective_func_params.all_rules))
            print("max nb of integers necessary:", str(rough_estimate_upperbound_nb_of_integers))

            int_byte_size = sys.getsizeof(int())
            rough_estimate_bytes_necessary = int_byte_size * rough_estimate_upperbound_nb_of_integers
            print("rough estimate nb of bytes necessary:", rough_estimate_bytes_necessary)

            self.f2_f3_cache = create_f2_f3_cache(self.objective_func_params.all_rules, overlap_checker=overlap_checker,
                                                  quant_dataframe=self.objective_func_params.quant_dataframe,
                                                  f2_f3_target_attr_to_upper_bound_map=objective_func_params.f2_f3_target_attr_to_upper_bound_map,
                                                  nb_of_target_attributes=objective_func_params.nb_of_target_attrs)
            print("FINISHED INITIALIZATION f2 f3 cache")

        self.cover_checker: CoverChecker = cover_checker
        self.overlap_checker: OverlapChecker = overlap_checker

    def _normalized_boundary_check(self, val, func_name):
        epsilon = 0.00000000000001

        if val < 0 and abs(val - 0) > epsilon:
            raise Exception(str(func_name) + " < 0:", str(val))
        if self.normalize:
            if val > 1 and abs(val - 1) > epsilon:
                raise Exception(str(func_name) + " > 1:", str(val))

    def f0_minimize_rule_set_size(self, ground_set_size: int, current_nb_of_rules: int):
        """
        Minimize the number of rules in the rule set

        :return:
        """

        f0_unnormalized = ground_set_size - current_nb_of_rules

        if self.normalize and ground_set_size > 0:
            f0 = f0_unnormalized / ground_set_size
        else:
            f0 = f0_unnormalized
        self._normalized_boundary_check(f0, 'f0')
        return f0
