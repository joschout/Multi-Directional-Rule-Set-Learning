import math
import random
import time
from collections import deque
from typing import Optional, List

import pandas as pd

from mdrsl.data_structures.rules.multi_target_class_association_rule import MCAR
from mdrsl.rule_generation.association_rule_mining.apyori_impl.mine_mt_rules_from_transactions_with_apyori import (
    mine_MCARs_from_transactions_using_apyori)
from mdrsl.rule_generation.association_rule_mining.frequent_itemset_mining import dataframe_to_list_of_transactions


class MiningParameters:
    def __init__(self, support: float, confidence: float, max_rule_length, iteration=0):
        if not (0 <= support <= 1.0):
            raise Exception()
        self.support: float = support
        if not (0 <= confidence <= 1.0):
            raise Exception()
        self.confidence: float = confidence
        if max_rule_length <= 0:
            raise Exception()
        self.max_rule_length: int = max_rule_length
        self.iteration = iteration

    def get_copy(self):
        return MiningParameters(support=self.support, confidence=self.confidence, max_rule_length=self.max_rule_length,
                                iteration=self.iteration+1)

    def __eq__(self, other):
        return (
                math.isclose(self.support, other.support) and
                math.isclose(self.confidence, other.confidence) and
                self.max_rule_length == other.max_rule_length
        )

    def __str__(self):
        return f"it={self.iteration}, support={self.support}, confidence={self.confidence}," \
               f" max_rule_length={self.max_rule_length}"


class IterativeRuleMining:
    """
    Goal: find the n 'best' rules.
    NOTE: what is 'best' ?
    TODO


    """

    def __init__(self, verbose: bool = False, should_sample_if_found_too_many_rules: bool = False):
        self.verbose: bool = verbose
        self.random_seed: Optional[float] = None
        self.should_sample_if_found_too_many_rules: bool = should_sample_if_found_too_many_rules

        # --- General settings ---
        self.total_time_out_limit: float = 100.0  # max time in seconds

        self.init_support: float = 0.05
        self.init_confidence: float = 0.5

        self.confidence_step: float = 0.05
        self.support_step: float = 0.05

        self.min_length: int = 2
        self.init_max_length: int = 10

        self.max_iterations: int = 1

        self.max_wanted_rule_length: int = 10

        # --- Settings of a single run ---

        self.last_three_params = None

    # def _init_iteration_vars(self):
    #     self.current_support: float = self.init_support
    #     self.current_confidence: float = self.init_confidence
    #
    #     self.current_max_length: int = self.init_max_length
    #
    #     self.last_rule_count = -1
    #     self.current_rules: Optional[List[MCAR]] = None

    def __get_initial_mining_parameters(self):
        self.last_three_params = deque()
        self.last_three_params.append(MiningParameters(support=self.init_support, confidence=self.init_confidence,
                                                       max_rule_length=self.init_max_length))

    def get_current_mining_params(self) -> MiningParameters:
        if self.last_three_params is None or len(self.last_three_params) <= 0:
            raise Exception()
        else:
            return self.last_three_params[-1]

    def add_update_mining_params(self, new_mining_params: MiningParameters) -> None:
        if len(self.last_three_params) >= 3:
            self.last_three_params.popleft()
        self.last_three_params.append(new_mining_params)

    def __subsample_found_rules(self, best_rules, n_rules_to_sample: int) -> List[MCAR]:
        if self.random_seed is not None:
            random.seed(self.random_seed)
        return random.sample(best_rules, n_rules_to_sample)

    def mine_n_rules(self, df: pd.DataFrame, rule_cutoff: int) -> List[MCAR]:
        """
        Tries to find the 'rule_cutoff' best rules in using an iterative approach.
        """

        transactions: List[List[str]] = dataframe_to_list_of_transactions(df)
        best_rules: Optional[List[MCAR]] = self._iteratively_find_best_rules(
            transactions, target_rule_count=rule_cutoff)

        if best_rules is None:
            raise Exception("no rules found")

        if len(best_rules) > rule_cutoff:
            if self.should_sample_if_found_too_many_rules:
                rule_subset = self.__subsample_found_rules(best_rules, rule_cutoff)
            else:
                rule_subset = best_rules[:rule_cutoff]
                # TODO: this is pretty fishy
        else:
            rule_subset = best_rules
        return rule_subset

    def _is_target_rule_count_exceeded(self, target_rule_count, current_nb_of_rules: int) -> bool:
        if current_nb_of_rules >= target_rule_count:
            if self.verbose:
                print(f"\tTarget rule count satisfied: found {current_nb_of_rules} > target nb {target_rule_count}")
            return True
        return False

    def _is_time_out_exceeded(self, start_time: float) -> bool:
        current_execution_time = time.time() - start_time

        # if timeout limit exceeded
        if current_execution_time > self.total_time_out_limit:
            if self.verbose:
                print(f"Execution time exceeded:{current_execution_time:0.2f}s > {self.total_time_out_limit:0.2f}s")
            return True
        return False

    def _iteratively_find_best_rules(self,
                                     transactions: List[List[str]],
                                     target_rule_count: int = 1000
                                     ):
        """
        Function for finding the best n (target_rule_count) rules from transaction list.
            PROBLEM: how to define 'best'?

            Iteratively:
                Search for the rules under the current mining parameters.
                Check the properties of the found rules.
                If there is still room for improvement,
                    Then update the mining parameters,



            STOP if:
                - max nb of iterations is reached (default:  30).
                - the current nb of rules is more than the nb of rules we are looking for.
                - the time out is reach

            FIND all rules with as constraints:
                - min_support
                - min_confidence
                - max_length


        Parameters
        ----------
        :param transactions : 2D array of strings,  e.g. [["a:=:1", "b:=:3"], ["a:=:4", "b:=:2"]]
        :param target_rule_count : int - target number of rules to mine

        Returns
        -------
        list of mined rules. The rules are not ordered.

        """
        start_time: float = time.time()

        # the length of a rule is at most the length of a transaction. (All transactions have the same length.)
        TRANSACTION_LENGTH: int = len(transactions[0])
        rule_length_upper_boundary = min(TRANSACTION_LENGTH, self.max_wanted_rule_length)

        keep_mining: bool = True

        self.__get_initial_mining_parameters()

        # current_support: float = self.init_support
        # current_confidence: float = self.init_confidence
        # current_max_length: int = self.init_max_length

        last_rule_count = -1
        current_rules: Optional[List[MCAR]] = None

        current_iteration = 0

        if self.verbose:
            print("STARTING top_rules")
        while keep_mining and not self._is_max_n_iterations_reached(
                current_iteration) and not self._is_time_out_exceeded(start_time):
            current_iteration += 1
            if self._is_stuck_in_local_optimum():
                break

            current_mining_params: MiningParameters = self.get_current_mining_params()

            if self.verbose:
                print(f"--- iteration {current_iteration} ---")
                print((f"Running apriori with setting: "
                       f"confidence={current_mining_params.confidence}, "
                       f"support={current_mining_params.support}, "
                       f"min_length={self.min_length}, "
                       f"max_length={current_mining_params.max_rule_length}, "
                       f"TRANSACTION_LENGTH={TRANSACTION_LENGTH}",
                       f"max_wanted_rule_length={self.max_wanted_rule_length}"
                       ))

            current_rules: List[MCAR] = mine_MCARs_from_transactions_using_apyori(
                transactions, min_support=current_mining_params.support,
                min_confidence=current_mining_params.confidence,
                max_length=current_mining_params.max_rule_length)

            current_nb_of_rules = len(current_rules)

            if self.verbose:
                print(f"Rule count: {current_nb_of_rules}, Iteration: {current_iteration}")

            # if: nb of target rules is succeeded
            # then: increase the confidence

            if self._is_target_rule_count_exceeded(target_rule_count, current_nb_of_rules):
                if self.verbose:
                    print(f"Target rule count  exceeded: {current_nb_of_rules} > {target_rule_count}")
                if (1.0 - current_mining_params.confidence) > self.confidence_step:
                    next_mining_params = current_mining_params.get_copy()
                    next_mining_params.confidence += self.confidence_step
                    self.add_update_mining_params(next_mining_params)
                    if self.verbose:
                        print(f"\tINcreasing confidence to {next_mining_params.confidence}")
                elif (1.0 - current_mining_params.support) > self.support_step:
                    next_mining_params = current_mining_params.get_copy()
                    next_mining_params.support += self.support_step
                    self.add_update_mining_params(next_mining_params)
                    if self.verbose:
                        print(f"\tINcreasing support to {next_mining_params.support}")
                else:
                    if self.verbose:
                        print("Target rule count exceeded, no options left")
                    keep_mining = False
            else:
                if self.verbose:
                    print(f"Target rule count NOT exceeded: {current_nb_of_rules} < {target_rule_count}")
                # NB of rules is not exceeded!

                # what can we do?
                # * IF we have not reached the max rule length
                #      and the nb of rules went up since last time
                #   THEN:
                #      increase the rule length
                #

                # if we can still increase our rule length AND
                # the number of rules found has changed (increased?) since last time AND
                # there has
                if (
                        current_mining_params.max_rule_length < rule_length_upper_boundary and
                        last_rule_count != current_nb_of_rules
                ):
                    next_mining_params = current_mining_params.get_copy()
                    next_mining_params.max_rule_length += 1
                    self.add_update_mining_params(next_mining_params)
                    if self.verbose:
                        print(f"\tIncreasing max_length {next_mining_params.max_rule_length}")

                # if we can still increase our rule length AND
                #
                # we can still increase our support
                # THEN:
                # increase our support
                # increment our max length
                elif (
                        current_mining_params.max_rule_length < rule_length_upper_boundary and
                        current_mining_params.support <= 1 - self.support_step
                ):
                    next_mining_params = current_mining_params.get_copy()
                    next_mining_params.support -= self.support_step
                    self.add_update_mining_params(next_mining_params)
                    if self.verbose:
                        print(f"\tDecreasing minsup to {next_mining_params.support}")
                # IF we can still decrease our confidence
                # THEN decrease our confidence
                elif current_mining_params.confidence > self.confidence_step:
                    next_mining_params = current_mining_params.get_copy()
                    next_mining_params.confidence -= self.confidence_step
                    self.add_update_mining_params(next_mining_params)
                    if self.verbose:
                        print(f"\tDecreasing confidence to {next_mining_params.confidence}")
                else:
                    if self.verbose:
                        print("\tAll options exhausted")
                    keep_mining = False
            if self.verbose:
                end_of_current_iteration_message = f"--- end iteration {current_iteration} ---"
                print(end_of_current_iteration_message)
                print("-" * len(end_of_current_iteration_message))
            last_rule_count = current_nb_of_rules

        if self.verbose:
            print(f"FINISHED top_rules after {current_iteration} iterations")
        return current_rules

    def _is_max_n_iterations_reached(self, current_iteration):
        is_max_reached = current_iteration >= self.max_iterations
        if is_max_reached and self.verbose:
            print("Max iterations reached")
        return is_max_reached

    def _is_stuck_in_local_optimum(self):
        if len(self.last_three_params) == 3 and self.last_three_params[0] == self.last_three_params[2]:
            print(f"Stuck in local optimum, see iterations {self.last_three_params[0].iteration} and {self.last_three_params[2].iteration}")
            print("\t", str(self.last_three_params[0]))
            print("\t", str(self.last_three_params[2]))
            return True
        elif len(self.last_three_params) == 3 and self.last_three_params[1] == self.last_three_params[2]:
            print(f"Stuck in local optimum, see iterations {self.last_three_params[1].iteration} and {self.last_three_params[2].iteration}")
            print("\t", str(self.last_three_params[1]))
            print("\t", str(self.last_three_params[2]))
        else:
            return False
