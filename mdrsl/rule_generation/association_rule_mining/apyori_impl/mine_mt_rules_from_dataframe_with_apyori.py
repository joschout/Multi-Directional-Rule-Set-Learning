import random

import numpy as np
from typing import List, Optional, Dict
import pandas as pd
import time

from mdrsl.rule_generation.association_rule_mining.apyori_impl.mine_mt_rules_from_transactions_with_apyori import (
    mine_MCARs_from_transactions_using_apyori)
from mdrsl.rule_generation.association_rule_mining.frequent_itemset_mining import (
    dataframe_to_list_of_transactions, run_fim_apriori, dataframe_to_list_of_transactions_with_encoding)
from mdrsl.data_structures.rules.multi_target_class_association_rule import MCAR


def mine_MCARs_from_df_using_apyori(df,
                                    min_support: float = 0.1, min_confidence: float = 0.0, min_lift=0.0,
                                    max_length=None) -> List[MCAR]:
    transactions = dataframe_to_list_of_transactions(df)
    return mine_MCARs_from_transactions_using_apyori(
        transactions,
        min_support=min_support, min_confidence=min_confidence,
        min_lift=min_lift, max_length=max_length)


def mine_MCARs_from_df_using_apyori_with_encodings(df,
                                    min_support: float = 0.1, min_confidence: float = 0.0, min_lift=0.0,
                                    max_length=None) -> List[MCAR]:
    transactions, item_encoder = dataframe_to_list_of_transactions_with_encoding(df)
    return mine_MCARs_from_transactions_using_apyori(
        transactions,
        min_support=min_support, min_confidence=min_confidence,
        min_lift=min_lift, max_length=max_length, item_encoder=item_encoder)


def mine_MCARs(df, rule_cutoff: int,
               sample=False, random_seed=None,
               verbose: bool = True,
               **top_rules_kwargs) -> List[MCAR]:

    transactions: List[List[str]] = dataframe_to_list_of_transactions(df)
    mcars: Optional[List[MCAR]] = _top_rules_MIDS(transactions,
                                                  target_rule_count=rule_cutoff,
                                                  verbose=verbose)

    if mcars is None:
        raise Exception("no MCARs found as input for MIDS")

    if len(mcars) > rule_cutoff:
        if sample:
            if random_seed is not None:
                random.seed(random_seed)
            mcars_subset = random.sample(mcars, rule_cutoff)
        else:
            mcars_subset = mcars[:rule_cutoff]
    else:
        mcars_subset = mcars
    return mcars_subset


if __name__ == '__main__':

    df_total = pd.DataFrame({
        'A': np.array([1] * 4, dtype='float32'),
        'B': np.array([2] * 4, dtype='float32'),
        'C': np.array([3] * 4, dtype='float32'),
        'D': np.array([4] * 4, dtype='float32')
    })

    print(df_total)

    itemsets = dataframe_to_list_of_transactions(df_total)

    support_threshold = 0.1
    dataset_transactions = dataframe_to_list_of_transactions(df_total)  # type: List[List[str]]

    cars = mine_MCARs_from_transactions_using_apyori(dataset_transactions, min_support=support_threshold)
    for car in cars:
        print(car)

    print("---")
    fim_frequent_itemsets = run_fim_apriori(df_total, support_threshold)
    print(fim_frequent_itemsets)


def _top_rules_MIDS(transactions: List[List[str]],
                    appearance: Optional[Dict] = None,

                    target_rule_count: int = 1000,

                    init_support: float = 0.05,
                    init_confidence: float = 0.5,

                    confidence_step: float = 0.05,
                    support_step: float = 0.05,

                    min_length: int = 2,
                    init_max_length: int = 3,

                    total_timeout: float = 100.0,  # max time in seconds
                    max_iterations: int = 30,
                    verbose: bool = True
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
    :param appearance : dict - dictionary specifying rule appearance
    :param target_rule_count : int - target number of rules to mine
    :param init_support : float - support from which to start mining
    :param init_confidence : float - confidence from which to start mining
    :param confidence_step : float
    :param support_step : float
    :param min_length : int - minimum len of rules to mine
    :param init_max_length : int - maximum len from which to start mining
    :param total_timeout : float - maximum execution time of the function
    :param max_iterations : int - maximum iterations to try before stopping execution
    :param verbose : bool

    Returns
    -------
    list of mined rules. The rules are not ordered.

    """

    if appearance is None:
        appearance = {}

    start_time: float = time.time()

    # the length of a rule is at most the length of a transaction. (All transactions have the same length.)

    # max_rule_length_wanted = 10
    # MAX_RULE_LEN: int = min(len(transactions[0]), max_rule_length_wanted)
    MAX_RULE_LEN: int = len(transactions[0])


    current_support: float = init_support
    current_confidence: float = init_confidence

    current_max_length: int = init_max_length

    keep_mining: bool = True

    is_max_length_decreased_due_timeout = False
    current_iteration = 0

    last_rule_count = -1
    rules: Optional[List[MCAR]] = None

    if verbose:
        print("STARTING top_rules")
    while keep_mining:
        current_iteration += 1

        if current_iteration > max_iterations:
            if verbose:
                print("Max iterations reached")
            break

        if verbose:
            print(f"--- iteration {current_iteration} ---")
            print((f"Running apriori with setting: "
                   f"confidence={current_confidence}, "
                   f"support={current_support}, "
                   f"min_length={min_length}, "
                   f"max_length={current_max_length}, "
                   f"MAX_RULE_LEN={MAX_RULE_LEN}"
                   ))

        current_rules: List[MCAR] = mine_MCARs_from_transactions_using_apyori(
            transactions, min_support=current_support, min_confidence=current_confidence, max_length=current_max_length)
        # rules_current = fim.arules(transactions, supp=support, conf=conf, mode="o", report="sc", appear=appearance,
        #                            zmax=maxlen, zmin=minlen)

        current_nb_of_rules = len(current_rules)

        # assign
        rules = current_rules

        if verbose:
            print(f"Rule count: {current_nb_of_rules}, Iteration: {current_iteration}")

        if current_nb_of_rules >= target_rule_count:
            keep_mining = False
            if verbose:
                print(f"\tTarget rule count satisfied: {target_rule_count}")
        else:
            current_execution_time = time.time() - start_time

            # if timeout limit exceeded
            if current_execution_time > total_timeout:
                if verbose:
                    print("Execution time exceeded:", total_timeout)
                keep_mining = False

            # if we can still increase our rule length AND
            # the number of rules found has changed (increased?) since last time AND
            # there has
            elif current_max_length < MAX_RULE_LEN and last_rule_count != current_nb_of_rules and not is_max_length_decreased_due_timeout:
                current_max_length += 1
                last_rule_count = current_nb_of_rules
                if verbose:
                    print(f"\tIncreasing max_length {current_max_length}")

            # if we can still increase our rule length AND
            #
            # we can still increase our support
            # THEN:
            # increase our support
            # increment our max length
            elif current_max_length < MAX_RULE_LEN and is_max_length_decreased_due_timeout and current_support <= 1 - support_step:
                current_support += support_step
                current_max_length += 1
                last_rule_count = current_nb_of_rules
                is_max_length_decreased_due_timeout = False

                if verbose:
                    print(f"\tIncreasing maxlen to {current_max_length}")
                    print(f"\tIncreasing minsup to {current_support}")
            # IF we can still decrease our confidence
            # THEN decrease our confidence
            elif current_confidence > confidence_step:
                current_confidence -= confidence_step
                if verbose:
                    print(f"\tDecreasing confidence to {current_confidence}")
            else:
                if verbose:
                    print("\tAll options exhausted")
                keep_mining = False
        if verbose:
            end_of_current_iteration_message = f"--- end iteration {current_iteration} ---"
            print(end_of_current_iteration_message)
            print("-" * len(end_of_current_iteration_message))
    if verbose:
        print(f"FINISHED top_rules after {current_iteration} iterations")
    return rules
