import random
import time
from typing import List, Optional, Dict

import fim

import pandas as pd

from pyarc.data_structures.antecedent import Antecedent
from pyarc.data_structures.consequent import Consequent
from pyarc.data_structures.item import Item
from pyarc.data_structures.car import ClassAssocationRule
from pyarc.data_structures.transaction_db import TransactionDB


def mine_CARs(df: pd.DataFrame, rule_cutoff: int,
              sample=False, random_seed=None,
              verbose: bool = True,
              **top_rules_kwargs) -> List[ClassAssocationRule]:
    """


    :param df: the (training) data to mine rules on
    :param rule_cutoff: the maximum number of rules to return
    :param sample: bool - if the generate nb of rules is larger than the rule_cutoff and sample == True,
        a random sample of rules rule_cutoff rules is returned
    :param random_seed:
    :param verbose:
    :param top_rules_kwargs:
    :return:
    """
    txns = TransactionDB.from_DataFrame(df)
    rules = top_rules(txns.string_representation,
                      appearance=txns.appeardict,  # NOTE: THIS IS VERY IMPORTANT; without this, any attribute can be
                      # the target of the class association rule
                      target_rule_count=rule_cutoff,
                      verbose=verbose,
                      **top_rules_kwargs)
    cars: List[ClassAssocationRule] = createCARs(rules)

    cars_subset: List[ClassAssocationRule]
    if len(cars) > rule_cutoff:
        if sample:
            if random_seed is not None:
                random.seed(random_seed)
            cars_subset = random.sample(cars, rule_cutoff)
        else:
            cars_subset = cars[:rule_cutoff]
    else:
        cars_subset = cars

    return cars_subset


def mine_unrestricted_CARS(df: pd.DataFrame, min_support = 0.01, min_confidence = 0.5,
                           max_length=7) -> List[ClassAssocationRule]:
    """
    :param df: the (training) data to mine rules on
    :param rule_cutoff: the maximum number of rules to return
    :param sample: bool - if the generate nb of rules is larger than the rule_cutoff and sample == True,
        a random sample of rules rule_cutoff rules is returned
    :param random_seed:
    :param verbose:
    :param top_rules_kwargs:
    :return:
    """
    txns = TransactionDB.from_DataFrame(df)

    min_support_percents = min_support * 100
    min_confidence_percents = min_confidence * 100
    CARs: List[ClassAssocationRule] = generateCARs(txns,
                                                   support=min_support_percents,
                                                   confidence=min_confidence_percents, maxlen=max_length)
    return CARs


def createCARs(rules) -> List[ClassAssocationRule]:
    """Function for converting output from fim.arules or fim.apriori
    to a list of ClassAssociationRules

    Parameters
    ----------
    rules : output from fim.arules or from generateCARs


    Returns
    -------
    list of CARs

    """
    CARs: List[ClassAssocationRule] = []

    for rule in rules:
        con_tmp, ant_tmp, support, confidence = rule

        con = Consequent(*con_tmp.split(":=:"))

        ant_items = [Item(*i.split(":=:")) for i in ant_tmp]
        ant = Antecedent(ant_items)

        CAR = ClassAssocationRule(ant, con, support=support, confidence=confidence)
        CARs.append(CAR)

    CARs.sort(reverse=True)

    return CARs


def generateCARs(transactionDB: TransactionDB,
                 support: float = 1, confidence: float = 50, maxlen: int = 10, **kwargs):
    """Function for generating ClassAssociationRules from a TransactionDB

    Parameters
    ----------
    :param transactionDB : TransactionDB

    support : float
        minimum support in percents if positive
        absolute minimum support if negative

    confidence : float
        minimum confidence in percents if positive
        absolute minimum confidence if negative

    maxlen : int
        maximum length of mined rules

    **kwargs :
        arbitrary number of arguments that will be
        provided to the fim.apriori function

    Returns
    -------
    list of CARs

    """
    appear = transactionDB.appeardict

    rules = fim.apriori(transactionDB.string_representation, supp=support, conf=confidence, mode="o", target="r",
                        report="sc", appear=appear, **kwargs, zmax=maxlen)

    return createCARs(rules)


def top_rules(transactions,
              appearance: Optional[Dict] = None,
              target_rule_count: int = 1000,
              init_support: float = 0.05,
              init_confidence: float = 0.5,
              confidence_step: float = 0.05,
              support_step: float = 0.05,
              min_length: int = 2,
              init_max_length: int = 3,
              total_timeout: float = 100.0,
              max_iterations: int = 30,
              verbose: bool = True):
    """
    Function for finding the best n (target_rule_count) rules from transaction list
    Returns list of mined rules. The rules are not ordered.

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
    :param verbose: bool
    """

    if appearance is None:
        appearance = {}

    start_time: float = time.time()

    # the length of a rule is at most the length of a transaction. (All transactions have the same length.)
    MAX_RULE_LEN: int = len(transactions[0])

    current_support: float = init_support
    current_confidence: float = init_confidence

    current_max_length: int = init_max_length

    keep_mining: bool = True

    is_max_length_decreased_due_timeout: bool = False
    current_iteration: int = 0

    last_rule_count = -1
    rules: Optional[List] = None

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

        current_rules = fim.arules(transactions, supp=current_support, conf=current_confidence, mode="o", report="sc",
                                   appear=appearance,
                                   zmax=current_max_length, zmin=min_length)
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
                    print(f"\tExecution time exceeded: {total_timeout}")
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
