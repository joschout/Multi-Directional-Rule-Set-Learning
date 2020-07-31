from typing import Set

from mdrsl.data_structures.rules.multi_target_class_association_rule import MCAR
from mdrsl.data_structures.item import Literal
from mdrsl.data_structures.rules.rule_part import Antecedent, Consequent


def pretty_print_ids_car(car, prefix=None):
    if prefix is None:
        prefix = ""
    print(prefix + ids_car_to_pretty_string(car))


def ids_car_to_pretty_string(car) -> str:
    antecedent = car.antecedent
    consequent = car.consequent

    ant_items = list(antecedent.itemset.items())
    ant_items_str =  ["{}={}".format(key, val) for key, val in ant_items]
    ant_str = ", ".join(ant_items_str)

    cons_str = str(consequent.attribute) + "=" + str(consequent.value)

    args = [ant_str, cons_str]
    result = "{} -> {}".format(*args)
    return result


def pretty_print_mids_mcar(mcar: MCAR, prefix=None):
    if prefix is None:
        prefix = ""
    print(prefix + mids_mcar_to_pretty_string(mcar))


def mids_mcar_to_pretty_string(mcar: MCAR) -> str:
    antecedent: Antecedent = mcar.antecedent
    consequent: Consequent = mcar.consequent

    ant_literals: Set[Literal] = antecedent.get_literals()
    ant_literals_str = [str(lit) for lit in ant_literals]
    ant_literals_str.sort()
    ant_str = ", ".join(ant_literals_str)

    cons_literals: Set[Literal] = consequent.get_literals()
    cons_literals_str = [str(lit) for lit in cons_literals]
    cons_literals_str.sort()
    cons_str = ", ".join(cons_literals_str)

    args = [ant_str, cons_str]
    result = "{} -> {}".format(*args)
    return result
