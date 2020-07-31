from typing import List

from pyarc.data_structures.car import ClassAssocationRule
from pyarc.data_structures.antecedent import Antecedent as CARAntecedent
from pyarc.data_structures.consequent import Consequent as CARConsequent

from mdrsl.data_structures.rules.generalized_rule_part import GeneralizedAntecedent
from mdrsl.data_structures.rules.rule_part import Consequent as MCARConsequent
from mdrsl.data_structures.rules.multi_target_class_association_rule import MCAR
from mdrsl.data_structures.item import EQLiteral


def convert_single_target_car_to_multi_target_car(single_target_car: ClassAssocationRule) -> MCAR:

    st_antecedent: CARAntecedent = single_target_car.antecedent
    st_consequent: CARConsequent = single_target_car.consequent

    mcar_antecedent_literals: List[EQLiteral] = []

    for literal in st_antecedent:
        attribute, value = literal
        mcar_antecedent_literals.append(EQLiteral(attribute=attribute, value=value))

    mcar_antecedent = GeneralizedAntecedent(mcar_antecedent_literals)
    mcar_consequent: MCARConsequent = MCARConsequent(
        [EQLiteral(attribute=st_consequent.attribute, value=st_consequent.value)])
    return MCAR(antecedent=mcar_antecedent, consequent=mcar_consequent,
                support=single_target_car.support,
                confidence=single_target_car.confidence)
