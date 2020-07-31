# GOAL: given a multi-target rule set, get all single-target rules per target attribute:
from typing import List, Dict, Optional

from mdrsl.data_structures.rules.multi_target_class_association_rule import MCAR


TargetAttr = str


def multi_target_cars_to_st_mcars(multi_target_mcars: List[MCAR],
                                  target_attr_list: Optional[List[TargetAttr]] = None
                                  ) -> Dict[TargetAttr, List[MCAR]]:
    """
    Given a set of multi-target rules, get the single-target class association rules.

    :param target_attr_list:
    :param multi_target_mcars:
    :return:
    """

    target_attr_to_st_rule_list_map: Dict[TargetAttr, List[MCAR]] = dict()
    rule: MCAR
    for rule in multi_target_mcars:
        if len(rule.consequent) == 1:
            target_attr: TargetAttr = list(rule.consequent.get_attributes())[0]

            # check if it already exists
            target_attr_rule_list: Optional[List[MCAR]] = target_attr_to_st_rule_list_map.get(target_attr, None)
            if target_attr_rule_list is None:
                target_attr_rule_list: List[MCAR] = [rule]
                target_attr_to_st_rule_list_map[target_attr] = target_attr_rule_list
            else:
                target_attr_rule_list.append(rule)
        else:
            pass

    if target_attr_list is not None:
        for target_attr in target_attr_list:
            if target_attr not in target_attr_to_st_rule_list_map.keys():
                target_attr_to_st_rule_list_map[target_attr] = []

    return target_attr_to_st_rule_list_map
