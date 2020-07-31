from typing import List, Optional

from rule_generation.association_rule_mining.apyori_impl.apyori import apriori

from mdrsl.data_structures.rules.multi_target_class_association_rule import MCAR
from mdrsl.data_structures.rules.generalized_rule_part import GeneralizedAntecedent
from mdrsl.data_structures.item import EQLiteral
from mdrsl.data_structures.rules.rule_part import Consequent
from mdrsl.rule_generation.association_rule_mining.frequent_itemset_mining import attribute_value_separator, ItemEncoder


def mine_MCARs_from_transactions_using_apyori(transactions: List[List[str]],
                                              min_support: float = 0.1,
                                              min_confidence: float = 0.0,
                                              min_lift=0.0,
                                              max_length=None,
                                              item_encoder: Optional[ItemEncoder] = None) -> List[MCAR]:
    relation_record_generator = apriori(transactions,
                                        min_support=min_support,
                                        min_confidence=min_confidence,
                                        min_lift=min_lift,
                                        max_length=max_length)
    return __convert_apyori_result_to_MCARs(relation_record_generator, item_encoder)


def __convert_apyori_result_to_MCARs(relation_record_generator,
                                     item_encoder: Optional[ItemEncoder] = None) -> List[MCAR]:
    """
    Converts the output of apyori.apriori into MCARs

    :return:
    """

    mcars = []

    for relation_record in relation_record_generator:  # type: RelationRecord
        # print("-- relation record ---")
        # print_relation_record(relation_record)
        # print("----------------------")
        # items = relation_record.items

        support = relation_record.support
        for ordered_statistic in relation_record.ordered_statistics:
            antecedent_tmp = ordered_statistic.items_base
            consequent_tmp = ordered_statistic.items_add

            confidence = ordered_statistic.confidence
            lift = ordered_statistic.lift

            if item_encoder is not None:
                antecedent_tmp = [item_encoder.decode_item(encoding) for encoding in antecedent_tmp]
                consequent_tmp = [item_encoder.decode_item(encoding) for encoding in consequent_tmp]

            antecedent_items = [EQLiteral(*item.split(attribute_value_separator)) for item in antecedent_tmp]
            consequent_items = [EQLiteral(*item.split(attribute_value_separator)) for item in consequent_tmp]

            # antecedent_items = [Item(*item.split(attribute_value_separator)) for item in antecedent_tmp]
            # consequent_items = [Item(*item.split(attribute_value_separator)) for item in consequent_tmp]

            antecedent = GeneralizedAntecedent(antecedent_items)
            # antecedent = Antecedent(antecedent_items)
            consequent = Consequent(consequent_items)

            rule = MCAR(antecedent, consequent, support, confidence)
            mcars.append(rule)

    return mcars
