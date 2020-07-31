try:
    from rule_generation.association_rule_mining.apyori_impl.apyori import RelationRecord
except ModuleNotFoundError:
    from collections import namedtuple

    # Ignore name errors because these names are namedtuples.
    SupportRecord = namedtuple(  # pylint: disable=C0103
        'SupportRecord', ('items', 'support'))
    RelationRecord = namedtuple(  # pylint: disable=C0103
        'RelationRecord', SupportRecord._fields + ('ordered_statistics',))
    OrderedStatistic = namedtuple(  # pylint: disable=C0103
        'OrderedStatistic', ('items_base', 'items_add', 'confidence', 'lift',))


def print_relation_record(relation_record: RelationRecord):
    """
    From http://www.zaxrosenberg.com/unofficial-apyori-documentation/

    Each RelationRecord  reflects all rules associated with a specific itemset (items) that has relevant rules.
    Support (support), given that it’s simply a count of appearances of those items together,
    is the same for any rules involving those items, and so only appears once per RelationRecord.
    The ordered_statistic  reflects a list of all rules that met our min_confidence  and min_lift  requirements
    (parameterized when we called apriori() ).
    Each OrderedStatistic  contains the antecedent (items_base)
    and consequent (items_add) for the rule, as well as the associated confidence  and lift .


    :param relation_record:
    :return:
    """

    # first index of the inner list
    # Contains base item and add item=

    items = relation_record.items
    support = relation_record.support
    print("itemset:", items)
    print("support:", support)
    # ordered_statistics = record.ordered_statistics[0]

    print("Rules generated from itemset")
    for ordered_statistic in relation_record.ordered_statistics:
        antecedent = ordered_statistic.items_base
        consequent = ordered_statistic.items_add
        confidence = ordered_statistic.confidence
        lift = ordered_statistic.lift

        print("Rule: " + ",".join([str(i) for i in antecedent]) + " -> " + ",".join([str(i) for i in consequent]), ', conf:', confidence, ', lift:', lift)

