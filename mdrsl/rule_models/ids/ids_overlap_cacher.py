from typing import Optional

from pyids.data_structures.ids_ruleset import IDSRuleSet
from pyids.data_structures.ids_cacher import IDSCacher


def init_overlap_cacher(cacher: Optional[IDSCacher], all_rules: IDSRuleSet, quant_dataframe) -> IDSCacher:
    if cacher is None:
        cacher_to_use = IDSCacher()
        cacher_to_use.calculate_overlap(all_rules, quant_dataframe)
    else:
        cacher_to_use = cacher
    return cacher_to_use
