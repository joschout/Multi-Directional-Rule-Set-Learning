import gzip
import json
from dataclasses import dataclass
import jsonpickle


@dataclass
class CARGenTimingInfo:
    total_fim_time_s: float
    total_assoc_time_s: float


@dataclass
class TreeRuleGenTimingInfo:
    total_time_decision_tree_learning_s: float
    total_time_rf_conversion_s: float


def store_tree_rule_gen_timing_info(abs_file_name: str, tree_rule_gen_timing_info: TreeRuleGenTimingInfo) -> None:
    frozen = jsonpickle.encode(tree_rule_gen_timing_info)
    pretty_frozen = json.dumps(json.loads(frozen), indent=2, sort_keys=True)
    with gzip.open(abs_file_name, 'wt') as ofile:
        ofile.write(pretty_frozen)


def load_tree_rule_gen_timing_info(abs_file_name: str) -> TreeRuleGenTimingInfo:
    tree_rule_gen_timing_info: TreeRuleGenTimingInfo
    with gzip.open(abs_file_name, 'rt') as ifile:
        file_contents = ifile.read()
        tree_rule_gen_timing_info = jsonpickle.decode(file_contents)
    return tree_rule_gen_timing_info
