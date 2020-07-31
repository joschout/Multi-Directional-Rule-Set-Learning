import gzip
import json
from typing import List, Dict

import jsonpickle

from mdrsl.data_structures.rules.multi_target_class_association_rule import MCAR
from mdrsl.evaluation.predictive_performance_metrics import ScoreInfo
from mdrsl.rule_models.mids.model_evaluation.mids_interpretability_metrics import MIDSInterpretabilityStatistics
from mdrsl.rule_models.mids.mids_classifier import MIDSClassifier


def store_mcars(mcars_abs_file_name: str, mcars: List[MCAR]) -> None:
    frozen = jsonpickle.encode(mcars)
    pretty_frozen = json.dumps(json.loads(frozen), indent=2, sort_keys=True)
    with gzip.open(mcars_abs_file_name, 'wt') as ofile:
        ofile.write(pretty_frozen)


def load_mcars(mcars_abs_file_name: str) -> List[MCAR]:
    mcars: List[MCAR]
    with gzip.open(mcars_abs_file_name, 'rt') as ifile:
        file_contents = ifile.read()
        mcars = jsonpickle.decode(file_contents)
    return mcars


def store_mids_classifier(mids_classifier_abs_file_name: str, mids_classifier: MIDSClassifier) -> None:
    frozen = jsonpickle.encode(mids_classifier)
    with gzip.open(mids_classifier_abs_file_name, 'wt') as ofile:
        ofile.write(frozen)


def load_mids_classifier(mids_classifier_abs_file_name: str) -> MIDSClassifier:
    mids_classifier: MIDSClassifier
    with gzip.open(mids_classifier_abs_file_name, 'rt') as ifile:
        file_contents = ifile.read()
        mids_classifier = jsonpickle.decode(file_contents)
    return mids_classifier


def store_mids_target_attr_to_score_info(
        mids_target_attr_to_score_info_abs_file_name: str,
        target_attr_to_score_info_map: Dict[str, ScoreInfo],
) -> None:
    with gzip.open(mids_target_attr_to_score_info_abs_file_name, 'wt') as ofile:
        ofile.write(jsonpickle.encode(target_attr_to_score_info_map))


def load_mids_target_attr_to_score_info(mids_target_attr_to_score_info_abs_file_name: str) -> Dict[str, ScoreInfo]:
    target_attr_to_score_info_map:  Dict[str, ScoreInfo]
    with gzip.open(mids_target_attr_to_score_info_abs_file_name, 'rt') as ifile:
        file_contents = ifile.read()
        target_attr_to_score_info_map = jsonpickle.decode(file_contents)
    return target_attr_to_score_info_map


def store_mids_interpret_stats(
        mids_interpret_stats_abs_file_name: str, interpret_stats: MIDSInterpretabilityStatistics) -> None:
    with gzip.open(mids_interpret_stats_abs_file_name, 'wt') as ofile:
        ofile.write(jsonpickle.encode(interpret_stats))


def load_mids_interpret_stats(mids_interpret_stats_abs_file_name: str) -> MIDSInterpretabilityStatistics:
    interpret_stats:  MIDSInterpretabilityStatistics
    with gzip.open(mids_interpret_stats_abs_file_name, 'rt') as ifile:
        file_contents = ifile.read()
        interpret_stats = jsonpickle.decode(file_contents)
    return interpret_stats
