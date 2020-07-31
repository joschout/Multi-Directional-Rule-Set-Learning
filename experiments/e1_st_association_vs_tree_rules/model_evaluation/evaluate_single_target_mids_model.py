from typing import List, Dict

import pandas as pd

from experiments.utils.experiment_logging import create_logger, close_logger

from experiments.arcbench_data_preparation.reworked_one_hot_encoding import get_original_data_fold_abs_file_name, \
    TrainTestEnum

from mdrsl.rule_models.mids.io_mids import (
    store_mids_interpret_stats, store_mids_target_attr_to_score_info, load_mids_classifier)
from mdrsl.evaluation.predictive_performance_metrics import ScoreInfo
from mdrsl.rule_models.mids.model_evaluation.mids_interpretability_metrics import MIDSInterpretabilityStatistics, \
    MIDSInterpretabilityStatisticsCalculator
from mdrsl.rule_models.mids.mids_classifier import MIDSClassifier
from mdrsl.rule_models.mids.mids_ruleset import MIDSRuleSet
from mdrsl.rule_models.mids.model_evaluation.scoring_mids import score_MIDS_on_its_targets_without_nans
from mdrsl.rule_models.mids.model_fitting.mids_with_value_reuse import MIDSValueReuse

TargetAttr = str


def evaluate_single_target_mids_model_for_dataset_fold(
        dataset_name: str,
        fold_i: int,
        logger_name: str,
        logger_file_name: str,
        mids_classifier_abs_file_name: str,
        mids_target_attr_to_score_info_abs_file_name: str,
        mids_interpret_stats_abs_file_name: str
):
    logger = create_logger(
        logger_name=logger_name,
        log_file_name=logger_file_name
    )

    # --- load test data ----------------------------------------------------------------------------------------------
    # read in original (discretized) training data
    original_test_data_fold_abs_file_name = get_original_data_fold_abs_file_name(dataset_name, fold_i,
                                                                                 TrainTestEnum.test)
    df_test_original_column_order = pd.read_csv(original_test_data_fold_abs_file_name,
                                                delimiter=',')

    # --- load classifier ---------------------------------------------------------------------------------------------
    # mids_classifier_abs_file_name = get_mids_clf_abs_file_name(dataset_name, fold_i)
    logger.info(f"start loading MIDS model from {mids_classifier_abs_file_name}")
    mids_classifier: MIDSClassifier = load_mids_classifier(mids_classifier_abs_file_name)
    logger.info("finished loading MIDS model")
    logger.info(mids_classifier)
    reconstructed_mids = MIDSValueReuse()
    reconstructed_mids.classifier = mids_classifier

    # --- Evaluate and store interpretability statistics --------------------------------------------------------------
    filter_nans: bool = True
    target_attr_to_score_info_map: Dict[str, ScoreInfo] = score_MIDS_on_its_targets_without_nans(
        reconstructed_mids, df_test_original_column_order, filter_nans=filter_nans)
    logger.info("Evaluated MIDS classifier on predictive performance")

    target_attrs: List[TargetAttr] = mids_classifier.target_attrs
    for target_attr in target_attrs:
        target_attr_score_info: ScoreInfo = target_attr_to_score_info_map[target_attr]
        logger.info(f"\t{target_attr}:\n {target_attr_score_info.to_str('    ')}")
        logger.info("\t---")

    store_mids_target_attr_to_score_info(mids_target_attr_to_score_info_abs_file_name, target_attr_to_score_info_map)
    logger.info(f"Wrote MIDS Dict[TargetAttr, ScoreInfo] to {mids_target_attr_to_score_info_abs_file_name}")

    # --- Evaluate and store interpretability statistics --------------------------------------------------------------
    interpret_stats: MIDSInterpretabilityStatistics \
        = MIDSInterpretabilityStatisticsCalculator.calculate_ruleset_statistics(
            MIDSRuleSet(mids_classifier.rules), df_test_original_column_order, target_attributes=target_attrs)
    logger.info("Evaluated MIDS classifier on interpretability")
    logger.info(interpret_stats.to_str("\n"))

    store_mids_interpret_stats(mids_interpret_stats_abs_file_name, interpret_stats)
    logger.info(f"Wrote MIDSInterpretabilityStatistics to {mids_interpret_stats_abs_file_name}")
    logger.info("---")

    close_logger(logger)
