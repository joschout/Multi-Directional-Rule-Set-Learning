import os
from typing import Dict, List, Tuple

import distributed
import pandas as pd
from dask import delayed
from dask.delayed import Delayed

from experiments.dask_utils.computations import compute_delayed_functions


from experiments.dask_utils.dask_initialization import reconnect_client_to_ssh_cluster
from experiments.decision_tree_rule_learning.relative_file_naming import \
    get_tree_derived_rules_rel_file_name_without_extension
from experiments.file_naming.classifier_naming import SingleTargetClassifierIndicator

from experiments.arcbench_data_preparation.reworked_one_hot_encoding import get_original_data_fold_abs_file_name, \
    TrainTestEnum

from experiments.utils.experiment_logging import create_logger, close_logger
from experiments.file_naming.classifier_naming import get_tree_based_mids_dir, \
    get_tree_based_mids_clf_abs_file_name
from experiments.e2_multi_directional_model_comparison.file_naming.evaluation_naming import (
    get_tree_based_mids_target_attr_to_score_info_abs_file_name,
    get_tree_based_mids_interpret_stats_abs_file_name)

from mdrsl.rule_models.multi_target_rule_set_clf_utils.rule_combining_strategy import (
    WeightedVotingRuleCombinator, RuleCombiningStrategy)
from mdrsl.rule_models.mids.io_mids import (
    load_mids_classifier, store_mids_target_attr_to_score_info, store_mids_interpret_stats)
from mdrsl.evaluation.predictive_performance_metrics import ScoreInfo
from mdrsl.rule_models.mids.model_evaluation.mids_interpretability_metrics import MIDSInterpretabilityStatistics, \
    MIDSInterpretabilityStatisticsCalculator
from mdrsl.rule_models.mids.model_evaluation.scoring_mids import score_MIDS_on_its_targets_without_nans
from mdrsl.rule_models.mids.mids_classifier import MIDSClassifier
from mdrsl.rule_models.mids.mids_ruleset import MIDSRuleSet
from mdrsl.rule_models.mids.model_fitting.mids_with_value_reuse import MIDSValueReuse

TargetAttr = str


def evaluate_mids_model_for_dataset_fold_target_attribute(
        dataset_name: str,
        fold_i: int,
        classifier_indicator: SingleTargetClassifierIndicator,
        nb_of_trees_per_model: int,
        nb_of_original_targets_to_predict: int,
        min_support: float,
        max_depth: int
):
    logger = create_logger(
        logger_name=f'evaluate_mids_model_tree_derived_' + get_tree_derived_rules_rel_file_name_without_extension(
            dataset_name=dataset_name, fold_i=fold_i, classifier_indicator=classifier_indicator,
            nb_of_trees_per_model=nb_of_trees_per_model,
            nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
            min_support=min_support, max_depth=max_depth),
        log_file_name=os.path.join(get_tree_based_mids_dir(),
                                   get_tree_derived_rules_rel_file_name_without_extension(
                                       dataset_name=dataset_name, fold_i=fold_i,
                                       classifier_indicator=classifier_indicator,
                                       nb_of_trees_per_model=nb_of_trees_per_model,
                                       nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
                                       min_support=min_support, max_depth=max_depth)
                                   + '_model_evaluation_tree_derived_rules.log')
    )

    # --- load test data ----------------------------------------------------------------------------------------------
    # read in original (discretized) training data
    original_test_data_fold_abs_file_name = get_original_data_fold_abs_file_name(dataset_name, fold_i,
                                                                                 TrainTestEnum.test)
    df_test_original_column_order = pd.read_csv(original_test_data_fold_abs_file_name,
                                                delimiter=',')

    # --- load classifier ---------------------------------------------------------------------------------------------
    tree_based_mids_classifier_abs_file_name = get_tree_based_mids_clf_abs_file_name(
        dataset_name=dataset_name, fold_i=fold_i,
        classifier_indicator=classifier_indicator, nb_of_trees_per_model=nb_of_trees_per_model,
        nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
        min_support=min_support, max_depth=max_depth
    )

    # mids_classifier_abs_file_name = get_mids_clf_abs_file_name(dataset_name, fold_i)
    logger.info(f"start loading MIDS model from {tree_based_mids_classifier_abs_file_name}")
    mids_classifier: MIDSClassifier = load_mids_classifier(tree_based_mids_classifier_abs_file_name)
    logger.info("finished loading MIDS model")
    logger.info(mids_classifier)
    reconstructed_mids = MIDSValueReuse()
    reconstructed_mids.classifier = mids_classifier

    mids_classifier.rule_combination_strategy = RuleCombiningStrategy.WEIGHTED_VOTE
    mids_classifier.rule_combinator = WeightedVotingRuleCombinator()

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

    # mids_target_attr_to_score_info_abs_file_name: str = get_mids_target_attr_to_score_info_abs_file_name(
    #     dataset_name, fold_i)

    tree_based_mids_target_attr_to_score_info_abs_file_name: str = \
        get_tree_based_mids_target_attr_to_score_info_abs_file_name(
            dataset_name=dataset_name, fold_i=fold_i,
            classifier_indicator=classifier_indicator, nb_of_trees_per_model=nb_of_trees_per_model,
            nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
            min_support=min_support, max_depth=max_depth
        )

    store_mids_target_attr_to_score_info(tree_based_mids_target_attr_to_score_info_abs_file_name,
                                         target_attr_to_score_info_map)
    logger.info(f"Wrote MIDS Dict[TargetAttr, ScoreInfo] to {tree_based_mids_target_attr_to_score_info_abs_file_name}")

    # --- Evaluate and store interpretability statistics --------------------------------------------------------------
    interpret_stats: MIDSInterpretabilityStatistics \
        = MIDSInterpretabilityStatisticsCalculator.calculate_ruleset_statistics(
            MIDSRuleSet(mids_classifier.rules), df_test_original_column_order, target_attributes=target_attrs)
    logger.info("Evaluated MIDS classifier on interpretability")
    logger.info(interpret_stats.to_str("\n"))

    # mids_interpret_stats_abs_file_name: str = get_mids_interpret_stats_abs_file_name(
    #     dataset_name, fold_i)
    tree_based_mids_interpret_stats_abs_file_name: str = get_tree_based_mids_interpret_stats_abs_file_name(
        dataset_name=dataset_name, fold_i=fold_i,
        classifier_indicator=classifier_indicator, nb_of_trees_per_model=nb_of_trees_per_model,
        nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
        min_support=min_support, max_depth=max_depth
    )
    store_mids_interpret_stats(tree_based_mids_interpret_stats_abs_file_name, interpret_stats)
    logger.info(f"Wrote MIDSInterpretabilityStatistics to {tree_based_mids_interpret_stats_abs_file_name}")
    logger.info("---")

    close_logger(logger)


def main():
    from experiments.arcbench_data_preparation.dataset_info import datasets
    datasets = [dict(filename="iris", targetvariablename="class", numerical=True)]
    from experiments.dask_utils.dask_initialization import scheduler_host_name
    scheduler_host: str = scheduler_host_name
    list_of_computations: List[Tuple[Delayed, Dict]] = []

    nb_of_folds: int = 10
    classifier_indicator = SingleTargetClassifierIndicator.random_forest
    nb_of_original_targets_to_predict: int = 2

    nb_of_trees_per_model_list: List[int] = [5, 10]
    min_support: float = 0.1  # min_samples_leaf must be at least 1 or in (0, 0.5], got 0

    max_depth: int = 7 - nb_of_original_targets_to_predict

    use_dask = False
    if use_dask:
        client = reconnect_client_to_ssh_cluster(scheduler_host)

    for dataset_info in datasets:
        dataset_name = dataset_info['filename']

        for fold_i in range(nb_of_folds):

            for nb_of_trees_per_model in nb_of_trees_per_model_list:

                if use_dask:
                    func_args = dict(
                        dataset_name=dataset_name,
                        fold_i=fold_i,
                        classifier_indicator=classifier_indicator,
                        nb_of_trees_per_model=nb_of_trees_per_model,
                        nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
                        min_support=min_support,
                        max_depth=max_depth
                    )

                    delayed_func = \
                        delayed(evaluate_mids_model_for_dataset_fold_target_attribute)(
                            **func_args
                        )
                    list_of_computations.append((delayed_func, func_args))
                else:
                    evaluate_mids_model_for_dataset_fold_target_attribute(
                        dataset_name=dataset_name,
                        fold_i=fold_i,
                        classifier_indicator=classifier_indicator,
                        nb_of_trees_per_model=nb_of_trees_per_model,
                        nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
                        min_support=min_support,
                        max_depth=max_depth
                    )

    if use_dask:
        log_file_dir: str = get_tree_based_mids_dir()

        logger_name: str = 'model_evaluation_tree_derived_rules_ERROR_LOGGER'
        logger_file_name: str = os.path.join(
            log_file_dir,
            f'ERROR_LOG_model_evaluation_tree_derived_rules.log'
        )

        compute_delayed_functions(
            list_of_computations=list_of_computations,
            client=client,
            nb_of_retries_if_erred=5,
            error_logger_name=logger_name,
            error_logger_file_name=logger_file_name
        )
    if use_dask:
        nb_of_retries_if_erred = 2
        print("start compute")
        print(list_of_computations)
        distributed.wait(client.compute(list_of_computations, retries=nb_of_retries_if_erred))
        print("end compute")
        # distributed.wait(list_of_computations)


if __name__ == '__main__':
    main()
