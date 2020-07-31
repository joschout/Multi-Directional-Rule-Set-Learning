import os
import time
from typing import List, Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from experiments.arcbench_data_preparation.reworked_one_hot_encoding import \
    get_one_hot_encoded_data_fold_abs_file_name
from experiments.arcbench_data_preparation.reworked_one_hot_encoding import get_original_data_fold_abs_file_name, \
    TrainTestEnum
from experiments.e2_multi_directional_model_comparison.file_naming.rules.single_target_tree_rule_naming import (
    get_single_target_tree_rules_relative_file_name_without_extension,
    get_single_target_tree_rule_dir,
    get_single_target_tree_rules_abs_file_name,
    get_single_target_tree_rules_gen_timing_info_abs_file_name
)
from experiments.file_naming.column_encodings import get_encodings_book_keeper_abs_file_name_for
from experiments.file_naming.single_target_classifier_indicator import SingleTargetClassifierIndicator

from experiments.decision_tree_rule_learning.data_preparation import PreparedDataForTargetSet
from experiments.decision_tree_rule_learning.timing_utils import store_tree_rule_gen_timing_info, TreeRuleGenTimingInfo
from experiments.utils.experiment_logging import create_logger, close_logger

from experiments.typing_utils import TimeDiffSec

from mdrsl.data_handling.one_hot_encoding.encoding_io import load_encoding_book_keeper
from mdrsl.rule_models.mids.io_mids import store_mcars
from mdrsl.rule_models.mids.cover.cover_checker import CoverChecker
from mdrsl.rule_models.mids.mids_rule import MIDSRule
from mdrsl.data_structures.rules.multi_target_class_association_rule import MCAR
from mdrsl.data_handling.one_hot_encoding.encoding_book_keeping import EncodingBookKeeper
from mdrsl.rule_generation.decision_tree_conversion.support_confidence_adding import \
    add_support_and_confidence_to_MIDSRule
from mdrsl.rule_generation.decision_tree_conversion.tree_to_rule_set_conversion import \
    convert_decision_tree_to_mids_rule_list


Attr = str


def learn_and_convert_single_target_tree_ensemble_to_rules(
        dataset_name: str,
        fold_i: int,
        target_attribute: str,
        nb_of_trees_per_model: int,
        min_support: float,
        max_depth: int
):
    classifier_indicator = SingleTargetClassifierIndicator.random_forest
    train_test = TrainTestEnum.train

    relative_name: str = get_single_target_tree_rules_relative_file_name_without_extension(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
        classifier_indicator=classifier_indicator,
        nb_of_trees_per_model=nb_of_trees_per_model,
        min_support=min_support, max_depth=max_depth,
    )

    logger_dir: str = get_single_target_tree_rule_dir()
    logger = create_logger(
        logger_name=f'mine_single_target_tree_rules_' + relative_name,
        log_file_name=os.path.join(logger_dir,
                                   f"{relative_name}_single_tree_rule_generation.log")
    )

    # --- load train data ---------------------------------------------------------------------------------------------
    df_original = pd.read_csv(get_original_data_fold_abs_file_name(dataset_name, fold_i, train_test),
                              delimiter=',')
    df_one_hot_encoded = pd.read_csv(get_one_hot_encoded_data_fold_abs_file_name(dataset_name, fold_i, train_test),
                                     delimiter=",")

    encoding_book_keeper: EncodingBookKeeper = load_encoding_book_keeper(
        get_encodings_book_keeper_abs_file_name_for(dataset_name, fold_i))

    cover_checker = CoverChecker()

    original_group_to_predict: List[Attr] = [target_attribute]
    original_target_attr_set = set(original_group_to_predict)

    logger.info(f"Fetching the necessary columns for {dataset_name}{fold_i} {original_target_attr_set}")

    prepared_data: PreparedDataForTargetSet = PreparedDataForTargetSet.prepare_data_for_target_set(
        df_original=df_original,
        df_one_hot_encoded=df_one_hot_encoded,
        encoding_book_keeper=encoding_book_keeper,
        original_target_attr_set=original_target_attr_set,
    )

    # --- Fit and save classifier ---------------------------------------------------------------------------------

    start_time_decision_tree_learning_s = time.time()
    classifier: RandomForestClassifier = RandomForestClassifier(n_estimators=nb_of_trees_per_model,
                                                                min_samples_leaf=min_support,
                                                                max_depth=max_depth
                                                                )

    classifier.fit(
        X=prepared_data.df_one_hot_encoded_descriptive_attributes,
        y=prepared_data.df_one_hot_encoded_target_attributes
    )
    end_time_decision_tree_learning_s = time.time()
    total_time_decision_tree_learning_s: float = end_time_decision_tree_learning_s - start_time_decision_tree_learning_s

    logger.info(f"Fitted a {classifier_indicator.value} model predicting {original_target_attr_set}"
                f" for {dataset_name}{fold_i}")

    total_time_rf_conversion_s: TimeDiffSec = 0

    complete_rule_list: List[MCAR] = []
    tree_classifiers = classifier.estimators_
    for tree_clf in tree_classifiers:
        list_of_dt_rules: Optional[List[MIDSRule]] = None
        try:
            start_time_clf_conversion_s = time.time()
            list_of_dt_rules: List[MIDSRule] = convert_decision_tree_to_mids_rule_list(
                tree_classifier=tree_clf,
                one_hot_encoded_feature_names=prepared_data.descriptive_one_hot_encoded_columns,
                target_attribute_names=prepared_data.target_one_hot_encoded_columns,
                encoding_book_keeper=encoding_book_keeper)
        except NotImplementedError as err:
            logger.error(str(err))

        if list_of_dt_rules is not None:

            # --- adding support and confidence to rules
            mids_rule: MIDSRule
            for mids_rule in list_of_dt_rules:
                add_support_and_confidence_to_MIDSRule(
                    prepared_data.df_original_without_nans_for_targets,
                    mids_rule,
                    cover_checker=cover_checker)

            # logger.info(f"found {len(list_of_dt_rules)} rules,"
            #             f" updated total rule set size: {len(complete_rule_list)}")
            mids_rules_as_mcars = [mids_rule.car for mids_rule in list_of_dt_rules]
            complete_rule_list.extend(mids_rules_as_mcars)

            end_time_clf_conversion_s = time.time()
            total_time_clf_conversion_s = end_time_clf_conversion_s - start_time_clf_conversion_s
            total_time_rf_conversion_s += total_time_clf_conversion_s

    logger.info(f"Complete set size: {len(complete_rule_list)}")

    for i in range(0, len(complete_rule_list)):
        logger.info(f"rule {i}: {str(complete_rule_list[i])}")
        if i > 10:
            break
    # --- Save rules to file ---------------------------------------------------------------------------------
    tree_clf_derived_rules_abs_file_name: str = get_single_target_tree_rules_abs_file_name(
        dataset_name=dataset_name, fold_i=fold_i, target_attribute=target_attribute,
        classifier_indicator=classifier_indicator, nb_of_trees_per_model=nb_of_trees_per_model,
        min_support=min_support, max_depth=max_depth
    )
    store_mcars(tree_clf_derived_rules_abs_file_name, complete_rule_list)
    logger.info(f"finished writing single-target tree rules to file: {tree_clf_derived_rules_abs_file_name}")

    tree_rule_gen_timing_info = TreeRuleGenTimingInfo(
        total_time_decision_tree_learning_s=total_time_decision_tree_learning_s,
        total_time_rf_conversion_s=total_time_rf_conversion_s
    )

    tree_rule_gen_timing_info_abs_file_name: str = get_single_target_tree_rules_gen_timing_info_abs_file_name(
        dataset_name=dataset_name, fold_i=fold_i, target_attribute=target_attribute,
        classifier_indicator=classifier_indicator, nb_of_trees_per_model=nb_of_trees_per_model,
        min_support=min_support, max_depth=max_depth
    )
    store_tree_rule_gen_timing_info(tree_rule_gen_timing_info_abs_file_name, tree_rule_gen_timing_info)

    logger.info("==================================================================")
    close_logger(logger)
