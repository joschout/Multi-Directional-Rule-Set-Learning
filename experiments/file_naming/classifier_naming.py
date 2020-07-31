import os

from experiments.decision_tree_rule_learning.relative_file_naming import \
    get_tree_derived_rules_rel_file_name_without_extension
from experiments.file_naming.single_target_classifier_indicator import SingleTargetClassifierIndicator

from project_info import project_dir


def get_models_dir() -> str:
    models_dir: str = os.path.join(project_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    return models_dir


def get_single_target_classifier_dir_for_dataset_fold(classifier_indicator: SingleTargetClassifierIndicator,
                                                      dataset_name: str, fold_i: int) -> str:
    classifier_indicator_value: str = classifier_indicator.value
    classifier_dir: str = os.path.join(get_models_dir(),
                                       f'{classifier_indicator_value}_classifier/{dataset_name}{fold_i}')
    if not os.path.exists(classifier_dir):
        os.makedirs(classifier_dir)
    return classifier_dir


def get_single_target_classifier_abs_file_name(classifier_indicator: SingleTargetClassifierIndicator,
                                               dataset_name: str, fold_i: int, target_attribute) -> str:
    classifier_dir: str = get_single_target_classifier_dir_for_dataset_fold(classifier_indicator, dataset_name, fold_i)

    if (classifier_indicator == SingleTargetClassifierIndicator.ids):
        classifier_abs_file_name: str = os.path.join(classifier_dir, f'{target_attribute}_clf.json.gz')

    elif (classifier_indicator == SingleTargetClassifierIndicator.logistic_regression or
          classifier_indicator == SingleTargetClassifierIndicator.decision_tree or
          classifier_indicator == SingleTargetClassifierIndicator.random_forest):
        classifier_abs_file_name = os.path.join(classifier_dir, f'{target_attribute}_clf.pickle')
    else:
        raise Exception(f"SingleTargetClassifierIndicator {classifier_indicator} is unaccounted for")
    return classifier_abs_file_name


# --- MIDS ------------------------------------------------------------------------------------------------------------
def get_mids_dir() -> str:
    mids_classifier_dir: str = os.path.join(get_models_dir(), 'mids_classifier')
    if not os.path.exists(mids_classifier_dir):
        os.makedirs(mids_classifier_dir)
    return mids_classifier_dir


def get_mids_clf_abs_file_name(dataset_name: str, fold_i: int) -> str:
    mids_classifier_dir: str = get_mids_dir()
    mids_classifier_abs_file_name = os.path.join(mids_classifier_dir, f'{dataset_name}{fold_i}_clf.json.gz')
    return mids_classifier_abs_file_name


# --- tree-based MIDS -------------------------------------------------------------------------------------------------
def get_tree_based_mids_dir() -> str:
    tree_based_mids_classifier_dir: str = os.path.join(get_models_dir(), 'mids_classifier_tree_based')
    if not os.path.exists(tree_based_mids_classifier_dir):
        os.makedirs(tree_based_mids_classifier_dir)
    return tree_based_mids_classifier_dir


def get_tree_based_mids_clf_abs_file_name(
        dataset_name: str, fold_i: int,
        classifier_indicator: SingleTargetClassifierIndicator,
        nb_of_trees_per_model: int,
        nb_of_original_targets_to_predict: int,
        min_support: float,
        max_depth: int
) -> str:
    tree_based_mids_classifier_dir: str = get_tree_based_mids_dir()

    relative_file_name = get_tree_derived_rules_rel_file_name_without_extension(
            dataset_name=dataset_name, fold_i=fold_i,
            classifier_indicator=classifier_indicator, nb_of_trees_per_model=nb_of_trees_per_model,
            nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
            min_support=min_support, max_depth=max_depth
            )
    mids_classifier_abs_file_name = os.path.join(tree_based_mids_classifier_dir, f'{relative_file_name}_clf.json.gz')
    return mids_classifier_abs_file_name


if __name__ == '__main__':
    dir_to_print = get_single_target_classifier_dir_for_dataset_fold(SingleTargetClassifierIndicator.ids,
                                                                     dataset_name='iris', fold_i=1)
    print(dir_to_print)
