from experiments.file_naming.single_target_classifier_indicator import SingleTargetClassifierIndicator


def get_tree_derived_rules_rel_file_name_without_extension(
        dataset_name: str, fold_i: int,
        classifier_indicator: SingleTargetClassifierIndicator,
        nb_of_trees_per_model: int,
        nb_of_original_targets_to_predict: int,
        min_support: float,
        max_depth: int
) -> str:
    return (
        f"{dataset_name}{fold_i}_{str(classifier_indicator.value)}"
        f"_{nb_of_trees_per_model}trees"
        f"_{nb_of_original_targets_to_predict}targets"
        f"_{min_support}supp_{max_depth}depth"
    )
