from typing import List

from sklearn.tree import DecisionTreeClassifier

from mdrsl.data_handling.one_hot_encoding.encoding_book_keeping import EncodingBookKeeper
from mdrsl.rule_generation.decision_tree_conversion.consequent_building import ConsequentBuilder
from mdrsl.rule_generation.decision_tree_conversion.path_to_rule_conversion import PathToRuleConverter
from mdrsl.rule_generation.decision_tree_conversion.tree_edge import AntecedentBuilder
from mdrsl.rule_generation.decision_tree_conversion.tree_path import TreePath
from mdrsl.rule_generation.decision_tree_conversion.tree_to_paths_conversion import TreeToPathsConverter
from mdrsl.rule_models.mids.mids_rule import MIDSRule


def convert_decision_tree_to_mids_rule_list(tree_classifier: DecisionTreeClassifier,
                                            one_hot_encoded_feature_names: List[str],
                                            target_attribute_names: List[str],
                                            encoding_book_keeper: EncodingBookKeeper,
                                            # training_dataframe: Optional[pd.DataFrame]
                                            ) -> List[MIDSRule]:
    tree_to_paths_converter = TreeToPathsConverter(tree_classifier)
    list_of_tree_paths: List[TreePath] = tree_to_paths_converter.convert()

    antecedent_builder: AntecedentBuilder = AntecedentBuilder(
        one_hot_encoded_feature_names=one_hot_encoded_feature_names,
        ohe_prefix_separator=encoding_book_keeper.ohe_prefix_separator
    )
    consequent_builder: ConsequentBuilder = ConsequentBuilder(dt_target_attr_names=target_attribute_names,
                                                              encoding_book_keeper=encoding_book_keeper)

    path_to_rule_converter = PathToRuleConverter(
        antecedent_builder=antecedent_builder,
        consequent_builder=consequent_builder,
        # total_nb_of_training_examples=tree_to_paths_converter.total_nb_of_training_samples
    )

    list_of_rules: List[MIDSRule] = path_to_rule_converter.convert_to_mids_rules(list_of_tree_paths)
    return list_of_rules
