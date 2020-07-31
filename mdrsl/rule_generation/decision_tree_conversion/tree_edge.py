from typing import List
from mdrsl.data_structures.rules.generalized_rule_part import GeneralizedAntecedent
from mdrsl.data_structures.item import Literal, NEQLiteral, EQLiteral
from mdrsl.rule_generation.decision_tree_conversion.attribute_id_to_name_conversion import DecisionTreeFeatureIDConverter


class TreeEdge:
    def __init__(self, feature_id: int, threshold: float, is_left: bool):
        self.feature_id: int = feature_id
        self.threshold: float = threshold
        self.is_left: bool = is_left

    def __str__(self):
        output_str = 'f(' + str(self.feature_id) + ')'
        if self.is_left:
            output_str += '<='
        else:
            output_str += '>'
        output_str += str(self.threshold)

        if self.is_left:
            output_str += ' (L)'
        else:
            output_str += ' (R)'
        return output_str

    def __repr__(self):
        return self.__str__()


class AntecedentBuilder:

    def __init__(self, one_hot_encoded_feature_names: List[str], ohe_prefix_separator: str):

        self.ohe_prefix_separator: str = ohe_prefix_separator
        self.decision_tree_feature_id_converter = DecisionTreeFeatureIDConverter(one_hot_encoded_feature_names)

    def convert_edges(self, edges: List[TreeEdge]):
        antecedent_literals: List[Literal] = []

        for tree_edge in edges:
            lit = self.convert(tree_edge)
            antecedent_literals.append(lit)
        antecedent = GeneralizedAntecedent(antecedent_literals)
        return antecedent

    def convert(self, tree_edge: TreeEdge):
        if tree_edge.threshold != 0.5:
            print("Unexpected tree edge threshold value: " + str(tree_edge.threshold))

        # find the descriptive attr as used for input for the decision tree
        dt_descriptive_attribute = self.decision_tree_feature_id_converter.convert(tree_edge.feature_id)

        splitted_string = dt_descriptive_attribute.split(self.ohe_prefix_separator)
        if len(splitted_string) == 1:
            feature_name = dt_descriptive_attribute
            if tree_edge.is_left:
                feature_value = str(0)
            else:
                feature_value = str(1)

            return EQLiteral(attribute=feature_name, value=feature_value)

        elif len(splitted_string) == 2:
            feature_name = splitted_string[0]
            feature_value = splitted_string[1]

            if tree_edge.is_left:
                return NEQLiteral(attribute=feature_name, value=feature_value)
            else:
                return EQLiteral(attribute=feature_name, value=feature_value)
        else:
            raise Exception("Unexpected feature name:" + dt_descriptive_attribute)


