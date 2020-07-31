from typing import List

from mdrsl.data_structures.rules.multi_target_class_association_rule import MCAR
from mdrsl.data_structures.rules.generalized_rule_part import GeneralizedAntecedent
from mdrsl.data_structures.item import Literal
from mdrsl.data_structures.rules.rule_part import Consequent
from mdrsl.rule_generation.decision_tree_conversion.leaf_info import LeafInfo
from mdrsl.rule_generation.decision_tree_conversion.consequent_building import ConsequentBuilder
from mdrsl.rule_generation.decision_tree_conversion.tree_edge import AntecedentBuilder
from mdrsl.rule_generation.decision_tree_conversion.tree_path import TreePath
from mdrsl.rule_models.mids.mids_rule import MIDSRule

NodeId = int


class PathToRuleConverter:
    """
    Converts a Path obtained from a learned Scikit-learn DecisionTreeClassifier into a MIDS rule.

    NOTE: A Path still contains information about the training data as stored in the DecisionTreeClassifier.

    When converting it to a MIDS rule, the feature ids and such will need to be renamed to attribute names.

    Note that there may be a difference between attributes in the original training data,
    and the attributes in the training data as input to the DecisionTreeClassifier, which might be one-hot encoded.

    """

    support_dummy: float = 0.0
    confidence_dummy: float = 0.0

    def __init__(self,
                 antecedent_builder: AntecedentBuilder,
                 consequent_builder: ConsequentBuilder,
                 # total_nb_of_training_examples: Optional[int] = None
                 ):

        self.antedent_builder: AntecedentBuilder = antecedent_builder
        self.consequent_builder: ConsequentBuilder = consequent_builder
        # self.total_nb_of_training_examples_examples: Optional[int] = total_nb_of_training_examples

    def convert_to_mids_rules(self, tree_paths: List[TreePath]) -> List[MIDSRule]:
        rule_list: List[MIDSRule] = []

        for path in tree_paths:
            rule = self.convert_path(path)
            rule_list.append(rule)
        return rule_list

    def convert_path(self, tree_path: TreePath) -> MIDSRule:
        try:
            antecedent: GeneralizedAntecedent = self.antedent_builder.convert_edges(tree_path.edges)
        except AttributeError as err:
            print("AttributeError")
        consequent: Consequent = self._convert_leaf_info(tree_path.leaf_info)

        # if self.total_nb_of_training_examples_examples is not None:
        #     nb_of_samples_in_leaf: int = tree_path.leaf_info.get_nb_of_training_examples_in_leaf()
        #     support = nb_of_samples_in_leaf / self.total_nb_of_training_examples_examples
        # else:
        #     support = PathToRuleConverter.support_dummy

        mcar = MCAR(antecedent=antecedent, consequent=consequent,
                    support=PathToRuleConverter.support_dummy, confidence=PathToRuleConverter.confidence_dummy)
        mids_rule = MIDSRule(mcar)

        return mids_rule

    # def _build_antecedent(self, edges: List[TreeEdge]) -> GeneralizedAntecedent:
    #
    #     antecedent_literals: List[Literal] = []
    #
    #     for tree_edge in edges:
    #         lit = self.antedent_builder.convert(tree_edge)
    #         antecedent_literals.append(lit)
    #     antecedent = GeneralizedAntecedent(antecedent_literals)
    #     return antecedent

    def _convert_leaf_info(self, leaf_info: LeafInfo) -> Consequent:
        consequent_literals: List[Literal] = self.consequent_builder.convert(leaf_info)
        return Consequent(consequent_literals)
