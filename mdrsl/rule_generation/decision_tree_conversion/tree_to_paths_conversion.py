from typing import List, Optional

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from mdrsl.rule_generation.decision_tree_conversion.tree_branch import TreeBranch
from mdrsl.rule_generation.decision_tree_conversion.tree_edge import TreeEdge
from mdrsl.rule_generation.decision_tree_conversion.tree_path import TreePath

NodeId = int


class TreeToPathsConverter:
    """
    Converts a single Scikit-learn DecisionTreeClassifier in a list of paths.

    NOTE: the paths will still use the information as was stored in the Scikit-learn decision tree.
    Any renaming should be done at a later point.


    """
    def __init__(self, classifier: DecisionTreeClassifier):
        self.tree: DecisionTreeClassifier = classifier

        # The number of nodes (internal nodes + leaves) in the tree.
        self.n_nodes: int = classifier.tree_.node_count
        self.children_left: np.ndarray = classifier.tree_.children_left
        self.children_right: np.ndarray = classifier.tree_.children_right

        # feature : array of int, shape [node_count]
        #             feature[i] holds the feature to split on, for the internal node i.
        self.feature: np.ndarray = classifier.tree_.feature
        self.threshold: np.ndarray = classifier.tree_.threshold

        # n_node_samples : array of int, shape [node_count]
        #       n_node_samples[i] holds the number of training samples reaching node i.
        self.nb_training_samples_reaching_node: np.ndarray = classifier.tree_.n_node_samples

        self.total_nb_of_training_samples: int = 0  # counted when walking the tree

        # value : array of double, shape [node_count, n_outputs, max_n_classes]
        #     Contains the constant prediction value of each node.
        self.value: np.array = classifier.tree_.value

        self.class_labels = classifier.classes_
        self.nb_of_target_attributes = classifier.n_outputs_

        self.list_of_paths: List[TreePath] = []

    def convert(self) -> List[TreePath]:
        root_node_id: NodeId = 0
        self.total_nb_of_training_samples: int = 0

        self._recursive_convert(root_node_id, parent_tree_branch=None)
        return self.list_of_paths

    def _recursive_convert(self, node_id: NodeId, parent_tree_branch: Optional[TreeBranch]) -> None:

        # check if the tnode is a test node, i.e. whether it has both children:
        left_child_node_id: NodeId = self.children_left[node_id]
        right_child_node_id: NodeId = self.children_right[node_id]
        # If we have a test node
        if left_child_node_id != right_child_node_id:

            # do something with the test of this node

            node_feature_id = self.feature[node_id]
            node_threshold = self.threshold[node_id]

            left_extended_branch = TreeBranch(parent_tree_branch=parent_tree_branch,
                                              edge=TreeEdge(
                                                  feature_id=node_feature_id, threshold=node_threshold, is_left=True))
            right_extended_branch = TreeBranch(parent_tree_branch=parent_tree_branch,
                                               edge=TreeEdge(
                                                   feature_id=node_feature_id, threshold=node_threshold, is_left=False))
            self._recursive_convert(left_child_node_id, left_extended_branch)
            self._recursive_convert(right_child_node_id, right_extended_branch)
        else:
            class_label_counts: np.ndarray = self.value[node_id]
            nb_of_training_samples_reaching_leaf: int = self.nb_training_samples_reaching_node[node_id]

            self.total_nb_of_training_samples += nb_of_training_samples_reaching_leaf

            all_zeros: bool = not np.any(class_label_counts)
            if all_zeros:
                raise Exception("all labels have count 0")

            tree_path: TreePath = self._convert_to_tree_path(parent_tree_branch,
                                                             nb_of_training_samples_reaching_leaf,
                                                             class_label_counts)
            self.list_of_paths.append(tree_path)

    def _convert_to_tree_path(self, tree_branch: Optional[TreeBranch], leaf_nb_training_samples: int,
                              leaf_class_label_counts: np.ndarray) -> TreePath:
        if tree_branch is None:
            raise NotImplementedError("Trees consisting of only the root as leaf are currently not supported")
        else:
            tree_branch_edges_as_list: List[TreeEdge] = tree_branch.to_list()
            return TreePath(edges=tree_branch_edges_as_list,
                            nb_training_samples_in_leaf=leaf_nb_training_samples,
                            leaf_class_label_counts=leaf_class_label_counts,
                            class_labels=self.class_labels,
                            nb_of_target_attributes=self.nb_of_target_attributes)
