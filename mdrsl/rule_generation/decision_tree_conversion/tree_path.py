from typing import List

import numpy as np


from mdrsl.rule_generation.decision_tree_conversion.leaf_info import LeafInfo
from mdrsl.rule_generation.decision_tree_conversion.tree_edge import TreeEdge


class TreePath:
    def __init__(self,
                 edges: List[TreeEdge],
                 nb_training_samples_in_leaf: int,
                 leaf_class_label_counts: np.ndarray,
                 class_labels: np.ndarray,
                 nb_of_target_attributes: int
                 ):
        self.edges: List[TreeEdge] = edges
        self.leaf_info = LeafInfo(nb_training_samples_in_leaf=nb_training_samples_in_leaf,
                                  class_label_counts=leaf_class_label_counts,
                                  class_labels=class_labels,
                                  nb_of_target_attributes=nb_of_target_attributes)

    def __str__(self):
        edges_str_list = [str(edge) for edge in self.edges]
        edges_str = ", ".join(edges_str_list)

        output_str = edges_str + " => " + str(self.leaf_info)
        return output_str

    def __repr__(self):
        return self.__str__()

    def predict(self, X) -> np.ndarray:
        return self.leaf_info.predict(X)
