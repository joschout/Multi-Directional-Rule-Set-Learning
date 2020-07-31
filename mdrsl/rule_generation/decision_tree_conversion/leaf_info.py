from typing import Union, List

import numpy as np
import pandas as pd

Attr = str

SingleAttrClassLabelsList = np.ndarray

ClassLabelsType = Union[
    SingleAttrClassLabelsList,  # array of shape = [n_classes]  -> the class labels (single output problem)
    List[SingleAttrClassLabelsList]  # a list of such arrays  -> a list of arrays of class labels (multi-output problem)
]


class LeafInfo:
    def __init__(self,
                 nb_training_samples_in_leaf: int,
                 class_label_counts: np.ndarray,
                 class_labels: ClassLabelsType,
                 nb_of_target_attributes: int
                 ):
        """
        classes_ == class_labels:
         - array of shape = [n_classes]  -> the class labels (single output problem)
           OR
         - a list of such arrays  -> a list of arrays of class labels (multi-output problem)

             The classes labels (single output problem), or a list of arrays of class labels (multi-output problem).
         I.E. the actual values, not the column names

        :param class_label_counts:
        :param class_labels:
        :param nb_of_target_attributes:
        """

        self.nb_training_samples_in_leaf: int = nb_training_samples_in_leaf

        # shape: n_outputs, max_n_classes
        self.class_label_counts: np.ndarray = class_label_counts
        all_zeros: bool = not np.any(class_label_counts)
        if all_zeros:
            raise Exception("all labels have count 0")

        # shape = [n_classes] OR
        # shape = n_outputs, n_classes
        self.possible_class_labels: ClassLabelsType = class_labels
        self.nb_of_target_attributes: int = nb_of_target_attributes

    def get_nb_of_training_examples_in_leaf(self):
        return self.nb_training_samples_in_leaf

    def get_possible_class_labels_for_decision_tree_attribute(self,
                                                              decision_tree_attribute_index: int
                                                              ) -> SingleAttrClassLabelsList:
        if self.nb_of_target_attributes == 1:
            return self.possible_class_labels
        else:
            return self.possible_class_labels[decision_tree_attribute_index]

    # def __str__(self):
    #     output_str = "#targets=" + str(self.nb_of_target_attributes) +"\n"
    #
    #     for i in range(self.nb_of_target_attributes):
    #         target_str = "t" + str(i) + ":"
    #
    #     return output_str

    def predict(self, X) -> np.ndarray:
        """
        n_outputs_ == nb of target_attributes

        classes_ == possible class labels
          - array of shape = [n_classes]  -> the class labels (single output problem)
           OR
          - a list of such arrays  -> a list of arrays of class labels (multi-output problem)

             The classes labels (single output problem), or a list of arrays of class labels (multi-output problem).

         I.E. the actual values, not the column names

        :param X:
        :return:
        """

        if isinstance(X, list):
            if len(X) == 0:
                raise Exception("empty list")
            else:
                first_instance = X[0]
                if not isinstance(first_instance, list):
                    raise Exception("should have a list of lists")
                else:
                    nb_of_instances = len(X)
        elif isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame):
            nb_of_instances = X.shape[0]
        else:
            raise Exception("unsupported input type: " + str(type(X)))

        nb_of_target_attrs = self.nb_of_target_attributes
        possible_class_labels = self.possible_class_labels

        if nb_of_target_attrs == 1:
            return possible_class_labels.take(
                np.argmax(self.class_label_counts, axis=1),
                axis=0)

        else:
            class_type = possible_class_labels[0].dtype
            predictions = np.zeros(
                (nb_of_instances, nb_of_target_attrs), dtype=class_type)

            for target_attr_index in range(nb_of_target_attrs):

                possible_class_labels_for_target = possible_class_labels[target_attr_index]
                counts_per_label_for_target = self.class_label_counts[target_attr_index]

                index_largest_count = np.argmax(counts_per_label_for_target)

                label_largest_count = possible_class_labels_for_target.take(index_largest_count)

                predictions[:, target_attr_index] = label_largest_count

            return predictions


