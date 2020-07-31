from typing import List

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

import graphviz


def render_decision_tree_classifier_to_file(
        clf: DecisionTreeClassifier, decision_tree_descriptive_attribute_names: List[str],
        image_absolute_file_path: str) -> None:
    # %%
    # classes_ : array of shape = [n_classes] or a list of such arrays
    #
    #     The classes labels (single output problem), or a list of arrays of class labels (multi-output problem).
    #
    class_labels = clf.classes_

    class_labels: List[str] = list(map(str, class_labels))

    dot_data = tree.export_graphviz(clf, feature_names=decision_tree_descriptive_attribute_names,
                                    class_names=class_labels)
    graph = graphviz.Source(dot_data)
    # this will create an .pdf file
    graph.render(image_absolute_file_path)
