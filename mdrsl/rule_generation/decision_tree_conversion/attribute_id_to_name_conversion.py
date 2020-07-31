from typing import List


class DecisionTreeFeatureIDConverter:
    """
    Converts an attribute id (as found in a scikit-learn decision tree) into the corresponding attribute name,
     as found in the training data fed into the decision tree.
    """
    def __init__(self, dt_descriptive_atts: List[str]):
        self.dt_descriptive_atts = dt_descriptive_atts

    def convert(self, feature_id: int):
        # find the descriptive attr as used for input for the decision tree
        return self.dt_descriptive_atts[feature_id]
