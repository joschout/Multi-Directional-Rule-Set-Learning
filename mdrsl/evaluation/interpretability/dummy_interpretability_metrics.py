from mdrsl.evaluation.interpretability.basic_rule_set_stats import AbstractModelStatistics


class ConstantPredictionInterpretabilityStatistics(AbstractModelStatistics):
    def __init__(self, constant):
        super().__init__(model_abbreviation='constant')
        self.constant = constant

    def get_model_size(self) -> int:
        return 1
