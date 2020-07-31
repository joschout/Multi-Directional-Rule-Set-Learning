from mdrsl.utils.value_collection import ValueCollector


def is_valid_fraction(val: float, lower: float = 0, higher: float = 1) -> bool:
    return lower <= val <= higher


class AbstractModelStatistics:
    def __init__(self, model_abbreviation: str):
        self.model_abbreviation: str = model_abbreviation

    def get_model_size(self) -> int:
        raise NotImplementedError('Abstract method')


class BasicRuleSetStatistics(AbstractModelStatistics):

    def __init__(self, rule_length_counter: ValueCollector, model_abbreviation: str):
        super().__init__(model_abbreviation)
        self.rule_length_counter: ValueCollector = rule_length_counter

    def ruleset_size(self) -> int:
        """
        Returns the size of the rule set.
        :return:
        """
        return self.rule_length_counter.count

    def total_nb_of_literals(self) -> int:
        """
        Returns the total nb of __literals in the rule set.
        """
        return self.rule_length_counter.sum

    def get_model_size(self) -> int:
        return self.total_nb_of_literals()

    def avg_nb_of_literals_per_rule(self) -> float:
        """
        Returns the avg nb of __literals over the rules in the rule set.
        """
        return self.rule_length_counter.get_avg()

    def min_nb_of_literals(self) -> float:
        """
        Returns the nb of __literals in the shortest rule
        """
        return self.rule_length_counter.min

    def max_nb_of_literals(self) -> float:
        """
        Returns the nb of __literals in the longest rule
        """
        return self.rule_length_counter.max


class SingleTargetRuleSetStatistics(BasicRuleSetStatistics):
    def __init__(self,
                 rule_length_collector: ValueCollector,
                 model_abbreviation: str,
                 fraction_bodily_overlap: float,
                 fraction_uncovered_examples: float,
                 frac_predicted_classes: float
                 ):
        super().__init__(rule_length_collector, model_abbreviation=model_abbreviation)
        self.fraction_bodily_overlap: float = fraction_bodily_overlap
        self.fraction_uncovered_examples: float = fraction_uncovered_examples
        self.frac_predicted_classes: float = frac_predicted_classes

    def to_str(self, indentation: str = "") -> str:
        output_string = (
                indentation + "Rule length stats: " + str(self.rule_length_counter) + "\n"
                + indentation + "Fraction bodily overlap: " + str(self.fraction_bodily_overlap) + "\n"
                + indentation + "Fraction uncovered examples: " + str(self.fraction_uncovered_examples) + "\n"
                + indentation + "Fraction predicted classes: " + str(self.frac_predicted_classes) + "\n"
        )

        return output_string

    def __str__(self):
        return self.to_str()


if __name__ == '__main__':
    value_collector = ValueCollector()
    stats = BasicRuleSetStatistics(value_collector, "test")
    print(stats.rule_length_counter)
