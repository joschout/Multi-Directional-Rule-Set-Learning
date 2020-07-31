from mdrsl.data_structures.rules.generalized_rule_part import GeneralizedAntecedent
from mdrsl.data_structures.rules.rule_part import Consequent


class MCAR:
    id = 0

    def __init__(self, antecedent: GeneralizedAntecedent, consequent: Consequent, support: float, confidence: float):
        self.antecedent: GeneralizedAntecedent = antecedent
        self.consequent: Consequent = consequent
        self.support: float = support
        self.confidence: float = confidence
        self.id: int = MCAR.id

        MCAR.id += 1

    def __len__(self):
        """
        returns
        -------

        length of this rule
        """
        return len(self.antecedent) + len(self.consequent)

    def __repr__(self):
        args = [self.antecedent.string(), self.consequent.string(), self.support, self.confidence, len(self), self.id]
        text = "MCAR {} => {} sup: {:.2f} conf: {:.2f} len: {}, id: {}".format(*args)

        return text

    def __gt__(self, other):
        """
        Precedence operator. Determines if this rule has higher precedence.

        Rules are sorted according to
            * their confidence,
            * support,
            * length and id.
        """
        if self.confidence > other.confidence:
            return True
        elif (self.confidence == other.confidence and
              self.support > other.support):
            return True
        elif (self.confidence == other.confidence and
              self.support == other.support and
              len(self) < len(other)):
            return True
        elif (self.confidence == other.confidence and
              self.support == other.support and
              len(self) == len(other) and
              self.id < other.id):
            return True
        else:
            return False

    def __lt__(self, other):
        """
        rule precedence operator
        """
        return not self > other

    def __eq__(self, other):
        if not isinstance(other, MCAR):
            return False
        elif self is other:
            return True
        else:
            return (
                self.antecedent == other.antecedent
                and self.consequent == other.consequent
                and self.support == other.support
                and self.confidence == other.confidence
                and self.id == other.id
            )

    def equal_without_id(self, other):
        if not isinstance(other, MCAR):
            return False
        elif self is other:
            return True
        else:
            return (
                self.antecedent == other.antecedent
                and self.consequent == other.consequent
                and abs(self.support - other.support) < 0.01
                and abs(self.confidence - other.confidence) < 0.01
            )

    def __hash__(self):
        return hash((self.antecedent, self.consequent, self.support, self.confidence, self.id))
