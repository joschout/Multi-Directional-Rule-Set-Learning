from typing import Iterable, Set

from mdrsl.data_structures.item import Literal


class GeneralizedAntecedent:
    """RulePart represents either the left-hand side or right-hand side of the association rule.
    It is a set of __literals (Items).

    Parameters
    ----------
    literals: 1D array of Items


    Attributes
    ----------
    itemset: 1D array of Items
        dictionary of unique attributes, such as: {a: 1, b: 3}

    frozenset: frozenset of Items
        this attribute is vital for determining if antecedent
        is a subset of transaction and, consequently, if transaction
        satisfies antecedent


    """

    class_name = 'GeneralizedAntecedent'

    def __init__(self, literals: Iterable[Literal]):

        # extract unique attributes and convert them to dict
        # such as: {a: 1, b: 3, c: 4}
        self.__attributes: Set[str] = set()
        self.__literals: Set[Literal] = set()

        for lit in literals:
            self.__literals.add(lit)
            self.__attributes.add(lit.get_attribute())

    def __len__(self):
        """
        Returns
        -------
        length of the itemset
        """
        return len(self.__literals)

    def __class_name__(self) -> str:
        return GeneralizedAntecedent.class_name

    def __repr__(self):
        str_array = [str(literal) for literal in self.__literals]

        text = ", ".join(str_array)
        return self.__class_name__() + "({})".format(text)

    def __hash__(self):
        return hash(tuple(self.__literals))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __contains__(self, literal: Literal):
        """Checks whether the attribute occurs in the itemset
        """
        return literal in self.__literals

    def string(self):
        string_items = [str(literal) for literal in self.__literals]

        string_ant = ",".join(string_items)

        return "{" + string_ant + "}"

    def get_literals(self) -> Set[Literal]:
        return self.__literals

    def get_attributes(self) -> Set[str]:
        return self.__attributes
