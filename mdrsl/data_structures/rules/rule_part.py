from typing import Iterable, Dict, Optional, List, Tuple, ValuesView, KeysView

from mdrsl.data_structures.comparable_itemset import ComparableItemSet
from mdrsl.data_structures.item import Literal


class RulePart(ComparableItemSet):
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

    class_name = 'RulePart'

    def __init__(self, literals: Iterable[Literal]):

        # extract unique attributes and convert them to dict
        # such as: {a: 1, b: 3, c: 4}

        self.__literals: Dict[str, Literal] = {}

        for lit in literals:
            attribute = lit.get_attribute()
            if self.__literals.get(attribute, None) is not None:
                raise Exception("Multiple conditions for the same attribute")

            self.__literals[attribute] = lit

        # self.itemset = dict(list(set(items)))

    def __getattr__(self, attr_name: str) -> Literal:
        """
        Parameters
        ----------
        attr_name: str
            name of desired attribute

        Returns
        -------
        Attribute of given name, otherwise an AttributeError
        """
        lit: Optional[Literal] = self.__literals.get(attr_name, None)

        if lit is not None:
            return lit
        else:
            raise AttributeError("No attribute of that name")

    def __getitem__(self, idx: int) -> Tuple[str, Literal]:
        """Method which allows indexing on antecedent's itemset
        """
        literals: List[Tuple[str, Literal]] = list(self.__literals.items())

        if idx <= len(literals):
            return literals[idx]
        else:
            raise IndexError("No value at the specified index")

    def __len__(self):
        """
        Returns
        -------
        length of the itemset
        """
        return len(self.__literals)

    def __class_name__(self) -> str:
        return RulePart.class_name

    def __repr__(self):
        str_array = [str(literal) for literal in self.__literals.values()]

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
        string_items = [str(literal) for literal in self.__literals.values()]

        string_ant = ",".join(string_items)

        return "{" + string_ant + "}"

    def get_literals(self) -> ValuesView[Literal]:
        return self.__literals.values()

    def get_attributes(self) -> KeysView[str]:
        return self.__literals.keys()

    def get_literal(self, attr_name: str) -> Literal:
        """
        Parameters
        ----------
        attr_name: str
            name of desired attribute

        Returns
        -------
        Attribute of given name, otherwise an AttributeError
        """
        lit: Optional[Literal] = self.__literals.get(attr_name, None)

        if lit is not None:
            return lit
        else:
            raise AttributeError("No attribute of that name")


class Antecedent(RulePart):

    class_name = 'Antecedent'

    def __init__(self, literals: Iterable[Literal]):
        super().__init__(literals)

        # this part is important for better performance
        # of M1 and M2 algorithms
        self.frozenset = frozenset(self)

    def __class_name__(self) -> str:
        return Antecedent.class_name


class Consequent(RulePart):
    class_name = 'Consequent'

    def __init__(self, literals:  Iterable[Literal]):
        super().__init__(literals)

        # this part is important for better performance
        # of M1 and M2 algorithms
        self.frozenset = frozenset(self)

    def __class_name__(self) -> str:
        return Consequent.class_name

    def get_predicted_value(self, attr_name: str) -> str:
        return self.__getattr__(attr_name).get_value()
