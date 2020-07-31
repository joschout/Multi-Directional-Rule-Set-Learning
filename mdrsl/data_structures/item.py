import numpy as np
import pandas as pd


class Literal:
    def __init__(self, attribute, value):
        # convert attribute and value so that
        # Item("a", 1) == Item("a", "1")
        self.__attribute = repr(attribute) if type(attribute) != str else attribute
        self.__value = repr(value) if type(value) != str else value

    def get_attribute(self) -> str:
        return self.__attribute

    def get_value(self) -> str:
        return self.__value

    def get_operator(self) -> str:
        raise NotImplementedError('abstract method')

    def __hash__(self):
        return hash(tuple([self.get_operator(), self.get_attribute(), self.get_value()]))

    def __str__(self):
        return "{}{}{}".format(self.get_attribute(), self.get_operator(), self.get_value())

    def is_satisfied_for(self, value_array: np.ndarray) -> np.ndarray:
        raise NotImplementedError('abstract method')

    def __repr__(self):
        return self.__str__()

    def does_literal_hold_for_instance(self, instance: pd.Series) -> bool:
        raise NotImplementedError('abstract method')


class EQLiteral(Literal):
    operator = "="

    def is_satisfied_for(self, value_array: np.ndarray) -> np.ndarray:
        value = self.get_value()
        boolean_array: np.ndarray = value_array == value
        return boolean_array

    def get_operator(self) -> str:
        return EQLiteral.operator

    def does_literal_hold_for_instance(self, instance: pd.Series) -> bool:
        value_for_instance = instance[self.get_attribute()]
        return value_for_instance == self.get_value()


class NEQLiteral(Literal):
    operator = "!="

    def is_satisfied_for(self, value_array: np.ndarray) -> np.ndarray:
        value = self.get_value()
        boolean_array: np.ndarray = value_array != value
        return boolean_array

    def get_operator(self) -> str:
        return NEQLiteral.operator

    def does_literal_hold_for_instance(self, instance: pd.Series) -> bool:
        value_for_instance = instance[self.get_attribute()]
        return value_for_instance != self.get_value()


class Item(Literal):
    pass
