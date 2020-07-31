from typing import List, Dict, Optional

import numpy as np

from mdrsl.data_structures.item import Literal, EQLiteral
from mdrsl.rule_generation.decision_tree_conversion.leaf_info import Attr, LeafInfo, SingleAttrClassLabelsList
from mdrsl.data_handling.one_hot_encoding.encoding_book_keeping import EncodingBookKeeper


class ConsequentBuilder:

    def __init__(self, dt_target_attr_names: List[Attr], encoding_book_keeper: EncodingBookKeeper):
        self.dt_target_attr_names: List[Attr] = dt_target_attr_names
        self.encoding_book_keeper: EncodingBookKeeper = encoding_book_keeper

    def __check_correct_number_of_target_attr(self, nb_of_target_attrs: int) -> None:
        given_nb_of_attr_names: int = len(self.dt_target_attr_names)

        if given_nb_of_attr_names != nb_of_target_attrs:
            raise Exception("Expected " + str(nb_of_target_attrs) + " attribute names, got "
                            + str(given_nb_of_attr_names))

    def convert_old(self, leaf_info: LeafInfo) -> List[Literal]:
        nb_of_target_attrs: int = leaf_info.nb_of_target_attributes
        self.__check_correct_number_of_target_attr(nb_of_target_attrs)

        if nb_of_target_attrs == 1:
            return self.__convert_single_dt_target_attribute(leaf_info)
        else:
            return self.__convert_multiple_dt_target_attributes(leaf_info, nb_of_target_attrs)

    def __convert_multiple_dt_target_attributes_old(self, leaf_info: LeafInfo, nb_of_target_attrs: int) -> List[
        Literal]:

        literals: List[Literal] = []
        possible_class_labels: np.ndarray = leaf_info.possible_class_labels

        for target_attr_index in range(nb_of_target_attrs):
            possible_class_labels_for_target = possible_class_labels[target_attr_index]
            counts_per_label_for_target = leaf_info.class_label_counts[target_attr_index]
            all_zeros: bool = not np.any(counts_per_label_for_target)
            if all_zeros:
                raise Exception("all labels have count 0")

            index_largest_count = np.argmax(counts_per_label_for_target)

            label_largest_count = possible_class_labels_for_target.take(index_largest_count)
            predicted_value = label_largest_count
            predicted_attribute = self.dt_target_attr_names[target_attr_index]
            predictive_literal = EQLiteral(attribute=predicted_attribute, value=predicted_value)

            # TODO: this is still very filthy
            predictive_literal.counts = list(zip(possible_class_labels_for_target, counts_per_label_for_target))

            literals.append(predictive_literal)

        return literals

    def get_original_attr_to_decision_tree_attr_mapping(self) -> Dict[Attr,
                                                                      List[int]]:
        # get the indices for the decision tree target attributes corresponding to the original attributes:
        original_target_attr_to_decision_tree_target_attr_indices: Dict[Attr, List[int]] = {}
        for index, dt_target_attr in enumerate(self.dt_target_attr_names):
            original_target_attr: Attr = self.encoding_book_keeper.get_original(dt_target_attr)
            if original_target_attr_to_decision_tree_target_attr_indices.get(original_target_attr, None) is None:
                original_target_attr_to_decision_tree_target_attr_indices[original_target_attr] = [index]
            else:
                original_target_attr_to_decision_tree_target_attr_indices[original_target_attr].append(index)
        return original_target_attr_to_decision_tree_target_attr_indices

    def convert(self, leaf_info: LeafInfo) -> List[EQLiteral]:
        """
        This method converts a LeafInfo object to a Consequent of a rule.
        Note: a consequent can predict multiple attributes.
        It does this by storing one EQLiteral per predicted original attribute.

        A Scikit-learn DecisionTreeClassifier can predict multiple attributes.

        -- ONE-HOT ENCODED ATTRIBUTES --

        In our framework, some of these target attributes belong to the same original attribute,
        using one-hot encodings.
        e.g. the following one-hot encoding attributes:
            Passenger_Cat=1st_class
            Passenger_Cat=2nd_class
            Passenger_Cat=3rd_class
            Passenger_Cat=crew
        all correspond to the same original target attribute: Passenger_Cat.

        We should group the decision tree target attributes per original target attribute.
        For each original attribute, we have to create an EQLiteral.

        Initial idea:
        Each one-hot encoding attribute has 2 possible class labels: [0, 1].
        Each of these two possible class labels has a count: [c0, c1].
        --> 1. take for each one-hot encoding target attribute the class label with the highest count,
        and create a literal for it, e.g.:
                Passenger_Cat=1st_class=0,  (0 has higher count than 1)
                Passenger_Cat=2nd_class=0,  (0 has higher count than 1)
                Passenger_Cat=3rd_class=0,  (0 has higher count than 1)
                Passenger_Cat=crew=1        (1 has higher count than 0)
            2. Only keep the literal with class label 1:
                Passenger_Cat=crew=1
            3. Convert that literal back to the original attribute:
                Passenger_cat = crew

        PROBLEM: it might occur that for each one-hot encoded attribute, class label 0 has the highest count, e.g.
                Passenger_Cat=1st_class=0,  (0 has higher count than 1)
                Passenger_Cat=2nd_class=0,  (0 has higher count than 1)
                Passenger_Cat=3rd_class=0,  (0 has higher count than 1)
                Passenger_Cat=crew=0        (0 has higher count than 1)
            --> NO literal with class value 1
            --> EMPTY CONSEQUENT.

        SOLUTION:
            1. for each one-hot encoding attribute, look at the count of class label 1
                e.g. the following one-hot encoding attributes:
                    Passenger_Cat=1st_class  --> c11 for class label 1
                    Passenger_Cat=2nd_class  --> c12 for class label 1
                    Passenger_Cat=3rd_class  --> c13 for class label 1
                    Passenger_Cat=crew       --> c14 for class label 1
            2. pick the one-hot encoding attribute with the largest count for class label 1

        --  BINARY ATTRIBUTES --




        Parameters
        ----------
        leaf_info
        nb_of_dt_target_attrs

        Returns
        -------

        """
        nb_of_target_attrs: int = leaf_info.nb_of_target_attributes
        self.__check_correct_number_of_target_attr(nb_of_target_attrs)

        literals: List[EQLiteral] = []

        # get the indices for the decision tree target attributes corresponding to the original attributes:
        original_target_attr_to_decision_tree_target_attr_indices: Dict[Attr, List[int]] \
            = self.get_original_attr_to_decision_tree_attr_mapping()

        original_target_attr: Attr
        dt_target_attr_indices: List[int]
        for original_target_attr, dt_target_attr_indices in \
                original_target_attr_to_decision_tree_target_attr_indices.items():

            if not self.__is_original_attr_binary(dt_target_attr_indices):
                opt_predictive_literal: Optional[EQLiteral] = self.__convert_n_ary_original_attibute(
                    original_target_attr, dt_target_attr_indices, leaf_info)
                if opt_predictive_literal is not None:
                    literals.append(opt_predictive_literal)

            elif self.__is_original_attr_binary(dt_target_attr_indices):
                predictive_literal: EQLiteral = self.__convert_binary_original_attribute(
                    original_target_attr, dt_target_attr_indices, leaf_info)
                literals.append(predictive_literal)

            else:
                raise Exception("should not be in this situation")

        return literals

    def __convert_binary_original_attribute(self,
                                            original_target_attr: Attr,
                                            dt_target_attr_indices: List[int],
                                            leaf_info: LeafInfo
                                            ) -> EQLiteral:
        dt_target_attr_index: int = dt_target_attr_indices[0]
        dt_target_attr: Attr = self.dt_target_attr_names[dt_target_attr_index]

        # the possible class labels for the decision tree attribute, an array [0, 1]
        if len(self.dt_target_attr_names) == 1:
            possible_class_labels_for_dt_target: SingleAttrClassLabelsList = leaf_info.possible_class_labels
            print(possible_class_labels_for_dt_target)
            predicted_value: np.ndarray = possible_class_labels_for_dt_target.take(
                np.argmax(leaf_info.class_label_counts, axis=1),
                axis=0)

            if len(predicted_value) > 1:
                raise Exception("Expected one predicted value, but got " + str(predicted_value))
            else:
                predicted_value = predicted_value[0]
                predictive_literal = EQLiteral(attribute=dt_target_attr, value=predicted_value)
                print(f"predictive literal:{predictive_literal} \t(IF-branch)\n")
                return predictive_literal
        else:
            # THIS IS STILL INCORRECT

            # print("\n---")

            possible_class_labels_for_dt_target: SingleAttrClassLabelsList \
                = leaf_info.get_possible_class_labels_for_decision_tree_attribute(dt_target_attr_index)
            # print(f"possible class labels for target {self.dt_target_attr_names[dt_target_attr_index]}"
            #       f" (original: {original_target_attr}): {possible_class_labels_for_dt_target}")
            # counts per possible target label [0, 1]
            counts_per_label_for_target: np.ndarray = leaf_info.class_label_counts[dt_target_attr_index]
            if counts_per_label_for_target.size != 2:
                raise Exception(
                    f"expected two counts, for class labels 0 and 1, got {counts_per_label_for_target} for labels {possible_class_labels_for_dt_target}")
            all_zeros: bool = not np.any(counts_per_label_for_target)
            if all_zeros:
                raise Exception("all labels have count 0")

            predicted_value: np.ndarray = possible_class_labels_for_dt_target[np.argmax(counts_per_label_for_target)]

            # ----
            possibly_splitted_attr = dt_target_attr.split(
                self.encoding_book_keeper.ohe_prefix_separator)
            if len(possibly_splitted_attr) != 2:
                # do what we previously did
                real_target_attr = dt_target_attr
                real_target_value = predicted_value
            else:
                real_target_attr = possibly_splitted_attr[0]
                real_target_value = possibly_splitted_attr[1]
            # ---

            predictive_literal = EQLiteral(attribute=real_target_attr, value=real_target_value)
            # print(f"predictive literal:{predictive_literal} \t(ELSE-branch)\n")
            return predictive_literal

    def __convert_n_ary_original_attibute(self,
                                          original_target_attr: Attr,
                                          dt_target_attr_indices: List[int],
                                          leaf_info: LeafInfo
                                          ) -> Optional[EQLiteral]:

        dt_target_attr_index_with_current_highest_count: Optional[int] = \
            self.__get_decision_tree_attribute_with_highest_count_for_class_label_one(
                original_target_attr, dt_target_attr_indices, leaf_info)

        if dt_target_attr_index_with_current_highest_count is None:
            print(f"something went wrong with finding the decision tree target attribute"
                  f" with the highest count for class label 1"
                  f" for original target attribute {original_target_attr}.")
            return None
        else:
            dt_target_attr_with_highest_count_for_class_label_one \
                = self.dt_target_attr_names[dt_target_attr_index_with_current_highest_count]

            possibly_splitted_attr = dt_target_attr_with_highest_count_for_class_label_one.split(
                self.encoding_book_keeper.ohe_prefix_separator)
            if len(possibly_splitted_attr) != 2:
                raise Exception("Expected " + str(dt_target_attr_index_with_current_highest_count)
                                + " to be split into 2 using separator "
                                + str(self.encoding_book_keeper.ohe_prefix_separator))
            else:
                # deal with the one-hot encoding
                real_target_attr = possibly_splitted_attr[0]
                real_target_value = possibly_splitted_attr[1]
                if real_target_attr != original_target_attr:
                    raise Exception("something went wrong: got " + real_target_attr + " and " + original_target_attr)
                predictive_literal = EQLiteral(attribute=real_target_attr, value=real_target_value)
                return predictive_literal

    def __get_decision_tree_attribute_with_highest_count_for_class_label_one(self,
                                                                             original_target_attr: Attr,
                                                                             dt_target_attr_indices: List[int],
                                                                             leaf_info: LeafInfo
                                                                             ) -> Optional[int]:
        """
        Find the decision tree attribute with the highest count for class label 1
        """

        current_highest_count_for_class_label_one = float('-inf')
        dt_target_attr_index_with_current_highest_count: Optional[int] = None

        dt_target_attr_index: int
        for dt_target_attr_index in dt_target_attr_indices:

            # the possible class labels for the decision tree attribute, an array [0, 1]
            possible_class_labels_for_dt_target: SingleAttrClassLabelsList = \
                leaf_info.get_possible_class_labels_for_decision_tree_attribute(dt_target_attr_index)

            # NOTE: it could be possible that the one-hot encoded lable does not occur.
            # in that case, the possible class labels for the decision tree attribute are [1]
            if possible_class_labels_for_dt_target.size > 2:
                raise Exception(
                    f"expected two possible class labels (0 and 1) for "
                    f"{self.dt_target_attr_names[dt_target_attr_index]} (original: {original_target_attr}),"
                    f" got {possible_class_labels_for_dt_target}")
            elif possible_class_labels_for_dt_target.size == 2:
                # counts per possible target label [0, 1]
                counts_per_label_for_target: np.ndarray = leaf_info.class_label_counts[dt_target_attr_index]
                if counts_per_label_for_target.size != 2:
                    raise Exception(
                        f"expected two counts for class labels 0 and 1 for "
                        f"{self.dt_target_attr_names[dt_target_attr_index]} (original: {original_target_attr}),"
                        f" got {counts_per_label_for_target}")
                all_zeros: bool = not np.any(counts_per_label_for_target)
                if all_zeros:
                    raise Exception("all labels have count 0")

                indexes_of_class_label_one = np.where(possible_class_labels_for_dt_target == 1)
                index_of_class_label_one: int = indexes_of_class_label_one[0][0]
                if index_of_class_label_one != 0 and index_of_class_label_one != 1:
                    raise Exception(
                        f"expected the index of class label 1 to be either 0 or 1 for "
                        f"{self.dt_target_attr_names[dt_target_attr_index]} (original: {original_target_attr}),"
                        f" got {index_of_class_label_one}")

                count_of_class_label_one: int = counts_per_label_for_target[index_of_class_label_one]
                if count_of_class_label_one > current_highest_count_for_class_label_one:
                    current_highest_count_for_class_label_one = count_of_class_label_one
                    dt_target_attr_index_with_current_highest_count = dt_target_attr_index
                else:
                    pass
            elif possible_class_labels_for_dt_target.size == 1:
                dt_target_attr = self.dt_target_attr_names[dt_target_attr_index]
                print(f"possible class labels for {dt_target_attr}: {possible_class_labels_for_dt_target}")

                counts_per_label_for_target: np.ndarray = leaf_info.class_label_counts[dt_target_attr_index]
                print(f"counts: {counts_per_label_for_target}")
                if counts_per_label_for_target.size != 2:
                    raise Exception(
                        f"expected two count for class labels 0 and 1"
                        f"{self.dt_target_attr_names[dt_target_attr_index]} (original: {original_target_attr}),"
                        f" got {counts_per_label_for_target}")
                all_zeros: bool = not np.any(counts_per_label_for_target)
                if all_zeros:
                    raise Exception("all labels have count 0")

                print("ONEONEONE")
                pass

        if dt_target_attr_index_with_current_highest_count is None:
            print(
                f"WARNING: no highest count for any attribute of {original_target_attr}. It might be constant in the training data.")

        return dt_target_attr_index_with_current_highest_count

    def __is_original_attr_binary(self, dt_target_attr_indices: List[int]) -> bool:
        nb_of_encodings_for_original_attr: int = len(dt_target_attr_indices)
        if nb_of_encodings_for_original_attr <= 0:
            raise Exception("incorrect encoding")
        return nb_of_encodings_for_original_attr == 1

    def __convert_single_dt_target_attribute_old(self, leaf_info: LeafInfo) -> List[Literal]:

        possible_class_labels: np.ndarray = leaf_info.possible_class_labels
        predicted_value: np.ndarray = possible_class_labels.take(
            np.argmax(leaf_info.class_label_counts, axis=1),
            axis=0)

        if len(predicted_value) > 1:
            raise Exception("Expected one predicted value, but got " + str(predicted_value))
        else:
            predicted_attribute = self.dt_target_attr_names[0]
            predicted_value = predicted_value[0]
            predictive_literal = EQLiteral(attribute=predicted_attribute, value=predicted_value)
            return [predictive_literal]
