from mdrsl.data_structures.comparable_itemset import ComparableItemSet
from mdrsl.data_structures.item import Item


class Transaction(ComparableItemSet):
    """Transaction represents one instance in a dataset.
    Transaction is hashed based on its items and class
    value.

    Parameters
    ----------

    row: array of ints or strings

    header: array of strings
        Represents column labels.


    Attributes
    ----------
    items: array of Items

    tid: int
        Transaction ID.

    string_items: two dimensional array of strings
        e.g. [["a:=:1", "b:=:2"]]




    """

    id_ = 0

    def __init__(self, row, header):
        self.items = []
        self.tid = Transaction.id_
        Transaction.id_ += 1

        # eg. [pay=high, eyes=green]
        self.string_items = []

        for idx, val in enumerate(row):
            header_label = header[idx]

            item = Item(header_label, val)

            self.string_items.append("{}:=:{}".format(header_label, val))

            self.items.append(item)

        self.frozenset = frozenset(self)

    def __repr__(self):
        string = ", ".join(self.string_items)
        return "{" + string + "}"

    def __hash__(self):
        return hash(tuple(self.items))
        # return hash((self.tid, tuple(self.items)))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __getitem__(self, idx):
        return self.items[idx]

    def getclass(self):
        return self.class_val


class UniqueTransaction(Transaction):
    """Same as Transaction class except for
    hashing by Transaction id.

    """

    def __hash__(self):
        return hash(self.tid)
