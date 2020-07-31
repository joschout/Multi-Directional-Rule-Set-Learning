from mdrsl.data_structures.transactions.transaction import UniqueTransaction, Transaction


class TransactionDB:

    def __init__(self, dataset, header, unique_transactions=True):
        """TransactionDB represents a list of Transactions that can be
        passed to CBA algorithm as a training or a test set. 

        Parameters
        ----------
        
        dataset: two dimensional array of strings or ints
    
        header: array of strings
            Represents column labels.
        
        unique_transactions: bool
            Determines if UniqueTransaction or Transaction class
            should be used for individual instances.


        Attributes
        ----------
        header: array of strings
            Column labels.

        data: array of Transactions
            Individual instances.

        string_representation: two dimensional array of strings
            e.g. [["food:=:schitzel", "mood:=:happy"], ["food:=:not_schitzel], ["mood:=:unhappy"]]


        """

        TransactionClass = UniqueTransaction if unique_transactions else Transaction

        self._dataset_param = dataset
        self.header = header

        new_dataset = []

        for row in dataset:
            new_row = TransactionClass(row[:-1], header[:-1])

            new_dataset.append(new_row)

        self.data = new_dataset

        get_string_items = lambda transaction: transaction.string_items

        mapped = map(get_string_items, self)

        self.string_representation = list(mapped)

    def __getitem__(self, idx):
        return self.data[idx]

    @classmethod
    def from_DataFrame(clazz, df, unique_transactions=False):
        """
        Allows the conversion of pandas DataFrame class to 
        TransactionDB class.
        """

        rows = df.values
        header = list(df.columns.values)

        return clazz(rows, header, unique_transactions=unique_transactions)

    def __repr__(self):
        return repr(self.string_representation)

    def __len__(self):
        return len(self.data)
