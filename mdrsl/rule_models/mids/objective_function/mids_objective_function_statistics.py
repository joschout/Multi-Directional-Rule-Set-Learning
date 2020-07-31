from typing import Optional, Dict

from tabulate import tabulate
import pandas as pd

from utils.value_collection import ValueCollector


class MIDSObjectiveFunctionStatistics:
    def __init__(self):

        self.last_f0: Optional[int] = None
        self.last_f1: Optional[int] = None
        self.last_f2: Optional[int] = None
        self.last_f3: Optional[int] = None
        self.last_f4: Optional[int] = None
        self.last_f5: Optional[int] = None
        self.last_f6: Optional[int] = None
        self.last_f7: Optional[int] = None
        self.last_f_total: Optional[int] = None

        self.value_collectors = dict(
            f0=ValueCollector(),
            f1=ValueCollector(),
            f2=ValueCollector(),
            f3=ValueCollector(),
            f4=ValueCollector(),
            f5=ValueCollector(),
            f6=ValueCollector(),
            f_total=ValueCollector()
        )

    def add_values(self, f0, f1, f2, f3, f4, f5, f6, f_total):

        self.last_f0 = f0
        self.last_f1 = f1
        self.last_f2 = f2
        self.last_f3 = f3
        self.last_f4 = f4
        self.last_f5 = f5
        self.last_f6 = f6
        self.last_f_total = f_total

        self.value_collectors['f0'].add_value(f0)
        self.value_collectors['f1'].add_value(f1)
        self.value_collectors['f2'].add_value(f2)
        self.value_collectors['f3'].add_value(f3)
        self.value_collectors['f4'].add_value(f4)
        self.value_collectors['f5'].add_value(f5)
        self.value_collectors['f6'].add_value(f6)
        self.value_collectors['f_total'].add_value(f_total)

    def values_to_pandas_dataframe(self) -> Optional[pd.DataFrame]:
        if ValueCollector.collect_values:
            columns = ['type', 'value']
            data = []
            for function_name, value_collector in self.value_collectors.items():
                for value in value_collector.values:
                    data.append([function_name, value])

            df = pd.DataFrame(data=data, columns=columns)
            return df
        else:
            return None

    def values_to_pandas_dataframe2(self) -> Optional[pd.DataFrame]:
        if ValueCollector.collect_values:
            columns = ['call_index', 'type', 'value']
            data = []

            for function_name, value_collector in self.value_collectors.items():
                for call_index, value in enumerate(value_collector.values):
                    data.append([call_index, function_name, value])

            df = pd.DataFrame(data=data, columns=columns)
            return df
        else:
            return None

    def get_last_f_values(self) -> Dict[str, float]:
        return dict(
            f0=self.last_f0,
            f1=self.last_f1,
            f2=self.last_f2,
            f3=self.last_f3,
            f4=self.last_f4,
            f5=self.last_f5,
            f6=self.last_f6,
            f_total=self.last_f_total)

    def __str__(self):
        table_str = tabulate(
            [
                ['count',
                 self.value_collectors['f0'].count,
                 self.value_collectors['f1'].count,
                 self.value_collectors['f2'].count,
                 self.value_collectors['f3'].count,
                 self.value_collectors['f4'].count,
                 self.value_collectors['f5'].count,
                 self.value_collectors['f6'].count,
                 self.value_collectors['f_total'].count
                 ],
                ['sum',
                 self.value_collectors['f0'].sum,
                 self.value_collectors['f1'].sum,
                 self.value_collectors['f2'].sum,
                 self.value_collectors['f3'].sum,
                 self.value_collectors['f4'].sum,
                 self.value_collectors['f5'].sum,
                 self.value_collectors['f6'].sum,
                 self.value_collectors['f_total'].sum
                 ],
                ['min',
                 self.value_collectors['f0'].min,
                 self.value_collectors['f1'].min,
                 self.value_collectors['f2'].min,
                 self.value_collectors['f3'].min,
                 self.value_collectors['f4'].min,
                 self.value_collectors['f5'].min,
                 self.value_collectors['f6'].min,
                 self.value_collectors['f_total'].min
                 ],
                ['avg',
                 self.value_collectors['f0'].get_avg(),
                 self.value_collectors['f1'].get_avg(),
                 self.value_collectors['f2'].get_avg(),
                 self.value_collectors['f3'].get_avg(),
                 self.value_collectors['f4'].get_avg(),
                 self.value_collectors['f5'].get_avg(),
                 self.value_collectors['f6'].get_avg(),
                 self.value_collectors['f_total'].get_avg()
                 ],
                ['max',
                 self.value_collectors['f0'].max,
                 self.value_collectors['f1'].max,
                 self.value_collectors['f2'].max,
                 self.value_collectors['f3'].max,
                 self.value_collectors['f4'].max,
                 self.value_collectors['f5'].max,
                 self.value_collectors['f6'].max,
                 self.value_collectors['f_total'].max
                 ],
                ['last_val',
                 self.last_f0,
                 self.last_f1,
                 self.last_f2,
                 self.last_f3,
                 self.last_f4,
                 self.last_f5,
                 self.last_f6,
                 self.last_f_total
                 ]
            ],
            headers=['type', 'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f_total']
        )
        return table_str


if __name__ == '__main__':
    vc = ValueCollector()
    vc.add_value(1)
    vc.add_value(2)
    vc.add_value(3)
    print(vc)
