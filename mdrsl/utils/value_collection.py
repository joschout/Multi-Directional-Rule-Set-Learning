class ValueCollector:
    collect_values = False

    """
    Based on IntSummaryStatistics from the Java SDK
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.min = float('+inf')
        self.max = float('-inf')

        if ValueCollector.collect_values:
            self.values = []

    def add_value(self, value: float) -> None:
        self.count += 1
        self.sum += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)

        if ValueCollector.collect_values:
            self.values.append(value)

    def get_avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

    def __str__(self):
        return 'count={}, sum={}, min={}, average={}, max={}'.format(self.count, self.sum, self.min, self.get_avg(), self.max)