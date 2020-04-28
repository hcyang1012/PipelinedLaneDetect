from Pipeline import Pipeline

class AllNumbers(Pipeline):
    def generator(self):
        value = 0
        while True:
            yield value
            value += 1


class Evens(Pipeline):
    def filter(self, value):
        return value % 2 == 0


class MultipleOf(Pipeline):
    def __init__(self, factor=1):
        self.factor = factor
        super(MultipleOf, self).__init__()
        
    def filter(self, value):
        return value % self.factor == 0


class Printer(Pipeline):
    def map(self, value):
        print (value)
        return value


class First(Pipeline):
    def __init__(self, total=10):
        self.total = total
        self.count = 0
        super(First, self).__init__()
    
    def map(self, value):
        self.count += 1
        if self.count > self.total:
            raise StopIteration
        return value


def main():
    all_numbers = AllNumbers()
    evens = MultipleOf(2)
    multiple_of_3 = MultipleOf(3)
    printer = Printer()
    first_10 = First(10)
    pipeline = all_numbers | evens | multiple_of_3 | first_10 | printer
    
    for i in pipeline:
        pass


if __name__ == '__main__':
    main()