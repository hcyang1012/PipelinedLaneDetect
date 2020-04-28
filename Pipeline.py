
class Pipeline(object):
    def __init__(self):
        self.source = None
    
    def __iter__(self):
        return self.generator()
    
    def generator(self):
        while True:
            value = self.source.__next__()
            if self.filter(value):
                yield self.map(value)
    
    def __or__(self, other):
        other.source = self.generator()
        return other
    
    def filter(self, value):
        return True
    
    def map(self, value):
        return value

