
class History(object):

    def __init__(self, intensity=2):
        self.hist = []
        self.read_count = 0
        self.intensity = intensity

    def __len__(self):
        return len(self.hist)

    def add(self, solution):
        self.hist.append(solution)

    def get(self):
        self.read_count += 1
        if self.read_count % self.intensity != 0:
            res = self.hist[-1]
            self.hist = self.hist[:-1]
        else:
            res = self.hist[0]
            self.hist = self.hist[1:]
        return res
