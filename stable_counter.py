class StableCounter():
    def __init__(self):
        self.count = 0
        self.active = True

    def add(self, increment):
        if self.active:
            self.count += increment

    def __iadd__(self, increment):
        if self.active:
            self.count += increment
        return self

    def reset(self):
        self.count = 0
        self.active = True

    def set(self, val):
        self.count = val
        self.active = False

    def stop(self):
        self.active = False

    def start(self):
        self.active = True

    def value(self):
        return self.count

    def __call__(self):
        if self.active:
            return None
        return self.count

    def __str__(self):
        return str(self.count)
