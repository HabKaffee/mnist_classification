import numpy as np

class CyclicLR:
    def __init__(self, min_lr, max_lr, step_size):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.iteration = 0

    def update(self):
        self.iteration += 1
        cycle = np.floor(1 + self.iteration / (2 * self.step_size))
        x = np.abs(self.iteration / self.step_size - 2 * cycle + 1)
        return self.min_lr + (self.max_lr - self.min_lr) * np.maximum(0, (1 - x))
