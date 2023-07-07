import numpy as np

class UniformNoise:
    def __init__(self, mu=0, sigma=0.5):
        self.mu = mu
        self.sigma = sigma

    def noise(self):
        # x = self.mu + self.sigma * np.random.normal(size=2)
        x = np.random.uniform(-self.sigma, self.sigma, 2)
        return x
    
    def step(self):
        #decay sigma over time
        self.sigma = self.sigma * 0.999