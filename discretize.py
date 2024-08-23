import numpy
from numbers import Number
import multiprocessing
import torch
import os



# Markovian process generator
def generator(random_state, size, T):
    omega = 3.0397692859811784e-05
    alpha_1 = 0.200052672189836
    alpha_2 = 0.699953910465642
    mu = 0.003469667713479039
    phi = -0.1392222068214604
    # (r_Mt, epsilon_Mt, sigma^2_Mt)
    epsilon = random_state.normal.Normal(0,1).sample((T,size))
    process = torch.zeros((size,T,3))
    process[:,0,0] = -0.006
    process[:,0,2] = omega/(1-alpha_1-alpha_2)
    for t in range(1,T):
        process[:,t,2] = omega + alpha_1*process[:,t-1,1]**2 + alpha_2*process[:,t-1,2]
        process[:,t,1] = torch.sqrt(process[:,t,2]) * epsilon[t]
        process[:,t,0] = mu + phi*process[:,t-1,0] + process[:,t,1]
    return process


class MarkovSampler(object):
    def __init__(self, generator, n_Markov_states: list, n_sample_paths: int):
        self.samples = generator(torch.distributions, n_sample_paths, len(n_Markov_states))
        shape = self.samples.shape
        self.T, self.dim_Markov_states = shape[1:]
        self.n_Markov_states = n_Markov_states
        self.generator = generator
        self.n_samples = n_sample_paths
        # initialize the states [1, 100, 100, 100, ...] where init state and n_states at each time step
        self.Markov_states = [None for _ in range(self.T)]
        self.Markov_states[0] = self.samples[0,0,:].reshape(1,-1)

    def _initialize_states(self):
        # initialization of the Markov states
        for t in range(1, self.T):
            self.Markov_states[t] = self.samples[:self.n_Markov_states[t],t,:]

    def _initialize_matrix(self):
        # initialization of the transition matrix
        self.transition_matrix = [torch.tensor([[1]])]
        self.transition_matrix += [torch.zeros(self.n_Markov_states[t-1],self.n_Markov_states[t]) for t in range(1, self.T)]

    def SA(self):
        # Use stochastic approximation to compute the partition
        self._initialize_states()
        for idx, sample in enumerate(self.samples):
            step_size = 1.0/(idx+1)
            for t in range(1,self.T):
                temp = self.Markov_states[t] - sample[t]
                idx = torch.argmin(torch.sum(temp**2, axis=1))
                self.Markov_states[t][idx] += ((sample[t]-self.Markov_states[t][idx]) * step_size)
        self.train_transition_matrix()
        return (self.Markov_states,self.transition_matrix)

    def train_transition_matrix(self):
        # Use the generated sample to train the transition matrix by frequency counts
        labels = torch.zeros(self.n_samples, self.T, dtype=torch.int)
        # stay only unique states
        for t in range(1, self.T):
            self.Markov_states[t] = torch.unique(self.Markov_states[t], dim=0)
            self.n_Markov_states[t] = len(self.Markov_states[t])
        for t in range(1, self.T):
            dist = torch.empty(self.n_samples, self.n_Markov_states[t])
            for idx, markov_state in enumerate(self.Markov_states[t]):
                temp = self.samples[:, t, :] - markov_state
                dist[:, idx] = torch.sum(temp**2, axis=1)
            labels[:, t] = torch.argmin(dist, axis=1)
        self._initialize_matrix()
        for k in range(self.n_samples):
            for t in range(1, self.T):
                self.transition_matrix[t][labels[k, t-1], labels[k, t]] += 1
        for t in range(1, self.T):
            counts = torch.sum(self.transition_matrix[t], axis=1)
            idx = (counts == 0)
            if idx.any():
                self.Markov_states[t-1] = self.Markov_states[t-1][~idx]
                self.n_Markov_states[t-1] -= torch.sum(idx).item()
                self.transition_matrix[t-1] = self.transition_matrix[t-1][:, ~idx]
                self.transition_matrix[t] = self.transition_matrix[t][~idx, :]
                counts = counts[~idx]
            self.transition_matrix[t] /= counts.reshape(-1, 1)

    def write(self, path):
        os.system('mkdir ' + path)
        for t in range(self.T):
            numpy.savetxt(path + "Markov_states_{}.txt".format(t), self.Markov_states[t])
            numpy.savetxt(path + "transition_matrix_{}.txt".format(t), self.transition_matrix[t])

    def simulate(self, n_samples):
        """A utility function. Generate a three dimensional array
        (n_samples * T * n_states) representing n_samples number of sample paths.
        Can be used to generate fan plot to compare with the historical data."""
        sim = numpy.empty([n_samples,self.T,self.dim_Markov_states])
        for i in range(n_samples):
            state = 0
            random_state = numpy.random.RandomState(i)
            for t in range(self.T):
                state = random_state.choice(range(self.n_Markov_states[t]),p=self.transition_matrix[t][state],)
                sim[i][t]=self.Markov_states[t][state]
        return sim
    

if __name__ == "__main__":
    T = 25
    temp = MarkovSampler(generator, [1] + [100] * (T - 1), 1000, 25)
    temp.SA()
    temp.write('/home/adanilishin/sddp/logs/')