import torch.distributions as D
from torch import nn
import torch

#weight to point can be added

def get_all_vects(a, v, i, u):
    while v[i] != u:
        if i != 0:
            get_all_vects(a, v, i-1, u)
        else:
            a.append(v.copy())
        v[i] += 1
    v[i] = 0

class Gaussian_Grid(nn.Module):
    def __init__(self, dimensions, grid_size, variance, weighting = 'uniform', rand_seed = None):
        super(Gaussian_Grid, self).__init__()
        self.grid_size = grid_size
        self.variance = variance
        self.dimensions = dimensions
        self.grid_distr = list()
        #uniform distribution on knots
        if weighting == 'uniform':
            u = 1/(self.grid_size**self.dimensions)
            self.grid_distr = [u for i in range(self.grid_size**self.dimensions)]
        #random distribution on knots (seed can be specified)
        if weighting == 'random':
            if rand_seed is not None:
                torch.manual_seed(rand_seed)
            cur_sum = 0
            for i in range(self.grid_size**self.dimensions):
                u = D.Uniform(0, 1-cur_sum).sample().item()
                self.grid_distr.append(u)
                cur_sum += u
        m = D.Categorical(torch.tensor(self.grid_distr))

        all_vects = list()
        get_all_vects(all_vects, [0.0 for i in range(self.dimensions)], self.dimensions-1, grid_size)
        comp = D.Independent(D.Normal(
             torch.tensor(all_vects), 
             self.variance*torch.ones(self.grid_size**self.dimensions,self.dimensions)), 1)
        self.gmm = D.MixtureSameFamily(m, comp)


    def sampler(self, sample_amnt):
        return self.gmm.sample((sample_amnt,))

    def log_pdf(self, x):
        return self.gmm.log_prob(x)

class Simple_Gaussian(nn.Module):
    def __init__(self, dimensions, mean = None, variance = None, dev = None):
        super(Simple_Gaussian, self).__init__()
        self.mean = None
        if mean is None:
            self.mean = torch.zeros(dimensions)
        else:
            self.mean = torch.FloatTensor(mean)
        self.variance = None
        if variance is None:
            self.variance = torch.eye(dimensions)
        else:
            self.variance = torch.FloatTensor(variance)
        if dev is None:
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.norml = D.MultivariateNormal(self.mean.to(dev), self.variance.to(dev)) # * torch.eye(dimensions)

    def sampler(self, sample_amnt):
        return self.norml.sample((sample_amnt,))

    def log_pdf(self, x):
        return self.norml.log_prob(x)

class Banana_Gaussian(nn.Module):
    def __init__(self, p=100, b=0.1, dev = None):
        super(Banana_Gaussian, self).__init__()
        self.mean = torch.FloatTensor([0, 0])
        self.variance = torch.FloatTensor([[p, 0], [0, 1]])
        if dev is None:
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.norml = D.MultivariateNormal(self.mean.to(dev), self.variance.to(dev)) # * torch.eye(dimensions)
        self.p = p
        self.b = b
        self.samps = None

    def sampler(self, sample_amnt):
        samps = self.norml.sample((sample_amnt,))
        samps[:, 1] = samps[:, 1] + self.b*samps[:, 0]**2-self.p*self.b
        return samps

    def log_pdf(self, x):
        y = x.clone().detach()
        y[:, 1] = y[:, 1] - self.b*y[:, 0]**2 +self.p*self.b
        return self.norml.log_prob(y)
