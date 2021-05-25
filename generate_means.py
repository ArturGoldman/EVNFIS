import torch
import torch.distributions as D
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import math

def MH(f, log_p, sample_amnt, dimensions, burn_in = 100):
    ans = []

    for resamp in tqdm(range(10)):
        # multistart

        x = torch.zeros([1, dimensions])
        while f(x) == 0:
            x = D.MultivariateNormal(loc = torch.tensor([[0., 0.]]),
                                    covariance_matrix = 2*torch.eye(dimensions)).sample()
        cur = []
        for _ in range(burn_in):
            x_new = D.MultivariateNormal(loc = x,
                                         covariance_matrix = 2*torch.eye(dimensions)).sample()
            u = D.Uniform(0, 1).sample()
            a = min(1, torch.abs(f(x_new)/f(x))*torch.exp(log_p(x_new)-log_p(x) ) )
            if u < a:
                x = x_new

        while len(cur) != math.ceil(sample_amnt/10):
            x_new = D.MultivariateNormal(loc = x,
                                         covariance_matrix = torch.eye(dimensions)).sample()
            u = D.Uniform(0, 1).sample()
            a = min(1, torch.abs(f(x_new)/f(x))*torch.exp(log_p(x_new)-log_p(x) ) )
            if u < a:
                cur.append(x_new[0].tolist())
                x = x_new
        ans += cur

    return torch.tensor(ans)


def regular_expectancy(f, p, test_amnt, sample_amnt):
    results = list()
    for i in tqdm(range(test_amnt)):
        samples = p.sampler(sample_amnt)
        mean = f(samples).mean()
        results.append(mean.item())
    return results

def new_expectancy(f, p, q, test_amnt, sample_amnt, model):
    results = list()
    for i in tqdm(range(test_amnt)):
        mean = 0
        z = q.sampler(sample_amnt)
        x, log_det = model(z, mode='inverse')
        mean = (f(x)*torch.exp(p.log_pdf(x)-q.log_pdf(z) + log_det.reshape(1, -1))).mean()
        results.append(mean.item())
    return results

def box_comp(a, b):
    data = [a, b]
    plt.figure(figsize=(12,8))
    plt.boxplot(data, showfliers = False, labels =
                ["MC Vanila", "norm_flow"])
    plt.grid()
    plt.show()
