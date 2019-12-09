import torch

import numpy as np

import pyro
import pyro.distributions as dist
import torch.distributions.constraints as constraints
from pyro.infer import SVI, Trace_ELBO, MCMC, HMC, EmpiricalMarginal
from pyro.optim import Adam

from random import shuffle

# sample as many thetas as in observtions
# compare if the bags of observations and thetas match up

## Alpha ################
alpha = torch.tensor(10.0) # should not be learned

true_alpha_sampler = dist.Uniform(-alpha, alpha) # uniform from -alpha to alpha, holds universe of possible values?
true_alpha = true_alpha_sampler.sample()
print('true_alpha', true_alpha) # this is where boundary line lies?
dim = 1
#########################

## Data #################
coherence = 0.2
samples = 100
data = []

coherent_dist = dist.Normal(true_alpha, 1) # constant * 1 / coherence
for i in range(int(coherence * samples)):
    sampled_point = pyro.sample("coherent_point", coherent_dist)
    print(sampled_point)
    data.append(sampled_point)

incoherent_dist = dist.Uniform(-alpha, alpha) # incoherence chosen from all possible vectors
for i in range(samples - int(coherence * samples)):
    data.append(pyro.sample("incoherent_point", incoherent_dist))

shuffle(data)
data = torch.tensor(data)
print('data shape', data.shape)
#########################


## Model ################
def model(data):
    coefs_mean = torch.zeros(dim) # (-alpha + alpha) / 2
    coefs = pyro.sample('beta', dist.Normal(coefs_mean, torch.tensor([alpha] * dim)))
    y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1))) #obs=labels
    return y

    # hypothesis_prob = torch.tensor(0.5)
    # h = pyro.sample("h", dist.Bernoulli(hypothesis_prob))
    # print('h', h)
    # if h == 1:
    #     theta = pyro.sample("theta", dist.Uniform(0, alpha))
    # else:
    #     theta = pyro.sample("theta", dist.Uniform(-alpha, 0))

    # # loop over the observed data
    # for i in range(len(data)):
    #     pyro.sample(f"obs_{i}", theta, obs=data[i]) # convert to MCMC, create sample function for one sample, run n times (right now it's one theta)


hmc_kernel = HMC(model, step_size=0.1, num_steps=50)
mcmc = MCMC(hmc_kernel, num_samples=100, warmup_steps=100)
mcmc.run(data)
print(mcmc.get_samples()['beta'], mcmc.get_samples()['beta'].shape)

# mcmc.get_samples()['beta'].mean(0)  # doctest: +SKIP


# n_steps = 50
# # do gradient steps
# for _ in range(n_steps):
#     svi.step(data)


# grab the learned variational parameters
# hypothesis_prob_q = pyro.param("hypothesis_prob_q").item()
# alpha_q = pyro.param("alpha_q").item()

# print(hypothesis_prob_q, alpha_q)