import torch

import numpy as np

import pyro
import pyro.distributions as dist
import torch.distributions.constraints as constraints
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from random import shuffle


true_alpha = 10
coherence = 0.2
samples = 100
data = []

coherent_dist = dist.Normal(true_alpha, 5)
for i in range(int(0.2 * samples)):
    data.append(pyro.sample("coherent_point", coherent_dist))

incoherent_dist = dist.Uniform(-90, 90)
for i in range(samples - int(0.2 * samples)):
    data.append(pyro.sample("incoherent_point", incoherent_dist))

shuffle(data)


def model(data):
    hypothesis_prob = torch.tensor(0.5)
    alpha = torch.tensor(10.0)

    h = pyro.sample("h", dist.Bernoulli(hypothesis_prob))

    if h == 1:
        theta = pyro.sample("theta", dist.Uniform(0, alpha))
    else:
        theta = pyro.sample("theta", dist.Uniform(-alpha, 0))

    # loop over the observed data
    for i in range(len(data)):
        pyro.sample("obs_{}".format(i), dist.Normal(0, alpha), obs=data[i])


def guide(data):
    hypothesis_prob_q = pyro.param("hypothesis_prob_q", torch.tensor(0.5),
                         constraint=constraints.unit_interval)
    alpha_q = pyro.param("alpha_q", torch.tensor(10.0),
                        constraint=constraints.positive)

    h_q = pyro.sample("h", dist.Bernoulli(hypothesis_prob_q))

    if h_q == 1:
        pyro.sample("theta", dist.Uniform(0, alpha_q))
    else:
        pyro.sample("theta", dist.Uniform(-alpha_q, 0))


# set up the optimizer
adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

n_steps = 50
# do gradient steps
for step in range(n_steps):
    svi.step(data)


# grab the learned variational parameters
hypothesis_prob_q = pyro.param("hypothesis_prob_q").item()
alpha_q = pyro.param("alpha_q").item()

print(hypothesis_prob_q, alpha_q)