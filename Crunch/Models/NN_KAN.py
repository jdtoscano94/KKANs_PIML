
import jax
import jax.numpy as np
from jax import random, grad, vmap, jit, hessian, lax
from jax.example_libraries import optimizers
from jax.nn import relu, tanh
#from jax.config import config
from jax.numpy import index_exp as index
from jax.flatten_util import ravel_pytree
import itertools
from functools import partial
from tqdm import trange, tqdm
import numpy as np0
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import scipy.io as sio
import tqdm as tqdm
import sys
import os

from flax import linen as nn
import jax.numpy as jnp
from typing import Sequence, Callable

from Crunch.Models.layers import  *
from Crunch.Models.polynomials import  *
from Crunch.Auxiliary.metrics import  *


class eMLP(nn.Module):
    layers: Sequence[int]
    activation: Callable = nn.relu
    degree: int=5
    @nn.compact
    def __call__(self, x):
        init = nn.initializers.glorot_normal()
        X =Polynomial_Embedding_Layer(degree=self.degree)(x)
        H = nn.activation.tanh(WN_layer(self.layers[0], kernel_init=init)(X))
        for feat in self.layers[1:-1]:
            H = AdaptiveResNet(out_features=feat)(H) 
        H = Polynomial_Embedding_Layer(degree=self.degree)(H)
        H = WN_layer(self.layers[-1], kernel_init=init)(H)
        return H

class get_Psi(nn.Module):
    degree: int
    features: Sequence[int]
    M: int = 10

    def setup(self):
        # Set up the Chebyshev functions (T_funcs) based on the degree
        self.T_funcs = [globals()[f"T{i}"] for i in range(self.degree + 1)]
    @nn.compact
    def __call__(self, inputs):
        init = nn.initializers.glorot_normal()
        sum_psi = 0
        for i, X in enumerate(inputs):
            X =Polynomial_Embedding_Layer(degree=self.degree)(X)
            # Pass through WN_layer and apply tanh activation
            H = nn.activation.tanh(WN_layer(self.features[0], kernel_init=init)(X))
            # Pass through AdaptiveResNet layers
            for fs in self.features[1:-1]:
                H = AdaptiveResNet(out_features=fs)(H) 
            # Stack transformations using the T_funcs
            H = Polynomial_Embedding_Layer(degree=self.degree)(H)
            H = WN_layer(self.features[-1], kernel_init=init)(H)
            # Accumulate results
            sum_psi += H
        return sum_psi


def KAN_5(layers, activation=tanh):
  ''' Vanilla KAN'''
  def init(rng_key):
      def init_layer(key, in_dim, out_dim,degree=5):
          std_kan=1 / (in_dim * (degree + 1))
          k1, k2 = random.split(key)
          W = std_kan * random.normal(k1, (in_dim, out_dim,degree+1))
          b = np.zeros(out_dim)
          return W, b
      key, *keys = random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      return params
      
  def apply(params, x):
      for W, b in params:
        # Read chebyshev coefficients:
        nfx=x.shape[0]
        ny=x.shape[1]
        cheby_coeffs= W
        inputdim=cheby_coeffs.shape[0]
        outdim=cheby_coeffs.shape[1]
        # Normalize 
        x = activation(x)
        # Reshape
        x = x.reshape((-1, inputdim, 1))
        x=np.stack((T0(x),
                    T1(x),
                    T2(x),
                    T3(x),
                    T4(x),
                    T5(x)),axis=2)# Discard dummy dimension
        # Compute the Chebyshev interpolation
        x = np.einsum("bid0,iod->bo", x, cheby_coeffs)  # shape = (batch_size, output_dim)
        # Remove extra dimension
        x=  np.reshape(x, (nfx,outdim))
      return x
  return init, apply


def KAN_5_trunk(layers, activation=tanh):
  ''' Vanilla KAN'''
  def init(rng_key):
      def init_layer(key, in_dim, out_dim,degree=5):
          std_kan=1 / (in_dim * (degree + 1))
          k1, k2 = random.split(key)
          W = std_kan * random.normal(k1, (in_dim, out_dim,degree+1))
          b = np.zeros(out_dim)
          return W, b
      key, *keys = random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      return params
      
  def apply(params, x):
      x=np.stack((x[0],x[1],np.cos(2*np.pi*x[1]),np.cos(4*np.pi*x[1]),np.sin(2*np.pi*x[1]),np.sin(4*np.pi*x[1])))
      for W, b in params:
        # Read chebyshev coefficients:
        cheby_coeffs= W
        inputdim=cheby_coeffs.shape[0]
        outdim=cheby_coeffs.shape[1]
        # Normalize 
        x = activation(x)
        # Reshape
        x = x.reshape((-1, inputdim, 1))
        x=np.stack((T0(x),
                    T1(x),
                    T2(x),
                    T3(x),
                    T4(x),
                    T5(x)),axis=2)# Discard dummy dimension
        # Compute the Chebyshev interpolation
        x = np.einsum("bid0,iod->bo", x, cheby_coeffs)  # shape = (batch_size, output_dim)
        # Remove extra dimension
        x=  np.reshape(x, (outdim,))
      return x
  return init, apply


def init_A(rng_key, N,K):
    k1, k2 = random.split(rng_key)
    glorot_stddev = 1. / np.sqrt((N + K) / 2.)
    A= glorot_stddev * random.normal(k1, (N, K))
    return A

# Chebyshev's Polynomials
def T0(x):
    return x*0+1
def T1(x):
    return x
def T2(x):
    return 2*x**2-1
def T3(x):
    return 4*x**3-3*x
def T4(x):
    return 8*x**4-8*x**2+1
def T5(x):
    return 16*x**5-20*x**3+5*x
def T6(x):
    return 32*x**6-48*x**4+18*x**2-1
def T7(x):
    return 64*x**7-112*x**5+56*x**3-7*x
def T8(x):
    return 128*x**8-256*x**6+160*x**4-32*x**2+1
def T9(x):
    return 256*x**9-576*x**7+432*x**5-120*x**3+9*x
def T10(x):
    return 512*x**10-1280*x**8+1120*x**6-400*x**4+50*x**2-1
def T11(x):
    return 1024*x**11-2816*x**9+2816*x**7-1232*x**5+220*x**3-11*x
def T12(x):
    return 2048*x**12-6144*x**10+6912*x**8-3584*x**6+840*x**4-72*x**2+1