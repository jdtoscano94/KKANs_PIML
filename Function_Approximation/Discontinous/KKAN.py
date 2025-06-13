# %%
import os
from tqdm import tqdm
import sys
import os
file_path = os.getcwd()
project_root = os.path.dirname(os.path.dirname(file_path))
print(f"Project root: {project_root}")
if project_root not in sys.path:
    sys.path.append(project_root)
    
import time
import jax
from jax import lax
from jax import flatten_util
from Crunch.Models.layers import  *
from Crunch.Models.polynomials import  *
from Crunch.Auxiliary.metrics import  *
from jax import vmap
import argparse
import jax.numpy as jnp
import jaxopt
import optax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import trange
from jax import jvp, vjp, value_and_grad
from flax import linen as nn
from typing import Sequence
from functools import partial
import scipy
from pyDOE import lhs
import scipy.io as sio

import jax
import jax.numpy as jnp
import jaxopt
from jaxopt import LBFGS
import random


# %%
from jax import config
config.update("jax_default_matmul_precision", "float32")

# %%
cmap = 'RdBu_r'
num_colors=8
# Create a colormap
cmap = plt.get_cmap(cmap)
colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
colors=colors[:num_colors//4]+colors[3*num_colors//4:]
print(len(colors))
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 14

# %%
# Set up argument parser
parser = argparse.ArgumentParser(description='Tuning Parameters')
parser.add_argument('--Equation', type=str, default='Disc3', help='Name of equation')
parser.add_argument('--Name', type=str, default='KKAN', help='Name of the experiment')
parser.add_argument('--NC', type=int, default=10000, help='Number of samples for training')
parser.add_argument('--NI', type=int, default=512, help='Number of iterations')
parser.add_argument('--NB', type=int, default=512, help='Batch size')
parser.add_argument('--NC_TEST', type=int, default=100, help='Number of test samples')
parser.add_argument('--SEED', type=int, default=444, help='Random seed')
parser.add_argument('--EPOCHS', type=int, default=200000, help='Number of training epochs')
parser.add_argument('--N_LAYERS', type=int, default=4, help='Number of layers in the network')
parser.add_argument('--HIDDEN', type=int, default=32, help='Number of hidden units per layer')
parser.add_argument('--FEATURES', type=int, default=64, help='Feature size')
parser.add_argument('--degree', type=int, default=7, help='Degree of outer')
parser.add_argument('--degree_T', type=int, default=7, help='Degree of polynomial')
parser.add_argument('--lr_fact', type=float, default=0.2, help='Scale Lr')
parser.add_argument('--eta', type=float, default=0.01, help='Learning rate or step size for adaptive gamma')
parser.add_argument('--gamma', type=float, default=0.999, help='Decay rate for adaptive gamma')
parser.add_argument('--gamma_grads', type=float, default=0.99, help='Decay rate for adaptive gamma')
parser.add_argument('--alpha', type=float, default=0.999750, help='Decay rate for exponential moving average')
parser.add_argument('--cap_RBA', type=float, default=20, help='Cap limit for RBA')
parser.add_argument('--max_RBA', type=float, help='Maximum RBA value, default calculated as eta / (1 - gamma)')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for learning rate schedule')
parser.add_argument('--LR', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--decay_step', type=int,default=5000, help='Decay step size')
parser.add_argument('--Note', type=str, default='', help='In case')
parser.add_argument('--basis', type=str, default='sin_series', help='basis selection for g')


# Parse arguments and display them
args, unknown = parser.parse_known_args()
for arg, value in vars(args).items():
    print(f'{arg}: {value}')

# Initialize parameters with parsed or default values
NC = args.NC
NI = args.NI
NB = args.NB
NC_TEST = args.NC_TEST
SEED = args.SEED
EPOCHS = args.EPOCHS
N_LAYERS = args.N_LAYERS
HIDDEN = args.HIDDEN
FEATURES = args.FEATURES
degree = args.degree
degree_T = args.degree_T
eta = args.eta
#RBA Params
gamma = args.gamma
alpha = args.alpha
max_RBA0 = args.max_RBA if args.max_RBA is not None else eta / (1 - gamma)
cap_RBA = args.cap_RBA
# Global weights
gamma_grads=args.gamma_grads
# Optimizer parameters
decay_rate = args.decay_rate
LR = args.LR
lr0 = LR
decay_step = args.decay_step# if args.decay_step is not None else int(EPOCHS * jnp.log(decay_rate) / jnp.log(lrf / lr0))
args.Name=args.Name+f'g:{args.basis}-[d:{degree},lr:{args.lr_fact}]_psi[N:{N_LAYERS},H:{HIDDEN},T:{degree_T},F:{FEATURES},lr:{lr0:.1e},rate:{decay_rate},step:{decay_step}]_RBA[{max_RBA0:.2f}-{cap_RBA:.2f}]__GW:[{alpha:.6f},{args.gamma_grads:.4f}]_Seed:{SEED}'+args.Note
print(args.Name)
# random key
key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)
# Initialize NumPy seed
np.random.seed(SEED)
# Initialize Python's random module seed
random.seed(SEED)


# %% [markdown]
# ## 1. PINN

# %%
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
            H = Polynomial_Embedding_Layer(degree=self.degree)(X) 
            for fs in self.features[:-1]:
                H = nn.activation.tanh(nn.Dense(fs)(H))
            H = Polynomial_Embedding_Layer(degree=self.degree)(H)
            H = nn.Dense(self.features[-1])(H)
            sum_psi += H
        return sum_psi


# The PINN class integrates GetPhi and RBF_KAN_layer to compute the final output
class PINN(nn.Module):
    degree: int
    degree_T: int
    features: Sequence[int]
    M: int = 10
    basis: str= 'rbf'
    out_dim: int =1
    def setup(self):
        # Initialize the GetPhi submodule
        self.get_Psi = get_Psi(degree=self.degree_T, features=self.features, M=self.M)
        if self.basis.lower()=='rbf':
            self.g_fx= RBF_KAN_layer(out_dim=self.out_dim, degree=self.degree)
        elif self.basis.lower()=='chebyshev':
            self.g_fx= Polynomial_KAN_layer(out_dim=self.out_dim, degree=self.degree)
        elif self.basis.lower()=='legendre':
            self.g_fx= Polynomial_KAN_layer(out_dim=self.out_dim, degree=self.degree,polynomial_type='L')
        elif self.basis.lower()=='sin_series':
            self.g_fx= AcNet_KAN_layer(out_dim=self.out_dim, degree=self.degree)
        elif self.basis.lower()=='chebyshev_grid':
            self.g_fx= Polynomial_grid_KAN_layer(out_dim=self.out_dim, degree=self.degree,polynomial_type='T')
        elif self.basis.lower()=='rbf_single':
            self.g_fx= RBF_KAN_single_layer(out_dim=self.out_dim, degree=self.degree)
        else:
            print(f'the desired basis:{self.basis.lower()} is not available.')
    @nn.compact
    def __call__(self, t, x):
        # Process inputs through the GetPhi function
        inputs = [t, x]
        sum_psi = self.get_Psi(inputs)
        sum_Phi = self.g_fx(sum_psi)
        return sum_Phi


# %%
@partial(jax.jit, static_argnums=(0, 1))  # key and optimizer are static
def update_model(key, optimizer, gradient, params, state):
    # Perform updates using the specified optimizer and key
    updates, new_state = optimizer.update(gradient['params'][key], state)
    new_params = optax.apply_updates(params['params'][key], updates)
    # Return updated parameters and state for this key only
    params['params'][key] = new_params
    return params, new_state


# %%
@partial(jax.jit, static_argnums=(0,))
def apply_model(apply_fn, params, lambdas, gamma, eta,lamE,lamB,all_grads,alpha, *train_data):
    # Unpack data
    def data_loss(params, t, x, u,lambdas):
        return jnp.mean((apply_fn(params, t, x) - u)**2)
    # unpack data
    ti,xi,ui = train_data
    #Update RBA
    r_i=jnp.abs((apply_fn(params, ti, xi) - ui))
    new_lambdas=gamma*lambdas+eta*(r_i/r_i.max())
    #Function
    loss_fn = lambda params: data_loss(params, ti,xi,ui,new_lambdas)
    loss, total_gradient = jax.value_and_grad(loss_fn)(params)

    #Store
    all_loss={
        'loss_data':loss,
        'loss_BCs':0.0,
        'Loss':loss,
    }
    all_grads={
        'grad_bar_PDE':0.0,
        'grad_bar_BCs':0.0,
        'grad_PDE':0.0,
        'grad_BCs':0.0,
    }
    return all_loss, total_gradient, new_lambdas,lamB,all_grads

# %%
def get_gamma(eta, max_RBA):
    gamma_it = 1-eta/max_RBA
    return gamma_it

# %% [markdown]
# # Generate Data

# %%

# Define the piecewise function
def disc_fx(x):
    if x < 0:
        return (5 + sum([np.sin(k * x) for k in range(1, 5)]) ) # Summing sin(kx) from k=1 to 4
    else:
        return (np.cos(10 * x))  # cos(10x) for x >= 0

# Create a vectorized version of the function to handle arrays
disc_vect = np.vectorize(disc_fx)

# Generate values of x from -5 to 5 for plotting
x_values = np.linspace(-5, 5, 200)
y_values = disc_vect(x_values)

# Plot the function
plt.plot(x_values, y_values, label='y(x)')
plt.axhline(0, color='black',linewidth=0)
plt.axvline(0, color='black',linewidth=0)
plt.title('Piecewise Function y(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()


# %%
x1_vals = np.linspace(-2, 2, 256)
x2_vals = np.linspace(-2, 2, 256)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Flatten the full grid for testing data
t = X1.flatten()[:, None]
x = X2.flatten()[:, None]

# Evaluate ground truth at all grid points (testing data)
U_x = disc_vect(x)
U_t = disc_vect(t)
U_gt = U_x * U_t
Exact0 = U_gt.reshape(X1.shape)
u_gt = Exact0.flatten()[:, None]

# Collocation points (training data) using LHS
lb = jnp.array([t.min(), x.min()])
ub = jnp.array([t.max(), x.max()])
X_c = lb + (ub - lb)*lhs(2, NC)
tc = X_c[:, 0:1]
xc = X_c[:, 1:2]

# Evaluate ground truth at training points
U_t_train = disc_vect(tc)
U_x_train = disc_vect(xc)
u_train = (U_t_train * U_x_train).reshape(-1,1)


# Create the training and testing datasets
train_data = (tc, xc, u_train)
lambdas = u_train * 0   # same size as training outputs
test_data = t, x, u_gt

# %%
# Plotting code
fig = plt.figure(figsize=(9, 3))
# Scatter plot for collocation, initial, and boundary points

# Surface plot for Exact solution
ax2 = fig.add_subplot(122)
contour = ax2.contourf(X1, X2, Exact0, cmap='jet', levels=50)  # Filled contour plot with 50 levels
fig.colorbar(contour)  # Add color bar to show scale
ax2.set_xlabel('t')
ax2.set_ylabel('x')
ax2.set_title('Exact Solution')
plt.show()

# %%
# Batches SNR
n_batches=256
X_all=jnp.hstack([t, x, u_gt])
X_batches=np.array(np.split(X_all,n_batches))
X_batches.shape


# %% [markdown]
# ## Initialize Model

# %%

# force jax to use one device
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

# feature sizes
feat_sizes = tuple([HIDDEN for _ in range(N_LAYERS)] + [FEATURES])
print(feat_sizes)
# make & init model
model = PINN(degree,degree_T,feat_sizes,basis=args.basis)
params = model.init(subkey, jnp.ones((NC, 1)), jnp.ones((NC, 1)))

optimizers = {}
for key in params['params'].keys():
    if key=='g_fx':
        print(f'Kart model with basis:{args.basis}')
        optimizers[key]=optax.adam(optax.exponential_decay(lr0*args.lr_fact, decay_step, decay_rate, staircase=False))
    else:
        optimizers[key]=optax.adam(optax.exponential_decay(lr0, decay_step, decay_rate, staircase=False))

# Initialize optimizer states for each parameter group
states = {key: optim.init(params['params'][key]) for key, optim in optimizers.items()}

# forward & loss function
apply_fn = jax.jit(model.apply)


total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(total_params )

# %%
def get_g_x(params, t, x):
    # Compute u
    u = apply_fn(params, t, x)
    # Compute derivatives
    v_t = jnp.ones_like(t)
    v_x = jnp.ones_like(x)
    u_t = jvp(lambda t_val: apply_fn(params, t_val, x), (t,), (v_t,))[1]
    u_x = jvp(lambda x_val: apply_fn(params, t, x_val), (x,), (v_x,))[1]
    return jnp.hstack([u_t,u_x])
def get_GC(params,t,x):
    g_x=get_g_x(params, t, x)[:,:,None]
    norm_f_tx = jnp.linalg.norm(g_x, ord='fro', axis=(1, 2))**2
    model_gc=jnp.mean(norm_f_tx)
    return model_gc
print(get_GC(params,t,x))



# %%
all_errors = []
all_its = []
all_loss = []
all_gamma = []
all_lamB = []
all_max_RBA = []
all_lamE = []
all_lambdas=[]
all_gc = []

start = time.time()
pbar = tqdm(range(1, EPOCHS + 1), desc='Training Progress')
gamma_it=get_gamma(eta, max_RBA0)
#Global weights
max_RBA=max_RBA0
lamE,lamB=1,max_RBA**2
#RBA
step_RBA=(cap_RBA-max_RBA0)/((EPOCHS)/50000-1)
# initialize grads container
all_grads={
    'grad_bar_PDE':1,
    'grad_bar_BCs':1,
}
for e in pbar:
    # single run
    all_loss_it, gradient, lambdas,lamB,all_grads = apply_model(apply_fn, params, lambdas,gamma_it,eta,lamE,lamB,all_grads,alpha, *train_data)
    for key in params['params']:
        params, states[key] = update_model(key, optimizers[key], gradient, params, states[key])
    log_frequency = 1 if e < 500 else 100 if e <= 5000 else 500
    if e % log_frequency == 0:
        # Update RBA
        max_RBA=max_RBA0+step_RBA*e//50000
        if max_RBA>cap_RBA+1:
            max_RBA=cap_RBA+1
            alpha=1
        gamma_it=get_gamma(eta, max_RBA)
        all_lambdas.append(np.array(lambdas))  # JAX to NumPy conversion
        #Compute errors
        error = relative_l2(apply_fn(params, t, x), u_gt)
        # Geometric Complexity
        model_gc=get_GC(params,t,x)
        # Updating the tqdm progress bar with loss and other metrics
        pbar.set_description(f"It: {e}/{EPOCHS} | Error: {error:.3e} | lam_max: {lambdas.max():.3f}| max_RBA: {max_RBA:.3f}| lamB: {lamB:.3f}|")    
        # Append metrics to lists
        all_errors.append(error)
        all_its.append(e)
        all_lambdas.append(np.array(lambdas))
        all_loss.append(all_loss_it['Loss'])
        all_gamma.append(gamma_it)
        all_lamB.append(lamB)
        all_max_RBA.append(max_RBA)
        all_lamE.append(lamE)
        all_gc.append(model_gc)
end = time.time()
print(f'Runtime: {((end - start) / EPOCHS * 1000):.2f} ms/iter.')

# %% [markdown]
# # Save Results

# %%
results_dict = {
    'model_name': args.Name,
    'all_errors': all_errors,
    'all_its': all_its,
    'all_loss': all_loss,
    'all_gamma': all_gamma,
    'all_lamB': all_lamB,
    'all_max_RBA': all_max_RBA,
    'all_gc': all_gc,
}

# Save dictionary as a .mat file
scipy.io.savemat(args.Name + 'Log_files.mat', results_dict)

# %% [markdown]
# # Errors
# 

# %%
# Adjusting the plot to use a log scale for both loss and error

plt.figure(figsize=(12, 6))

# Plotting loss history with log scale
plt.subplot(2, 2, 1)
plt.plot(all_its, all_loss, label='Loss', color='blue')
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Loss (log scale)')
plt.title('Loss History (Log Scale)')
plt.grid(True)
plt.legend()

# Plotting error history with log scale
plt.subplot(2, 2, 2)
plt.plot(all_its, all_errors, label='Error', color='red')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Iterations')
plt.ylabel('Error (log scale)')
plt.title('Error History (Log Scale)')
plt.grid(True)
plt.legend()

# Plotting error history with log scale
plt.subplot(2, 2, 3)
plt.plot(all_its, all_lamB, label='LamB', color='orange')
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Global Weight')
plt.title('Global Weight History')
plt.grid(True)
plt.legend()



# Show the plots
plt.tight_layout()
plt.savefig(args.Name+'_Loss.png')
plt.show()


# %%
error = relative_l2(apply_fn(params, t, x), u_gt)
print(f'RL2 error: {error:.8f}') 

# %%
it=-1
print('Solution:')
T,X=X1, X2 
u = apply_fn(params, t, x)
u = u.reshape(T.shape)
lambdas_grid=all_lambdas[it]
# Plotting code
fig = plt.figure(figsize=(12, 3))
levels=50
# Scatter plot for collocation, initial, and boundary points
ax1 = fig.add_subplot(131)
contour = ax1.contourf(T ,X, u, cmap='jet', levels=levels)  # Filled contour plot with 50 levels
fig.colorbar(contour)  # Add color bar to show scale
ax1.set_xlabel('t')
ax1.set_ylabel('x')
ax1.set_title('Prediction')

# Surface plot for Exact solution
ax2 = fig.add_subplot(132)
contour = ax2.contourf(T, X, Exact0, cmap='jet', levels=levels)  # Filled contour plot with 50 levels
fig.colorbar(contour)  # Add color bar to show scale
ax2.set_xlabel('t')
ax2.set_ylabel('x')
ax2.set_title('Reference')
# Surface plot for Exact solution
ax3 = fig.add_subplot(133)
contour = ax3.contourf(T, X, np.abs(Exact0-u), cmap='jet', levels=levels)  # Filled contour plot with 50 levels
fig.colorbar(contour)  # Add color bar to show scale
ax3.set_xlabel('t')
ax3.set_ylabel('x')
ax3.set_title('Error')

plt.tight_layout()
plt.savefig(args.Name+'_Results.png')
plt.show()


