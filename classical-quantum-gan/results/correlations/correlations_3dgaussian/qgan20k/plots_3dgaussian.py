#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from numpy.random import randn
from qibo import gates, hamiltonians, models, set_backend
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from scipy.stats import entropy
import argparse  

def set_params(circuit, params, x_input, i, nqubits, layers, latent_dim):
    p = []
    index = 0
    noise = 0
    for l in range(layers):
        for q in range(nqubits):
            p.append(params[index]*x_input[noise][i] + params[index+1])
            index+=2
            noise=(noise+1)%latent_dim
            p.append(params[index]*x_input[noise][i] + params[index+1])
            index+=2
            noise=(noise+1)%latent_dim
        if l==1 or l==5 or l==9 or l==13 or l==17:
            p.append(params[index]*x_input[noise][i] + params[index+1])
            index+=2
            noise=(noise+1)%latent_dim
            p.append(params[index]*x_input[noise][i] + params[index+1])
            index+=2
            noise=(noise+1)%latent_dim
        if l==3 or l==7 or l==11 or l==15 or l==19:
            p.append(params[index]*x_input[noise][i] + params[index+1])
            index+=2
            noise=(noise+1)%latent_dim
            p.append(params[index]*x_input[noise][i] + params[index+1])
            index+=2
            noise=(noise+1)%latent_dim
    for q in range(nqubits):
        p.append(params[index]*x_input[noise][i] + params[index+1])
        index+=2
        noise=(noise+1)%latent_dim
    circuit.set_parameters(p)  

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, samples):
    # generate points in the latent space
    x_input = randn(latent_dim * samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(samples, latent_dim)
    return x_input

# use the generator to generate fake examples, with class labels
def generate_fake_samples(circuit, params, latent_dim, samples, nqubits, layers, hamiltonian1, hamiltonian2, hamiltonian3):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, samples)
    x_input = np.transpose(x_input)
    # generator outputs
    X1 = []
    X2 = []
    X3 = []
    # quantum generator circuit
    for i in range(samples):
        set_params(circuit, params, x_input, i, nqubits, layers, latent_dim)
        circuit_execute = circuit.execute()
        X1.append(hamiltonian1.expectation(circuit_execute))
        X2.append(hamiltonian2.expectation(circuit_execute))
        X3.append(hamiltonian3.expectation(circuit_execute))
    # shape array
    X = tf.stack((X1, X2, X3), axis=1)
    # create class labels
    y = np.zeros((samples, 1))
    return X, y

def generate_real_samples(samples):
  # generate training samples from the distribution
    s = []
    mean = [0, 0, 0]
    #cov = [[0.5, 0.1, 0.25], [0.5, 0.25, 0.1], [0.5, 0.5, 0.1]]  
    cov = [[0.5, 0.1, 0.25], [0.1, 0.5, 0.1], [0.25, 0.1, 0.5]]      
    x, y, z = np.random.multivariate_normal(mean, cov, samples).T/4
    s1 = np.reshape(x, (samples,1))
    s2 = np.reshape(y, (samples,1))
    s3 = np.reshape(z, (samples,1))
    s = np.hstack((s1,s2,s3))
    return s

def plot_loss(g_loss, d_loss, nqubits, samples, bins, latent_dim, layers, training_samples, batch_samples, lr):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')    
    fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
    plt.ylabel('Loss', fontsize=17)
    plt.xlabel('Epoch', fontsize=17)
    plt.plot(np.linspace(0, len(g_loss), len(g_loss)), g_loss, label='generator')
    plt.plot(np.linspace(0, len(d_loss), len(d_loss)), d_loss, label='discriminator')
    plt.rcParams["axes.linewidth"]  = 1.25
    plt.legend()
    #plt.show()
    fig.savefig(f"loss_{samples}_{bins}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}.pdf", bbox_inches='tight')
    plt.show(block=False)
    plt.pause(1)
    plt.close()


def main(samples, bins, latent_dim, layers, training_samples, batch_samples, lr, plot_real):
    
    # define hamiltonian to generate fake samples
    def hamiltonian1():
        id = [[1, 0], [0, 1]]
        m0 = hamiltonians.Z(1).matrix
        m0 = np.kron(id, np.kron(id, m0))
        ham = hamiltonians.Hamiltonian(3, m0)
        return ham
    
    def hamiltonian2():
        id = [[1, 0], [0, 1]]
        m0 = hamiltonians.Z(1).matrix
        m0 = np.kron(id, np.kron(m0, id))
        ham = hamiltonians.Hamiltonian(3, m0)
        return ham
    
    def hamiltonian3():
        id = [[1, 0], [0, 1]]
        m0 = hamiltonians.Z(1).matrix
        m0 = np.kron(m0, np.kron(id, id))
        ham = hamiltonians.Hamiltonian(3, m0)
        return ham
    
    # number of qubits generator
    nqubits = 3
    # create hamiltonians
    hamiltonian1 = hamiltonian1()
    hamiltonian2 = hamiltonian2()
    hamiltonian3 = hamiltonian3()
    # create quantum generator
    circuit = models.Circuit(nqubits)
    for l in range(layers):
        for q in range(nqubits):
            circuit.add(gates.RY(q, 0))
            circuit.add(gates.RZ(q, 0))
        if l==1 or l==5 or l==9 or l==13 or l==17:
            circuit.add(gates.CRY(0, 1, 0))
            circuit.add(gates.CRY(0, 2, 0))
        if l==3 or l==7 or l==11 or l==15 or l==19:
            circuit.add(gates.CRY(1, 2, 0))
            circuit.add(gates.CRY(2, 0, 0))
    for q in range(nqubits):
        circuit.add(gates.RY(q, 0))
    
    nqubits = 3
    d_loss = np.loadtxt(f"dloss_3Dgaussian_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}")
    g_loss = np.loadtxt(f"gloss_3Dgaussian_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}")
    params = np.loadtxt(f"PARAMS_3Dgaussian_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}")
    
    plot_loss(g_loss, d_loss, nqubits, samples, bins, latent_dim, layers, training_samples, batch_samples, lr)
    
    print('generating real samples')
    x_real = generate_real_samples(samples)
    x_real1 = []
    x_real2 = []
    x_real3 = []
    for i in range(samples):
        x_real1.append(x_real[i][0])
        x_real2.append(x_real[i][1])
        x_real3.append(x_real[i][2])
    
    print('generating fake samples')   
    x_fake, _ = generate_fake_samples(circuit, params, latent_dim, samples, nqubits, layers, hamiltonian1, hamiltonian2, hamiltonian3)
    x_fake1 = []
    x_fake2 = []
    x_fake3 = []
    for i in range(samples):
        x_fake1.append(x_fake[i][0])
        x_fake2.append(x_fake[i][1])
        x_fake3.append(x_fake[i][2])
    
    print('1D distributions')
    real, _ = np.array(np.histogram(x_real1, np.linspace(-1.0, 1.0, bins)))/samples
    fake, _ = np.array(np.histogram(x_fake1, np.linspace(-1.0, 1.0, bins)))/samples
    for i in range(len(real)):
        if real[i] == 0:
            real[i] = 1e-100        
        if fake[i] == 0:
            fake[i] = 1e-100
    kl_divergence = np.around(entropy(real, fake), 3)
    
    fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
    plt.ylabel('Samples', fontsize=17)
    plt.xlabel('', fontsize=17)
    plt.hist(x_real1, np.linspace(-1.0, 1.0, bins), color='red', histtype='step', label='real', alpha=0.5)
    plt.hist(x_fake1, np.linspace(-1.0, 1.0, bins), color='blue', histtype='step', label='fake', alpha=0.5)
    plt.rcParams["axes.linewidth"]  = 1.25
    plt.legend()
    plt.text(-1.0, max(fake.flatten())*samples, 'KL divergence = '+str(kl_divergence), bbox=dict(fill=False, edgecolor='black', linewidth=2))
    plt.show()
    fig.savefig(f"1-distribution_{samples}_{bins}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}.pdf", bbox_inches='tight')
    
    real, _ = np.array(np.histogram(x_real2, np.linspace(-1.0, 1.0, bins)))/samples
    fake, _ = np.array(np.histogram(x_fake2, np.linspace(-1.0, 1.0, bins)))/samples
    for i in range(len(real)):
        if real[i] == 0:
            real[i] = 1e-100        
        if fake[i] == 0:
            fake[i] = 1e-100
    kl_divergence = np.around(entropy(real, fake), 3)
    
    fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
    plt.ylabel('Samples', fontsize=17)
    plt.xlabel('', fontsize=17)
    plt.hist(x_real2, np.linspace(-1.0, 1.0, bins), color='red', histtype='step', label='real', alpha=0.5)
    plt.hist(x_fake2, np.linspace(-1.0, 1.0, bins), color='blue', histtype='step', label='fake', alpha=0.5)
    plt.rcParams["axes.linewidth"]  = 1.25
    plt.legend()
    plt.text(-1.0, max(fake.flatten())*samples, 'KL divergence = '+str(kl_divergence), bbox=dict(fill=False, edgecolor='black', linewidth=2))
    plt.show()
    fig.savefig(f"2-distribution_{samples}_{bins}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}.pdf", bbox_inches='tight')
        
    real, _ = np.array(np.histogram(x_real3, np.linspace(-1.0, 1.0, bins)))/samples
    fake, _ = np.array(np.histogram(x_fake3, np.linspace(-1.0, 1.0, bins)))/samples
    for i in range(len(real)):
        if real[i] == 0:
            real[i] = 1e-100        
        if fake[i] == 0:
            fake[i] = 1e-100
    kl_divergence = np.around(entropy(real, fake), 3)
    
    fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
    plt.ylabel('Samples', fontsize=17)
    plt.xlabel('', fontsize=17)
    plt.hist(x_real3, np.linspace(-1.0, 1.0, bins), color='red', histtype='step', label='real', alpha=0.5)
    plt.hist(x_fake3, np.linspace(-1.0, 1.0, bins), color='blue', histtype='step', label='fake', alpha=0.5)
    plt.rcParams["axes.linewidth"]  = 1.25
    plt.legend()
    plt.text(-1.0, max(fake.flatten())*samples, 'KL divergence = '+str(kl_divergence), bbox=dict(fill=False, edgecolor='black', linewidth=2))
    plt.show()
    fig.savefig(f"3-distribution_{samples}_{bins}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}.pdf", bbox_inches='tight')
    
    if plot_real==True:
        print('Real 2D distributions')
        fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
        plt.ylabel('', fontsize=17)
        plt.xlabel('', fontsize=17)
        plt.hist2d(x_real1, x_real2, np.linspace(-1.0, 1.0, bins), color='red', label='real', alpha=1.0, linewidths=0)
        plt.colorbar()
        plt.rcParams["axes.linewidth"]  = 1.25
        plt.show()
        fig.savefig(f"1-2_REAL_{samples}_{bins}.pdf", bbox_inches='tight')
        
        fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
        plt.ylabel('', fontsize=17)
        plt.xlabel('', fontsize=17)
        plt.hist2d(x_real2, x_real3, np.linspace(-1.0, 1.0, bins), color='red', label='real', alpha=1.0, linewidths=0)
        plt.colorbar()
        plt.rcParams["axes.linewidth"]  = 1.25
        plt.show()
        fig.savefig(f"2-3_REAL_{samples}_{bins}.pdf", bbox_inches='tight')
        
        fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
        plt.ylabel('', fontsize=17)
        plt.xlabel('', fontsize=17)
        plt.hist2d(x_real3, x_real1, np.linspace(-1.0, 1.0, bins), color='red', label='real', alpha=1.0, linewidths=0)
        plt.colorbar()
        plt.rcParams["axes.linewidth"]  = 1.25
        plt.show()
        fig.savefig(f"3-1_REAL_{samples}_{bins}.pdf", bbox_inches='tight')
    
    print('Fake 2D distributions')
    H, xedges, yedges = np.histogram2d(x_real1, x_real2, np.linspace(-1.0, 1.0, bins))
    fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
    plt.ylabel('', fontsize=17)
    plt.xlabel('', fontsize=17)
    plt.hist2d(x_fake1, x_fake2, np.linspace(-1.0, 1.0, bins), color='red', label='real', alpha=1.0, vmax=max(H.flatten()), linewidths=0)
    plt.colorbar()
    plt.rcParams["axes.linewidth"]  = 1.25
    plt.show()
    fig.savefig(f"1-2_FAKE_{samples}_{bins}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}.pdf", bbox_inches='tight')
    
    H, xedges, yedges = np.histogram2d(x_real2, x_real3, np.linspace(-1.0, 1.0, bins))
    fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
    plt.ylabel('', fontsize=17)
    plt.xlabel('', fontsize=17)
    plt.hist2d(x_fake2, x_fake3, np.linspace(-1.0, 1.0, bins), color='red', label='real', alpha=1.0, vmax=max(H.flatten()), linewidths=0)
    plt.colorbar()
    plt.rcParams["axes.linewidth"]  = 1.25
    plt.show()
    fig.savefig(f"2-3_FAKE_{samples}_{bins}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}.pdf", bbox_inches='tight')
    
    H, xedges, yedges = np.histogram2d(x_real3, x_real1, np.linspace(-1.0, 1.0, bins))
    fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
    plt.ylabel('', fontsize=17)
    plt.xlabel('', fontsize=17)
    plt.hist2d(x_fake3, x_fake1, np.linspace(-1.0, 1.0, bins), color='red', label='real', alpha=1.0, vmax=max(H.flatten()), linewidths=0)
    plt.colorbar()
    plt.rcParams["axes.linewidth"]  = 1.25
    plt.show()
    fig.savefig(f"3-1_FAKE_{samples}_{bins}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}.pdf", bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=1000, type=int)
    parser.add_argument("--bins", default=100, type=int)
    parser.add_argument("--latent_dim", default=3, type=int)
    parser.add_argument("--layers", default=20, type=int)
    parser.add_argument("--training_samples", default=10000, type=int)
    parser.add_argument("--batch_samples", default=128, type=int)
    parser.add_argument("--lr", default=0.5, type=float)
    parser.add_argument("--plot_real", default=False, type=bool)
    args = vars(parser.parse_args())
    main(**args)