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
from main import readInit, readEvent, GetEnergySquared, GetMandelT, GetRapidity
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
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


# generate real samples with class labels
def generate_real_samples(samples, distribution, total_samples):
    # generate samples from the distribution
    idx = np.random.randint(total_samples, size=samples)
    X = distribution[idx,:]
    # generate class labels
    y = np.ones((samples, 1))
    return X, y

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

def load_events(filename, samples=10000):
    init = readInit(filename)
    evs = list(readEvent(filename))

    invar = np.zeros((len(evs),3))
    for ev in range(len(evs)):
         invar[ev, 0] = GetEnergySquared(evs[ev])
         invar[ev, 1] = GetMandelT(evs[ev])
         invar[ev, 2] = GetRapidity(init, evs[ev])
         
    return invar[:samples, :]

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
    plt.show()
    fig.savefig(f"loss_{samples}_{bins}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}.pdf", bbox_inches='tight')

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
    d_loss = np.loadtxt(f"dloss_LHCdata_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}")
    g_loss = np.loadtxt(f"gloss_LHCdata_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}")
    params = np.loadtxt(f"PARAMS_LHCdata_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}")
    
    plot_loss(g_loss, d_loss, nqubits, samples, bins, latent_dim, layers, training_samples, batch_samples, lr)
    
    print('generating real samples')
    x_real = load_events('data/ppttbar_10k_events.lhe')
    x_real1 = []
    x_real2 = []
    x_real3 = []
    for i in range(samples):
        x_real1.append(x_real[i][0])
        x_real2.append(x_real[i][1])
        x_real3.append(x_real[i][2])
    
    print('generating fake samples')   
    x_fake, _ = generate_fake_samples(circuit, params, latent_dim, samples, nqubits, layers, hamiltonian1, hamiltonian2, hamiltonian3)
    init = readInit('data/ppttbar_10k_events.lhe')
    evs = list(readEvent('data/ppttbar_10k_events.lhe'))    
    invar = np.zeros((len(evs),3))
    for ev in range(len(evs)):
         invar[ev, 0] = GetEnergySquared(evs[ev])
         invar[ev, 1] = GetMandelT(evs[ev])
         invar[ev, 2] = GetRapidity(init, evs[ev])         
    pt = PowerTransformer()
    print(pt.fit(invar[:10000, :]))
    print(pt.lambdas_)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    print(scaler.fit(pt.transform(invar[:10000, :])))
    
    x_fake = pt.inverse_transform(scaler.inverse_transform(x_fake))
    x_fake1 = []
    x_fake2 = []
    x_fake3 = []
    for i in range(samples):
        x_fake1.append(x_fake[i][0])
        x_fake2.append(x_fake[i][1])
        x_fake3.append(x_fake[i][2])
    
    print('1D distributions')
    real, _ = np.array(np.histogram(x_real1, np.logspace(np.log10(min(x_real1)),np.log10(max(x_real1)),bins)))/samples
    fake, _ = np.array(np.histogram(x_fake1, np.logspace(np.log10(min(x_real1)),np.log10(max(x_real1)),bins)))/samples
    for i in range(len(real)):
        if real[i] == 0:
            real[i] = 1e-100        
        if fake[i] == 0:
            fake[i] = 1e-100
    kl_divergence = np.around(entropy(real, fake), 3)
    
    fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
    plt.ylabel('Samples', fontsize=17)
    plt.xlabel('', fontsize=17)
    plt.hist(x_real1, np.logspace(np.log10(min(x_real1)),np.log10(max(x_real1)),bins), histtype='step', color='red', label='real', alpha=0.5)
    plt.hist(x_fake1, np.logspace(np.log10(min(x_real1)),np.log10(max(x_real1)),bins), histtype='step', color='blue', label='fake', alpha=0.5)
    plt.rcParams["axes.linewidth"]  = 1.25
    plt.legend(loc='best')
    plt.xscale('log')
    plt.text(0.5e6, max(fake.flatten())*samples, 'KL divergence = '+str(kl_divergence), bbox=dict(fill=False, edgecolor='black', linewidth=2))
    plt.show()
    fig.savefig(f"s-distribution_{samples}_{bins}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}.pdf", bbox_inches='tight')
    
    x_real2 = -1*np.array(x_real2)
    x_fake2 = -1*np.array(x_fake2)    
    real, _ = np.array(np.histogram(-1*x_real2, -1*np.logspace(np.log10(max(x_real2)),np.log10(min(x_real2)),bins)))/samples
    fake, _ = np.array(np.histogram(-1*x_fake2, -1*np.logspace(np.log10(max(x_real2)),np.log10(min(x_real2)),bins)))/samples
    for i in range(len(real)):
        if real[i] == 0:
            real[i] = 1e-100        
        if fake[i] == 0:
            fake[i] = 1e-100
    kl_divergence = np.around(entropy(real, fake), 3)
    
    fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
    plt.ylabel('Samples', fontsize=17)
    plt.xlabel('', fontsize=17)
    plt.hist(-1*x_real2, -1*np.logspace(np.log10(max(x_real2)),np.log10(min(x_real2)),bins), histtype='step', color='red', label='real', alpha=0.5)
    plt.hist(-1*x_fake2, -1*np.logspace(np.log10(max(x_real2)),np.log10(min(x_real2)),bins), histtype='step', color='blue', label='fake', alpha=0.5)
    plt.rcParams["axes.linewidth"]  = 1.25
    plt.legend(loc=2)
    plt.xscale('symlog')
    plt.text(-0.5e6, max(fake.flatten())*samples, 'KL divergence = '+str(kl_divergence), bbox=dict(fill=False, edgecolor='black', linewidth=2))
    plt.show()
    fig.savefig(f"t-distribution_{samples}_{bins}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}.pdf", bbox_inches='tight')
        
    real, _ = np.array(np.histogram(x_real3, np.linspace(min(x_real3),max(x_real3),bins)))/samples
    fake, _ = np.array(np.histogram(x_fake3, np.linspace(min(x_real3),max(x_real3),bins)))/samples
    for i in range(len(real)):
        if real[i] == 0:
            real[i] = 1e-100        
        if fake[i] == 0:
            fake[i] = 1e-100
    kl_divergence = np.around(entropy(real, fake), 3)
    
    fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
    plt.ylabel('Samples', fontsize=17)
    plt.xlabel('', fontsize=17)
    plt.hist(x_real3, np.linspace(min(x_real3),max(x_real3),bins), histtype='step', color='red', label='real', alpha=0.5)
    plt.hist(x_fake3, np.linspace(min(x_real3),max(x_real3),bins), histtype='step', color='blue', label='fake', alpha=0.5)
    plt.rcParams["axes.linewidth"]  = 1.25
    plt.legend()
    plt.text(-2, max(fake.flatten())*samples, 'KL divergence = '+str(kl_divergence), bbox=dict(fill=False, edgecolor='black', linewidth=2))
    plt.show()
    fig.savefig(f"y-distribution_{samples}_{bins}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}.pdf", bbox_inches='tight')
    
    if plot_real==True:
        print('Real 2D distributions')
        fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
        plt.ylabel('', fontsize=17)
        plt.xlabel('', fontsize=17)
        plt.hist2d(x_real1, -1*x_real2, [np.logspace(np.log10(min(x_real1)),np.log10(max(x_real1)),bins),-1*np.logspace(np.log10(max(x_real2)),np.log10(min(x_real2)),bins)], color='red', label='real', alpha=1.0, linewidths=0)
        plt.colorbar()
        plt.rcParams["axes.linewidth"]  = 1.25
        plt.xscale('log')
        plt.yscale('symlog')
        plt.show()
        fig.savefig(f"s-t_REAL_{samples}_{bins}.pdf", bbox_inches='tight')
        
        fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
        plt.ylabel('', fontsize=17)
        plt.xlabel('', fontsize=17)
        plt.hist2d(-1*x_real2, x_real3, [-1*np.logspace(np.log10(max(x_real2)),np.log10(min(x_real2)),bins),np.linspace(min(x_real3),max(x_real3),bins)], color='red', label='real', alpha=1.0, linewidths=0)
        plt.colorbar()
        plt.rcParams["axes.linewidth"]  = 1.25
        plt.xscale('symlog')
        plt.show()
        fig.savefig(f"t-y_REAL_{samples}_{bins}.pdf", bbox_inches='tight')
        
        fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
        plt.ylabel('', fontsize=17)
        plt.xlabel('', fontsize=17)
        plt.hist2d(x_real3, x_real1, [np.linspace(min(x_real3),max(x_real3),bins),np.logspace(np.log10(min(x_real1)),np.log10(max(x_real1)),bins)], color='red', label='real', alpha=1.0, linewidths=0)
        plt.colorbar()
        plt.rcParams["axes.linewidth"]  = 1.25
        plt.yscale('log')
        plt.show()
        fig.savefig(f"y-s_REAL_{samples}_{bins}.pdf", bbox_inches='tight')
    
    print('Fake 2D distributions')
    H, xedges, yedges = np.histogram2d(x_real1, -1*x_real2, [np.logspace(np.log10(min(x_real1)),np.log10(max(x_real1)),bins),-1*np.logspace(np.log10(max(x_real2)),np.log10(min(x_real2)),bins)])    
    fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
    plt.ylabel('', fontsize=17)
    plt.xlabel('', fontsize=17)
    plt.hist2d(x_fake1, -1*x_fake2, [np.logspace(np.log10(min(x_real1)),np.log10(max(x_real1)),bins),-1*np.logspace(np.log10(max(x_real2)),np.log10(min(x_real2)),bins)], color='red', vmax=max(H.flatten()), label='real', alpha=1.0, linewidths=0)
    plt.colorbar()
    plt.rcParams["axes.linewidth"]  = 1.25
    plt.xscale('log')
    plt.yscale('symlog')
    plt.show()
    fig.savefig(f"s-t_FAKE_{samples}_{bins}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}.pdf", bbox_inches='tight')
    
    H, xedges, yedges = np.histogram2d(-1*x_real2, x_real3, [-1*np.logspace(np.log10(max(x_real2)),np.log10(min(x_real2)),bins),np.linspace(min(x_real3),max(x_real3),bins)])    
    fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
    plt.ylabel('', fontsize=17)
    plt.xlabel('', fontsize=17)
    plt.hist2d(-1*x_fake2, x_fake3, [-1*np.logspace(np.log10(max(x_real2)),np.log10(min(x_real2)),bins),np.linspace(min(x_real3),max(x_real3),bins)], color='red', vmax=max(H.flatten()), label='real', alpha=1.0, linewidths=0)
    plt.colorbar()
    plt.rcParams["axes.linewidth"]  = 1.25
    plt.xscale('symlog')
    plt.show()
    fig.savefig(f"t-y_FAKE_{samples}_{bins}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}.pdf", bbox_inches='tight')
    
    H, xedges, yedges = np.histogram2d(x_real3, x_real1, [np.linspace(min(x_real3),max(x_real3),bins),np.logspace(np.log10(min(x_real1)),np.log10(max(x_real1)),bins)])    
    fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
    plt.ylabel('', fontsize=17)
    plt.xlabel('', fontsize=17)
    plt.hist2d(x_fake3, x_fake1, [np.linspace(min(x_real3),max(x_real3),bins),np.logspace(np.log10(min(x_real1)),np.log10(max(x_real1)),bins)], color='red', vmax=max(H.flatten()), label='real', alpha=1.0, linewidths=0)
    plt.colorbar()
    plt.rcParams["axes.linewidth"]  = 1.25
    plt.yscale('log')
    plt.show()
    fig.savefig(f"y-s_FAKE_{samples}_{bins}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}.pdf", bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=1000, type=int)
    parser.add_argument("--bins", default=100, type=int)
    parser.add_argument("--latent_dim", default=3, type=int)
    parser.add_argument("--layers", default=20, type=int)
    parser.add_argument("--training_samples", default=10000, type=int)
    parser.add_argument("--batch_samples", default=128, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--plot_real", default=False, type=bool)
    args = vars(parser.parse_args())
    main(**args)