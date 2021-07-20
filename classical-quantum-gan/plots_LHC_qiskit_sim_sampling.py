#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from numpy.random import randn
import qiskit, qiskit.circuit.qpy_serialization
from qiskit import Aer, IBMQ
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from main import readInit, readEvent, GetEnergySquared, GetMandelT, GetRapidity
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from scipy.stats import entropy
import argparse  

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
def generate_fake_samples(circuit, backend, noise_params, samples, nqubits, layers, nshots):
    latent_dim = len(noise_params)
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, samples)
    def bind_params(i):
        return circuit.bind_parameters(dict(zip(noise_params, x_input[i])))
    circuits = [qiskit.transpile(bind_params(i), backend) for i in range(samples)]
    print("compiled circuit, noise id 0")
    print(circuits[0])
    circuits[0].draw(output='mpl', filename=f"compiled_generator_circuit_noise0_{samples}_{nqubits}_{latent_dim}.pdf")

    # run the simulation
    job = qiskit.execute(circuits, backend=backend, shots=nshots)
    result = job.result()
    with open(f"compiled_generator_circuits_{job.job_id()}_{samples}_{nqubits}_{latent_dim}.qpy", 'wb') as f:
        qiskit.circuit.qpy_serialization.dump(circuits, f)
    np.savez(f"job_results_{job.job_id()}_{samples}_{nqubits}_{latent_dim}.npy", **result.to_dict())
    counts = result.get_counts()

    # generator outputs
    X1 = []
    X2 = []
    X3 = []
    # quantum generator circuit
    for c in counts:
        X3.append((-c.get('000', 0)+c.get('001', 0)-c.get('010', 0)+c.get('011', 0)-c.get('100', 0)+c.get('101', 0)-c.get('110', 0)+c.get('111', 0))/nshots)
        X2.append((-c.get('000', 0)-c.get('001', 0)+c.get('010', 0)+c.get('011', 0)-c.get('100', 0)-c.get('101', 0)+c.get('110', 0)+c.get('111', 0))/nshots)
        X1.append((-c.get('000', 0)-c.get('001', 0)-c.get('010', 0)-c.get('011', 0)+c.get('100', 0)+c.get('101', 0)+c.get('110', 0)+c.get('111', 0))/nshots)

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

def describe_qubit(qubit, properties):
    """Print a string describing some of reported properties of the given qubit."""

    # Conversion factors from standard SI units
    us = 1e6
    ns = 1e9
    GHz = 1e-9

    print(f"Qubit {qubit} has a ")
    try:
        print(f"  - T1 time of {properties.t1(qubit) * us} microseconds")
    except AttributeError:
        pass
    try:
        print(f"  - T2 time of {properties.t2(qubit) * us} microseconds")
    except AttributeError:
        pass
    try:
        print(f"  - U2 gate error of {properties.gate_error('u2', qubit)}")
    except (AttributeError, qiskit.providers.exceptions.BackendPropertyError):
        pass
    try:
        print(f"  - U2 gate duration of {properties.gate_length('u2', qubit) * ns} nanoseconds")
    except (AttributeError, qiskit.providers.exceptions.BackendPropertyError):
        pass
    try:
        print(f"  - resonant frequency of {properties.frequency(qubit) * GHz} GHz")
    except AttributeError:
        pass

def print_backend_info(backend):
    config = backend.configuration()
    # Basic Features
    print("This backend is called {0}, and is on version {1}. It has {2} qubit{3}. It "
          "{4} OpenPulse programs. The basis gates supported on this device are {5}."
          "".format(config.backend_name,
                    config.backend_version,
                    config.n_qubits,
                    '' if config.n_qubits == 1 else 's',
                    'supports' if config.open_pulse else 'does not support',
                    config.basis_gates))

    props = backend.properties()
    describe_qubit(0, props)
    describe_qubit(1, props)
    describe_qubit(2, props)

def main(samples, bins, latent_dim, layers, training_samples, batch_samples, lr, plot_real, nshots, backend, noise_model):
    
    # number of qubits generator
    nqubits = 3

    # angle paramters for the qiskit circuit
    circuit_params = []
    def add_param():
        par = qiskit.circuit.Parameter(f'p[{len(circuit_params)}]')
        circuit_params.append(par)
        return par

    circuit_noise_params = qiskit.circuit.ParameterVector('r', latent_dim)
    import itertools
    circuit_noise_params_cycle = itertools.cycle(circuit_noise_params)

    def add_angle():
        return add_param()*next(circuit_noise_params_cycle)+add_param()

    # create quantum generator with qiskit
    qubits = qiskit.QuantumRegister(nqubits, 'q')
    bits = qiskit.ClassicalRegister(nqubits, 'c')
    circuit = qiskit.QuantumCircuit(nqubits, nqubits)
    circuit.reset(qubits)
    for l in range(layers):
        for qb in qubits:
            circuit.ry(add_angle(), qb)
            circuit.rz(add_angle(), qb)
        if l==1 or l==5 or l==9 or l==13 or l==17:
            circuit.cry(add_angle(), qubits[0], qubits[1])
            circuit.cry(add_angle(), qubits[0], qubits[2])
        if l==3 or l==7 or l==11 or l==15 or l==19:
            circuit.cry(add_angle(), qubits[1], qubits[2])
            circuit.cry(add_angle(), qubits[2], qubits[0])
#        circuit.barrier(qubits)
    for qb in qubits:
        circuit.ry(add_angle(), qb)
#    circuit.barrier(qubits) # are all these barries needed?
    circuit.measure(qubits, bits)
    print("base generator circuit")
    print(circuit)
    circuit.draw(output='mpl', filename=f"base_generator_circuit_{samples}_{nqubits}_{latent_dim}.pdf")

    d_loss = np.loadtxt(f"dloss_LHCdata_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}")
    g_loss = np.loadtxt(f"gloss_LHCdata_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}")
    params = np.loadtxt(f"PARAMS_LHCdata_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}")

    # assigning parameters from file
    circuit.assign_parameters(dict(zip(circuit_params, params)), inplace=True)
    print("optimized generator circuit")
    print(circuit)
    circuit.draw(output='mpl', filename=f"optimized_generator_circuit_{samples}_{nqubits}_{latent_dim}.pdf")
    
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
    simulator = Aer.get_backend('aer_simulator')
    if noise_model:
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
        try:
            backend = provider.get_backend(noise_model)
        except qiskit.providers.exceptions.QiskitBackendNotFoundError:
            print('available backends')
            for b in provider.backends():
                print(b)
            raise
        backend = simulator.from_backend(backend)
    else:
        backend = simulator
    print_backend_info(backend)
    x_fake, _ = generate_fake_samples(circuit, simulator, circuit_noise_params, samples, nqubits, layers, nshots)
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
    fig.savefig(f"s-distribution_{samples}_{bins}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}_{nshots}.pdf", bbox_inches='tight')
    
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
    fig.savefig(f"t-distribution_{samples}_{bins}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}_{nshots}.pdf", bbox_inches='tight')
        
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
    fig.savefig(f"y-distribution_{samples}_{bins}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}_{nshots}.pdf", bbox_inches='tight')
    
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
    fig.savefig(f"s-t_FAKE_{samples}_{bins}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}_{nshots}.pdf", bbox_inches='tight')
    
    H, xedges, yedges = np.histogram2d(-1*x_real2, x_real3, [-1*np.logspace(np.log10(max(x_real2)),np.log10(min(x_real2)),bins),np.linspace(min(x_real3),max(x_real3),bins)])    
    fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
    plt.ylabel('', fontsize=17)
    plt.xlabel('', fontsize=17)
    plt.hist2d(-1*x_fake2, x_fake3, [-1*np.logspace(np.log10(max(x_real2)),np.log10(min(x_real2)),bins),np.linspace(min(x_real3),max(x_real3),bins)], color='red', vmax=max(H.flatten()), label='real', alpha=1.0, linewidths=0)
    plt.colorbar()
    plt.rcParams["axes.linewidth"]  = 1.25
    plt.xscale('symlog')
    plt.show()
    fig.savefig(f"t-y_FAKE_{samples}_{bins}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}_{nshots}.pdf", bbox_inches='tight')
    
    H, xedges, yedges = np.histogram2d(x_real3, x_real1, [np.linspace(min(x_real3),max(x_real3),bins),np.logspace(np.log10(min(x_real1)),np.log10(max(x_real1)),bins)])    
    fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
    plt.ylabel('', fontsize=17)
    plt.xlabel('', fontsize=17)
    plt.hist2d(x_fake3, x_fake1, [np.linspace(min(x_real3),max(x_real3),bins),np.logspace(np.log10(min(x_real1)),np.log10(max(x_real1)),bins)], color='red', vmax=max(H.flatten()), label='real', alpha=1.0, linewidths=0)
    plt.colorbar()
    plt.rcParams["axes.linewidth"]  = 1.25
    plt.yscale('log')
    plt.show()
    fig.savefig(f"y-s_FAKE_{samples}_{bins}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}_{nshots}.pdf", bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=1000, type=int)
    parser.add_argument("--nshots", default=1000, type=int)
    parser.add_argument("--backend", default="simulator", type=str)
    parser.add_argument("--noise_model", default=None, type=str)
    parser.add_argument("--bins", default=100, type=int)
    parser.add_argument("--latent_dim", default=5, type=int)
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--training_samples", default=10000, type=int)
    parser.add_argument("--batch_samples", default=128, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--plot_real", default=False, type=bool)
    args = vars(parser.parse_args())
    main(**args)
