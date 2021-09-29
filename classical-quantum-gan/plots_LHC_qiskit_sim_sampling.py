#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import time
import numpy as np
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
def generate_fake_samples(circuit, backend, merge, noise_params, samples, batch_size, parallel, nqubits, layers, nshots, parallel_shots):
    if merge:
        latent_dim = len(noise_params[0])
        if samples%merge != 0:
            raise ValueError('the number of combined circuits must divide the number of samples')
        actual_samples = samples//merge
        # generate points in latent space
        x_input = generate_latent_points(latent_dim*merge, actual_samples)
    else:
        latent_dim = len(noise_params)
        actual_samples = samples
        # generate points in latent space
        x_input = generate_latent_points(latent_dim, samples)

    if nshots%parallel_shots != 0:
        raise ValueError('the number of parallel circuits must divide the number of shots')
    actual_nshots = nshots//parallel_shots

    def bind_params(i):
        if merge:
            return circuit.bind_parameters(dict(zip((p for np in noise_params for p in np), x_input[i])))
        return circuit.bind_parameters(dict(zip(noise_params, x_input[i])))

    def execute_or_wait(batch):
        wait_time = 60
        while True:
            try:
                return qiskit.execute(batch, backend=backend, shots=actual_nshots)
            except qiskit.providers.ibmq.exceptions.IBMQBackendJobLimitError:
                print(f'Reached maximum number of concurrent jobs. Retrying in {wait_time} seconds')
                time.sleep(wait_time)

    def submit_job(start, stop):
#        batch = [qiskit.transpile(bind_params(i), backend, optimization_level=3) for i in range(start, stop)]
        batch = [bind_params(i) for i in range(start, stop)]
        if start == 0:
            print("compiled circuit, noise id 0")
            print(batch[0])
            batch[0].draw(output='mpl', filename=f"compiled_generator_circuit_noise0_{samples}_{nqubits}_{latent_dim}.pdf")
        job = execute_or_wait(batch)
        print(f'job for batch from {start} to {stop-1} submitted', flush=True)
        with open(f"compiled_generator_circuits_{start}-{stop-1}_{job.job_id()}_{nqubits}_{latent_dim}.qpy", 'wb') as f:
            qiskit.circuit.qpy_serialization.dump(batch, f)
        return job

    def get_results(job):
        result = job.result()
        np.savez(f"job_results_{backend.name()}_{job.job_id()}_{nqubits}_{latent_dim}_{nshots}", **result.to_dict())
        return result

    # run the simulation in batches
    if parallel:
        jobs = [submit_job(i, min(i+batch_size, actual_samples)) for i in range(0, actual_samples, batch_size)]
        results = [get_results(job) for job in jobs]
    else:
        results = [get_results(submit_job(i, min(i+batch_size, actual_samples))) for i in range(0, actual_samples, batch_size)]
    counts = itertools.chain(*(res.get_counts() for res in results))

    # generator outputs
    X1 = []
    X2 = []
    X3 = []
    # quantum generator circuit
    for c in counts:
        for i in range(0, len(circuit.clbits), 3*parallel_shots):
            sum3 = 0
            sum2 = 0
            sum1 = 0
            for j in range(i, i+3*parallel_shots, 3):
                c3 = qiskit.result.marginal_counts(c, [j+0])
                sum3 += c3.get('1', 0)-c3.get('0', 0)
                c2 = qiskit.result.marginal_counts(c, [j+1])
                sum2 += c2.get('1', 0)-c2.get('0', 0)
                c1 = qiskit.result.marginal_counts(c, [j+2])
                sum1 += c1.get('1', 0)-c1.get('0', 0)
            X3.append(sum3/nshots)
            X2.append(sum2/nshots)
            X1.append(sum1/nshots)

    # shape array
    X = np.stack((X1, X2, X3), axis=1)
    # create class labels
    y = np.zeros((samples, 1))
    return X, y

# use the generator to generate fake examples, with class labels
def generate_fake_samples_pennylane(circuit, backend, merge, noise_params, samples, batch_size, parallel, nqubits, layers, nshots, parallel_shots):
    if merge:
        latent_dim = len(noise_params[0])
        if samples%merge != 0:
            raise ValueError('the number of combined circuits must divide the number of samples')
        actual_samples = samples//merge
        # generate points in latent space
        x_input = generate_latent_points(latent_dim*merge, actual_samples)
    else:
        latent_dim = len(noise_params)
        actual_samples = samples
        # generate points in latent space
        x_input = generate_latent_points(latent_dim, samples)

    if nshots%parallel_shots != 0:
        raise ValueError('the number of parallel circuits must divide the number of shots')
    actual_nshots = nshots//parallel_shots

    def bind_params(i):
        if merge:
            return circuit.bind_parameters(dict(zip((p for np in noise_params for p in np), x_input[i])))
        return circuit.bind_parameters(dict(zip(noise_params, x_input[i])))

    if merge or parallel_shots > 1:
        raise NotImplementedError

    import pennylane as qml
    qml_circuit_template = qml.from_qiskit(circuit)
    measures = {clbit[0].index: qubit[0].index for _, qubit, clbit in circuit.get_instructions('measure')}
    dev = qml.device('braket.local.qubit', wires=len(circuit.qubits), shots=actual_nshots)
    nw = len(circuit.qubits)
    wires=list(range(nw))

    @qml.qnode(dev)
    def qc_function(i):
        qml_circ = qml.from_qiskit(bind_params(i))
        qml_circ(wires=wires)
        return [qml.expval(qml.PauliZ(measures[w])) for w in range(3)]

    X = -np.asarray([qc_function(i) for i in range(actual_samples)])[:, ::-1]
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

def main(samples, bins, latent_dim, layers, training_samples, batch_samples, lr, plot_real, nshots, parallel_shots, provider, backend, noise_model, merge):
    
    # number of qubits generator
    nqubits = 3

    # angle paramters for the qiskit circuit
    circuit_params = []
    def add_param():
        par = qiskit.circuit.Parameter(f'p[{len(circuit_params)}]')
        circuit_params.append(par)
        return par

    circuit_noise_params = qiskit.circuit.ParameterVector('r', latent_dim)
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
    batch_size = 1_000_000
    parallel = False

    if provider == 'ibmq':
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    elif provider == 'ibmq-research-open':
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q-research-2', group='cern-3', project='main')
    elif provider == 'ibmq-cern-internal':
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q-cern', group='internal', project='qml4eg')
    elif provider == 'ionq':
        import qiskit_ionq
        provider = qiskit_ionq.IonQProvider()
    else:
        raise ValueError('unknow provider')

    def get_backend(name):
        try:
            return provider.get_backend(name)
        except qiskit.providers.exceptions.QiskitBackendNotFoundError:
            print('available backends')
            for b in provider.backends():
                print(b)
            raise
    if backend == 'simulator':
        if noise_model:
            backend = simulator.from_backend(get_backend(noise_model))
            batch_size = 1000 # limit the memory usage running jobs in smaller batches
        else:
            backend = simulator
    else:
        if noise_model is not None:
            raise ValueError('noise model can be specified only with the simulator backend')
        backend = get_backend(backend)
        batch_size = backend.configuration().max_experiments
        parallel = True
    print_backend_info(backend)

    if parallel_shots:
        qubits = qiskit.QuantumRegister(parallel_shots*nqubits, 'q')
        bits = qiskit.ClassicalRegister(parallel_shots*nqubits, 'c')
        parallel_circuit = qiskit.QuantumCircuit(qubits, bits)
        for i in range(parallel_shots):
            parallel_circuit.append(circuit, qubits[i*nqubits:i*nqubits+nqubits], bits[i*nqubits:i*nqubits+nqubits])
        circuit = parallel_circuit
    else:
        parallel_shots = 1

    if merge and isinstance(merge, int):
        pnq = parallel_shots*nqubits
        merged_circuits_noise_params = []
        qubits = qiskit.QuantumRegister(merge*pnq, 'q')
        bits = qiskit.ClassicalRegister(merge*pnq, 'c')
        merged_circuit = qiskit.QuantumCircuit(qubits, bits)
        for i in range(merge):
            noise_p = qiskit.circuit.ParameterVector(f'r[{i}]', latent_dim)
            merged_circuits_noise_params.append(noise_p)
            circ = circuit.assign_parameters(dict(zip(circuit_noise_params, noise_p)))
            merged_circuit.append(circ, qubits[i*pnq:i*pnq+pnq], bits[i*pnq:i*pnq+pnq])
        circuit = merged_circuit
        circuit_noise_params = merged_circuits_noise_params

    circuit = qiskit.transpile(circuit, backend, optimization_level=3)
    print("optimized generator circuit")
    print(circuit)
    circuit.draw(output='mpl', filename=f"generic_compiled_generator_circuit_{samples}_{nqubits}_{latent_dim}.pdf")

    if False: #provider.name == 'ionq_provider':
        x_fake, _ = generate_fake_samples_pennylane(circuit, backend, merge, circuit_noise_params, samples, batch_size, parallel, nqubits, layers, nshots, parallel_shots)
    else:
        x_fake, _ = generate_fake_samples(circuit, backend, merge, circuit_noise_params, samples, batch_size, parallel, nqubits, layers, nshots, parallel_shots)
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
    
    # saving samples to disk
    np.savetxt(f'real1_LHCttbar_{backend.name()}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{nshots}', x_real1)
    np.savetxt(f'real2_LHCttbar_{backend.name()}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{nshots}', x_real2)
    np.savetxt(f'real3_LHCttbar_{backend.name()}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{nshots}', x_real3)

    np.savetxt(f'fake1_LHCttbar_{backend.name()}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{nshots}', x_fake1)
    np.savetxt(f'fake2_LHCttbar_{backend.name()}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{nshots}', x_fake2)
    np.savetxt(f'fake3_LHCttbar_{backend.name()}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{nshots}', x_fake3)

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
    parser.add_argument("--parallel_shots", default=False, type=int)
    parser.add_argument("--provider", default="ibmq", type=str)
    parser.add_argument("--backend", default="simulator", type=str)
    parser.add_argument("--merge", default=False, type=int)
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
