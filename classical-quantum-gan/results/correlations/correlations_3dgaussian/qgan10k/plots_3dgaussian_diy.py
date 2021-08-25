#!/bin/bash


LAY='2 5 7 10'

for lay in $LAY
do
for (( lat=1; lat<=6; lat++))
do


python3.8 - << MARKER
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
    samples = 10000
    latent_dim = $lat #5
    layers = $lay #2
	
    print("Generate Nsamples={} with Nqubits={}, latent_dim={}, layers={}".format(samples,nqubits,latent_dim,layers))
	
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
    
    #nqubits = 3
    params = np.loadtxt(f"PARAMS_3Dgaussian_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}")
    
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
    
    outf1a = open(f"3dgaussian_circuit_{nqubits}_{latent_dim}_{layers}.smp", "w")
    for b in range(0,samples):
	    outf1a.write("%.14e %.14e %.14e   %.14e %.14e %.14e\n" % (x_real[b,0],x_real[b,1],x_real[b,2],x_fake[b,0],x_fake[b,1],x_fake[b,2]))	
    outf1a.close
	
   

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
	
MARKER

done
done


exit 0		
