#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# train a quantum-classical generative adversarial network on LHC data
import numpy as np
from numpy.random import rand
from numpy.random import randn
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Dropout, Reshape, LeakyReLU, Flatten, BatchNormalization
from qibo import gates, hamiltonians, models, set_backend, set_threads
from main import readInit, readEvent, GetEnergySquared, GetMandelT, GetRapidity
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
import argparse

set_backend('tensorflow')
set_threads(4)

# define the standalone discriminator model
def define_discriminator(lr, n_inputs=3, alpha=0.2, dropout=0.2):
    model = Sequential()
        
    model.add(Dense(200, use_bias=False, input_dim=n_inputs))
    model.add(Reshape((10,10,2)))
    
    model.add(Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=alpha))
    
    model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=alpha))

    model.add(Conv2D(16, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=alpha))

    model.add(Conv2D(8, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_normal'))

    model.add(Flatten())
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout)) 

    model.add(Dense(1, activation='sigmoid'))
    
    # compile model
    opt = Adadelta(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
 
# define the combined generator and discriminator model, for updating the generator
def define_cost_gan(params, discriminator, latent_dim, samples, circuit, nqubits, hamiltonian1, hamiltonian2, hamiltonian3):
    # generate fake samples
    x_fake, y_fake = generate_fake_samples(params, latent_dim, samples, circuit, nqubits, hamiltonian1, hamiltonian2, hamiltonian3)
    # create inverted labels for the fake samples
    y_fake = np.ones((samples, 1))
    # evaluate discriminator on fake examples
    disc_output = discriminator(x_fake)
    loss = tf.keras.losses.binary_crossentropy(y_fake, disc_output)
    loss = tf.reduce_mean(loss)
    return loss

def set_params(circuit, params, x_input, i, nqubits):
    p = []
    index = 0
    for l in range(2):
        for q in range(nqubits):
            p.append(params[index]*x_input[q][i] + params[index+1])
            index+=2
            p.append(params[index]*x_input[q][i] + params[index+1])
            index+=2
    for q in range(nqubits):
        p.append(params[index]*x_input[q][i] + params[index+1])
        index+=2
    for l in range(2):
        for q in range(nqubits):
            p.append(params[index]*x_input[q][i] + params[index+1])
            index+=2
            p.append(params[index]*x_input[q][i] + params[index+1])
            index+=2
    for q in range(nqubits):
        p.append(params[index]*x_input[q][i] + params[index+1])
        index+=2
    for l in range(2):
        for q in range(nqubits):
            p.append(params[index]*x_input[q][i] + params[index+1])
            index+=2
            p.append(params[index]*x_input[q][i] + params[index+1])
            index+=2
    for q in range(nqubits):
        p.append(params[index]*x_input[q][i] + params[index+1])
        index+=2
    circuit.set_parameters(p) 

def load_events(filename, real_samples):
    init = readInit(filename)
    evs = list(readEvent(filename))

    invar = np.zeros((len(evs),3))
    for ev in range(len(evs)):
         invar[ev, 0] = GetEnergySquared(evs[ev])
         invar[ev, 1] = GetMandelT(evs[ev])
         invar[ev, 2] = GetRapidity(init, evs[ev])
         
    pt = PowerTransformer()
    print(pt.fit(invar[:real_samples, :]))
    print(pt.lambdas_)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    print(scaler.fit(pt.transform(invar[:real_samples, :])))
    return scaler.transform(pt.transform(invar[:real_samples, :]))
 
# generate real samples with class labels
def generate_real_samples(samples, distribution, real_samples):
    # generate samples from the distribution
    idx = np.random.randint(real_samples, size=samples)
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
def generate_fake_samples(params, latent_dim, samples, circuit, nqubits, hamiltonian1, hamiltonian2, hamiltonian3):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, samples)
    x_input = np.transpose(x_input)
    # generator outputs
    X1 = []
    X2 = []
    X3 = []
    # quantum generator circuit
    for i in range(samples):
        set_params(circuit, params, x_input, i, nqubits)
        circuit_execute = circuit.execute()
        X1.append(hamiltonian1.expectation(circuit_execute))
        X2.append(hamiltonian2.expectation(circuit_execute))
        X3.append(hamiltonian3.expectation(circuit_execute))
    # shape array
    X = tf.stack((X1, X2, X3), axis=1)
    # create class labels
    y = np.zeros((samples, 1))
    return X, y

# train the generator and discriminator
def train(d_model, latent_dim, nqubits, training_samples, discriminator, circuit, n_epochs, samples, lr, hamiltonian1, hamiltonian2, hamiltonian3):
    d_loss = []
    g_loss = []
    # determine half the size of one batch, for updating the discriminator
    half_samples = int(samples / 2)
    initial_params = tf.Variable(np.random.uniform(0, 2*np.pi, 90))
    optimizer = tf.optimizers.Adadelta(learning_rate=lr)
    # prepare real samples
    s = load_events('data/ppttbar_10k_events.lhe', training_samples)
    # manually enumerate epochs
    for i in range(n_epochs):
        # prepare real samples
        x_real, y_real = generate_real_samples(half_samples, s, training_samples)
        # prepare fake examples
        x_fake, y_fake = generate_fake_samples(initial_params, latent_dim, half_samples, circuit, nqubits, hamiltonian1, hamiltonian2, hamiltonian3)
        # update discriminator
        d_loss_real, _ = d_model.train_on_batch(x_real, y_real)
        d_loss_fake, _ = d_model.train_on_batch(x_fake, y_fake)
        d_loss.append((d_loss_real + d_loss_fake)/2)
        # update generator
        with tf.GradientTape() as tape:
            loss = define_cost_gan(initial_params, d_model, latent_dim, samples, circuit, nqubits, hamiltonian1, hamiltonian2, hamiltonian3)
        grads = tape.gradient(loss, initial_params)
        optimizer.apply_gradients([(grads, initial_params)])
        g_loss.append(loss)
        np.savetxt(str(nqubits)+"_qGAN_IBM_"+str(latent_dim)+"_latent_PARAMS", [initial_params.numpy()], newline='')
        np.savetxt(str(nqubits)+"_qGAN_IBM_"+str(latent_dim)+"_latent_dloss", [d_loss], newline='')
        np.savetxt(str(nqubits)+"_qGAN_IBM_"+str(latent_dim)+"_latent_gloss", [g_loss], newline='')
        # serialize weights to HDF5
        discriminator.save_weights(f"discriminator_Quantum_{nqubits}_qubits_IBM_{latent_dim}_latent.h5")

def main(training_samples, n_epochs, batch_samples, lr):
    
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
    # latent dimension
    latent_dim = 3
    # create hamiltonians
    hamiltonian1 = hamiltonian1()
    hamiltonian2 = hamiltonian2()
    hamiltonian3 = hamiltonian3()
    # create quantum generator
    circuit = models.Circuit(nqubits)
    for l in range(2):
        for q in range(nqubits):
            circuit.add(gates.RY(q, 0))
            circuit.add(gates.RZ(q, 0))
    for q in range(nqubits):
        circuit.add(gates.RY(q, 0))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CNOT(1, 2))
    for l in range(2):
        for q in range(nqubits):
            circuit.add(gates.RY(q, 0))
            circuit.add(gates.RZ(q, 0))
    for q in range(nqubits):
        circuit.add(gates.RY(q, 0))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.CNOT(0, 1))
    for l in range(2):
        for q in range(nqubits):
            circuit.add(gates.RY(q, 0))
            circuit.add(gates.RZ(q, 0))
    for q in range(nqubits):
        circuit.add(gates.RY(q, 0))   
    # create classical discriminator
    discriminator = define_discriminator(lr)
    # train model
    train(discriminator, latent_dim, nqubits, training_samples, discriminator, circuit, n_epochs, batch_samples, lr, hamiltonian1, hamiltonian2, hamiltonian3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_samples", default=10000, type=int)
    parser.add_argument("--n_epochs", default=30000, type=int)
    parser.add_argument("--batch_samples", default=128, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    args = vars(parser.parse_args())
    main(**args)
