#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# train a quantum-classical generative adversarial network on a gaussian probability distribution
import numpy as np
from numpy.random import randn
from keras.models import Sequential
from keras.optimizers import Adadelta
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Dropout, Reshape, LeakyReLU, Flatten, BatchNormalization
from qibo import gates, hamiltonians, models, set_backend
from main import readInit, readEvent, GetEnergySquared, GetMandelT, GetRapidity
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
import argparse

set_backend('tensorflow')

# define the standalone discriminator model
def define_discriminator(n_inputs=3, alpha=0.2, dropout=0.2):
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
    opt = Adadelta(learning_rate=0.1)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def rotate(theta, x, latent_dim, layers, nqubits):
    theta = tf.reshape(theta, (latent_dim + 1, 4*layers*nqubits + nqubits + 4*layers))
    return tf.matmul(x, theta[1:]) + theta[0]
 
# define the combined generator and discriminator model, for updating the generator
def define_cost_gan(params, discriminator, latent_dim, samples, layers, nqubits):
    # generate fake samples
    x_fake, y_fake = generate_fake_samples(params, latent_dim, samples, layers, nqubits)
    # create inverted labels for the fake samples
    y_fake = np.ones((samples, 1))
    # evaluate discriminator on fake examples
    disc_output = discriminator(x_fake)
    loss = tf.keras.losses.binary_crossentropy(y_fake, disc_output)
    loss = tf.reduce_mean(loss)
    return loss

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

# define hamiltonian to generate fake samples
def hamiltonian1(nqubits):
    id = [[1, 0], [0, 1]]
    m0 = hamiltonians.Z(1, numpy=True).matrix
    m0 = np.kron(id, np.kron(id, m0))
    ham = hamiltonians.Hamiltonian(nqubits, m0)
    return ham

def hamiltonian2(nqubits):
    id = [[1, 0], [0, 1]]
    m0 = hamiltonians.Z(1, numpy=True).matrix
    m0 = np.kron(id, np.kron(m0, id))
    ham = hamiltonians.Hamiltonian(nqubits, m0)
    return ham

def hamiltonian3(nqubits):
    id = [[1, 0], [0, 1]]
    m0 = hamiltonians.Z(1, numpy=True).matrix
    m0 = np.kron(m0, np.kron(id, id))
    ham = hamiltonians.Hamiltonian(nqubits, m0)
    return ham
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, samples):
    # generate points in the latent space
    x_input = randn(latent_dim * samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(samples, latent_dim)
    return x_input
 
# use the generator to generate fake examples, with class labels
def generate_fake_samples(params, latent_dim, samples, layers, nqubits):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, samples)
    # quantum generator circuit
    circuit = models.Circuit(nqubits)
    for l in range(layers):
      circuit.add((gates.RY(q, theta=0) for q in range(nqubits)))
      circuit.add((gates.RZ(q, theta=0) for q in range(nqubits)))
      circuit.add(gates.CRY(0, 1, theta=0))
      circuit.add(gates.CRY(2, 0, theta=0))
      circuit.add((gates.RY(q, theta=0) for q in range(nqubits)))
      circuit.add((gates.RZ(q, theta=0) for q in range(nqubits)))
      circuit.add(gates.CRY(1, 2, theta=0))
      circuit.add(gates.CRY(2, 0, theta=0))
    circuit.add((gates.RY(q, theta=0) for q in range(nqubits))) 
    # generator outputs
    X1 = []
    X2 = []
    X3 = []
    newparams = rotate(params, x_input, latent_dim, layers, nqubits)
    for newparams_i in newparams:
        circuit.set_parameters(newparams_i)
        X1.append(hamiltonian1(nqubits).expectation(circuit.execute()))
        X2.append(hamiltonian2(nqubits).expectation(circuit.execute()))
        X3.append(hamiltonian3(nqubits).expectation(circuit.execute()))
    # shape array
    X = tf.stack((X1, X2, X3), axis=1)
    # create class labels
    y = np.zeros((samples, 1))
    return X, y
 
# train the generator and discriminator
def train(d_model, latent_dim, layers, nqubits, real_samples, discriminator, n_epochs=30000, samples=128):
    d_loss = []
    g_loss = []
    # determine half the size of one batch, for updating the discriminator
    half_samples = int(samples / 2)
    initial_params = tf.Variable(np.random.uniform(0, 2*np.pi, (latent_dim+1)*(4*layers*nqubits + nqubits + 4*layers)))
    optimizer = tf.optimizers.Adadelta(lr=0.1)
    # prepare real samples
    s = load_events('data/ppttbar_10k_events.lhe', real_samples)
    # manually enumerate epochs
    for i in range(n_epochs):
        # prepare real samples
        x_real, y_real = generate_real_samples(half_samples, s, real_samples)
        # prepare fake examples
        x_fake, y_fake = generate_fake_samples(initial_params, latent_dim, half_samples, layers, nqubits)
        # update discriminator
        d_loss_real, _ = d_model.train_on_batch(x_real, y_real)
        d_loss_fake, _ = d_model.train_on_batch(x_fake, y_fake)
        d_loss.append((d_loss_real + d_loss_fake)/2)
        # update generator
        with tf.GradientTape() as tape:
            loss = define_cost_gan(initial_params, d_model, latent_dim, samples, layers, nqubits)
        grads = tape.gradient(loss, initial_params)
        optimizer.apply_gradients([(grads, initial_params)])
        g_loss.append(loss)
        np.savetxt(str(nqubits)+"_qGAN_"+str(layers)+"_layers_"+str(latent_dim)+"_latent_PARAMS", [initial_params.numpy()], newline='')
        np.savetxt(str(nqubits)+"_qGAN_"+str(layers)+"_layers_"+str(latent_dim)+"_latent_dloss", [d_loss], newline='')
        np.savetxt(str(nqubits)+"_qGAN_"+str(layers)+"_layers_"+str(latent_dim)+"_latent_gloss", [g_loss], newline='')
        # serialize weights to HDF5
        discriminator.save_weights("discriminator_Quantum.h5")

def main(nqubits, latent_dim, layers, real_samples):        
    # create the discriminator
    discriminator = define_discriminator()
    # train model
    train(discriminator, latent_dim, layers, nqubits, real_samples, discriminator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", default=3, type=int)
    parser.add_argument("--latent_dim", default=1, type=int)
    parser.add_argument("--layers", default=5, type=int)
    parser.add_argument("--real_samples", default=10000, type=int)
    args = vars(parser.parse_args())
    main(**args)
