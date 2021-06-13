#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:29:11 2021

@author: carlos
"""
# train a quantum-classical generative adversarial network on a gaussian probability distribution
import numpy as np
import tensorflow as tf
from numpy.random import rand
from numpy.random import randn
from keras.models import Sequential, model_from_json
from keras.optimizers import Adadelta
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Dropout, Reshape, LeakyReLU, Flatten, BatchNormalization
from qibo import gates, hamiltonians, models, set_backend
from matplotlib import pyplot
from scipy.optimize import minimize

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

def rotate(theta, x):
    theta = tf.reshape(theta, (latent_dim + 1, 2*layers*nqubits))
    return tf.matmul(x, theta[1:]) + theta[0]
 
# define the combined generator and discriminator model, for updating the generator
def define_cost_gan(params, discriminator, latent_dim, samples):
    # generate fake samples
    x_fake, y_fake = generate_fake_samples(params, latent_dim, samples)
    # create inverted labels for the fake samples
    y_fake = np.ones((samples, 1))
    # evaluate discriminator on fake examples
    disc_output = discriminator(x_fake)
    loss = tf.keras.losses.binary_crossentropy(y_fake, disc_output)
    loss = tf.reduce_mean(loss)
    return loss

def generate_5k_real_samples(samples=5000, sigma=0.25, mu=0.0):
  # generate 5k samples from the distribution
    s = []
    s1 = np.reshape(np.random.normal(mu, sigma, samples), (samples,1))
    s2 = np.reshape(np.random.normal(mu, sigma, samples), (samples,1))
    s3 = np.reshape(np.random.normal(mu, sigma, samples), (samples,1))
    s = np.hstack((s1,s2,s3))
    return s
 
# generate real samples with class labels
def generate_real_samples(samples, distribution, total_samples=5000):
    # generate samples from the distribution
    idx = np.random.randint(total_samples, size=samples)
    X = distribution[idx,:]
    # generate class labels
    y = np.ones((samples, 1))
    return X, y

# define hamiltonian to generate fake samples
def hamiltonian1():
    id = [[1, 0], [0, 1]]
    m0 = hamiltonians.Z(1, numpy=True).matrix
    m0 = np.kron(id, np.kron(id, m0))
    ham = hamiltonians.Hamiltonian(nqubits, m0)
    return ham

def hamiltonian2():
    id = [[1, 0], [0, 1]]
    m0 = hamiltonians.Z(1, numpy=True).matrix
    m0 = np.kron(id, np.kron(m0, id))
    ham = hamiltonians.Hamiltonian(nqubits, m0)
    return ham

def hamiltonian3():
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
def generate_fake_samples(params, latent_dim, samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, samples)
    # quantum generator circuit
    circuit = models.Circuit(nqubits)
    for l in range(layers):
      circuit.add((gates.RY(q, theta=0) for q in range(nqubits)))
      circuit.add((gates.RZ(q, theta=0) for q in range(nqubits)))
    # generator outputs
    X1 = []
    X2 = []
    X3 = []
    newparams = rotate(params, x_input)
    for newparams_i in newparams:
        circuit.set_parameters(newparams_i)
        X1.append(hamiltonian1().expectation(circuit.execute()))
        X2.append(hamiltonian2().expectation(circuit.execute()))
        X3.append(hamiltonian3().expectation(circuit.execute()))
    # shape array
    X = tf.stack((X1, X2, X3), axis=1)
    # create class labels
    y = np.zeros((samples, 1))
    return X, y
 
# train the generator and discriminator
def train(d_model, latent_dim, n_epochs=30000, samples=128, nbins=49, n_eval=2):
    d_loss = []
    g_loss = []
    # determine half the size of one batch, for updating the discriminator
    half_samples = int(samples / 2)
    initial_params = tf.Variable(np.random.uniform(0, 2*np.pi, (latent_dim+1)*(2*layers*nqubits)))
    optimizer = tf.optimizers.Adadelta(lr=0.1)
    # prepare real samples
    s = generate_5k_real_samples()
    # manually enumerate epochs
    for i in range(n_epochs):
        # prepare real samples
        x_real, y_real = generate_real_samples(half_samples, s)
        # prepare fake examples
        x_fake, y_fake = generate_fake_samples(initial_params, latent_dim, half_samples)
        # update discriminator
        d_loss_real, _ = d_model.train_on_batch(x_real, y_real)
        d_loss_fake, _ = d_model.train_on_batch(x_fake, y_fake)
        d_loss.append((d_loss_real + d_loss_fake)/2)
        # update generator
        with tf.GradientTape() as tape:
            loss = define_cost_gan(initial_params, d_model, latent_dim, samples)
        grads = tape.gradient(loss, initial_params)
        optimizer.apply_gradients([(grads, initial_params)])
        g_loss.append(loss)
        np.savetxt(str(nqubits)+"_qGAN_"+str(layers)+"_layers_"+str(latent_dim)+"_latent_PARAMS", [initial_params.numpy()], newline='')
        np.savetxt(str(nqubits)+"_qGAN_"+str(layers)+"_layers_"+str(latent_dim)+"_latent_dloss", [d_loss], newline='')
        np.savetxt(str(nqubits)+"_qGAN_"+str(layers)+"_layers_"+str(latent_dim)+"_latent_gloss", [g_loss], newline='')
        # serialize weights to HDF5
        discriminator.save_weights("discriminator_Quantum.h5")

# size of the latent space
latent_dim = 1
# number of layers generator
layers = 10
# number of qubits generator
nqubits = 3
# create the discriminator
discriminator = define_discriminator()
# train model
train(discriminator, latent_dim)
