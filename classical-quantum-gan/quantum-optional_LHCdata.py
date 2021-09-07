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

#set_backend('tensorflow')
set_backend('matmuleinsum')

# define the classical discriminator and gan cost
# -----------------------------------------------

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

# define the combined generator and discriminator model, for updating the generator in the quantum-classical configuration
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
    
    
# define the quantum discriminator and gan cost
# ---------------------------------------------

# discriminator parameter rotation
def drotate(theta, dlatent_dim, dlayers, x):
    theta = tf.reshape(theta, (dlatent_dim + 1, 2*dlayers)) # factor three because there are 3 datasets
    return tf.matmul(x, theta[1:]) + theta[0]
        
# Fermi-Dirac transformation for circuit output, 
# Disriminator decides between 0 and 1, 0.5 is the center, 12.=steepness (empirically set, to do: make option)
def FermiDirac(x):
    return 1/(tf.math.exp((x-0.5)*12.) +1)     
    
# define discriminator hamiltonian (takes role of fidelity in usual re-uploading)   
def fidelity_ham(state):
    m0 = hamiltonians.Z(1, numpy=True).matrix
    ham = hamiltonians.Hamiltonian(1, m0)    
    num = ham.expectation(state, normalize=True)   
    res = FermiDirac((num +1)/2)
    return res    
    
# discriminator cost function, needs both real and fake data, discriminator always has 1 qubit                  
def discriminator_cost(params, dlatent_dim, dlayers, x_real, x_fake):
    # quantum discriminator circuit
    circuit = models.Circuit(1)
    for l in range(dlayers):
      circuit.add((gates.RY(0, theta=0) ))
      circuit.add((gates.RZ(0, theta=0) ))
    # set parameters and outputs
    tots=int(tf.size(x_real))
    cf1=0
    cf2=0
    # need to flatten the real data array
    x_real_flat = tf.reshape(x_real,[tots,1])    
    # real data labels
    y_real=np.ones(tots) 
    newparams = drotate(params, dlatent_dim, dlayers, x_real_flat)
    for newparams_i in newparams:
        circuit.set_parameters(newparams_i)
        state = circuit.execute()
        cf1 += tf.keras.losses.binary_crossentropy(fidelity_ham(state),y_real ) 
    # need to flatten the fake data array
    x_fake_flat = tf.reshape(x_fake,[tots,1])        
    # fake data labels    
    y_fake=np.zeros(tots)  
    newparams = drotate(params, dlatent_dim, dlayers, x_fake_flat)
    for newparams_i in newparams:
        circuit.set_parameters(newparams_i)
        state = circuit.execute()
        cf2 += tf.keras.losses.binary_crossentropy(fidelity_ham(state),y_fake )
    cf=0.25*(cf1+cf2)/tots        
    tflabel = tf.convert_to_tensor(cf, dtype=tf.float64)
    cf=(tflabel)
    return cf    
    
# get labels by evaluating discriminator, they will be between 0 and 1    
# In case a definite answer is desired a simple trick is to run sth like
#    y_guess = run_discriminator(dparams,x_fake)
#    for j in range(0,samples):
#      y_tmp=tf.cast( (y_guess[j]+0.5), dtype=tf.int64 )
def run_discriminator(params, dlatent_dim, dlayers, x_val):              
    # quantum discriminator circuit
    circuit = models.Circuit(1)
    for l in range(dlayers):
      circuit.add((gates.RY(0, theta=0) ))
      circuit.add((gates.RZ(0, theta=0) ))   
    # set parameters and outputs
    labels = []
    # need to flatten the data array
    x_val_flat = tf.reshape(x_val,[tf.size(x_val),1])
    newparams = drotate(params, dlatent_dim, dlayers, x_val_flat)
    for newparams_i in newparams:
        circuit.set_parameters(newparams_i)
        state = circuit.execute()
        label_prob=fidelity_ham(state)
        labels.append(label_prob)          
    # return labels    
    labels = tf.stack(labels)[:, tf.newaxis]
    return labels    
    
# define the combined generator and discriminator model, for updating the generator in the quantum-quantum configuration
def define_cost_qq_gan(params, dparams, latent_dim, dlatent_dim, samples, layers, dlayers, nqubits):
    # generate fake samples
    x_fake, y_fake = generate_fake_samples(params, latent_dim, samples, layers, nqubits)
    # create inverted labels for the fake samples, there have to be as many as the flattened data is long!
    #y_fake = np.ones((samples, 1))
    y_fake = np.ones((samples*3, 1)) # hard coded 3D data
    # evaluate discriminator on fake examples
    disc_output = run_discriminator(dparams, dlatent_dim, dlayers, x_fake)
    # output loss
    loss = tf.keras.losses.binary_crossentropy(y_fake, disc_output)
    loss = tf.reduce_mean(loss)
    return loss
     

# common in both configurations   
# -----------------------------

# load LHC event data
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

# define rotation of parameters for generator 
def rotate(theta, x, latent_dim, layers, nqubits):
    theta = tf.reshape(theta, (latent_dim + 1, 4*layers*nqubits + nqubits + 4*layers))
    return tf.matmul(x, theta[1:]) + theta[0]
 
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
 
 
# separate training routines
# --------------------------
 
# train the generator and discriminator in the quantum-classical configuration
def train_qc(d_model, latent_dim, layers, nqubits, real_samples, discriminator, n_epochs=30000, samples=128):
    d_loss = []
    g_loss = []
    # determine half the size of one batch, for updating the discriminator
    half_samples = int(samples / 2)
    initial_params = tf.Variable(np.random.uniform(0, 2*np.pi, (latent_dim+1)*(4*layers*nqubits + nqubits + 4*layers)))
    optimizer = tf.optimizers.Adadelta(lr=0.1)
    # prepare real samples
    s = load_events('./../data/ppttbar_10k_events.lhe', real_samples)
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
        
# train the generator and discriminator in the quantum-quantum configuration, prefix "d" indicates discriminator
def train_qq(latent_dim, layers, dlayers, nqubits, real_samples, n_epochs=30000, samples=128):
    d_loss = []
    g_loss = []
    # determine half the size of one batch, for updating the discriminator
    half_samples = int(samples / 2)
    # initial parameters for generator
    initial_params = tf.Variable(np.random.uniform(0, 2*np.pi, (latent_dim+1)*(4*layers*nqubits + nqubits + 4*layers)))
    # initial parameters for discriminator, latent_dim=1 set here (to do: make option), nqubits=1 hard coded
    dlatent_dim=1
    initial_dparams = tf.Variable(np.random.uniform(0, 2*np.pi, (dlatent_dim+1)*(2*dlayers*1)))    
    # set optimizers
    optimizer = tf.optimizers.Adadelta(lr=0.1)
    doptimizer = tf.optimizers.SGD(lr=0.1) 
    # prepare real samples
    s = load_events('./../data/ppttbar_10k_events.lhe', real_samples)
    # manually enumerate epochs
    for i in range(n_epochs):
        # prepare real samples
        x_real, y_real = generate_real_samples(half_samples, s, real_samples)
        # prepare fake examples
        x_fake, y_fake = generate_fake_samples(initial_params, latent_dim, half_samples, layers, nqubits)
        # update discriminator
        with tf.GradientTape() as tape:
            dloss = discriminator_cost(initial_dparams, dlatent_dim, dlayers, x_real, x_fake)
        dgrads = tape.gradient(dloss, initial_dparams)
        doptimizer.apply_gradients([(dgrads, initial_dparams)])
        d_loss.append(dloss)
        # update generator
        with tf.GradientTape() as tape:
            loss = define_cost_qq_gan(initial_params, initial_dparams, dlatent_dim, latent_dim, samples, layers, dlayers, nqubits)
        grads = tape.gradient(loss, initial_params)
        optimizer.apply_gradients([(grads, initial_params)])
        g_loss.append(loss)
        np.savetxt(str(nqubits)+"_qGAN_"+str(layers)+"_layers_"+str(latent_dim)+"_latent_PARAMS", [initial_params.numpy()], newline='')
        np.savetxt(str(nqubits)+"_qGAN_"+str(layers)+"_layers_"+str(latent_dim)+"_latent_DPARAMS", [initial_dparams.numpy()], newline='')
        np.savetxt(str(nqubits)+"_qGAN_"+str(layers)+"_layers_"+str(latent_dim)+"_latent_dloss", [d_loss], newline='')
        np.savetxt(str(nqubits)+"_qGAN_"+str(layers)+"_layers_"+str(latent_dim)+"_latent_gloss", [g_loss], newline='')
 
def main(nqubits, latent_dim, layers, real_samples, discriminator_type, discriminator_layers): 
    
    if discriminator_type == 'c':
        print('# Running quantum-classical GAN')  
        print('# --nqubits=', nqubits) 
        print('# --latent_dim=', latent_dim)
        print('# --layers=', layers)
        print('# --real_samples=', real_samples)
        # create the discriminator
        discriminator = define_discriminator()
        # train model
        train_qc(discriminator, latent_dim, layers, nqubits, real_samples, discriminator)
    elif discriminator_type == 'q':
        print('# Running quantum-quantum GAN')  
        print('# --nqubits=', nqubits) 
        print('# --latent_dim=', latent_dim)
        print('# --layers=', layers)
        print('# --real_samples=', real_samples)
        print('# --discriminator_layers=', discriminator_layers)
        # train model
        train_qq(latent_dim, layers, discriminator_layers, nqubits, real_samples)
    else:
        print('# Invalid discriminator_type. Choose c=classical or q=quantum') 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", default=3, type=int)
    parser.add_argument("--latent_dim", default=1, type=int)
    parser.add_argument("--layers", default=5, type=int)
    parser.add_argument("--real_samples", default=10000, type=int)
    parser.add_argument("--discriminator_type", default='c', type=str)
    parser.add_argument("--discriminator_layers", default=5, type=int)
    args = vars(parser.parse_args())
    main(**args)
