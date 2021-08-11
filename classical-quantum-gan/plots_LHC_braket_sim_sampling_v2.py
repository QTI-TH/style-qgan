#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import boto3
from boto3 import Session

# Import AWS Braket packages
from braket.aws import AwsSession, AwsDevice
from braket.devices import LocalSimulator
from braket.circuits import Circuit
# Define AWS profile to be used
# see https://docs.aws.amazon.com/braket/latest/developerguide/braket-using-boto3-profiles.html
boto_sess = Session(profile_name="name_of_your_aws-cli_profile")
# Initialize an AWS Braket session with boto3 Session credentials
aws_session = AwsSession(boto_session=boto_sess)
# Instantiate any Braket QPU device with the previously initiated AwsSession
#sim_arn = 'arn:aws:braket:::device/quantum-simulator/amazon/sv1'
#device = AwsDevice(sim_arn, aws_session=aws_session)

# Define the AWS S3 folder to store results in case actual QPU is used
bucket = "amazon-braket-ionq" # the name of the bucket [update to the name of your own bucket]
prefix = "test2-ondevice" # the name of the folder in the bucket  [update to the name of your own folder]
s3_folder = (bucket, prefix)

s3 = boto3.resource('s3') # s3 system on AWS

import itertools
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from kinematics import readInit, readEvent, GetEnergySquared, GetMandelT, GetRapidity
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from scipy.stats import entropy
import argparse


#########################################

s3_paginator = boto3.client('s3').get_paginator('list_objects_v2')
def keys(bucket_name, prefix='/', delimiter='/', start_after=''):
    prefix = prefix[1:] if prefix.startswith(delimiter) else prefix
    start_after = (start_after or prefix) if prefix.endswith(delimiter) else start_after
    for page in s3_paginator.paginate(Bucket=bucket_name, Prefix=prefix, StartAfter=start_after):
        for content in page.get('Contents', ()):
            yield content['Key']


# create quantum generator with AWS Braket. It outputs a given circuit for given parameters specified
# by params (trained hyperparameters) and x_input (random noise)
def create_circuit(params, x_input, i, nqubits, layers, latent_dim):
    circuit = Circuit()
    p = []
    index = 0
    noise = 0
    k = 0
    for l in range(layers):
        for q in range(nqubits):
            p.append(params[index]*x_input[noise][i] + params[index+1])
            circuit = circuit.ry(q, p[k])
            k+=1
            index+=2
            noise=(noise+1)%latent_dim
            p.append(params[index]*x_input[noise][i] + params[index+1])
            circuit = circuit.rz(q, p[k])
            k+=1
            index+=2
            noise=(noise+1)%latent_dim
        if l==1 or l==5 or l==9 or l==13 or l==17:
            p.append(params[index]*x_input[noise][i] + params[index+1])
            circuit = circuit.ry(1, p[k]/2.0)
            circuit = circuit.cnot(control=0, target=1)
            circuit = circuit.ry(1, -p[k]/2.0)
            circuit = circuit.cnot(control=0, target=1)
            k+=1                
            index+=2
            noise=(noise+1)%latent_dim
            p.append(params[index]*x_input[noise][i] + params[index+1])
            circuit = circuit.ry(2, p[k]/2.0)
            circuit = circuit.cnot(control=0, target=2)
            circuit = circuit.ry(2, -p[k]/2.0)
            circuit = circuit.cnot(control=0, target=2)
            k+=1 
            index+=2
            noise=(noise+1)%latent_dim
        if l==3 or l==7 or l==11 or l==15 or l==19:
            p.append(params[index]*x_input[noise][i] + params[index+1])
            circuit = circuit.ry(2, p[k]/2.0)
            circuit = circuit.cnot(control=1, target=2)
            circuit = circuit.ry(2, -p[k]/2.0)
            circuit = circuit.cnot(control=1, target=2)
            k+=1
            index+=2
            noise=(noise+1)%latent_dim
            p.append(params[index]*x_input[noise][i] + params[index+1])
            circuit = circuit.ry(0, p[k]/2.0)
            circuit = circuit.cnot(control=2, target=0)
            circuit = circuit.ry(0, -p[k]/2.0)
            circuit = circuit.cnot(control=2, target=0)
            k+=1
            index+=2
            noise=(noise+1)%latent_dim
    for q in range(nqubits):
        p.append(params[index]*x_input[noise][i] + params[index+1])
        index+=2
        noise=(noise+1)%latent_dim
        circuit = circuit.ry(q, p[k])
        k+=1
    return circuit

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, samples):
    # generate points in the latent space
    x_input = randn(latent_dim * samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(samples, latent_dim)
    return x_input

# use the generator to generate fake examples, with class labels
def generate_fake_samples(backend, params, latent_dim, samples, batch_size, nqubits, layers, nshots):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, samples)
    x_input = np.transpose(x_input) # not in Marco's qiskit version, why?

    if backend=='ionQ':
        device=AwsDevice("arn:aws:braket:::device/qpu/ionq/ionQdevice", aws_session = aws_session)

    if backend=='simulator':
        device=LocalSimulator()
    
    def submit_job(start, stop):
        batch = []
        for i in range(start, stop):
            circparam = create_circuit(params, x_input, i, nqubits, layers, latent_dim)
            batch.append(circparam)
        if start == 0:
#            print("compiled circuit, noise id 0")
            print("Circuit for first element of batch, noise id 0")
            print(batch[0])
        job = device.run_batch(batch, s3_folder, shots=nshots)
        print(f'job for batch from {start} to {stop-1} submitted', flush=True)
        return job

    # run the simulation in batches
    if(backend=='ionQ'):
        jobs = [submit_job(i, min(i+batch_size, samples)) for i in range(0, samples, batch_size)]
        results = []

        indexkey=0
        for resultsfiles in keys(bucket,prefix=prefix):
            if(indexkey>0): # the first key is the name of the "directory" on S3; skip it
                fileobject = s3.Object(bucket, resultsfiles)
                inputdata=fileobject.get()['Body'].read().decode('utf-8')
                awsdata = json.loads(inputdata)
                results.append(awsdata["measurementProbabilities"])
            indexkey += 1

        # generator outputs
        X1 = []
        X2 = []
        X3 = []
        # quantum generator circuit
        for c in results:
            X1.append(-c.get('000', 0)+c.get('001', 0)-c.get('010', 0)+c.get('011', 0)-c.get('100', 0)+c.get('101', 0)-c.get('110', 0)+c.get('111', 0))
            X2.append(-c.get('000', 0)-c.get('001', 0)+c.get('010', 0)+c.get('011', 0)-c.get('100', 0)-c.get('101', 0)+c.get('110', 0)+c.get('111', 0))
            X3.append(-c.get('000', 0)-c.get('001', 0)-c.get('010', 0)-c.get('011', 0)+c.get('100', 0)+c.get('101', 0)+c.get('110', 0)+c.get('111', 0))
    else:
        # generator outputs
        X1 = []
        X2 = []
        X3 = []
        for i in range(samples):
            circparam = create_circuit(params, x_input, i, nqubits, layers, latent_dim)
            if i==0:
                print("Circuit for first sample, noise id 0")
                print(circparam)
            circuit_execute = device.run(circparam, shots=nshots).result().measurement_counts            
            X1.append((-circuit_execute['000']+circuit_execute['001']-circuit_execute['010']+circuit_execute['011']-circuit_execute['100']+circuit_execute['101']-circuit_execute['110']+circuit_execute['111'])/nshots)
            X2.append((-circuit_execute['000']-circuit_execute['001']+circuit_execute['010']+circuit_execute['011']-circuit_execute['100']-circuit_execute['101']+circuit_execute['110']+circuit_execute['111'])/nshots)
            X3.append((-circuit_execute['000']-circuit_execute['001']-circuit_execute['010']-circuit_execute['011']+circuit_execute['100']+circuit_execute['101']+circuit_execute['110']+circuit_execute['111'])/nshots)

    # shape array
    X = np.stack((X1, X2, X3), axis=1)
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


def main(samples, bins, latent_dim, layers, training_samples, batch_samples, lr, plot_real, nshots, backend):
    
    # number of qubits generator
    nqubits = 3

    params = np.loadtxt(f"PARAMS_LHCdata_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}")

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
    batch_size = 1000

    if backend == 'ionQ':
#        batch_size = 900 # it should be possible to get this programmatically
        batch_size = 100 # it should be possible to get this programmatically

    x_fake, _ = generate_fake_samples(backend, params, latent_dim, samples, batch_size, nqubits, layers, nshots)

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
    outf = open("./out_aws_generator_samples.txt", "w")
    outf.write("# s\t t\t y\n")
    for i in range(samples):
        x_fake1.append(x_fake[i][0])
        x_fake2.append(x_fake[i][1])
        x_fake3.append(x_fake[i][2])
        outf.write("%.7e %.7e %.7e\n" % ( x_fake1[i],x_fake2[i],x_fake3[i] ))

    outf.close()

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
#    plt.show()
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
#    plt.show()
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
#    plt.show()
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
#        plt.show()
        fig.savefig(f"s-t_REAL_{samples}_{bins}.pdf", bbox_inches='tight')
        
        fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
        plt.ylabel('', fontsize=17)
        plt.xlabel('', fontsize=17)
        plt.hist2d(-1*x_real2, x_real3, [-1*np.logspace(np.log10(max(x_real2)),np.log10(min(x_real2)),bins),np.linspace(min(x_real3),max(x_real3),bins)], color='red', label='real', alpha=1.0, linewidths=0)
        plt.colorbar()
        plt.rcParams["axes.linewidth"]  = 1.25
        plt.xscale('symlog')
#        plt.show()
        fig.savefig(f"t-y_REAL_{samples}_{bins}.pdf", bbox_inches='tight')
        
        fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
        plt.ylabel('', fontsize=17)
        plt.xlabel('', fontsize=17)
        plt.hist2d(x_real3, x_real1, [np.linspace(min(x_real3),max(x_real3),bins),np.logspace(np.log10(min(x_real1)),np.log10(max(x_real1)),bins)], color='red', label='real', alpha=1.0, linewidths=0)
        plt.colorbar()
        plt.rcParams["axes.linewidth"]  = 1.25
        plt.yscale('log')
#        plt.show()
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
#    plt.show()
    fig.savefig(f"s-t_FAKE_{samples}_{bins}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}_{nshots}.pdf", bbox_inches='tight')
    
    H, xedges, yedges = np.histogram2d(-1*x_real2, x_real3, [-1*np.logspace(np.log10(max(x_real2)),np.log10(min(x_real2)),bins),np.linspace(min(x_real3),max(x_real3),bins)])    
    fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
    plt.ylabel('', fontsize=17)
    plt.xlabel('', fontsize=17)
    plt.hist2d(-1*x_fake2, x_fake3, [-1*np.logspace(np.log10(max(x_real2)),np.log10(min(x_real2)),bins),np.linspace(min(x_real3),max(x_real3),bins)], color='red', vmax=max(H.flatten()), label='real', alpha=1.0, linewidths=0)
    plt.colorbar()
    plt.rcParams["axes.linewidth"]  = 1.25
    plt.xscale('symlog')
#    plt.show()
    fig.savefig(f"t-y_FAKE_{samples}_{bins}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}_{nshots}.pdf", bbox_inches='tight')
    
    H, xedges, yedges = np.histogram2d(x_real3, x_real1, [np.linspace(min(x_real3),max(x_real3),bins),np.logspace(np.log10(min(x_real1)),np.log10(max(x_real1)),bins)])    
    fig = plt.figure(figsize=[10.5/1.5, 7.5/1.5])
    plt.ylabel('', fontsize=17)
    plt.xlabel('', fontsize=17)
    plt.hist2d(x_fake3, x_fake1, [np.linspace(min(x_real3),max(x_real3),bins),np.logspace(np.log10(min(x_real1)),np.log10(max(x_real1)),bins)], color='red', vmax=max(H.flatten()), label='real', alpha=1.0, linewidths=0)
    plt.colorbar()
    plt.rcParams["axes.linewidth"]  = 1.25
    plt.yscale('log')
#    plt.show()
    fig.savefig(f"y-s_FAKE_{samples}_{bins}_{nqubits}_{latent_dim}_{layers}_{training_samples}_{batch_samples}_{lr}_{nshots}.pdf", bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=1000, type=int)
    parser.add_argument("--nshots", default=1000, type=int)
    parser.add_argument("--backend", default="simulator", type=str)
    parser.add_argument("--bins", default=100, type=int)
    parser.add_argument("--latent_dim", default=5, type=int)
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--training_samples", default=10000, type=int)
    parser.add_argument("--batch_samples", default=128, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--plot_real", default=False, type=bool)
    args = vars(parser.parse_args())
    main(**args)
