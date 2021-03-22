#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from qibo.models import Circuit
from qibo import hamiltonians, gates, models, matrices
from qibo.hamiltonians import Hamiltonian
from scipy.optimize import minimize
import argparse


def main(layers, autoencoder, nlambda, ncompress):
    
    def encoder_hamiltonian_simple(nqubits, ncompress):
        """Creates the encoding Hamiltonian.
        Args:
            nqubits (int): total number of qubits.
            ncompress (int): number of discarded/trash qubits.
        Returns:
            Encoding Hamiltonian.
        """
        m0 = hamiltonians.Z(ncompress, numpy=True).matrix
        m1 = np.eye(2 ** (nqubits - ncompress), dtype=m0.dtype)
        ham = hamiltonians.Hamiltonian(nqubits, np.kron(m1, m0))
        return 0.5 * (ham + ncompress)
    
    # classical part of the EF-QAE: Rotate for new angles, pass them through and minimise
    def rotate(theta, x):
        new_theta = []
        index = 0
        for l in range(layers):
            for q in range(nqubits):
               new_theta.append(theta[index]*x + theta[index+1])
               index += 2
            for q in range(nqubits):
               new_theta.append(theta[index]*x + theta[index+1])
               index += 2
        for q in range(nqubits-compress, nqubits, 1):
            new_theta.append(theta[index]*x + theta[index+1])
            index += 2
        return new_theta


    # open some files
    if autoencoder == 1:
        outf_ising_in   = open("ising.QAE.ly{}.nlb{}.ncp{}.in".format(layers,nlambda,ncompress), "w")
        outf_ising_mid  = open("ising.QAE.ly{}.nlb{}.ncp{}.mid".format(layers,nlambda,ncompress), "w")
        outf_ising_out  = open("ising.QAE.ly{}.nlb{}.ncp{}.out".format(layers,nlambda,ncompress), "w")
        outf_loss_out   = open("loss.QAE.ly{}.nlb{}.ncp{}.dat".format(layers,nlambda,ncompress), "w")
    if autoencoder == 0:
        outf_ising_in   = open("ising.EF-QAE.ly{}.nlb{}.ncp{}.in".format(layers,nlambda,ncompress), "w")
        outf_ising_mid  = open("ising.EF-QAE.ly{}.nlb{}.ncp{}.mid".format(layers,nlambda,ncompress), "w")
        outf_ising_out  = open("ising.EF-QAE.ly{}.nlb{}.ncp{}.out".format(layers,nlambda,ncompress), "w")
        outf_loss_out   = open("loss.EF-QAE.ly{}.nlb{}.ncp{}.dat".format(layers,nlambda,ncompress), "w")
        

    # some variables, goal: compress 2 qubits from 6 qubit input    
    cost_function_steps = []
    nqubits = 6
    compress = ncompress
    nstop = 8000 # stop the calculation after a given set of iterations
    encoder = encoder_hamiltonian_simple(nqubits, compress) 
    count = [0]
    
    # Run the Ising model example
    ising_groundstates = []
    # just a single lambda for now
    lambdas = np.linspace(0.5, 1.0, nlambda)
    for lamb in lambdas:
        ising_ham = -1 * hamiltonians.TFIM(nqubits, h=lamb)
        ising_groundstates.append(ising_ham.eigenvectors()[0])
        #print(lamb,np.real(ising_groundstates)) -- don't quite understand how this is listed

        
    # There's a choice of two autoencoders, 1 = the QAE (supposed to be not as good), 0= EF-QAE   
    if autoencoder == 1:
        circuit = models.Circuit(nqubits)
        for l in range(layers):
            for q in range(nqubits):
                circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.CZ(5, 4))
            circuit.add(gates.CZ(5, 3))
            circuit.add(gates.CZ(5, 1))
            circuit.add(gates.CZ(4, 2))
            circuit.add(gates.CZ(4, 0))
            for q in range(nqubits):
                circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.CZ(5, 4))
            circuit.add(gates.CZ(5, 2))
            circuit.add(gates.CZ(4, 3))
            circuit.add(gates.CZ(5, 0))
            circuit.add(gates.CZ(4, 1))
        for q in range(nqubits-compress, nqubits, 1):
            circuit.add(gates.RY(q, theta=0))
            
        def cost_function_QAE_Ising(params, count):
            """Evaluates the cost function to be minimized for the QAE and Ising model.
    
            Args:
                params (array or list): values of the parameters.
    
            Returns:
                Value of the cost function.
            """                    
            cost = 0
            # The following two lines are what makes this a QAE: Quantum circuit needs to update angles
            circuit.set_parameters(params) # this will change all thetas to the appropriate values
            for i in range(len(ising_groundstates)):
                final_state = circuit.execute(np.copy(ising_groundstates[i]))
                cost += encoder.expectation(final_state).numpy().real
                
            cost_function_steps.append(cost/len(ising_groundstates)) # save cost function value after each step
    
            if count[0] % 50 == 0:
                print(count[0], cost/len(ising_groundstates))
                outf_loss_out.write("%d %.13e\n" % (count[0], cost/len(ising_groundstates)) )
            count[0] += 1
            
            # kill it after nstop iterations
            if count[0] > nstop:
                return 0
            else:
                return cost/len(ising_groundstates)
    
        nparams = 2 * nqubits * layers + compress
        initial_params = np.random.uniform(0, 2*np.pi, nparams)
    
        result = minimize(cost_function_QAE_Ising, initial_params,
                          args=(count), method='BFGS', options={'maxiter': 5.0e4}) # 5.0e4 is set in the example
                          
    # There's a choice of two autoencoders, 1 = the QAE (supposed to be not as good), 0= EF-QAE       
    elif autoencoder == 0:
        circuit = models.Circuit(nqubits)
        for l in range(layers):
            for q in range(nqubits):
                circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.CZ(5, 4))
            circuit.add(gates.CZ(5, 3))
            circuit.add(gates.CZ(5, 1))
            circuit.add(gates.CZ(4, 2))
            circuit.add(gates.CZ(4, 0))
            for q in range(nqubits):
                circuit.add(gates.RY(q, theta=0))
            circuit.add(gates.CZ(5, 4))
            circuit.add(gates.CZ(5, 2))
            circuit.add(gates.CZ(4, 3))
            circuit.add(gates.CZ(5, 0))
            circuit.add(gates.CZ(4, 1))
        for q in range(nqubits-compress, nqubits, 1):
            circuit.add(gates.RY(q, theta=0))
            
        def cost_function_EF_QAE_Ising(params, count):
            """Evaluates the cost function to be minimized for the EF-QAE and Ising model.
    
            Args:
                params (array or list): values of the parameters.
    
            Returns:
                Value of the cost function.
            """                                                               
            cost = 0
            # The following two lines are what makes this an EF-QAE: New angles are suggested via "rotate" and the circuit set externally
            for i in range(len(ising_groundstates)):
                newparams = rotate(params, lambdas[i])
                circuit.set_parameters(newparams)
                final_state = circuit.execute(np.copy(ising_groundstates[i]))
                cost += encoder.expectation(final_state).numpy().real
                
            cost_function_steps.append(cost/len(ising_groundstates)) # save cost function value after each step
    
            if count[0] % 50 == 0:
                print(count[0], cost/len(ising_groundstates))
                outf_loss_out.write("%d %.13e\n" % (count[0], cost/len(ising_groundstates)) )
            count[0] += 1

            # kill it after nstop iterations
            if count[0] > nstop:
                return 0
            else:
                return cost/len(ising_groundstates)

    
            return cost/len(ising_groundstates)
    
        
        nparams = 4 * nqubits * layers + 2 * compress
        initial_params = np.random.uniform(0, 2*np.pi, nparams)
    
        result = minimize(cost_function_EF_QAE_Ising, initial_params,
                          args=(count), method='BFGS', options={'maxiter': 5.0e4}) # 5.0e4 is set in the example
        
    else:
        raise ValueError("You have to introduce a value of 0 or 1 in the autoencoder argument.")

               
         
    print('Final parameters: ', result.x)
    print('Final cost function: ', result.fun)
    
    
    outf_ising_in.close 
    outf_ising_mid.close 
    outf_ising_out.close 
    outf_loss_out.close 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default=3, type=int, help='(int): number of ansatz layers')
    parser.add_argument("--autoencoder", default=0, type=int, help='(int): 0 to run the EF-QAE or 1 to run the QAE')
    parser.add_argument("--nlambda", default=20, type=int, help='(int): number if Ising lambda (couplings) between 0.5 and 1.0')
    parser.add_argument("--ncompress", default=2, type=int, help='(int): number if qubits to compress to, max=6')
    args = parser.parse_args()
    main(**vars(args))
