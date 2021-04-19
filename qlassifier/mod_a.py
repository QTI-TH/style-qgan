# /usr/bin/env python
import datasets_a as ds
import numpy as np
from qlassifier_a import single_qubit_classifier


# Some parameters

layers=1

# output

# Code, based on qibo qlassifier tutorial

"""Perform classification for a given problem and number of layers.
"""
ql = single_qubit_classifier("gauss",layers)  # Define classifier


print('Problem never solved, finding optimal parameters...')
result, parameters = ql.minimize(method='l-bfgs-b', options={'disp': True})
 
ql.set_parameters(parameters)
value_loss = ql.cost_function_fidelity()
print('The value of the cost function achieved is %.6f' % value_loss.numpy())


