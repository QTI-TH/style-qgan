#!/usr/bin/env python3
import numpy as np

def scalarProd(evp1, evp2):
    prod = evp1[0]*evp2[0] - np.dot(evp1[1:],evp2[1:])
    return prod


def GetEnergySquared(event):
    try:
        p1 = event.particles[0].p
        p2 = event.particles[1].p
        return 2*scalarProd(p1,p2)
    except (AttributeError, TypeError):
        raise AssertionError("Input file is not an Event")


def GetMandelT(event):
    try:
        p1 = event.particles[0].p
        p3 = event.particles[2].p
        return -2*scalarProd(p1,p3)
    except (AttributeError, TypeError):
        raise AssertionError("Input file is not an Event")


def GetRapidity(init, event):
    try:
        p1 = np.sqrt(np.sum(np.array(event.particles[0].p)**2))
        p2 = np.sqrt(np.sum(np.array(event.particles[1].p)**2))
        x1 = p1/init.beamA
        x2 = p2/init.beamB
        return np.log(x1/x2)/2
    except (AttributeError, TypeError):
        raise AssertionError("Input file is not an Event")


def GetTransverseMomentum(event):
    try:
        p1 = event.particles[2].p
        return np.sqrt(p1[1]**2 + p1[2]**2)
    except (AttributeError, TypeError):
        raise AssertionError("Input file is not an Event")
