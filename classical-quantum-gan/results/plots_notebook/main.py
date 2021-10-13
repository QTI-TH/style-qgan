#!/usr/bin/env python
import xml
import xml.etree.ElementTree
import numpy as np
import matplotlib.pyplot as plt


class Event():
    def __init__(self, particles):
        self.particles = particles
        for p in particles:
            p.event = self


class Particle():
    def __init__(self, val_labels):
        for key, val in val_labels.items():
            setattr(self, key, val)
        self.p = [self.e, np.sqrt(self.px**2 + self.py**2), self.pz]


class InitInfo():
    def __init__(self, init):
        for key, val in init.items():
            setattr(self, key, val)


def readEvent(file):
    labels = ['id', 'status', 'mother1', 'mother2', 'color1', 'color2', 'px', 'py', 'pz', 'e', 'm', 'time', 'spin']
    for _, block in xml.etree.ElementTree.iterparse(file):
        if block.tag == 'event':
            data = block.text.split('\n')[1:-1]
            val = {}
            part = []
            for p in data[1:]:
                line = [float(i) for i in p.split()]
                ind = np.arange(len(labels))
                for lin, i in zip(line,ind):
                    val[labels[i]] = lin
                part.append(Particle(val))
                val = {}
            yield Event(part)


def readInit(file):
    d_init = {}
    for _, block in xml.etree.ElementTree.iterparse(file):
        if block.tag == 'init':
            d_init['beamA'] = float(block.text.split()[2])
            d_init['beamB'] = float(block.text.split()[3])
    init = InitInfo(d_init)
    return init


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


def load_events(filename):
    init = readInit(filename)
    evs = list(readEvent(filename))

    invar = np.zeros((len(evs),3))
    for ev in range(len(evs)):
         invar[ev, 0] = GetEnergySquared(evs[ev])
         invar[ev, 1] = GetMandelT(evs[ev])
         invar[ev, 2] = GetRapidity(init, evs[ev])
    return invar


def main():
    invar = load_events('data/ppttbar_10k_events.lhe')

    # debug plot
    bins = [np.arange(0, 1e6, 1e4), np.arange(-1e6, 0, 1e4), np.arange(-3.5, 3.5, 0.1)]
    labels = ['s', 't', 'y']
    fig, ax = plt.subplots(1,3, figsize=(10,4))
    for i in range(3):
        ax[i].hist(invar[:, i], bins=bins[i], histtype='step', label='MadGraph', density=True)
        ax[i].set_title(labels[i])
        ax[i].set_xlabel(labels[i])
        ax[i].legend()
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
