#!/usr/bin/env python3
import numpy as np
import read_event as lhe
import kinematics as kin
import matplotlib.pyplot as plt
import csv


def load_events(filename):
    init = lhe.readInit(filename)
    evs  = list(lhe.readEvent(filename))

    invar = np.zeros((len(evs),3))

    for ev in range(len(evs)):
         invar[ev, 0] = kin.GetEnergySquared(evs[ev])
         invar[ev, 1] = kin.GetMandelT(evs[ev])
         invar[ev, 2] = kin.GetRapidity(init, evs[ev])
    return invar

def rescale1(x):
    xmin = min(x)
    xmax = max(x)
    print("min = ",xmin," and max = ",xmax)
    a = 1/(xmax-xmin)
    b = (0.5*xmax-1.5*xmin)/(xmax-xmin)
    tempx = a*x + b
    tempx = np.exp(-25*tempx)
    tempxmin = min(tempx)
    tempxmax = max(tempx)
    a = 2/(tempxmax-tempxmin)
    b = -(tempxmax+tempxmin)/(tempxmax-tempxmin)
    return a*tempx + b

def rescale2(x):
    xmin = min(x)
    xmax = max(x)
    print("min = ",xmin," and max = ",xmax)
    a = 1/(xmax-xmin)
    b = (0.5*xmax-1.5*xmin)/(xmax-xmin)
    tempx = a*x + b
    tempx = np.exp(25*tempx)
    tempxmin = min(tempx)
    tempxmax = max(tempx)
    a = 2/(tempxmax-tempxmin)
    b = -(tempxmax+tempxmin)/(tempxmax-tempxmin)
    return a*tempx + b

def rescale3(x):
    xmin = min(x)
    xmax = max(x)
    print("min = ",xmin," and max = ",xmax)
    a = 2/(xmax-xmin)
    b = -(xmax+xmin)/(xmax-xmin)
    return a*x + b

def main():
    invar = load_events('./data/ppttbar_10k_events.lhe')
#    invar = load_events('./data/ppWW_10k_unweighted_events.lhe')

    # Plot
#    bins = [np.arange(0, 1e6, 1e4), np.arange(-1e6, 0, 1e4), np.arange(-3.5, 3.5, 0.1)]
    bins = [np.arange(-1., 1., 0.05), np.arange(-1., 1., 0.05), np.arange(-1., 1., 0.05)]
    labels = ['s', 't', 'y']
    fig, ax = plt.subplots(1,3, figsize=(10,4))

    print("Number of events: ",len(invar[:, 0]))

    invar[:, 0] = rescale1(invar[:, 0])
    invar[:, 1] = rescale2(invar[:, 1])
    invar[:, 2] = rescale3(invar[:, 2])

    for i in range(3):
        ax[i].hist(invar[:, i], bins=bins[i], histtype='step', label='MadGraph', density=True)
        ax[i].set_title(labels[i])
        ax[i].set_xlabel(labels[i])
        ax[i].legend()
    fig.tight_layout()
#    plt.show()
    fig.savefig('data.png')
    plt.close(fig)
    with open('rescaled_data.txt', 'w') as dumped:
        dumped.write("# s\t t\t y\n")
        csv.writer(dumped, delimiter='\t').writerows(invar)


if __name__ == '__main__':
    main()


