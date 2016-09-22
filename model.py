#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import emcee
from astropy.io import ascii
import matplotlib.pyplot as plt
from astropy.table import Table, Column
import math
import scipy as sp
import sys
from PyAstronomy import pyasl
from PyAstronomy.pyasl import foldAt
import scipy.optimize as op
import corner
import time as chrono
from sympy.solvers import solve
from sympy import Symbol
try:
    from tqdm import tqdm
except ImportError:
    raise ImportError('You don t have the package tqdm installed. Try pip install tqdm.')


#################
# Take the Data #
#################

ks = pyasl.MarkleyKESolver()

GJ180 = 'datafiles/GJ180_weight_022_wave_4404.rv'
GJ180_PFS = 'datafiles/GJ180_PFS.dat'
GJ180_UVES = 'datafiles/GJ180_UVES.dat'
# GJ180_KECK = 'HIP22762_KECK.vels'

PEG51_1 = 'datafiles/51Peg_1_LICK.vels'
PEG51_2 = 'datafiles/51Peg_2_ELODIE.vels'

UPSAND_1 = 'datafiles/upsAnd_1_LICK.vels'
UPSAND_2 = 'datafiles/upsAnd_2_ELODIE.vels'
UPSAND_3 = 'datafiles/upsAnd_3_HJS.vels'


# data para gj180
dataset_1 = np.loadtxt(GJ180, usecols=(0,1,2))
dataset_2 = np.loadtxt(GJ180_PFS)
dataset_3 = np.loadtxt(GJ180_UVES)
# dataset_4 = np.loadtxt(GJ180_KECK, usecols=(0,1,2))

# data para 51peg
dataset_5 = np.loadtxt(PEG51_1)
dataset_6 = np.loadtxt(PEG51_2)

# data para upsilon andromeda
dataset_7 = np.loadtxt(UPSAND_1)

# HERE you select the current dataset

data_1 = np.vstack((dataset_1, dataset_2, dataset_3))  # GJ180
data_2 = np.vstack((dataset_5, dataset_6))  # peg51
data_3 = dataset_7  # UPSAND
# we merge the data, sort it, and get time, Radial_Velocity and the error (Err)


def data(data, PLOT=True, SAVE=True):
    time = data[:, 0]
    orden = np.argsort(time)
    Data = data[orden]
    time = Data[:, 0]
    time = time - time[0]
    Radial_Velocity = Data[:, 1]  # aca se pone la rv
    Err = Data[:, 2]  # el error de la rv
    dps = len(time)


    if PLOT:
        plt.clf()
        plt.errorbar(time, Radial_Velocity, Err, marker="o", linestyle="", label='Data')
        plt.xlabel('time [days]')
        plt.ylabel('RV [m/s]')
        plt.title('True Data', fontsize=22)
        plt.legend(numpoints=1)
        if SAVE:
            plt.savefig("true_data.jpg")
        plt.draw()
        plt.show()

    return time, Radial_Velocity, Err


time, Radial_Velocity, Err = data(data_2)

def model(A, P, w, phase, ecc, offset, jitt, time):
    freq = 2. * np.pi / P
    M = freq * time + phase
    E = np.array([ks.getE(m, ecc) for m in M])
    f = (np.arctan(np.sqrt((1. + ecc) / (1. - ecc)) * np.tan(E / 2.)) * 2)
    return A * (np.cos(f + w) + ecc * np.cos(w)) + offset


#fullmodel
modeltime = sp.linspace(0, time[len(time)-1], len(time))
model_full = model(60, 4.23, 3.05, 3.05, 0.00, -9.3, 6.099, modeltime)
'''
fig = plt.figure(figsize=(10, 10))
plt.clf()
plt.errorbar(time, Radial_Velocity, marker='o', color='red', label='Data', linestyle='')
plt.errorbar(time, model_full, color='blue', label='Model')

plt.xlabel('time [days]', fontsize=18)
plt.ylabel('RV [m/s]', fontsize=18)
plt.title('Raw Data vs Model', fontsize=22)
plt.legend(numpoints=1)
plt.draw()
plt.show()
'''
#phase-fold

phases = foldAt(time, 4.23, T0=0.)  # folds at the Period found
sortIndi = sp.argsort(phases)  # sorts the points
Phases = phases[sortIndi]  # gets the indices so we sort the RVs correspondingly(?)
rv_phased = Radial_Velocity[sortIndi]
err_phased = Err[sortIndi]
time_phased = Phases * 4.23  #After anti-logging P!!!
model_phased = model(65, 4.23, 2.05, 2.05, 0.00, -9.3, 6.099, time_phased)


fig2 = plt.figure(figsize=(10, 10))
plt.clf()

plt.errorbar(time_phased, rv_phased, err_phased, color='red', marker='o', linestyle="", label='Data')
plt.errorbar(time_phased, model_phased, color='blue', label='Model')
plt.xlabel('time [days]', fontsize=18)
plt.ylabel('RV [m/s]', fontsize=18)
plt.title('Phase Fold', fontsize=22)
plt.legend(numpoints=1)
plt.draw()
plt.show()
