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
from normalpdf import normal_pdf

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
dataset_1 = sp.loadtxt(GJ180, usecols=(0,1,2))
dataset_2 = sp.loadtxt(GJ180_PFS)
dataset_3 = sp.loadtxt(GJ180_UVES)
# dataset_4 = sp.loadtxt(GJ180_KECK, usecols=(0,1,2))

# data para 51peg
dataset_5 = sp.loadtxt(PEG51_1)
dataset_6 = sp.loadtxt(PEG51_2)

# data para upsilon andromeda
dataset_7 = sp.loadtxt(UPSAND_1)

# HERE you select the current dataset

data_1 = sp.vstack((dataset_1, dataset_2, dataset_3))  # GJ180
data_2 = sp.vstack((dataset_5, dataset_6))  # peg51
data_3 = dataset_7  # UPSAND
# we merge the data, sort it, and get Time, Radial_Velocity and the error (Err)

def data(data, PLOT=False, SAVE=True):
    Time = data[:, 0]
    orden = sp.argsort(Time)
    Data = data[orden]
    Time = Data[:, 0]
    Time = Time - Time[0]
    Radial_Velocity = Data[:, 1]  # aca se pone la rv
    Err = Data[:, 2]  # el error de la rv
    dps = len(Time)

    if PLOT:
        plt.clf()
        plt.errorbar(Time, Radial_Velocity, Err, marker="o", linestyle="", label='Data')
        plt.xlabel('Time [days]')
        plt.ylabel('RV [m/s]')
        plt.title('True Data', fontsize=22)
        plt.legend(numpoints=1)
        if SAVE:
            plt.savefig("true_data.jpg")
        plt.draw()
        plt.show()

    return Time, Radial_Velocity, Err


#[A1,A2, P1,P2, w1,w2, phase1,phase2, ecc1,ecc2, jitt1,jitt2, offset1,offset2]
mod_lims = [0.01, 100, 0.1, 1000, 0., 2. * sp.pi, 0., 2. * sp.pi, 0.0, 1.0, 0.0001,15., -50.0,50.0]
mod_lims[10] = sp.log(mod_lims[10])
mod_lims[11] = sp.log(mod_lims[11])
mod_lims[2] = sp.log(mod_lims[2])
mod_lims[3] = sp.log(mod_lims[3])


def model(A, P, w, phase, ecc, jitt, offset, time):
    freq = 2. * sp.pi / P
    M = freq * time + phase
    E = sp.array([ks.getE(m, ecc) for m in M])
    f = (sp.arctan(sp.sqrt((1. + ecc) / (1. - ecc)) * sp.tan(E / 2.)) * 2)
    return A * (sp.cos(f + w) + ecc * sp.cos(w)) + offset


def lnlike(theta, time, rv, rv_err):
    A, P, w, phase, ecc, lnjit, offset = theta
    per = sp.exp(P)
    JIT = sp.exp(lnjit)  # Convert the lnj to J!!
    MODEL = model(A, per, w, phase, ecc, JIT, offset, time)
    inv_sigma2 = 1.0 / (rv_err**2 + MODEL**2 * np.exp(2*lnjit))
    print(-0.5 * sp.sum(((rv - MODEL) ** 2 * inv_sigma2) - sp.log(inv_sigma2)))
    return -0.5 * sp.sum(((rv - MODEL) ** 2 * inv_sigma2) - sp.log(inv_sigma2))


def lnprior(theta):
    A, P, w, phase, ecc, lnjit, offset = theta
    per, JIT = sp.exp(P), sp.exp(lnjit)
    if mod_lims[0] <= A <= mod_lims[1] and mod_lims[2] <= P <= mod_lims[3] and mod_lims[4] <= w <= mod_lims[5] and mod_lims[6] <= phase <= mod_lims[7] and mod_lims[8] <= ecc <= mod_lims[9] and  mod_lims[10] <= lnjit <= mod_lims[11] and mod_lims[12] <= offset <= mod_lims[13]:
        return 0.0
    return -sp.inf


def lnprob(theta, time, rv, rv_err):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, time, rv, rv_err)



time, rv, rv_err = data(data_2)
ndim, nwalkers = 7, 100

pos = [sp.zeros(ndim) for i in range(nwalkers)]
k = 0
#Create the starting positions!!
for j in xrange(0, ndim):
    if j > 0:
        k += 2
    fact = sp.absolute(mod_lims[k] - mod_lims[k+1])/nwalkers
    dif = sp.arange(nwalkers) * fact
    for i in xrange(0, nwalkers):
        pos[i][j] = mod_lims[k] + (dif[i] + fact/2.0)


import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time, rv, rv_err))
sampler.run_mcmc(pos, 200)
samples = sampler.chain[:, 20:, :].reshape((-1, ndim))
ln_post = sampler.flatlnprobability

titles = sp.array(["Amplitude","Period","Longitude", "Phase","Eccentricity", "Jitter","Offset"])
A_max = sampler.flatchain[ln_post == np.max(ln_post),0][0]
P_max = sampler.flatchain[ln_post == np.max(ln_post),1][0]
w_max = sampler.flatchain[ln_post == np.max(ln_post),2][0]
phase_max = sampler.flatchain[ln_post == np.max(ln_post),3][0]
ecc_max = sampler.flatchain[ln_post == np.max(ln_post),4][0]
jitt_max = sampler.flatchain[ln_post == np.max(ln_post),5][0]
offset_max = sampler.flatchain[ln_post == np.max(ln_post),6][0]

print('--------------------------------------------------------')
print("The most probable flatchain values are as follows...")
print('A [JD]:          ', A_max)
print('P [days]:        ', P_max)  #After anti-logging P!!!
print('Longitude:       ', w_max)
print('Phase [m/s]:     ', phase_max)
print('Eccentricity:    ', ecc_max)
print('Jitter [m/s]:    ', sp.exp(jitt_max))
print('Offset [m/s]:    ', offset_max)
print('--------------------------------------------------------')


for i in range(ndim):
    pointnum = len(sampler.flatchain[:,i])
    fig1 = plt.figure(figsize=(10, 10))
    sorting = sp.arange(pointnum)
    if i == 5:
        plt.scatter(sp.arange(pointnum), sp.exp(sampler.flatchain[:,i]), c=sorting, lw=0.01)  # After anti-logging P!!!
    else:
        plt.scatter(sp.arange(pointnum), sampler.flatchain[:,i], c=sorting, lw=0.01)
    plt.colorbar()
    plt.ylabel(titles[i])
    plt.xlabel("N")
    plt.title(titles[i])

# Plot the Posteriors out for each parameter, to check the areas of high probability!!
for i in range(ndim):
    fig2 = plt.figure(figsize=(10, 10))
    sorting = sp.arange(len(sampler.flatchain[:,i]))
    if i == 5:
        plt.scatter(sp.exp(sampler.flatchain[:,i]),abs(ln_post), c=sorting, lw=0.01)  #After anti-logging P!!!
    else:
        plt.scatter(sampler.flatchain[:,i],abs(ln_post), c=sorting, lw=0.01)
    plt.colorbar()
    plt.yscale('log')
    plt.xlabel(titles[i])
    plt.ylabel("Posterior")
    plt.title(titles[i])

plt.show()
