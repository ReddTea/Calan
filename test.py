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
from normalpdf import normal_pdf
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

def data(data, PLOT=True, SAVE=True):
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


#Here the model limits are
#[A1,A2, P1,P2, w1,w2, phase1,phase2, ecc1,ecc2, jitt1,jitt2, offset1,offset2]
#Cirrently only P up to 1000d so we should change this!!
mod_lims = [0.01, 100, 0.1, 1000, 0., 2. * sp.pi, 0., 2. * sp.pi, 0.0, 1.0, 0.0001,15., -50.0,50.0]
mod_lims[2] = sp.log(mod_lims[2])
mod_lims[3] = sp.log(mod_lims[3])

#######################
## Bayesian Analysis ##
#######################

# Initiatiate the True Anomaly solver

counter = 0
pbar = tqdm(total=1)

def model(A, P, w, phase, ecc, offset, jitt, time):
    freq = 2. * sp.pi / P
    M = freq * time + phase
    E = sp.array([ks.getE(m, ecc) for m in M])
    f = (sp.arctan(sp.sqrt((1. + ecc) / (1. - ecc)) * sp.tan(E / 2.)) * 2)
    return A * (sp.cos(f + w) + ecc * sp.cos(w)) + offset

##################################
# Define the likelihood Function #
#################################

def lnlike(theta, time, rv, err):

    global counter
    global pbar
    A, P, w, phase, ecc, jitt, offset = theta
    # Convert the lnP to P!!
    per = sp.exp(P)

    MODEL = model(A, per, w, phase, ecc, offset, jitt, time)
    invsigma2 = 1.0 / (err**2 + jitt**2)
    counter += 1
    if counter == 100:
        counter = 0
        pbar.update(100)

    #print(-0.5 * sp.sum(((rv - MODEL) ** 2 * invsigma2) - sp.log(invsigma2)))

    return -0.5 * sp.sum(((rv - MODEL) ** 2 * invsigma2) - sp.log(invsigma2))


def lnprior_flat(theta):
    A, P, w, phase, ecc, jitt, offset = theta
    if mod_lims[0] <= A <= mod_lims[1] and mod_lims[2] <= P <= mod_lims[3] and mod_lims[4] <= w <= mod_lims[5] and mod_lims[6] <= phase <= mod_lims[7] and mod_lims[8] <= ecc <= mod_lims[9] and  mod_lims[10] <= jitt <= mod_lims[11] and mod_lims[12] <= offset <= mod_lims[13]:
        return 0.0
    per = sp.exp(P)  # what's up with this?

    return -sp.inf


def lnprior(theta):
    A, P, w, phase, ecc, jitt, offset = theta
    per = sp.exp(P)  # what's up with this?

    # Amplitude (Modified?) Jeffreys Prior
    if mod_lims[0] <= A <= mod_lims[1]:
        #print('LIMITE DE A MAYOR')
        #print(mod_lims[1] * (sp.exp(mod_lims[2]) / per) ** (1. / 3) * (1. / sp.sqrt(1. - ecc ** 2.)))
        #lp_semi = (semiamp+k0)**(-1.0) / sp.log( 1.0+(kmax/k0) * (pmin/period_lp)**(1.0/3.0) * (1.0 / sp.sqrt(1.0 - eccentric**2.0)) )
        #lp_amp = (A + mod_lims[0]) ** -1. / sp.log(1.0 + mod_lims[1]/mod_lims[0] * (sp.exp(mod_lims[2])/sp.exp(mod_lims[3]) ** 1./3 * (1./ sp.sqrt(1. - ecc ** 2.)) ))
        lp_amp = 0  # (A + mod_lims[0]) ** -1. * sp.log(1. + mod_lims[1] / mod_lims[0])
    else:
        return - sp.inf

    # Period Jeffreys
    if mod_lims[2] <= P <= mod_lims[3]:
        lp_per = 0  # 1. / (P * sp.log10(sp.exp(mod_lims[3]) / sp.exp(mod_lims[2]) ))
    else:
        return - sp.inf

    # Longitude
    if mod_lims[4] <= w <= mod_lims[5]:
        lp_w = 0  # 1.0 / (2.0 * sp.pi)
    else:
        return - sp.inf

    # Phase
    if mod_lims[6] <= phase <= mod_lims[7]:
        lp_pha = 0.0
    else:
        return - sp.inf

    # Eccentricity
    if mod_lims[8] <= ecc <= mod_lims[9]:
        norm = normal_pdf(ecc)
        lp_ecc = 0
        #if ecc < 0.1:
        #    lp_ecc = 50000 #np.log(norm)
    else:
        return -sp.inf

    # Jitter
    if mod_lims[10] <= jitt <= mod_lims[11]:  # jitmax and not ampmax?
        lp_jit = 0  # (jitt + mod_lims[10]) ** (-1.0) / sp.log(1. + mod_lims[11] / mod_lims[10])
    else:
        return -sp.inf
    # Offset
    if mod_lims[12] <= offset <= mod_lims[13]:
        lp_off = 0 #  1.0 / (mod_lims[12] - mod_lims[13])
    else:
        return - sp.inf

    return lp_amp + lp_per + lp_w + lp_pha + lp_ecc + lp_jit + lp_off


def lnprob(theta, time, rv, err):
    lp = lnprior(theta)
    if not sp.isfinite(lp):
        return -sp.inf
    #print(lp)
    #print(lp + lnlike(theta, time, rv, err))
    return lp + lnlike(theta, time, rv, err)


def mcmc(Time, Radial_Velocity, Err, limits, nwalkers=300, chainlength=120,
         BURNIN=True, CHAINS=True, POST= True, CORNER=True, PHASEFOLD=True,
         SAVE=True):
    global pbar
    start_chrono = chrono.time()  # Chronometer
    ndim = 7
    # 28000 con nwalkers = 100, chainlength = 300
    chain_length = chainlength * nwalkers
    print('The chain length is = '+str(chain_length))

    if BURNIN:
        burn_out = chain_length // 10
        pbar=tqdm(total= chain_length - burn_out)
    else:
        pbar=tqdm(total= chain_length)

    ##################################
    # Run the Sampler                #
    # Evenly spaced over the prior   #
    ##################################

    pos = [sp.zeros(ndim) for i in range(nwalkers)]
    k = 0
    #Create the starting positions!!
    for j in xrange(0, ndim):
        if j > 0:
            k += 2
        fact = sp.absolute(limits[k] - limits[k+1])/nwalkers
        dif = sp.arange(nwalkers) * fact
        for i in xrange(0, nwalkers):
            pos[i][j] = limits[k] + (dif[i] + fact/2.0)


    print("The MCMC Starting Positions for Walker1 for A, ln(P), w, Phase, e, Jitt and Offset are: ")
    print(pos[0])
    print("         ")

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(Time, Radial_Velocity, Err))
    sampler.run_mcmc(pos, chainlength) ##CAMBIOS AQUI, (pos, nwalks) el 100 funciona

    ###########
    # Burn-in #
    ###########
    if BURNIN:
        samples = sampler.chain[:, burn_out:, :].reshape((-1, ndim))

    ############
    # Plotting #
    ############

    print("Mean acceptance fraction: {0:.3f}"
                    .format(sp.mean(sampler.acceptance_fraction)))

    '''
    print "The estimated Autocorrelation times for each of the model parameters are:"
    for i in range(ndim):
        print sampler.acor[i]
    '''
    ln_post = sampler.flatlnprobability

    #############################################################

    #############
    # P L O T S #
    #############

    titles = sp.array(["Amplitude","Period","Longitude", "Phase","Eccentricity", "Jitter","Offset"])
    # Plot the chains out to investigate the walkers!!
    # with cmap = <color> you can change the colormap
    # like cmap=plt.cm.hot, gist_heat, binary, autumn, etc
    if CHAINS:
        for i in range(ndim):
            pointnum = len(sampler.flatchain[:,i])
            fig1 = plt.figure(figsize=(10, 10))
            sorting = sp.arange(pointnum)
            if i == 1:
                plt.scatter(sp.arange(pointnum), sp.exp(sampler.flatchain[:,i]), c=sorting, lw=0.01)  # After anti-logging P!!!
            else:
                plt.scatter(sp.arange(pointnum), sampler.flatchain[:,i], c=sorting, lw=0.01)
            plt.colorbar()
            plt.ylabel(titles[i])
            plt.xlabel("N")
            plt.title(titles[i])
            if SAVE:
                fig1.savefig("chains"+str(i)+".jpg")

    # Plot the Posteriors out for each parameter, to check the areas of high probability!!
    if POST:
        for i in range(ndim):
            fig2 = plt.figure(figsize=(10, 10))
            sorting = sp.arange(len(sampler.flatchain[:,i]))
            if i == 1:
                plt.scatter(sp.exp(sampler.flatchain[:,i]),abs(ln_post), c=sorting, lw=0.01)  #After anti-logging P!!!
            else:
                plt.scatter(sampler.flatchain[:,i],abs(ln_post), c=sorting, lw=0.01)
            plt.colorbar()
            plt.yscale('log')
            plt.xlabel(titles[i])
            plt.ylabel("Posterior")
            plt.title(titles[i])
            if SAVE:
                fig2.savefig("posteriors"+str(i)+".jpg")



    ########################################################

    ########################################################
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
    print('P [days]:        ', sp.exp(P_max))  #After anti-logging P!!!
    print('Longitude:       ', w_max)
    print('Phase [m/s]:     ', phase_max)
    print('Eccentricity:    ', ecc_max)
    print('Jitter [m/s]:    ', jitt_max)
    print('Offset [m/s]:    ', offset_max)
    print('--------------------------------------------------------')

    # Plotting


    #fullmodel
    modeltime = sp.linspace(0, Time[len(Time)-1], len(Time))
    model_full = model(A_max,P_max,w_max,phase_max,ecc_max, offset_max, jitt_max, modeltime)

    fig = plt.figure(figsize=(10, 10))
    plt.clf()
    plt.errorbar(Time, Radial_Velocity, marker='o', color='red', label='Data', linestyle='')
    plt.errorbar(Time, model_full, color='blue', label='Model')

    plt.xlabel('Time [days]', fontsize=18)
    plt.ylabel('RV [m/s]', fontsize=18)
    plt.title('Raw Data vs Model', fontsize=22)
    plt.legend(numpoints=1)
    if SAVE:
        plt.savefig("notphase_folded_rv.jpg")
    plt.draw()
    plt.show()

    #phase-fold
    if PHASEFOLD:
        per_max = sp.exp(P_max)

        phases = foldAt(Time, per_max, T0=0.)  # folds at the Period found
        sortIndi = sp.argsort(phases)  # sorts the points
        Phases = phases[sortIndi]  # gets the indices so we sort the RVs correspondingly(?)
        rv_phased = Radial_Velocity[sortIndi]
        err_phased = Err[sortIndi]

        time_phased = Phases * per_max  #After anti-logging P!!!

        model_phased = model(A_max, per_max, w_max, phase_max, ecc_max, offset_max, jitt_max, time_phased)


        fig2 = plt.figure(figsize=(10, 10))
        plt.clf()

        plt.errorbar(time_phased, rv_phased, err_phased, color='red', marker='o', linestyle="", label='Data')
        plt.errorbar(time_phased, model_phased, color='blue', label='Model')

        plt.xlabel('Time [days]', fontsize=18)
        plt.ylabel('RV [m/s]', fontsize=18)
        plt.title('Phase Fold', fontsize=22)
        plt.legend(numpoints=1)
        if SAVE:
            plt.savefig("phase_folded_rv.jpg")
        plt.draw()
        plt.show()
        chronometer = np.round(chrono.time() - start_chrono)
        minutes = chronometer // 60
        seconds = chronometer % 60
    pbar.close()
    print("--- %s minutes and %s seconds ---" % (minutes, seconds)) # Chronometer

time, rv, er = data(data_2)
mcmc(time, rv, er, mod_lims)

#tuomi anglada scude feroz hobson
