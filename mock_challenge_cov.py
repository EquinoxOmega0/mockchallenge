#!/home/users/csaulder/anaconda3/envs/desimock/bin/python

import pycorr
import cosmoprimo
import configparser
import argparse
from pycorr import TwoPointCorrelationFunction,project_to_multipoles
import scipy as sp
import numpy as np
import glob
import sys, getopt 
import os
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib as mpl


#to load and convert the correlation function
def mockchallenge_dataloader(filename,rebinfactor_s):
    alldata=TwoPointCorrelationFunction.load(filename)
    #rebinning
    alldata.rebin((rebinfactor_s,1))
    #calculate multipoles
    poles=project_to_multipoles(alldata)
    return poles,alldata




#define hartlap correction
def hartlap_correction(n_mocks,n_bins):
    hl_cor=float(n_mocks-n_bins-2)/float(n_mocks-1)
    return hl_cor


#define percival correction
def percival_correction(nb,ns,npara):
    A = 2./(float(ns-nb-1.)*float(ns-nb-4.))
    B = float(ns-nb-2.)/float((ns-nb-1.)*(ns-nb-4.))
    p_cor = np.sqrt(( 1.+(B*float(nb-npara)) )/( 1.+A+(B*float(npara+1.)) ))
    return p_cor





def get_cov(pathtwopoint,rebinfactor_s): 

    # get files and data format
    fnames_twopoint = glob.glob(pathtwopoint+'*.npy')
    n_mocks=len(fnames_twopoint)

    #get shape of the data
    poles,_=mockchallenge_dataloader(fnames_twopoint[0],rebinfactor_s)
    sbin_centres=poles[0]
    multipoles=poles[1]
    n_sbins=len(sbin_centres)
    multipole_n=len(multipoles)
    n_datapoint=n_sbins*multipole_n
    
    #create the right list of sbins for all multipoles
    sbin_centres_for_all=np.zeros(n_datapoint)
    for ip in range(multipole_n):
        sbin_centres_for_all[(n_sbins*ip):(n_sbins*(ip+1))]=sbin_centres
    
    #n_parameters=6

    #collect the data from all EZmocks into one array
    all_data = np.zeros((n_mocks,n_datapoint), dtype = [('s','f'),('xi','f')])    

    for i, fname in enumerate(fnames_twopoint):
        poles,_=mockchallenge_dataloader(fnames_twopoint[i],rebinfactor_s)
        multipoles=poles[1]
        all_data[i,:]['s']=sbin_centres_for_all
        all_data[i,:]['xi']=np.concatenate(multipoles)
    
    #calculate the covariance matrix and inverted covariance matrix
    xi_data=all_data['xi']
    cov_all=np.cov(xi_data,rowvar=False)
    cov_inv=np.linalg.inv(cov_all)
    
    #cov_inv_hartlap=cov_inv*hartlap_correction(n_mocks,n_datapoint)

    return cov_all,cov_inv,sbin_centres_for_all



#created plots for the covariance matrix and the inverted covariance matrix
def plot_cov(cov_all,cov_inv,pathtoplotoutput):

    cov_inv_norm=np.copy(cov_inv)
    cov_all_norm=np.copy(cov_all)

#normalize it for nice plots
    for i in range(cov_all.shape[0]):
        for j in range(cov_all.shape[0]):
            cov_all_norm[i,j]=cov_all[i,j]/(np.sqrt(cov_all[i,i])*np.sqrt(cov_all[j,j]))

    for i in range(cov_inv.shape[0]):
        for j in range(cov_inv.shape[0]):
            cov_inv_norm[i,j]=cov_inv[i,j]/(np.sqrt(cov_inv[i,i])*np.sqrt(cov_inv[j,j]))

#plot it
    plt.clf()
    plt.figure(figsize=[6.4, 4.8])
    plt.imshow(cov_all_norm,               cmap='RdBu_r')
    plt.tight_layout()
    plt.savefig(pathtoplotoutput+"cov.png",dpi=300, facecolor='w',edgecolor='w')
    plt.close()
    #norm=colors.SymLogNorm(linthresh=np.average(cov_all_norm))  , 

    plt.clf()
    plt.figure(figsize=[6.4, 4.8])
    plt.imshow(cov_inv_norm,           cmap='RdBu_r')
    plt.tight_layout()
    plt.savefig(pathtoplotoutput+"cov_inv.png",dpi=300, facecolor='w',edgecolor='w')
    plt.close()
    

#do everything for the covariance calculations
def cal_cov(pathtwopoint,savepath,plotpath,redshiftbin,rebinfactor_s):

#calcute the covariance matrix
    cov_all,cov_inv,sbin_centres_for_all=get_cov(pathtwopoint+redshiftbin,rebinfactor_s)
#plot it
    plot_cov(cov_all,cov_inv,plotpath+redshiftbin)

#save it in numpy format
    np.save(savepath+redshiftbin+'cov',cov_all)
    np.save(savepath+redshiftbin+'cov_inv',cov_inv)
    np.save(savepath+redshiftbin+'sbins',sbin_centres_for_all)

#save it in ascii format
    np.savetxt(savepath+redshiftbin+'cov.dat',cov_all)
    np.savetxt(savepath+redshiftbin+'cov_inv.dat',cov_inv)
    np.savetxt(savepath+redshiftbin+'sbins.dat',sbin_centres_for_all)


#converts the 25 abacus realisation to the same format as the covariance matrix
def convert_abacus(path_abacus_CF,savepath_abacus_CF,nrand,redshiftbin,rebinfactor_s):
    #loop over realisations
    for i_real in range(25):
        s_real=str(i_real).zfill(3)
        #determine filename from pycorr calculations and load file
        filename="results_realization"+s_real+"_rand"+str(nrand)+"_"+redshiftbin+".npy"
        poles,alldata=mockchallenge_dataloader(path_abacus_CF+filename,rebinfactor_s)
        
        #convert to standard ascii file for multipoles
        alldata.save_txt(savepath_abacus_CF+filename[:-4]+".dat",  ells=(0, 2, 4))
        
        #also save as numpy (nested array seem to be decrepit, hence also split parts)
        np.save(savepath_abacus_CF+filename[:-4],poles)
        np.save(savepath_abacus_CF+filename[:-4]+"_sbin",poles[0])
        np.save(savepath_abacus_CF+filename[:-4]+"_poles",poles[1])




#basic paths (for my filesystem)
rebinfactor_s=4
plotpath="plots/mockchallenge/cov/"
savepath="samples/mockchallenge/cov/"
pathtwopoint="samples/mockchallenge/EZmocks/EZmock_results_"



#calculate the covariance matrix for evey redshift slice
cal_cov(pathtwopoint,savepath,plotpath,"46_",rebinfactor_s)
cal_cov(pathtwopoint,savepath,plotpath,"68_",rebinfactor_s)
cal_cov(pathtwopoint,savepath,plotpath,"81_",rebinfactor_s)

#path to abacus data (for my file system)
path_abacus_CF="samples/mockchallenge/results/"
savepath_abacus_CF="samples/mockchallenge/CF_multipoles/"

#convert all CF with 20X randoms for all redshift slices
convert_abacus(path_abacus_CF,savepath_abacus_CF,20,"46",rebinfactor_s)
convert_abacus(path_abacus_CF,savepath_abacus_CF,20,"68",rebinfactor_s)
convert_abacus(path_abacus_CF,savepath_abacus_CF,20,"81",rebinfactor_s)

#convert all CF with 5X randoms for all redshift slices
convert_abacus(path_abacus_CF,savepath_abacus_CF,5,"46",rebinfactor_s)
convert_abacus(path_abacus_CF,savepath_abacus_CF,5,"68",rebinfactor_s)
convert_abacus(path_abacus_CF,savepath_abacus_CF,5,"81",rebinfactor_s)














