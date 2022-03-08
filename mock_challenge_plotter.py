#!/home/users/csaulder/anaconda3/envs/mockchallenge/bin/python

import pycorr
import cosmoprimo
import configparser
import argparse
from pycorr import TwoPointCorrelationFunction
from mpi4py import MPI
import scipy as sp
import numpy as np
import glob
import sys, getopt 
import os
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib as mpl
#from astropy.cosmology import FlatLambdaCDM


def mockchallenge_dataloader(filename,rebinfactor_s,rebinfactor_mu):
    alldata=TwoPointCorrelationFunction.load(filename)
    alldata.rebin((rebinfactor_s,rebinfactor_mu))
    cf=alldata.corr
    sbins=alldata.sep[:,0]
    mubins=alldata.seps[1][0]
    return cf,sbins,mubins



def plot_redshift_evolution(filename46,filename68,filename81,rebinfactor_s,rebinfactor_mu,plotpath,plotname):
    cf46_all,sbins,mubins=mockchallenge_dataloader(filename46,rebinfactor_s,rebinfactor_mu)
    cf68_all,_,_=mockchallenge_dataloader(filename68,rebinfactor_s,rebinfactor_mu)
    cf81_all,_,_=mockchallenge_dataloader(filename81,rebinfactor_s,rebinfactor_mu)
    
    nmu=len(mubins)
    ns=len(sbins)
    
    for imu in range(nmu):
        cf46=cf46_all[:,imu]
        cf68=cf68_all[:,imu]
        cf81=cf81_all[:,imu]
        
        current_mu=mubins[imu]
    
        minval=np.min([np.min(np.square(sbins)*cf46),np.min(np.square(sbins)*cf68),np.min(np.square(sbins)*cf81)])
        maxval=np.max([np.max(np.square(sbins)*cf46),np.max(np.square(sbins)*cf68),np.max(np.square(sbins)*cf81)])
    
        plt.clf()
        plt.figure(figsize=[6.4, 4.8])
        plt.title(r'redshift evolution $\bar{\mu}=$'+str(np.round(current_mu,3))+' ('+str(nmu)+' $\mu$-bins)')        
        plt.axis([np.min(sbins), np.max(sbins),minval, maxval]) 
        plt.scatter(sbins,np.square(sbins)*cf46,c='b',marker='o',label='0.4 < z < 0.6')
        plt.scatter(sbins,np.square(sbins)*cf68,c='g',marker='v',label='0.6 < z < 0.8')
        plt.scatter(sbins,np.square(sbins)*cf81,c='r',marker='+',label='0.8 < z < 1.1')
                        
        plt.xlabel(r's [$h^{-1}$ Mpc]')   
        plt.ylabel(r'$s^{2} \xi$ [$h^{-1}$ Mpc]') 
        plt.tight_layout()    
        plt.legend(loc='best',markerscale=1.,ncol=1)
        plt.savefig(plotpath+"redshiftevolution_"+plotname+"_sbin_"+str(ns)+"_mu_"+str(nmu)+"_"+str(imu)+".png",dpi=300, facecolor='w',edgecolor='w')
        plt.close()   
        
        

def plot_compare_seeds(filename_phase1,filename_phase2,filename_phase3,filename_phase4,filename_phase5,rebinfactor_s,rebinfactor_mu,plotpath,plotname):
    
    cf1_all,sbins,mubins=mockchallenge_dataloader(filename_phase1,rebinfactor_s,rebinfactor_mu)
    cf2_all,_,_=mockchallenge_dataloader(filename_phase2,rebinfactor_s,rebinfactor_mu)
    cf3_all,_,_=mockchallenge_dataloader(filename_phase3,rebinfactor_s,rebinfactor_mu)
    cf4_all,_,_=mockchallenge_dataloader(filename_phase4,rebinfactor_s,rebinfactor_mu)
    cf5_all,_,_=mockchallenge_dataloader(filename_phase5,rebinfactor_s,rebinfactor_mu)
    
    nmu=len(mubins)
    ns=len(sbins)
    
    for imu in range(nmu):
        cf1=cf1_all[:,imu]
        cf2=cf2_all[:,imu]
        cf3=cf3_all[:,imu]
        cf4=cf4_all[:,imu]
        cf5=cf5_all[:,imu]
        
        current_mu=mubins[imu]
    
        minval=np.min([np.min(np.square(sbins)*cf1),np.min(np.square(sbins)*cf2),np.min(np.square(sbins)*cf3),np.min(np.square(sbins)*cf4),np.min(np.square(sbins)*cf5)])
        maxval=np.max([np.max(np.square(sbins)*cf1),np.max(np.square(sbins)*cf2),np.max(np.square(sbins)*cf3),np.max(np.square(sbins)*cf4),np.max(np.square(sbins)*cf5)])
    
        plt.clf()
        plt.figure(figsize=[6.4, 4.8])
        plt.title(r'seed comparison $\bar{\mu}=$'+str(np.round(current_mu,3))+' ('+str(nmu)+')$\mu$-bins')        
        plt.axis([np.min(sbins), np.max(sbins),minval, maxval]) 
        plt.scatter(sbins,np.square(sbins)*cf1,c='b',marker='o',label='seed 001')
        plt.scatter(sbins,np.square(sbins)*cf2,c='g',marker='v',label='seed 002')
        plt.scatter(sbins,np.square(sbins)*cf3,c='r',marker='+',label='seed 003')
        plt.scatter(sbins,np.square(sbins)*cf4,c='m',marker='*',label='seed 004')
        plt.scatter(sbins,np.square(sbins)*cf5,c='c',marker='^',label='seed 005')
                        
        plt.xlabel(r's [$h^{-1}$ Mpc]')   
        plt.ylabel(r'$s^{2} \xi$ [$h^{-1}$ Mpc]') 
        plt.tight_layout()    
        plt.legend(loc='best',markerscale=1.,ncol=1)
        plt.savefig(plotpath+"seedcomparison_"+plotname+"_sbin_"+str(ns)+"_mu_"+str(nmu)+"_"+str(imu)+".png",dpi=300, facecolor='w',edgecolor='w')
        plt.close()   
        

def plot_compare_randoms(filename_rand1,filename_rand2,filename_rand3,filename_rand4,rebinfactor_s,rebinfactor_mu,plotpath,plotname):
    
    cf1_all,sbins,mubins=mockchallenge_dataloader(filename_rand1,rebinfactor_s,rebinfactor_mu)
    cf2_all,_,_=mockchallenge_dataloader(filename_rand2,rebinfactor_s,rebinfactor_mu)
    cf3_all,_,_=mockchallenge_dataloader(filename_rand3,rebinfactor_s,rebinfactor_mu)
    cf4_all,_,_=mockchallenge_dataloader(filename_rand4,rebinfactor_s,rebinfactor_mu)
    
    nmu=len(mubins)
    ns=len(sbins)
    
    for imu in range(nmu):
        cf1=cf1_all[:,imu]
        cf2=cf2_all[:,imu]
        cf3=cf3_all[:,imu]
        cf4=cf4_all[:,imu]
        
        current_mu=mubins[imu]
    
        minval=np.min([np.min(np.square(sbins)*cf1),np.min(np.square(sbins)*cf2),np.min(np.square(sbins)*cf3),np.min(np.square(sbins)*cf4)])
        maxval=np.max([np.max(np.square(sbins)*cf1),np.max(np.square(sbins)*cf2),np.max(np.square(sbins)*cf3),np.max(np.square(sbins)*cf4)])
    
        plt.clf()
        plt.figure(figsize=[6.4, 4.8])
        plt.title(r'n randoms comparison $\bar{\mu}=$'+str(np.round(current_mu,3))+' ('+str(nmu)+')$\mu$-bins')        
        plt.axis([np.min(sbins), np.max(sbins),minval, maxval]) 
        plt.scatter(sbins,np.square(sbins)*cf1,c='b',marker='o',label='5X random')
        plt.scatter(sbins,np.square(sbins)*cf2,c='g',marker='v',label='10X random')
        plt.scatter(sbins,np.square(sbins)*cf3,c='r',marker='+',label='15X random')
        plt.scatter(sbins,np.square(sbins)*cf4,c='m',marker='*',label='20X random')
                        
        plt.xlabel(r's [$h^{-1}$ Mpc]')   
        plt.ylabel(r'$s^{2} \xi$ [$h^{-1}$ Mpc]') 
        plt.tight_layout()    
        plt.legend(loc='best',markerscale=1.,ncol=1)
        plt.savefig(plotpath+"randomcomparison_"+plotname+"_sbin_"+str(ns)+"_mu_"+str(nmu)+"_"+str(imu)+".png",dpi=300, facecolor='w',edgecolor='w')
        plt.close()   
            






def scan_redshift_evolution(rebinfactor_s,rebinfactor_mu):
    for i in range(4):
        n_rand=(i+1)*5
        for iphase in range(5):
            phase=str(iphase+1).zfill(3)
            plotname="results_seed"+str(phase)+"_rand"+str(n_rand)+"_"
            filename46=loadpath+plotname+"46.sh.npy"
            filename68=loadpath+plotname+"68.sh.npy"
            filename81=loadpath+plotname+"81.sh.npy"

            plot_redshift_evolution(filename46,filename68,filename81,rebinfactor_s,rebinfactor_mu,plotpath,plotname)


def scan_seeds(rebinfactor_s,rebinfactor_mu):
    for i in range(4):
        n_rand=(i+1)*5
        for iredshift in range(3):
            current_redshift=str(redshiftrange["z_name"][iredshift],'utf-8')
            
            filename_phase1=loadpath+'results_seed001_rand'+str(n_rand)+'_'+current_redshift+'.sh.npy'
            filename_phase2=loadpath+'results_seed002_rand'+str(n_rand)+'_'+current_redshift+'.sh.npy'
            filename_phase3=loadpath+'results_seed003_rand'+str(n_rand)+'_'+current_redshift+'.sh.npy'
            filename_phase4=loadpath+'results_seed004_rand'+str(n_rand)+'_'+current_redshift+'.sh.npy'
            filename_phase5=loadpath+'results_seed005_rand'+str(n_rand)+'_'+current_redshift+'.sh.npy'
            
            plotname="results_rand"+str(n_rand)+"_"+current_redshift+"_"
            
            plot_compare_seeds(filename_phase1,filename_phase2,filename_phase3,filename_phase4,filename_phase5,rebinfactor_s,rebinfactor_mu,plotpath,plotname)
            
def scan_randoms(rebinfactor_s,rebinfactor_mu):
    for iphase in range(5):
        phase=str(iphase+1).zfill(3)
        for iredshift in range(3):
            current_redshift=str(redshiftrange["z_name"][iredshift],'utf-8')
            
            filename_rand1=loadpath+'results_seed'+phase+'_rand5_'+current_redshift+'.sh.npy'
            filename_rand2=loadpath+'results_seed'+phase+'_rand10_'+current_redshift+'.sh.npy'
            filename_rand3=loadpath+'results_seed'+phase+'_rand15_'+current_redshift+'.sh.npy'
            filename_rand4=loadpath+'results_seed'+phase+'_rand20_'+current_redshift+'.sh.npy'
 
 
            plotname="results_seed"+phase+"_"+current_redshift+"_"
            
            plot_compare_randoms(filename_rand1,filename_rand2,filename_rand3,filename_rand4,rebinfactor_s,rebinfactor_mu,plotpath,plotname)
            
                   
def sbin_tests(filename,rebinfactor_mu,plotpath,plotname):
    cf1_all,sbins1,mubins=mockchallenge_dataloader(filename_phase1,1,rebinfactor_mu)
    cf2_all,sbins2,_=mockchallenge_dataloader(filename_phase1,2,rebinfactor_mu)   
    cf5_all,sbins5,_=mockchallenge_dataloader(filename_phase1,5,rebinfactor_mu)   
    cf8_all,sbins8,_=mockchallenge_dataloader(filename_phase1,8,rebinfactor_mu)   
    cf10_all,sbins10,_=mockchallenge_dataloader(filename_phase1,10,rebinfactor_mu)   
    cf20_all,sbins20,_=mockchallenge_dataloader(filename_phase1,20,rebinfactor_mu)   

    nmu=len(mubins)
    
    for imu in range(nmu):
        cf1=cf1_all[:,imu]
        cf2=cf2_all[:,imu]
        cf5=cf5_all[:,imu]
        cf8=cf8_all[:,imu]
        cf10=cf10_all[:,imu]
        cf20=cf20_all[:,imu]
                
        current_mu=mubins[imu]
    
        minval=np.min([np.min(np.square(sbins1)*cf1),np.min(np.square(sbins2)*cf2),np.min(np.square(sbins5)*cf5),np.min(np.square(sbins8)*cf8),np.min(np.square(sbins10)*cf10),np.min(np.square(sbins20)*cf20)])
        maxval=np.max([np.max(np.square(sbins1)*cf1),np.max(np.square(sbins2)*cf2),np.max(np.square(sbins5)*cf5),np.max(np.square(sbins8)*cf8),np.max(np.square(sbins10)*cf10),np.max(np.square(sbins20)*cf20)])
    
        plt.clf()
        plt.figure(figsize=[6.4, 4.8])
        plt.title(r's bin comparison $\bar{\mu}=$'+str(np.round(current_mu,3))+' ('+str(nmu)+')$\mu$-bins')        
        plt.axis([np.min(sbins1), np.max(sbins1),minval, maxval]) 
        plt.scatter(sbins1,np.square(sbins1)*cf1,c='b',marker='o',label='1 Mpc/h',s=10)
        plt.scatter(sbins2,np.square(sbins2)*cf2,c='g',marker='v',label='2 Mpc/h',s=10)
        plt.scatter(sbins5,np.square(sbins5)*cf5,c='r',marker='+',label='5 Mpc/h',s=10)
        plt.scatter(sbins8,np.square(sbins8)*cf8,c='m',marker='*',label='8 Mpc/h',s=10)
        plt.scatter(sbins10,np.square(sbins10)*cf10,c='c',marker='^',label='10 Mpc/h',s=10)
        plt.scatter(sbins20,np.square(sbins20)*cf20,c='orange',marker='s',label='20 Mpc/h',s=10)
                                       
        plt.xlabel(r's [$h^{-1}$ Mpc]')   
        plt.ylabel(r'$s^{2} \xi$ [$h^{-1}$ Mpc]') 
        plt.tight_layout()    
        plt.legend(loc='best',markerscale=1.,ncol=1)
        plt.savefig(plotpath+"sbin_comparison_"+plotname+"_mu_"+str(nmu)+"_"+str(imu)+".png",dpi=300, facecolor='w',edgecolor='w')
        plt.close()   



FOR MULTIPOLES
#something like "s_d, xiell_d = project_to_multipoles(result_data, ells=ells)" where result_data is the computation of your 2pcf 




redshiftrange=np.zeros(3,dtype=[('z_name', 'S2'),('zmin', '<f8'), ('zmax', '<f8')])
redshiftrange["z_name"]=["46","68","81"]
redshiftrange["zmin"]=[0.4,0.6,0.8]
redshiftrange["zmax"]=[0.6,0.8,1.1]


plotpath="plots/mockchallenge/"
loadpath="samples/mockchallenge/results/"

#rebinfactor_s=1
#rebinfactor_mu=1

#scan_redshift_evolution(rebinfactor_s,rebinfactor_mu)
#scan_seeds(rebinfactor_s,rebinfactor_mu)
#scan_randoms(rebinfactor_s,rebinfactor_mu)



#rebinfactor_s=1
#rebinfactor_mu=6

#scan_redshift_evolution(rebinfactor_s,rebinfactor_mu)
#scan_seeds(rebinfactor_s,rebinfactor_mu)
#scan_randoms(rebinfactor_s,rebinfactor_mu)



rebinfactor_s=5
rebinfactor_mu=6

scan_randoms(rebinfactor_s,rebinfactor_mu)

scan_redshift_evolution(rebinfactor_s,rebinfactor_mu)

scan_seeds(rebinfactor_s,rebinfactor_mu)

        

filename=loadpath+'results_seed001_rand20_46.sh.npy'
plotname="results_seed001_rand20_46_"
sbin_tests(filename,rebinfactor_mu,plotpath,plotname)

filename=loadpath+'results_seed001_rand20_68.sh.npy'
plotname="results_seed001_rand20_68_"
sbin_tests(filename,rebinfactor_mu,plotpath,plotname)

filename=loadpath+'results_seed001_rand20_81.sh.npy'
plotname="results_seed001_rand20_81_"
sbin_tests(filename,rebinfactor_mu,plotpath,plotname)








