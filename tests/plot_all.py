'''
Created on Oct 2, 2013

@author: Steven
'''
from hod import HOD
import numpy as np
import matplotlib.pyplot as plt


def plot_all(r, corr, corr1, corr2, dmcorr, name):
    plt.clf()
    plt.plot(r, corr, label="Gal Corr")
    plt.plot(r, corr1, label="Gal Corr 1h")
    plt.plot(r, corr2, label="Gal Corr 2h")
    plt.plot(r, dmcorr, label="DM Corr")
    plt.legend()
    plt.yscale("log")
    plt.xscale('log', basex=2)
    plt.savefig("/Users/Steven/Documents/HOD_testplots/correlations_" + name + '.pdf')

r = np.logspace(-30.0, 25.0, 100, base=1.1)

h = HOD(r=r, scale_dependent_bias=False)

plot_all(h.r, h.corr_gal, h.corr_gal_1h, h.corr_gal_2h, h.dm_corr, "centralT")

#h.update(central=False)
#plot_all(h.r, h.corr_gal, h.corr_gal_1h, h.corr_gal_2h, h.dm_corr, "centralF")
#
#h.update(central=True)
#
#for bm in ["ST", 'seljak', "ma", "tinker"]:
#    h.update(bias_model=bm)
#
#    plot_all(h.r, h.corr_gal, h.corr_gal_1h, h.corr_gal_2h, h.dm_corr, bm)
#
#h.update(bias_model='ST', cm_relation="duffy")
#plot_all(h.r, h.corr_gal, h.corr_gal_1h, h.corr_gal_2h, h.dm_corr, 'duffy')


h.update(nonlinear=True)
#plt.clf()
#plt.plot(h.pert.lnk, h.pert._unnormalized_power, label="linear prenorm")
#plt.plot(h.pert.lnk, h.pert.power, label="linear")
#plt.plot(h.pert.lnk, h.matter_power, label="non-linear")
#plt.legend()
#plt.savefig("/Users/Steven/Documents/HOD_testplots/compare_power.pdf")
plot_all(h.r, h.corr_gal, h.corr_gal_1h, h.corr_gal_2h, h.dm_corr, 'NL')

#h.update(scale_dependent_bias=True)
#plot_all(h.r, h.corr_gal, h.corr_gal_1h, h.corr_gal_2h, h.dm_corr, 'SCALE')
#
#h.update(halo_exclusion="sphere_zheng")
#plot_all(h.r, h.corr_gal, h.corr_gal_1h, h.corr_gal_2h, h.dm_corr, 'sphere_zheng')

#
#h.update(halo_exclusion="ng_matched")
#plot_all(h.r, h.corr_gal, h.corr_gal_1h, h.corr_gal_2h, h.dm_corr, 'ng_matched')
#
#
#h.update(halo_exclusion="ellipsoid")
#plot_all(h.r, h.corr_gal, h.corr_gal_1h, h.corr_gal_2h, h.dm_corr, 'ellipsoid')
