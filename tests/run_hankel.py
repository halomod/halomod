'''
Created on Sep 9, 2013

@author: Steven
'''

from hmf import Perturbations
import numpy as np
import matplotlib.pyplot as plt

pert = Perturbations()
from hod import tools

R = np.arange(10.0, 200.0, 2)
res = tools.power_to_corr(pert.power, pert.lnk, R)

plt.plot(R, res)
plt.yscale('log')
plt.savefig("/Users/Steven/Documents/dm_corr_log.png")
plt.clf()

plt.plot(R, res)
#plt.yscale('log')
plt.savefig("/Users/Steven/Documents/dm_corr.png")
plt.clf()
