'''
Created on Sep 9, 2013

@author: Steven
'''

from hod import hod
import numpy as np
import matplotlib.pyplot as plt
R = np.arange(10, 200, 5)

x = hod(r=R)

plt.plot(np.exp(x.lnk), x.galaxy_power(), label="Total Power")
plt.plot(np.exp(x.lnk), x._power_1h(), label="1-halo")
plt.plot(np.exp(x.lnk), x._power_2h(), label="2-halo")
plt.plot(np.exp(x.lnk), np.exp(x.power), label="DM power")

plt.legend(loc=0)
plt.xscale('log');plt.yscale('log')
plt.savefig("/Users/Steven/Documents/HOD_testplots/galpower.png")
plt.clf()

plt.plot(x.r, x.galaxy_corr())
plt.savefig("/Users/Steven/Documents/HOD_testplots/galcorr.png")
