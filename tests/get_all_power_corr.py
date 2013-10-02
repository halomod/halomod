'''
Created on Sep 24, 2013

@author: Steven

This script gets the power and correlation several different ways and compares for consistency.
'''


import hod
import numpy as np
import matplotlib.pyplot as plt
import tools
import hmf.tools as ht
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.integrate as intg

#Set up our hod USE SAME R array as charles
r = np.zeros(100)
for i in range(len(r)):
    r[i] = (0.05 / 0.7) * 1.0513 ** i

#MM = np.zeros(225)
m1 = (10 ** 11.8363) / 0.7
MM = []
while m1 < 1.888E16:
    dm = m1 / 100.0
    MM.append(m1 + dm / 2)
    m1 += dm
MM = np.log10(np.array(MM))

h = hod.HOD(M=MM, r=r, central=True , H0=70.0, cm_relation='zehavi',
            M_1=10 ** 12.851, alpha=1.049, M_min=10 ** 11.6222,
            gauss_width=0.26, M_0=10 ** 11.5047, HOD_model='zehavi',
            profile='nfw', bias_model='tinker',
            transfer_fit="EH", omegab=0.05, omegac=0.25, sigma_8=0.8, n=1)

#MAKE PROPERTIES SIMPLER
m = h.M[np.logical_not(np.isnan(h.dndm))]
dndm = h.dndm[np.logical_not(np.isnan(h.dndm))]
n_c = h._n_cen()[np.logical_not(np.isnan(h.dndm))]
n_s = h._n_sat()[np.logical_not(np.isnan(h.dndm))]
n_t = h.n_tot()[np.logical_not(np.isnan(h.dndm))]
bias = h.biasmod.bias[np.logical_not(np.isnan(h.dndm))]

#for i, M in enumerate(m):
#    print M * 0.7, n_c[i], n_s[i]

#To make things "simpler", we do a further restriction above n_t = 0
#n_t could be 0 for a range of M for some HOD models (eg. if there is an M_0)
#NOTE: we don't care if n_s/n_c are zero, only n_t, because if n_t = 0, then
#n_c MUST be zero whether we require a central or not. The only time n_s appears
# on its own is in the case of non-central requirement, in which case, n_s also
#must be zero.
m = m[n_t > 0]
dndm = dndm[n_t > 0]
n_c = n_c[n_t > 0]
step_cut = len(m) < len(n_s)  #bool to tell whether we've cut
n_s = n_s[n_t > 0]
bias = bias[n_t > 0]
n_t = n_t[n_t > 0]

ng = h.mean_gal_den()

#Now we find the lower value of our integration. Generally we just set it to 0 (or -infinity in log space)
#However, if the integrand is 0 below a certain value of M, then we use that value rather.
if step_cut:
    m_min = np.log(m[0])
else:
    m_min = 5

M_new, dlogM = np.linspace(m_min, 60, 2000, retstep=True)

dm_corr = tools.power_to_corr(h.power, h.lnk, h.r)

print "min/max/len m: ", np.log(m[0]), np.log(m[-1]), len(m)
#===============================================================================
# NOW START THE DIFFERENT METHODS.
#
# For each, we want as many bits as we can get to crosscheck.
#===============================================================================

#===============================================================================
# CHARLES METHOD (no power steps)
#===============================================================================
charles_corr_1h = np.zeros(len(h.r))
charles_corr_2h = np.zeros(len(h.r))
charles_corr = np.zeros(len(h.r))

#here we follow charles and don't put a bottom limit of mvir on the integral
for i, r in enumerate(h.r):
    #GET 1-halo
    rho = h.profile.rho(r, m, h.z)
    lam = h.profile.lam(r, m , h.z)

    if h.central:
        integrand = n_c * (2 * n_s * rho + n_s * n_s * lam / m ** 2) * dndm
    else:
        integrand = (n_c * 2 * n_s * rho + n_s * n_s * lam / m ** 2) * dndm


    #Unfortunately, n_t>0 doesn't imply n_s > 0 , in which case 1-halo will be zero, so adjust...

    m_split = m[integrand > 0]
    integrand = integrand[integrand > 0]

#    if i == 10:
##        #plt.plot(m_split, integrand, label=str(r))
#        for j, ms in enumerate(m_split):
#            print ms * 0.7, integrand[j], rho[j], lam[j], n_c[j], n_s[j], dndm[j]
    if len(m_split) == len(m):
        #integ = spline(np.log(m_split), np.log(integrand), k=1)

        #integrand = integ(M_new)
        #cumint = intg.cumtrapz(np.exp(integrand + M_new), dx=dlogM)
        cumint = intg.cumtrapz(integrand, m_split)
        charles_corr_1h[i] = cumint[-1] / ng ** 2

    elif len(m_split) > 5:
 #       print "r=", r, ", len(m_split),min,max: ", len(m_split), m_split[0], m_split[-1]
#        integ = spline(np.log(m_split), np.log(integrand), k=1)
#        mnew, dlogm = np.linspace(m_split[0], m_split[-1], 500, retstep=True)
 #       integrand = integ(mnew)
  #      cumint = intg.cumtrapz(np.exp(integrand + mnew), dx=dlogm)
        cumint = intg.cumtrapz(integrand, m_split)
        charles_corr_1h[i] = cumint[-1] / ng ** 2

    else:
 #       print "r=", r, " empty"
        charles_corr_1h[i] = 0.0

#    if i == 0:
#        print len(cumint), len(m_split)
#        for j, ms in enumerate(m_split[1:]):
#            print ms / 0.7, cumint[j]

#GET 2-halo term
#First get power
power_charles_2h = np.zeros_like(h.lnk)
for i, lnk in enumerate(h.lnk):
    u = h.profile.u(np.exp(lnk), m , h.z)
    integrand = n_c * (1 + n_s * u) * dndm * bias

    cumint = intg.cumtrapz(integrand, m)
    power_charles_2h[i] = cumint[-1] / ng

#Now get correlation
power = np.exp(h.power) * power_charles_2h ** 2
charles_corr_2h = tools.power_to_corr(np.log(power), h.lnk, h.r)

#
#integrand = n_t * bias * dndm * rho
#m_split = m[integrand > 0]
#integrand = integrand[integrand > 0]
#if len(m_split) > 1:
#    integ = spline(np.log(m_split), np.log(integrand), k=1)
#    integrand = integ(M_new)
#    charles_corr_2h[i] = intg.simps(np.exp(integrand + M_new), dx=dlogM)
#else:
#    charles_corr_2h[i] = 0.0

#plt.xscale('log')
#plt.yscale('log')
#plt.legend()
#plt.savefig("integrands.pdf")
#plt.clf()

print "Gal Den: ", ng
print "1-h: ", charles_corr_1h
print "2-h: ", charles_corr_2h
#charles_corr_2h *= charles_corr_2h * dm_corr / ng ** 2
charles_corr = charles_corr_1h + charles_corr_2h


# Make plots
plt.plot(h.r, charles_corr_1h, label="Charles 1h")
plt.plot(h.r, charles_corr_2h, label="Charles 2h")
plt.plot(h.r, charles_corr, label="Charles Corr")
plt.plot(h.r, dm_corr, label="DM corr")
plt.legend(loc=0)
plt.yscale('log')
plt.xlabel("R")
plt.ylabel(r"$\xi(r)$")
plt.savefig("/Users/Steven/Documents/HOD_testplots/charles_corr.pdf")
plt.clf()

#Make plots of rho/lam to check if its doing what it should.
plt.plot(h.r, h.profile.rho(h.r, 10 ** 13, 0.0))
plt.yscale('log')
plt.savefig("/Users/Steven/Documents/HOD_testplots/rho.pdf")
plt.clf()

plt.plot(h.r, h.profile.lam(h.r, 10 ** 13, 0.0))
plt.yscale('log')
plt.savefig("/Users/Steven/Documents/HOD_testplots/lam.pdf")
plt.clf()
