"""A quick script to print out some intermediate results testing the Ludlow16 concentration
model vs. the Sigma_cMz_Py code, originally written by Ludlow et al.
"""
from halomod import concentration as cm
import numpy as np
from hmf import MassFunction
import matplotlib.pyplot as plt

# Use the actual power spectrum used in the Sigma_cMz_Py code.
# Not using this actual power spectrum results in fairly significant differences in sigma(m).
ludlow_pk, ludlow_k = np.genfromtxt(
    "../../Sigma_cMz_Py/PowerSpectra/Planck_camb_matterpower_z0_extrapolated.dat",
    skip_header=5,
    unpack=True,
)

mf = MassFunction(
    sigma_8=0.815,
    n=0.9677,
    cosmo_params={"Ob0": 0.0484, "Om0": 0.308, "H0": 67.8},
    transfer_model="FromArray",
    transfer_params={
        "k": ludlow_k[ludlow_pk > 0],
        "T": np.sqrt(ludlow_pk / ludlow_k ** 0.9677)[ludlow_pk > 0],
    },
)


plt.plot(ludlow_k, ludlow_pk)
plt.plot(mf.k, mf.power)
plt.xscale("log")
plt.yscale("log")
plt.savefig("my_power_vs_ludlow.pdf")

l16 = cm.Ludlow16(filter0=mf.normalised_filter)

m = np.logspace(10, 15, 100)

print(l16.cm(m))
