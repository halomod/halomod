'''
This script explores the projected differences between using an EH and CAMB
transfer function. 

For standard parameters, the difference seems to be about 20% max.
'''

from hod import HOD
import time
import numpy as np

omegab = [0.02, 0.05]  # , 0.1]
omegac = [0.2, 0.25]  # , 0.5]
H0 = [50, 70]  # , 90]
n = [0.8, 0.9, 1.0]

r = np.exp(np.linspace(-4, 4, 100))
print("r range", np.min(r), np.max(r))
HOD_camb = HOD(r=r, transfer_fit="CAMB", NonLinear=False, scale_dependent_bias=False, force_flat=True)
HOD_EH = HOD(r=r, transfer_fit="EH", NonLinear=False, scale_dependent_bias=False, force_flat=True)
camb_time = 0.0
eh_time = 0.0
for ob in omegab:
    for oc in omegac:
        for h in H0:
            for nn in n:
                HOD_camb.update(omegab=ob, omegac=oc, H0=h, n=nn)
                HOD_EH.update(omegab=ob, omegac=oc, H0=h, n=nn)

                start = time.time()
                camb = HOD_camb.corr_gal
                camb_time += time.time() - start

                start = time.time()
                eh = HOD_EH.corr_gal
                eh_time += time.time() - start

                print("For ", ob, oc, h, nn)
                print(camb[0] / eh[0], camb[50] / eh[50], camb[-1] / eh[-1], eh_time / camb_time)
