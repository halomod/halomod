import matplotlib.pyplot as plt
from matplotlib.style import use
from halomod import TracerHaloModel

use("seaborn-paper")

tr = TracerHaloModel()
plt.figure(figsize=(4, 2.8))
plt.loglog(tr.r, tr.corr_auto_tracer, label="z=0")
tr.z = 1
plt.loglog(tr.r, tr.corr_auto_tracer, label="z=1")
tr.hod_params = {"M_min": 8}
plt.loglog(tr.r, tr.corr_auto_tracer, label="Mmin=8")
plt.legend()
plt.xlabel("r [Mpc/h]")
plt.ylabel(r"$\xi_{gg}(r)$")
plt.tight_layout()
plt.savefig("/home/steven/Desktop/toy_example.pdf")
