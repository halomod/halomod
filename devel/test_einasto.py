import halomod
import numpy as np

model = halomod.TracerHaloModel(
    z=0.2,
    transfer_model="EH",
    rnum=30,
    rmin=0.1,
    rmax=30,
    hod_model="Zehavi05",
    hod_params={"M_min": 12.0, "M_1": 12.8, "alpha": 1.05},
    dr_table=0.1,
    dlnk=0.1,
    dlog10m=0.05,
)

model_ein = model.clone()
model_ein.halo_profile_model = "Einasto"

rhos = []
corrs = []
for a in np.linspace(0.08, 0.4, 20):
    model_ein.update(halo_profile_params={"alpha": a})

    rhos.append(model_ein.halo_profile_rho[:, 200])
    corrs.append(model_ein.corr_auto_tracer)

print(rhos)
print(corrs)
