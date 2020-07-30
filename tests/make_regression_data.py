"""
Make regression-testing data (this data is tested in `test_regression.py` -- see that
module for more information).
"""
from halomod import TracerHaloModel
from pathlib import Path
import numpy as np
from hashlib import md5
import sys

base_options = {
    "Mmin": 0,
    "Mmax": 18,
    "rmin": 0.1,
    "rmax": 50.0,
    "rnum": 20,
    "transfer_model": "EH",
    "dr_table": 0.1,
    "dlnk": 0.1,
    "dlog10m": 0.1,
}

tested_params = (
    [
        0,
        {
            "halo_profile_model": "NFW",
            "halo_concentration_model": "Duffy08",
            "bias_model": "Tinker10",
            "sd_bias_model": "TinkerSD05",
            "exclusion_model": "NgMatched_",
            "hc_spectrum": "nonlinear",
        },
    ],
    [
        1,
        {
            "halo_profile_model": "Einasto",
            "halo_concentration_model": "Bullock01Power",
            "bias_model": "Mo96",
            "sd_bias_model": "TinkerSD05",
            "exclusion_model": "Sphere",
            "hc_spectrum": "nonlinear",
        },
    ],
    [
        6,
        {
            "halo_profile_model": "CoredNFW",
            "halo_concentration_model": "Ludlow16",
            "bias_model": "Manera10",
            "sd_bias_model": None,
            "exclusion_model": "NoExclusion",
            "hc_spectrum": "filtered-lin",
        },
    ],
    [
        2,
        {
            "halo_profile_model": "NFW",
            "halo_concentration_model": "Ludlow16",
            "bias_model": "Tinker10PBSplit",
            "sd_bias_model": None,
            "exclusion_model": "NoExclusion",
            "hc_spectrum": "filtered-nl",
        },
    ],
)

quantities = [
    "r",
    "halo_bias",
    "cmz_relation",
    "corr_linear_mm",
    "corr_halofit_mm",
    "halo_profile_ukm",
    "halo_profile_rho",
    "halo_profile_lam",
    "power_1h_auto_matter",
    "corr_1h_auto_matter",
    "power_2h_auto_matter",
    "corr_2h_auto_matter",
    "corr_auto_matter",
    "power_auto_matter",
    "bias_effective_matter",
    "tracer_cmz_relation",
    "total_occupation",
    "satellite_occupation",
    "central_occupation",
    "mean_tracer_den",
    "bias_effective_tracer",
    "mass_effective",
    "satellite_fraction",
    "central_fraction",
    "power_1h_ss_auto_tracer",
    "corr_1h_ss_auto_tracer",
    "power_1h_cs_auto_tracer",
    "corr_1h_cs_auto_tracer",
    "power_1h_auto_tracer",
    "corr_1h_auto_tracer",
    "corr_2h_auto_tracer",  # 'power_auto_tracer',
    "corr_auto_tracer",
    "power_1h_cross_tracer_matter",
    "corr_1h_cross_tracer_matter",
    "power_2h_cross_tracer_matter",
    "corr_2h_cross_tracer_matter",
    "power_cross_tracer_matter",
    "corr_cross_tracer_matter",
]


def get_hash(z, params):
    return md5((str(z) + str(params)).encode()).hexdigest()


datadir = Path(__file__).parent / "data/regression"


def compress_data(data):
    if not np.isscalar(data):
        for i in range(data.ndim):
            slc = np.arange(data.shape[i], data.shape[i] // 5)
            data = np.take(data, slc, axis=i)
    return data


if __name__ == "__main__":
    import yaml

    # Read the command line instructions
    force = "--force" in sys.argv
    if "--param-sets" in sys.argv:
        param_sets = [
            int(x) for x in sys.argv[sys.argv.index("--param-sets") + 1].split(",")
        ]
    else:
        param_sets = range(len(tested_params))

    tr = TracerHaloModel(**base_options)

    if not datadir.exists():
        datadir.mkdir(parents=True)

    for indx in param_sets:

        z, params = tested_params[indx]
        print(f"Doing redshift {z} and params: ")
        print("\t" + "\t".join(f"{k}: {v}\n" for k, v in params.items()))

        tr.update(z=z, **params)

        hsh = get_hash(z, params)
        param_dir = datadir / hsh

        if not param_dir.exists():
            param_dir.mkdir()

        for quantity in quantities:
            fname = param_dir / quantity
            data = getattr(tr, quantity)

            # Quantities such as halo_profile_lam can be None for some models.
            if data is None:
                continue

            # We minimize the data to make things a lot smaller (i.e. take every 20th
            # value). We can't just initialize the hm like this because we might get
            # resolution issues.
            data = compress_data(data)

            # If data exists already, and is within 1%, we don't update it (to save
            # on making new git files).
            if fname.exists() and not force:
                prev_data = np.load(fname)
                if np.allclose(prev_data, data, rtol=1e-2):
                    continue

            np.save(fname, data)

        # Make a little readme in the directory
        with open(param_dir / "params.yaml", "w") as fl:
            params.update(z=z)
            yaml.dump(params, fl)
