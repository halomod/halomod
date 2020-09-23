from halomod.wdm import HaloModelWDM
from halomod import DMHaloModel
import numpy as np


def test_cmz_wdm():
    wdm = HaloModelWDM(
        hmf_model="SMT",
        z=0,
        hmf_params={"a": 1},
        filter_model="SharpK",
        filter_params={"c": 2.5},
        halo_concentration_model="Duffy08WDM",
        wdm_mass=3.3,
        Mmin=7.0,
        transfer_model="EH",
    )
    cdm = DMHaloModel(
        hmf_model="SMT",
        z=0,
        hmf_params={"a": 1},
        filter_model="SharpK",
        filter_params={"c": 2.5},
        halo_concentration_model="Duffy08",
        Mmin=7.0,
        transfer_model="EH",
    )

    assert np.all(
        cdm.cmz_relation[cdm.m <= wdm.wdm.m_hm]
        > wdm.cmz_relation[wdm.m <= wdm.wdm.m_hm]
    )


def test_ludlow_cmz_wdm():
    wdm = HaloModelWDM(
        hmf_model="SMT",
        z=0,
        hmf_params={"a": 1},
        filter_model="TopHat",
        mdef_model="SOCritical",
        halo_concentration_model="Ludlow16",
        halo_profile_model="Einasto",
        wdm_mass=3.3,
        Mmin=7.0,
        transfer_model="EH",
    )
    cdm = DMHaloModel(
        hmf_model="SMT",
        z=0,
        hmf_params={"a": 1},
        filter_model="TopHat",
        halo_concentration_model="Ludlow16",
        halo_profile_model="Einasto",
        Mmin=7.0,
        mdef_model="SOCritical",
        transfer_model="EH",
    )

    assert np.all(
        cdm.cmz_relation[cdm.m <= wdm.wdm.m_hm]
        > wdm.cmz_relation[wdm.m <= wdm.wdm.m_hm]
    )
