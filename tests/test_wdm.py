from halomod.wdm import HaloModelWDM
from halomod import DMHaloModel


def test_halo_model_wdm():
    hm_cdm = DMHaloModel(
        halo_concentration_model="Bullock01Power", mdef_model="SOCritical"
    )
    hm_wdm = HaloModelWDM(
        halo_concentration_model="Bullock01Power", mdef_model="SOCritical", wdm_mass=0.1
    )

    assert hm_cdm.power_auto_matter[-1] > hm_wdm.power_auto_matter[-1]
