import numpy as np
import pytest
from click.testing import CliRunner

from halomod._cli import run


def test_cli(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tmp")

    runner = CliRunner()
    result = runner.invoke(run, ["--outdir", str(tmp), "--", "--transfer_model", "EH"])
    assert result.exit_code == 0


@pytest.mark.filterwarnings("ignore:MassFunction.nu is:DeprecationWarning")
def test_cli_import_path_model_in_toml(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        'quantities = ["m", "nu2", "halo_bias"]\n'
        "[params]\n"
        'transfer_model = "EH"\n'
        'bias_model = "halomod.bias:Mo96"\n'
        'halo_profile_model = "halomod.profiles:Einasto"\n'
    )

    runner = CliRunner()
    result = runner.invoke(run, ["--config", str(cfg), "--outdir", str(tmp_path)])
    assert result.exit_code == 0, result.output

    nu = np.loadtxt(tmp_path / "halomod_nu2.txt")
    bias = np.loadtxt(tmp_path / "halomod_halo_bias.txt")

    # The Mo & White (1996) peak-background-split bias, b = 1 + (nu - 1) / delta_c,
    # with nu = (delta_c / sigma)^2 and the default (Einstein-de Sitter) critical
    # overdensity delta_c = 3/20 (12 pi)^(2/3). In particular, it is unity at nu = 1,
    # unlike the default Tinker10 bias.
    delta_c = 3 / 20 * (12 * np.pi) ** (2 / 3)
    np.testing.assert_allclose(bias, 1 + (nu - 1) / delta_c, rtol=1e-6)
    assert np.any(nu < 1)
    assert np.any(nu > 1)
    assert np.all(bias[nu < 1] < 1)
    assert np.all(bias[nu > 1] > 1)
