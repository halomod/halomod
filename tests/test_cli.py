from click.testing import CliRunner
from halomod._cli import run


def test_cli(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tmp")

    runner = CliRunner()
    result = runner.invoke(run, ["--outdir", str(tmp), "--", "--transfer_model", "EH"])
    assert result.exit_code == 0
