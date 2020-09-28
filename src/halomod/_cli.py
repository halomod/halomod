"""Module that contains the command line app."""
from hmf._cli import run_cli
import click
from .halo_model import TracerHaloModel
import hmf
import halomod

main = click.Group()


@main.command(
    context_settings={  # Doing this allows arbitrary options to override config
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
@click.option(
    "-i", "--config", type=click.Path(exists=True, dir_okay=False), default=None,
)
@click.option(
    "-o",
    "--outdir",
    type=click.Path(exists=True, dir_okay=True, file_okay=True),
    default=".",
)
@click.option(
    "-l", "--label", type=str, default="halomod",
)
@click.pass_context
def run(ctx, config, outdir, label):
    run_cli(config, "halomod", ctx.args, outdir, label, [halomod, hmf], TracerHaloModel)
