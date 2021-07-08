"""Module that contains the command line app."""
import click

import halomod
import hmf
from hmf._cli import run_cli

from .halo_model import TracerHaloModel

main = click.Group()


@main.command(
    context_settings={  # Doing this allows arbitrary options to override config
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
@click.option(
    "-i",
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "-o",
    "--outdir",
    type=click.Path(exists=True, dir_okay=True, file_okay=True),
    default=".",
)
@click.option(
    "-l",
    "--label",
    type=str,
    default="halomod",
)
@click.pass_context
def run(ctx, config, outdir, label):
    """Run halomod for a particular configuration."""
    run_cli(config, "halomod", ctx.args, outdir, label, [halomod, hmf], TracerHaloModel)
