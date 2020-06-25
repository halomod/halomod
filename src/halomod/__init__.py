try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass

from . import halo_model
from . import bias
from . import concentration
from . import cross_correlations
from . import functional
from . import halo_exclusion
from . import hod
from . import integrate_corr
from . import profiles
from . import tools
from . import wdm

from .halo_model import HaloModel, DMHaloModel, TracerHaloModel
from .integrate_corr import ProjectedCF, AngularCF, projected_corr_gal
