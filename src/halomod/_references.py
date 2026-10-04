"""Citation strings for halomod and its component models.

Each component model lists the papers to cite when it is used in a ``references``
class attribute, which hmf's ``Framework.get_acknowledgments`` collects (hmf>=3.7).
Keeping each string in one place makes sure that references used by more than one
component are deduplicated by ``get_acknowledgments(flat=True)``.

The strings follow the format hmf uses: ``"Authors, YEAR. Journal Volume, Page. URL"``.
"""

from __future__ import annotations

from collections.abc import Callable

#: The reference for halomod itself, always included by
#: :meth:`halomod.DMHaloModel.get_acknowledgments`.
HALOMOD = "Murray, S. G., Diemer, B., Chen, Z., et al., 2020. arXiv:2009.14066"

# ---------------------------------------------------------------------------------------
# Bias
# ---------------------------------------------------------------------------------------
MO96 = (
    "Mo, H. J., White, S. D. M., 1996. MNRAS 282, 347. "
    "https://ui.adsabs.harvard.edu/abs/1996MNRAS.282..347M"
)
JING98 = "Jing, Y. P., 1998. ApJ 503, L9. http://adsabs.harvard.edu/abs/1998ApJ...503L...9J"
ST99 = (
    "Sheth, R. K., Tormen, G., 1999. MNRAS 308, 119. "
    "https://ui.adsabs.harvard.edu/abs/1999MNRAS.308..119S"
)
SMT01 = (
    "Sheth, R. K., Mo, H. J., Tormen, G., 2001. MNRAS 323, 1. "
    "https://ui.adsabs.harvard.edu/abs/2001MNRAS.323....1S"
)
SELJAK04 = (
    "Seljak, U., Warren, M. S., 2004. MNRAS 355, 129. "
    "https://ui.adsabs.harvard.edu/abs/2004MNRAS.355..129S"
)
TINKER05 = "Tinker, J. L., et al., 2005. ApJ 631, 41. https://ui.adsabs.harvard.edu/abs/2005ApJ...631...41T"
MANDELBAUM05 = (
    "Mandelbaum, R., et al., 2005. MNRAS 362, 1451. "
    "https://ui.adsabs.harvard.edu/abs/2005MNRAS.362.1451M"
)
PILLEPICH10 = (
    "Pillepich, A., Porciani, C., Hahn, O., 2010. MNRAS 402, 191. "
    "https://ui.adsabs.harvard.edu/abs/2010MNRAS.402..191P"
)
MANERA10 = (
    "Manera, M., Sheth, R. K., Scoccimarro, R., 2010. MNRAS 402, 589. "
    "https://ui.adsabs.harvard.edu/abs/2010MNRAS.402..589M"
)
TINKER10 = (
    "Tinker, J. L., et al., 2010. ApJ 724, 878. "
    "https://ui.adsabs.harvard.edu/abs/2010ApJ...724..878T"
)

# ---------------------------------------------------------------------------------------
# Concentration-mass relations
# ---------------------------------------------------------------------------------------
BULLOCK01 = (
    "Bullock, J. S., et al., 2001. MNRAS 321, 559. "
    "https://ui.adsabs.harvard.edu/abs/2001MNRAS.321..559B"
)
MACCIO07 = (
    "Macciò, A. V., et al., 2007. MNRAS 378, 55. "
    "https://ui.adsabs.harvard.edu/abs/2007MNRAS.378...55M"
)
PADMANABHAN17 = (
    "Padmanabhan, H., et al., 2017. MNRAS 469, 2323. "
    "https://ui.adsabs.harvard.edu/abs/2017MNRAS.469.2323P"
)
DUFFY08 = (
    "Duffy, A. R., et al., 2008. MNRAS 390, L64. "
    "https://ui.adsabs.harvard.edu/abs/2008MNRAS.390L..64D"
)
ZEHAVI11 = (
    "Zehavi, I., et al., 2011. ApJ 736, 59. https://ui.adsabs.harvard.edu/abs/2011ApJ...736...59Z"
)
LUDLOW16 = (
    "Ludlow, A. D., et al., 2016. MNRAS 460, 1214. "
    "https://ui.adsabs.harvard.edu/abs/2016MNRAS.460.1214L"
)

# ---------------------------------------------------------------------------------------
# Halo profiles
# ---------------------------------------------------------------------------------------
NFW96 = (
    "Navarro, J. F., Frenk, C. S., White, S. D. M., 1996. ApJ 462, 563. "
    "https://ui.adsabs.harvard.edu/abs/1996ApJ...462..563N"
)
NFW97 = (
    "Navarro, J. F., Frenk, C. S., White, S. D. M., 1997. ApJ 490, 493. "
    "https://ui.adsabs.harvard.edu/abs/1997ApJ...490..493N"
)
HERNQUIST90 = (
    "Hernquist, L., 1990. ApJ 356, 359. https://ui.adsabs.harvard.edu/abs/1990ApJ...356..359H"
)
MOORE98 = (
    "Moore, B., et al., 1998. ApJ 499, L5. https://ui.adsabs.harvard.edu/abs/1998ApJ...499L...5M"
)
MOORE99 = (
    "Moore, B., et al., 1999. MNRAS 310, 1147. "
    "https://ui.adsabs.harvard.edu/abs/1999MNRAS.310.1147M"
)
ZHAO96 = "Zhao, H., 1996. MNRAS 278, 488. https://ui.adsabs.harvard.edu/abs/1996MNRAS.278..488Z"
EINASTO65 = "Einasto, J., 1965. Trudy Inst. Astrofiz. Alma-Ata 5, 87."
MALLER04 = (
    "Maller, A. H., Bullock, J. S., 2004. MNRAS 355, 694. "
    "https://ui.adsabs.harvard.edu/abs/2004MNRAS.355..694M"
)
SPINELLI20 = (
    "Spinelli, M., et al., 2020. MNRAS 493, 5434. "
    "https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.5434S"
)

# ---------------------------------------------------------------------------------------
# HODs
# ---------------------------------------------------------------------------------------
ZEHAVI05 = (
    "Zehavi, I., et al., 2005. ApJ 630, 1. https://ui.adsabs.harvard.edu/abs/2005ApJ...630....1Z"
)
ZHENG05 = (
    "Zheng, Z., et al., 2005. ApJ 633, 791. https://ui.adsabs.harvard.edu/abs/2005ApJ...633..791Z"
)
CONTRERAS13 = (
    "Contreras, S., et al., 2013. MNRAS 432, 2717. "
    "https://ui.adsabs.harvard.edu/abs/2013MNRAS.432.2717C"
)
GEACH12 = (
    "Geach, J. E., et al., 2012. MNRAS 426, 679. "
    "https://ui.adsabs.harvard.edu/abs/2012MNRAS.426..679G"
)
LEAUTHAUD11_FRAMEWORK = (
    "Leauthaud, A., Tinker, J., Behroozi, P. S., Busha, M. T., Wechsler, R. H., 2011. "
    "arXiv:1103.2077"
)
LEAUTHAUD11_COSMOS = "Leauthaud, A., et al., 2011. arXiv:1104.0928"
BEHROOZI10 = (
    "Behroozi, P. S., Conroy, C., Wechsler, R. H., 2010. ApJ 717, 379. "
    "https://ui.adsabs.harvard.edu/abs/2010ApJ...717..379B"
)


def with_halomod_reference(refs: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    """Return a copy of ``refs`` with a ``"halomod"`` entry right after ``"hmf"``.

    Parameters
    ----------
    refs
        References grouped by source, as returned by hmf's
        ``Framework.get_acknowledgments(flat=False)``.

    Returns
    -------
    dict
        The same groups, with ``"halomod": (HALOMOD,)`` inserted after the ``"hmf"``
        entry (or first, if there is no ``"hmf"`` entry).
    """
    out = {}
    if "hmf" in refs:
        out["hmf"] = refs["hmf"]
    out["halomod"] = (HALOMOD,)
    out.update((k, v) for k, v in refs.items() if k not in out)
    return out


def get_acknowledgments(
    parent: Callable[..., dict[str, tuple[str, ...]]], flat: bool
) -> dict[str, tuple[str, ...]] | list[str]:
    """Add halomod's own reference to the acknowledgments of a framework.

    Parameters
    ----------
    parent
        The bound ``get_acknowledgments`` method of the framework's hmf parent class.
    flat
        Whether to return a flat, deduplicated list rather than a dict.

    Returns
    -------
    dict or list
        See ``get_acknowledgments`` of :class:`halomod.DMHaloModel`.
    """
    refs = with_halomod_reference(parent(flat=False))
    return flatten(refs) if flat else refs


def flatten(refs: dict[str, tuple[str, ...]]) -> list[str]:
    """Flatten grouped references into one list, removing duplicates.

    Parameters
    ----------
    refs
        References grouped by source.

    Returns
    -------
    list of str
        All references, keeping the first occurrence of each.
    """
    return list(dict.fromkeys(ref for group in refs.values() for ref in group))
