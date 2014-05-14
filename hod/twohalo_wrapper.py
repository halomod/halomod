from fort.thalo import haloexclusion as thalo
import numpy as np
def twohalo_wrapper(excl, sdb, m, bias, ntot, dndm, lnk, dmpower, u, r, dmcorr,
                    nbar, dhalo, rhob):
    u = np.asfortranarray(u.T)

    # # Mask all arrays by ntot
    m = m[ntot > 0]
    bias = bias[ntot > 0]
    dndm = dndm[ntot > 0]
    u = u[ntot > 0, :]
    ntot = ntot[ntot > 0]

    corr = np.zeros(len(r))

    exc_type = {"None":1,
                "schneider":2,
                "sphere":3,
                "ng_matched":4,
                "ellipsoid":5}

    corr = thalo.twohalo(m, bias, ntot, dndm, lnk, dmpower, u, r, dmcorr, nbar,
                         dhalo, rhob, exc_type[excl], sdb)

    return corr
#     if excl == "None" and not sdb:
#         corr = thalo.noexclusion(m, bias, ntot, dndm, lnk, dmpower, u,
#                                  r, nbar)
#
#     elif excl == "None" and sdb:
#         corr = thalo.noexclusion_sd(m, bias, ntot, dndm, lnk, dmpower, u,
#                                  r, dmcorr, nbar)
#     elif excl == "schneider" and not sdb:
#         corr = thalo.schneider(m, bias, ntot, dndm, lnk, dmpower, u,
#                                r, nbar)
#
#     elif excl == "schneider" and sdb:
#         corr = thalo.schneider_sd(m, bias, ntot, dndm, lnk, dmpower, u,
#                                  r, dmcorr, nbar)
#
#     elif excl == "sphere" and not sdb:
#         corr = thalo.sphere(m, bias, ntot, dndm, lnk, dmpower, u,
#                             r, nbar, dhalo, rhob)
#
#     elif excl == "sphere" and sdb:
#         corr = thalo.sphere_sd(m, bias, ntot, dndm, lnk, dmpower, u,
#                                r, dmcorr, nbar, dhalo, rhob)
#
#     elif excl == "ng_matched" and not sdb:
#         corr = thalo.ng_matched(m, bias, ntot, dndm, lnk, dmpower, u,
#                                 r, nbar, dhalo, rhob)
#
#     elif excl == "ng_matched" and sdb:
#         corr = thalo.ng_matched_sd(m, bias, ntot, dndm, lnk, dmpower, u,
#                                    r, dmcorr, nbar, dhalo, rhob)
#
#     elif excl == "ellipsoid" and not sdb:
#         corr = thalo.ellipsoid(m, bias, ntot, dndm, lnk, dmpower, u,
#                                 r, nbar, dhalo, rhob)
#
#     elif excl == "ellipsoid" and sdb:
#         corr = thalo.ellipsoid_sd(m, bias, ntot, dndm, lnk, dmpower, u,
#                                    r, dmcorr, nbar, dhalo, rhob)
#
#     else:
#         raise ValueError("Halo Exclusion doesn't exist")
#     return corr
