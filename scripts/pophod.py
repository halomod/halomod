#!/usr/local/bin/python2.7
# encoding: utf-8
'''
scripts.pophod -- shortdesc

scripts.pophod is a description

It defines classes_and_methods
'''

import sys
import os
import traceback
import numpy as np
from hod import hod
from hod import profiles
from scipy.stats import poisson
import ast
import read_halo

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

__all__ = []
__version__ = 0.1
__date__ = "2014 - 01 - 21"
__updated__ = "2014 - 01 - 21"

DEBUG = 0
TESTRUN = 0
PROFILE = 0

class CLIError(Exception):
    '''Generic exception to raise and log different fatal errors.'''
    def __init__(self, msg):
        super(CLIError).__init__(type(self))
        self.msg = "E: %s" % msg
    def __str__(self):
        return self.msg
    def __unicode__(self):
        return self.msg

def main(argv=None):
    '''Process command line options.'''

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    program_name = os.path.basename(sys.argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
  #  program_license = '''%s
#
#   Created by Steven Murray on 21-01-2014.
#   Copyright 2014 organization_name. All rights reserved.
#
#   Licensed under the Apache License 2.0
#   http://www.apache.org/licenses/LICENSE-2.0
#
#   Distributed on an "AS IS" basis without warranties
#   or conditions of any kind, either express or implied.
#
# USAGE
# ''' % (program_shortdesc, str(__date__))

    try:
        # Setup argument parser
        parser = ArgumentParser(description=program_shortdesc, formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument("-v", "--verbose", dest="verbose", action="count", help="set verbosity level [default: %(default)s]")
        parser.add_argument('-V', '--version', action='version', version=program_version_message)
        parser.add_argument("simfile", help="the input FoF catalogue OR the file from Santi")
        parser.add_argument("--subfile", help="The subfind halo file, needed if using FOF")
        parser.add_argument("--idsfile", help="The subfind ids file, needed if using FOF")
        parser.add_argument("outfile", help="the output galaxy catalogue")
        parser.add_argument("hodmod_dict", help="dictionary of HOD parameters")
        parser.add_argument("--hodmod", default="zheng", help="the HOD model used")
        parser.add_argument("--delta_h", default=200.0, type=float, help="the average halo overdensity")
        parser.add_argument("--profile", default="NFW", help="the halo profile to use")
        parser.add_argument("--cm-relation", default="zehavi", help="the Concentration-Mass relation")
        parser.add_argument("--no-truncate", action="store_true", help="don't truncate profile")
        parser.add_argument("--omegam", type=float, default=0.3, help="the matter density")
        parser.add_argument("--redshift", type=float, default=0.0, help="the redshift")

        # Process arguments
        args = parser.parse_args()

        if args.subfile is not None:
            s, halos = read_halo.load(args.simfile, halofinder="subfind",
                                 subfile=args.subfile, idsfile=args.idsfile,
                                 normalise_halos=False, get_halo_particles=True,
                                 get_vel=False, get_inertia=False)

            centres = [h.properties["centre"] for h in halos]
            masses = [h.properties["mp"] for h in halos]
            omegam = s.properties["omegam"]
            z = s.properties['redshift']
        else:
            centres, masses = read(args.simfile)
            omegam = args.omegam
            z = args.redshift

        hodparams = ast.literal_eval(args.hodmod_dict)
        hodmod = hod.HOD(args.hodmod, **hodparams)

        pos = populate(centres, masses, args.delta_h, omegam, z,
                       profile=args.profile, cm_relation=args.cm_relation,
                       hodmod=hodmod, truncate=not args.no_truncate)

        np.savetxt(args.outfile, pos)

        return 0
    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    except Exception, e:
        if DEBUG or TESTRUN:
            raise(e)
        traceback.print_exc()
        indent = len(program_name) * " "
        sys.stderr.write(program_name + ": " + repr(e) + "\n")
        sys.stderr.write(indent + "  for help use --help\n")
        return 2


#===============================================================================
# FUNCTION CALLS
#===============================================================================
def read(filename):
    x = np.genfromtxt(filename)
    centres = [x[i, :3] for i in range(len(x[:, 0]))]
    masses = x[:, 3] * 5.58e11
    print centres.shape, masses.shape
    return centres, masses

def populate(centres, masses, delta_halo, omegam, z, profile, cm_relation, hodmod,
             truncate):

    # Define which halos have central galaxies.
    cgal = np.zeros_like(masses)
    masses = np.array(masses)
    cgal[np.random.rand() < hodmod.nc(masses)] = 1.0
    # Calculate the number of satellite galaxies in halos
    sgal = np.zeros_like(masses)
    for i, m in enumerate(masses):
        if cgal[i] == 1.0:
            sgal[i] = poisson.rvs(hodmod.ns(m))

    # Now go through each halo and calculate galaxy positions
    for i, m in enumerate(masses):
        if cgal[i] > 0 and sgal[i] > 0:
            prof = profiles.get_profile(profile, omegam, delta_halo, cm_relation, truncate)
            pos = np.concatenate((prof.populate(sgal[i], m, ba=1, ca=1, z=z) + centres[i], np.atleast_2d(centres[i])))
        elif cgal[i] == 1:
            pos = np.atleast_2d(centres[i])

        else:
            continue

        try:
            allpos = np.concatenate((pos, allpos))
        except UnboundLocalError:
            allpos = pos
    return allpos

if __name__ == "__main__":
    if DEBUG:
        sys.argv.append("-h")
        sys.argv.append("-v")
    if TESTRUN:
        import doctest
        doctest.testmod()
    if PROFILE:
        import cProfile
        import pstats
        profile_filename = 'scripts.pophod_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = open("profile_stats.txt", "wb")
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    sys.exit(main())

