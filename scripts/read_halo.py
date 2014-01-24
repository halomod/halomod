'''
Created on May 22, 2013

@author: Steven
'''
# import pynbody as pn
import numpy as np
import os
import glob
import re



def _read_gadget_binary(filename):
    """
    Reads a gadget binary specified by filename
    """
    positions_dtype = np.dtype([("x", 'f'),
                                ("y", 'f'),
                                ("z", 'f')
                                ])

    header_dtype = np.dtype([('npart', ('i', 6)),
                             ('massarr', ('d', 6)),
                             ('time', 'd'),
                             ('z', 'd'),
                             ('FlagSfr', 'i'),
                             ('FlagFeedback', 'i'),
                             ('n_all', ('i', 6)),
                             ('FlagCooling', 'i'),
                             ('num_files', 'i'),
                             ('boxsize', 'd'),
                             ('omegam', 'd'),
                             ('omegav', 'd'),
                             ('h', 'd'),
                             ('FlagAge', 'i'),
                             ('FlagMetals', 'i'),
                             ('n_all_HW', ('i', 6)),
                             ('flag_entr_ics', 'i')
                             ])

    with open(filename, "rb") as f:
        # Header block
        f.read(4)
        header = np.fromfile(f, dtype=header_dtype, count=1)
        # So far only 196 out of 256 so read the rest
        f.read(256 - 196)
        f.read(4)

        # Positions Block
        f.read(4)
        pos = np.fromfile(f, dtype=positions_dtype, count=header['npart'][0][1])
        f.read(4)

        # Velocities Block
        f.read(4)
        vel = np.fromfile(f, dtype=positions_dtype, count=header['npart'][0][1])
        f.read(4)

        # ID's block
        f.read(4)
        ids = np.fromfile(f, dtype='i', count=header['npart'][0][1])


    ids = np.array(ids - 1)

    indices = np.argsort(ids)
    pos = pos[indices]
    vel = vel[indices]
    ids = sorted(ids)

    header_dict = {}
    for name in header.dtype.names:
        header_dict[name] = header[name][0]

    return pos.view(np.float32).reshape(pos.shape + (-1,)) , vel.view(np.float32).reshape(vel.shape + (-1,)) , header_dict, ids

def _run_ahf(filename):
    """
    Runs AHF halofinder on the sim
    """
    # find AHF
    for directory in os.environ["PATH"].split(os.pathsep):
        ahfs = glob.glob(os.path.join(directory, "AHF*"))
        for iahf, ahf in enumerate(ahfs):
            # if there are more AHF*'s than 1, it's not the last one, and
            # it's AHFstep, then continue, otherwise it's OK.
            if ((len(ahfs) > 1) & (iahf != len(ahfs) - 1) &
                    (os.path.basename(ahf) == 'AHFstep')):
                continue
            else:
                groupfinder = ahf
                break


    # make input file
    with open('AHF.in', 'w') as f:

        # Setup some variables for the file.
        lgmax = 131072  # 2^17
        typecode = 60  # For gadget "single" files

        ahf_config = """
[AHF]
ic_filename = %s
ic_filetype = %d
outfile_prefix = %s
LgridDomain = 256
LgridMax = %d
NperDomCell = 5
NperRefCell = 5
VescTune = 1.5
NminPerHalo = 50
RhoVir = 0
Dvir = 200
MaxGatherRad = 10.0
""" % (filename, typecode, filename, lgmax)

        print >> f, ahf_config

        gadget_config = """
[GADGET]
GADGET_LUNIT = 1
GADGET_MUNIT = 1e10
"""
        print >> f, gadget_config

    if os.path.exists(groupfinder):
        # run it
        os.system(groupfinder + " AHF.in")
        return


def _import_ahf_haloes(filename, get_particles=True):

    #=======================================================================
    # Import halo properties
    #=======================================================================
    # If there is no AHFfile, run AHF
    if len(glob.glob(filename + '*z*halos')) == 0:
        _run_ahf()

    ahf_base = glob.glob(filename + '*z*halos')[0][:-5]

    with open(ahf_base + "halos") as f:
        # get all the property names from the first, commented line
        keys = [re.sub('\([0-9]*\)', '', field)
                for field in f.readline().split()]

        halo_props = []
        for h, line in enumerate(f):
            values = [float(x) if '.' in x or 'e' in x or 'nan'
                       in x else int(x) for x in line.split()]

            halo_props.append(values)


    # provide translations
    for i, key in enumerate(keys):
        if(key == '#npart'):
            keys[i] = 'npart'
        if(key == 'a'):
            keys[i] = 'a_axis'
        if(key == 'b'):
            keys[i] = 'b_axis'
        if(key == 'c'):
            keys[i] = 'c_axis'

    # fix for column 0 being a non-column in some versions of the AHF
    # output
    if keys[0] == '#':
        keys = keys[1:]

    # Turn the lists into a list of dictionaries
    properties = []
    for i, val in enumerate(halo_props):
        properties.append({})
        for j, key in enumerate(keys):
            properties[i].update({key:val[j]})

    #=======================================================================
    # Import halo particles
    #=======================================================================
    ids = []
    if get_particles:
        with open(ahf_base + "particles") as f:

            nhalos = int(f.readline())

            for h in xrange(nhalos):
                nparts = int(f.readline().split()[0])

                ids.append((np.fromfile(f, dtype=int,
                                   sep=" ", count=nparts * 2).reshape(nparts, 2))[:, 0])

        return properties, ids

    return properties


def _import_subfind_halos(subfile, idsfile, get_particles=True):
    import struct
    with open(subfile) as f:
        # Get number of halos
        f.read(4)
        nhalos = struct.unpack('i', f.read(4))[0]
        f.read(4)

        # Get halo sizes
        f.read(4)
        grouplen = np.fromfile(f, dtype='i', count=nhalos)
        f.read(4)

        # Get group id offsets
        f.read(4)
        groupoffsets = np.fromfile(f, dtype='i', count=nhalos)
        f.read(4)

    with open(idsfile) as f:
        # Get total number of particles in groups
        f.read(4)
        total_particles_in_groups = struct.unpack('i', f.read(4))[0]
        f.read(4)

        # Get ids of particles in groups
        f.read(4)
        part_ids = np.fromfile(f, dtype='i', count=total_particles_in_groups)

    properties = []

    for h in xrange(nhalos):
        properties.append({'npart':grouplen[h], '#ID':h})

    if get_particles:
        ids = []
        for h in xrange(nhalos):
            ids.append(part_ids[groupoffsets[h]:groupoffsets[h] + grouplen[h]])
        return properties, ids

    return properties


def load(filename, halofinder="AHF", num_halos=None, subfile=None, idsfile=None,
         normalise_halos=True, get_halo_particles=True, get_vel=False,
         get_inertia=True):
    """
    Imports a simulation and its haloes
    
    Returns a simulation object
    """

    sim_pos, sim_vel, sim_header, sim_ids = _read_gadget_binary(filename)

    # If we don't care about velocities, delete them straight off
    if not get_vel:
        del sim_vel

    # We get specific parts of the sim and halo and give them to our Sim object
    # This way we always know what we are dealing with.
    s = Simulation(sim_pos, sim_header["omegam"], sim_header['omegav'],
                   sim_header['boxsize'], sim_header['z'], sim_header['h'])

    if halofinder == "AHF":
        if get_halo_particles:
            properties, ids = _import_ahf_haloes(filename, get_halo_particles)
        else:
            properties = _import_ahf_haloes(filename)
    elif halofinder == "subfind":
        if get_halo_particles:
            properties, ids = _import_subfind_halos(subfile, idsfile, get_halo_particles)
        else:
            properties = _import_subfind_halos(subfile, idsfile)

    halos = []

    for i, halo in enumerate(properties):
        # This breaks out if we have the number of halos requested
        if num_halos is not None:
            if i == num_halos:
                break

        if halofinder == "AHF":
            inertia_tensor = np.array([[halo['Eax'], halo['Ebx'], halo['Ecx']],
                                       [halo['Eay'], halo['Eby'], halo['Ecy']],
                                       [halo['Eaz'], halo['Ebz'], halo['Ecz']],
                                       ])

            ba = halo['b_axis']
            ca = halo['c_axis']
            rvir = halo["Rvir"] / 1000.0
            mvir = halo["Mvir"]
            centre = np.array([halo['Xc'], halo['Yc'], halo['Zc']]) / 1000

        else:
            inertia_tensor = None
            ba = None
            ca = None
            rvir = None
            mvir = halo["npart"] * sim_header["massarr"][1] * 10 ** 10
            centre = None

        if not get_halo_particles:
            normalise_halos = False
            pos = None
        else:
            pos = sim_pos[ids[i]]

        halos.append(BaseHalo(pos, sim_header['omegam'], sim_header["omegav"], sim_header['h'] * 100,
                              sim_header['boxsize'], redshift=sim_header['z'],
                              rvir=rvir, mass=mvir, centre=centre,
                              ba=ba, ca=ca, inertia_tensor=inertia_tensor,
                              centred=False, rotated=False, normalised=False,
                              get_inertia=get_inertia))

        if normalise_halos:
            halos[i].prepare()

    return s, halos

class Simulation(object):
    """
    A simulation with specified properties
    """

    def __init__(self, pos, omegam, omegav, boxsize, z, h):
        self.properties = {"omegam": omegam,
                           "omegav": omegav,
                           "boxsize": boxsize,
                           "redshift": z,
                           "H0": 100 * h}
        self.pos = pos

class BaseHalo(object):
    """
    Base properties that a halo must have
    
    Decided not to use the pynbody object (SimSnap) because I am then
    restricted in making mock halos etc
    """

    def __init__(self, pos, omegam, omegav, H0, boxsize=None, rvir=None, mass=None, redshift=None, centre=None, ba=None, ca=None,
                 inertia_tensor=None, centred=False, rotated=False, normalised=False, get_inertia=True):

        self.pos = pos

        self.properties = {"omegam":omegam,
                           "omegav":omegav,
                           "H0":H0,
                           "boxsize": boxsize,
                           'rvir':rvir,
                           "mp":mass}

        if redshift is not None:
            self.properties.update({'redshift':redshift})
        else:
            self.properties.update({"redshift":None})

        if centre is not None:
            self.properties.update({'centre':centre})
        else:
            self.properties.update({"centre":self.get_centre_of_mass()})

        if ba is not None and ca is not None:
            self.properties.update({"ba":ba, "ca":ca})
        elif get_inertia:
            print pos.shape
            print type(pos)
            ba, ca, eigval, inertia_tensor = calc_ratios(self.pos, self.properties['mp'], tol=0.001)
            self.properties.update({"ba":ba, "ca":ca, "inertia_tensor":inertia_tensor})

        if inertia_tensor is not None:
            self.properties.update({"inertia_tensor":inertia_tensor})

        self.centred = centred
        self.rotated = rotated
        self.normalised = normalised

    def get_centre_of_mass(self):
        return np.mean(self.pos, axis=0)

    def recentre(self):
        """
        Centres the particles on the centre of mass at (0,0,0).
        """

        if not self.centred:
            for coord in range(3):
                if "boxsize" in self.properties:  # If its not we can assume its a mock and will already be centred
                    self.pos[:, coord][self.pos[:, coord] > self.properties["centre"][coord] + self.properties['boxsize'] / 2.0] -= self.properties['boxsize']
                    self.pos[:, coord][self.pos[:, coord] < self.properties["centre"][coord] - self.properties['boxsize'] / 2.0] += self.properties['boxsize']
                self.pos[:, coord] -= self.properties["centre"][coord]

        self.centred = True

    def rotate(self):
        """
        Rotates the particles (pos and vel) by the eigenvectors from the shape
        """
        # Must be recentred before rotation
        if not self.centred:
            self.recentre()

        if not self.rotated:
            self.pos = np.dot(self.properties['inertia_tensor'], self.pos.T).T
        self.rotated = True

    def normalise(self):
        """
        Divides distances in the halo by the virial radius
        """
        # Must be recentred first
        if not self.centred:
            self.recentre()

        if not self.normalised:
            self.pos /= self.properties['rvir']
        self.normalised = True

    def prepare(self):
        """
        Convenience function to recentre, rotate and normalise the halo
        """
        self.recentre()
        self.rotate()
        self.normalise()


def find_inertia(pos, mass):
    """
    Calculates the inertia tensor given an N*3 matrix of positions, and their mass
    """
    inertia = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            inertia[i, j] = np.sum(pos[:, i] * pos[:, j])

    inertia = inertia * mass
    return inertia


def calc_ratios(pos, mass, tol=0.001):
    """
    Calculates the axis ratios (and eigenvalues and eigenvectors) of a halo which is pre-centred.
    """
    # We calculate ratios until convergence
    err = 1
    ba = 1
    ca = 1
    counter = 0
    eigvec = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])

    while err > tol:
        counter = counter + 1
        pos = np.dot(eigvec, pos)
        # pos = np.array((eigvec * pos.view(np.matrix).T).T)

        # Define the ellipse
        r_ell = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2 / ba ** 2 + pos[:, 2] ** 2 / ca ** 2)

        mask = r_ell <= 1

        # Calculate the Moment of Inertia Tensor
        inertia = find_inertia(pos[mask, :], mass)

        # Diagonalize the IT and order it.
        eigval, eigvec = np.linalg.eigh(inertia)
        order = np.argsort(-eigval)
        eigvec = eigvec[:, order]
        eigval = eigval[order]

        # Determine Axis Ratios
        ba_new = np.sqrt(eigval[1] / eigval[0])
        ca_new = np.sqrt(eigval[2] / eigval[0])

        err_ba = abs((ba_new - ba) / ba_new)
        err_ca = abs((ca_new - ca) / ca_new)

        err = max(err_ba, err_ca)
        ba = ba_new
        ca = ca_new


    # Project positions and velocities onto new axes
    print 'Required ', counter, ' iteration(s) to find ellipticity'
    return ba, ca, eigval, eigvec

def concentration(pos, ba, ca, mass, r_vir, bins=25, min_bin=0.005):
    """
    Calcualtes the concentration of a pre-centred halo
    
    We need the axis ratios to get a good fit for the concentration and the profile
    """
    start = np.log10(min_bin)
    end = 0
    bins = np.logspace(start, end, num=bins + 1)

    # We use the ellipticity information to return an elliptic profile
    r_eff = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2 / ba ** 2 + pos[:, 2] ** 2 / ca ** 2)

    hgram, edges = np.histogram(r_eff, bins=bins)

    dens = mass * hgram / ((4. / 3.) * np.pi * r_vir ** 3 * (bins[1:] ** 3 - bins[:-1] ** 3))

    centres = (0.5 * bins[:-1] ** 3 + bins[1:] ** 3) ** (1. / 3.)


    # This is a hack to normalize the curve. THIS HAS THE BA and CA dependence.
    # It is equal to the ratio of particles that are pushed outside the sphere
    # by making r the 'effective' radius.
    dens = dens * len(r_eff) / len(r_eff[r_eff <= 1])

    rho_r_squared = dens * centres ** 2
    c = 1.0 / centres[rho_r_squared.argmax()]
    return c, dens, centres, rho_r_squared

