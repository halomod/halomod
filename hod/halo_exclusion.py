'''
Created on Oct 3, 2013

@author: Steven
'''

class HaloExclusion(object):
    '''
    Wraps up several methods of halo exclusion in the 2h term of the gal power
    '''


    def __init__(self, m, dndm, nt, u, exclusion_type="sphere_zheng"):

        '''
        INPUT
        dndm - the number density of haloes at a vector M
        nt   - the number of galaxies in haloes at M
        u    - the normalised fourier transform of the halo profile
        exclusion_type - one of 'sphere_zheng', 'sphere_tinker', 'ng_matched','ellipsoid'  (from lowest to highest accuracy)
        '''

        self.m = m
        self.dndm = dndm
        self.nt = nt
        self.u = u
        self.exclusion_type = exclusion_type


