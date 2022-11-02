import numpy as np
import pkfire.ptl_lib.particle as ptllib
import illustris_python as il
    
def addK(grid, k, amp, x0, y0):
    xs = np.linspace(0, grid.box, grid.getShape()[0])
    X, Y, Z = np.meshgrid(xs, xs, xs)
    R = np.sqrt(X**2 + Y**2 + Z**2)
    grid.grid += amp * np.sin(k * R - x0) + y0
    return

def random(grid, nptls, mass_range = [6, 13], z_range = [-np.inf, np.inf]):
    dim = len(grid.getShape())
    mr = mass_range
    pos = np.random.rand(nptls, dim) * grid.box
    mass = np.random.rand(nptls) * (mr[1] - mr[0]) + mr[0]
    mass = 10**mass
    pos_mask = (pos[:, 2] > z_range[0]) & (pos[:, 2] < z_range[1])
    ptlList = ptllib.Particles(pos[pos_mask], mass[pos_mask])
    return ptlList

def random_linear(grid, nptls, mass_range = [0, 1], z_range = [-np.inf, np.inf]):
    dim = len(grid.getShape())
    mr = mass_range
    pos = np.random.rand(nptls, dim) * grid.box
    mass = np.random.rand(nptls) * (mr[1] - mr[0]) + mr[0]
    pos_mask = (pos[:, 2] > z_range[0]) & (pos[:, 2] < z_range[1])
    ptlList = ptllib.Particles(pos[pos_mask], mass[pos_mask])
    return ptlList

def gaussian(grid, mean = 0, stdev = 1):
    rng_array = np.random.normal(mean, stdev, grid.getShape())
    grid.setDelta(rng_array)
    return

def stellarMassFunction():
    # using definitions from Weigel 2016
    return


SSP = 'SubhaloStellarPhotometrics'
SGN = 'SubhaloGrNr'
POS = 'SubhaloPos'
SMT = 'SubhaloMassType'
SPT = 'SubhaloParent'
GM = 'GroupMass'
GFS = 'GroupFirstSub'
GM500 = 'Group_M_Crit500'
GR500 = 'Group_R_Crit500'
class Illustris():

    def __init__(self, simpath, snapshot, add_sub_fields = [], add_grp_fields = []):
        fields = ['SubhaloGrNr', 'SubhaloPos', 'SubhaloMassType',
                'SubhaloStellarPhotometrics', 'SubhaloParent']
        fields.extend(add_sub_fields)
        subs = il.groupcat.loadSubhalos(simpath, snapshot, 
                fields = fields)
        head = il.groupcat.loadHeader(simpath, snapshot)
        subs['SubhaloMassType'] *= 1e10/head['HubbleParam']
        subs['SubhaloPos'] /= 1e3
        fields = [GM, GFS, GM500, GR500]
        fields.extend(add_grp_fields)
        grps = il.groupcat.loadHalos(simpath, snapshot, fields = fields)
        grps[GM] *= 1e10/head['HubbleParam']
        self.grpdata = grps
        self.snap = snapshot
        self.subdata = subs
        self.box = head['BoxSize'] / 1e3
        return
    
    def maskMass(self, idx = 4, mass_min = 2e8, mass_max = np.inf):
        mass = self.subdata[SMT][:, idx]
        return (mass >= mass_min) & (mass < mass_max)
    
    def maskColor(self, gr_min = None, gr_max = np.inf):
        if gr_min is None:
            if self.snap == 99:
                gr_min = 0.6
            # TODO add the other snapshots if needed
        gr = self.subdata[SSP][:, 4] - self.subdata[SSP][:, 5]
        return (gr > gr_min) & (gr < gr_max)
    
    def maskHalo(self, halo_id):
        return np.ma.masked_equal(self.subdata[SGN], np.ones_like(self.subdata[SGN]) * halo_id)
    
    def maskGroupMass(self, mass_min = 0, mass_max = np.inf):
        mass = self.grpdata[GM][:]
        return (mass >= mass_min) & (mass < mass_max)

    def getPos(self):
        return self.subdata[POS]
    
    def getMass(self, idx = 4):
        return self.subdata[SMT][:, idx]
    
    def getGroupMass(self):
        return self.grpdata[GM]

    def getBox(self):
        return self.box