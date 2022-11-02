import numpy as np

class Particles():

    def __init__(self, pos, mass = None, attrs = {}, **kwargs):
        attrs.update(kwargs)
        self.pos = pos
        if mass is None:
            self.mass = np.ones(len(pos))
        else:
            self.mass = mass
        self.attrs = attrs
        self.nptls = len(self.mass)
        return
    
    def getPos(self, grid = None): # gives flexibility for subclasses
        return self.pos
    
    def getMass(self):
        return self.mass
    
    def addAttr(self, key, val, overwrite = True):
        if not key in self.attrs or overwrite:
            self.attrs[key] = val
        return
    
    def update(self, attr_dict):
        self.attrs.update(attr_dict)
        return
    
    def getMatches(self, inattrs):
        mask = np.ones(self.nptls, dtype = bool)
        for k,v in inattrs.items():
            if k in self.attrs:
                mask = mask & (self.attrs[k] == v)
            else:
                return np.zeros(self.nptls, dtype = bool)
        return mask