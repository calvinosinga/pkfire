

class Particle():

    def __init__(self, pos, mass = 1, attrs = {}, **kwargs):
        attrs.update(kwargs)
        self.pos = pos
        self.mass = mass
        self.attrs = attrs
        return
    
    def addAttr(self, key, val, overwrite = True):
        if not key in self.attrs or overwrite:
            self.attrs[key] = val
        return
    
    def update(self, attr_dict):
        self.attrs.update(attr_dict)
        return
    
    def isMatch(self, inattrs):
        isMatch = True
        for k,v in inattrs.items():
            if k in self.attrs:
                isMatch = isMatch and self.attrs[k] == v
            else:
                isMatch = False
        return isMatch