import Rhino.Geometry as rg
import math

class Edge():
    
    def __init__(self, v1, v2):
        
        if v1 < v2:
            self.v1 = v1
            self.v2 = v2
        else:
            self.v1 = v2
            self.v2 = v1
        
        self.faces = []
    
        self.male_created = False
    
    def add_face(self, face):
        
        self.faces.append(face)
    
    def __eq__(self, other):
        
        return isinstance(other, Edge) and self.v1 == other.v1 and self.v2 == other.v2

    def __hash__(self):
        
        return self.key    

    @property
    def key(self):
        
        return self.v1 * 100000 + self.v2
    
    @staticmethod
    def generate_key(v1, v2):
        
        v1, v2 = (v1, v2) if v1 < v2 else (v2, v1)
        
        return v1 * 100000 + v2