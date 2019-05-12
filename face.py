import Rhino.Geometry as rg
import math

class Face():
    
    def __init__(self, f, mesh):
        
        self.v1 = f.A
        self.v2 = f.B
        self.v3 = f.C

        self.mesh = mesh
        self.edges = []
    
    @property
    def indexes(self):
        
        return [self.v1, self.v2, self.v3]


    @property
    def vertices(self):

        return [self.mesh.Vertices[self.v1], self.mesh.Vertices[self.v2], self.mesh.Vertices[self.v3]]
    
    def __hash__(self):

        return hash('%d-%d-%d' % (self.v1, self.v2, self.v3))

    def contains_index(self, i):

        return True if i == self.v1 or i == self.v2 or i == self.v3 else False

    def contains_plane(self, value, axis=0, epsilon=0.01):

        min_x, min_y, min_z = 9999999, 9999999, 9999999
        max_x, max_y, max_z = -9999999, -9999999, -9999999

        for v in self.vertices:

            if v.X < min_x:
                min_x = v.X
            elif v.X > max_x:
                max_x = v.X

            if v.Y < min_y:
                min_y = v.Y
            elif v.Y > max_y:
                max_y = v.Y

            if v.Z< min_z:
                min_z = v.Z
            elif v.Z > max_z:
                max_z = v.Z

        if axis == 0 and min_x - epsilon <= value and value <= max_x + epsilon:
            return True 

        if axis == 1 and min_y - epsilon <= value and value <= max_y + epsilon:
            return True 

        if axis == 2 and min_y - epsilon <= value and value <= max_z + epsilon:
            return True 

        return False

    @property
    def normal(self):
        
        vec1 = rg.Vector3d(self.mesh.Vertices[self.v1])
        vec2 = rg.Vector3d(self.mesh.Vertices[self.v2])
        vec3 = rg.Vector3d(self.mesh.Vertices[self.v3])
        
        vector1 = vec2 - vec1
        vector2 = vec3 - vec1
        
        vector1.Unitize()
        vector2.Unitize()
        
        normal = rg.Vector3d.CrossProduct(vector1, vector2)
        normal.Unitize()
        
        return normal
    
    def compute_outer_vertices(self):
        
        vertices_sets = []
        
        normal = self.normal
        
        arr = [self.v1, self.v2, self.v3]
        
        for i in range(len(arr)):
            
            v1 = self.mesh.Vertices[arr[i]]
            v2 = self.mesh.Vertices[arr[(i + 1) % len(arr)]]
            v3 = self.mesh.Vertices[arr[i - 1]]
            
            vec1 = rg.Vector3d(v1) - rg.Vector3d(v2)
            vec2 = rg.Vector3d(v1) - rg.Vector3d(v3)
            vec1.Unitize()
            vec2.Unitize()
            
            vec = vec1 + vec2
            vec.Unitize()
            
            pt = rg.Point3d.Add(v1, vec)
                        
            vertices_sets.append((arr[i], pt))
        
        return vertices_sets