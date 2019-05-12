
import Rhino.Geometry as rg
import math
import random
from face import Face
from edge import Edge

class Mesh():

    def __init__(self, meshes):

        # join meshes into a single mesh
        mesh = rg.Mesh()

        for m in meshes:
            mesh.Append(m)

        # weld a mesh
        mesh.Weld(math.pi)

        # triangulate
        mesh.Faces.ConvertQuadsToTriangles()

        # weld a mesh
        mesh.Weld(math.pi)

        self.mesh = mesh

        self.edges = {}
        self.faces = []
        self.outer_vertices = {}
        self.inner_vertices = {}

        for f in mesh.Faces:
            
            if f.C != f.D:
                
                raise ValueError('The input mesh must be a collection of trigonal faces.')
            
            face = Face(f, mesh)
            self.faces.append(face)
            
            for v1, v2 in [(f.A, f.B), (f.B, f.C), (f.C, f.A)]:
                
                edge = Edge(v1, v2)
                key  = edge.key
                
                if not key in self.edges:
                    
                    self.edges[key] = edge
                
                else:

                    edge = self.edges[key]

                self.edges[key].add_face(face)
                face.edges.append(edge)
                
                self.inner_vertices[v1] = mesh.Vertices[v1]
            
            vertices_sets = face.compute_outer_vertices()
            
            for i, v in vertices_sets:
                
                if i in self.outer_vertices:
                    
                    self.outer_vertices[i].append(v)
                
                else:
                    
                    self.outer_vertices[i] = [v]
    
    @property
    def vertices(self):

        return list(self.mesh.Vertices)

    def separate(self):

        avg = rg.Point3d(0, 0, 0)

        for v in self.vertices:
            
            avg.X += v.X
            avg.Y += v.Y
            avg.Z += v.Z
        
        avg.X /= len(self.vertices)
        avg.Y /= len(self.vertices)
        avg.Z /= len(self.vertices)
        
        class Node():

            def __init__(self, index):

                self.index = index
                self.visited = False
                self.faces = []
            
            def add_face(self, f):

                self.faces.append(f)

        candidate_vertices = {}
        vertices_by_face = {}
        nodes = {}

        first_node, last_node = None, None
        for f in self.faces:

            if f.contains_plane(value=avg.Y, axis=1):

                if not f in vertices_by_face:
                    vertices_by_face[f] = []

                for e in f.edges:

                    if not e.v1 in candidate_vertices:
                        candidate_vertices[e.v1] = []

                    if not e.v2 in candidate_vertices:
                        candidate_vertices[e.v2] = []

                    if not e.v1 in nodes:
                        nodes[e.v1] = Node(e.v1)

                    if not e.v2 in nodes:
                        nodes[e.v2] = Node(e.v2)

                    node1 = nodes[e.v1]
                    node2 = nodes[e.v2]

                    if not f in node1.faces:
                        node1.add_face(f)

                    if not f in node2.faces:
                        node2.add_face(f)

                    if not node1 in vertices_by_face[f]:
                        vertices_by_face[f].append(node1)

                    if not node2 in vertices_by_face[f]:
                        vertices_by_face[f].append(node2)

                    candidate_vertices[e.v1].append(node2)
                    candidate_vertices[e.v2].append(node1)

        longest_path = []
        trial = 0
        while True:

            closed = False

            # get the first & last node at random

            is_searchable = False
            while not is_searchable:

                first_node = random.choice(nodes.values())
                shared_face = random.choice(first_node.faces)

                while True:
                    last_node = random.choice(vertices_by_face[shared_face])
                    if last_node != first_node:
                        break

                for k in nodes:

                    node = nodes[k]
                    
                    if shared_face in node.faces:

                        node.visited = True
                    
                    else:

                        node.visited = False

                # need one vertex to visit from the starting point at least
                candidates = candidate_vertices[first_node.index]
                for c in candidates:
                    if not c.visited:
                        is_searchable = True
                        break

            path = [first_node]

            found = True

            while found:

                found = False

                candidates = candidate_vertices[path[-1].index]

                for c in candidates:

                    if c == last_node and len(path) > 1:
                        closed = True
                        found = False
                        path.append(c)
                        print('closed', len(path))

                        break

                    if c.visited:
                        # print('visited', len(candidates))
                        continue

                    c.visited = True
                    
                    path.append(c)
                    found = True

                    #let all the vertices of the face visited
                    for f in c.faces:
                        
                        vertices = vertices_by_face[f]

                        if c in vertices and path[-2] in vertices:

                            for n in vertices:
                                n.visited = True

                    break
                
                if closed:
                    break

            if len(path) > len(longest_path) and closed:

                longest_path = path
        
            trial += 1
            if trial > len(candidate_vertices) * 5:
                print('finish search')
                break
        
        if len(longest_path) == 0:
            longest_path = [first_node]
        print(longest_path)
        a = []
        for i in range(len(longest_path)):
            v1 = longest_path[i].index
            v2 = longest_path[(i + 1) % len(longest_path) ].index
            
            a.append(rg.Line(self.vertices[v1], self.vertices[v2]))

        a.append(self.vertices[longest_path[0].index])

        if len(longest_path) > 1:
            a.append(self.vertices[longest_path[-1].index])



        cut_edges = {}
        path_indexes = []
        for i in range(len(longest_path)):

            node1 = longest_path[i]
            node2 = longest_path[(i + 1) % len(longest_path)]

            key = Edge.generate_key(node1.index, node2.index)

            cut_edges[key] = self.edges[key]
            path_indexes.append(node1.index)
        
        initial_face = None
        for f in self.faces:
            
            for v in f.indexes:

                if v not in path_indexes:

                    initial_face = f
                    break
        
        print(initial_face)
        print(cut_edges)

        available_faces = {}
        for f in self.faces:
            available_faces[f] = False

        def search(face):

            available_faces[face] = True

            for edge in face.edges:

                if edge.key in cut_edges:
                    continue

                for ff in edge.faces:

                    if not available_faces[ff]:
                        search(ff)

        search(initial_face)

        mesh_1 = rg.Mesh()
        mesh_2 = rg.Mesh()

        index_table_1 = {}
        index_table_2 = {}
        j = 0
        k = 0
        for f in available_faces:
            
            for i in f.indexes:
                    
                if available_faces[f] and i not in index_table_1:

                    index_table_1[i] = j
                    j += 1
                    mesh_1.Vertices.Add(self.vertices[i])

                elif not available_faces[f] and i not in index_table_2:

                    index_table_2[i] = k
                    k += 1
                    mesh_2.Vertices.Add(self.vertices[i])

        for f in available_faces:

            index1 = f.v1
            index2 = f.v2
            index3 = f.v3

            if available_faces[f]:

                new_index1 = index_table_1[index1]
                new_index2 = index_table_1[index2]
                new_index3 = index_table_1[index3]

                mesh_1.Faces.AddFace(new_index1, new_index2, new_index3)

            else:

                new_index1 = index_table_2[index1]
                new_index2 = index_table_2[index2]
                new_index3 = index_table_2[index3]

                mesh_2.Faces.AddFace(new_index1, new_index2, new_index3)

        return mesh_1, mesh_2


    def thinken(self, thickness=1, joint_type=0, joint_length=1, joint_width=1, origin=None, interval=10):
        
        if not origin:
            origin = rg.Point3d(30, 0, 0)

        rigid_faces = []
        oriented_rigid_faces = []

        merged_outer_vertices = []

        for key in self.outer_vertices:
            
            points = self.outer_vertices[key]
            
            inner_point = self.inner_vertices[key]
            
            x, y, z = 0, 0, 0
            
            for p in points:
                
                x += p.X
                y += p.Y
                z += p.Z
                
            x /= len(points)
            y /= len(points)
            z /= len(points)
            
            point = rg.Point3d(x, y, z)
            
            vector = rg.Vector3d(point) - rg.Vector3d(inner_point)
            vector.Unitize()
            vector *= thickness
            
            new_pt = rg.Point3d.Add(inner_point, vector)
            
            merged_outer_vertices.append(new_pt)
        


        base_planes = []

        for face in self.faces:
            
            outer_points = []
            inner_points = [] 
            
            for i in face.indexes:
                
                outer_point = merged_outer_vertices[i]
                
                outer_points.append(outer_point)
                
                inner_points.append(rg.Point3d(self.vertices[i]))

            local_edges = []
            for i in range(len(face.indexes)):
                
                v1 = i
                v2 = (i + 1) % len(face.indexes)
                
                key = Edge.generate_key(face.indexes[v1], face.indexes[v2])
                
                local_edges.append(self.edges[key])
            
            m = rg.Mesh()
            
            for p in inner_points + outer_points:
                
                m.Vertices.Add(p.X, p.Y, p.Z)

            m.Faces.AddFace(0, 1, 2)
            m.Faces.AddFace(3, 4, 5)
            
            loop_sets = zip([0, 1, 2], [[0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]])

            for edge_index, original_indexes in loop_sets:
            
                edge = local_edges[edge_index]
            
                if len(edge.faces) != 2:
                    
                    m.Faces.AddFace(original_indexes[0], original_indexes[1], original_indexes[2], original_indexes[3])
                
                elif joint_type == 0:
                    
                    m.Faces.AddFace(original_indexes[0], original_indexes[1], original_indexes[2], original_indexes[3])
                
                elif joint_type == 1:
                    
                    p0 = rg.Vector3d(m.Vertices[original_indexes[0]])
                    p1 = rg.Vector3d(m.Vertices[original_indexes[1]])
                    p2 = rg.Vector3d(m.Vertices[original_indexes[2]])
                    p3 = rg.Vector3d(m.Vertices[original_indexes[3]])
                    
                    vec1 = rg.Vector3d(p1) - rg.Vector3d(p0)
                    vec2 = rg.Vector3d(p3) - rg.Vector3d(p0)
                    vec3 = rg.Vector3d(p2) - rg.Vector3d(p1)
                    
                    normal = rg.Vector3d.CrossProduct(vec1, vec2)
                    normal.Unitize()

                    if not edge.male_created:
                        
                        alpha = 1
                        beta  = 0.36
                        gamma = 0.64
                        zeta  = 0.5
                        
                    else:
                        
                        alpha = -1
                        beta  = 0.35
                        gamma = 0.65
                        zeta  = 0.5
                        
                    normal *= joint_length
                    p4 = p0 + vec1 * beta
                    p5 = p0 + vec1 * gamma
                    p6 = (p5 + p4) * zeta + (vec2 * 0.25 + vec3 * 0.25)
                    p7  = p4 + normal
                    p8  = p5 + normal
                    p9  = p6 + normal
            
                    p10 = p0 + vec2 * zeta
                    p11 = p1 + vec3 * zeta
                    
                    current_length = len(m.Vertices) - 1
            
                    m.Vertices.Add(p4.X, p4.Y, p4.Z) # 1
                    m.Vertices.Add(p5.X, p5.Y, p5.Z) # 2
                    m.Vertices.Add(p6.X, p6.Y, p6.Z) # 3
                    m.Vertices.Add(p7.X, p7.Y, p7.Z) # 4
                    m.Vertices.Add(p8.X, p8.Y, p8.Z) # 5
                    m.Vertices.Add(p9.X, p9.Y, p9.Z) # 6
                    m.Vertices.Add(p10.X, p10.Y, p10.Z) # 8
                    m.Vertices.Add(p11.X, p11.Y, p11.Z) # 7
                    
                    m.Faces.AddFace(original_indexes[0], current_length + 1, current_length + 3, current_length + 7)
                    m.Faces.AddFace(original_indexes[1], current_length + 8, current_length + 3, current_length + 2)
                    m.Faces.AddFace(original_indexes[2], original_indexes[3], current_length + 7, current_length + 8)
                    m.Faces.AddFace(current_length + 4, current_length + 5, current_length + 6)
                    m.Faces.AddFace(current_length + 1, current_length + 2, current_length + 5, current_length + 4)
                    m.Faces.AddFace(current_length + 2, current_length + 3, current_length + 6, current_length + 5)
                    m.Faces.AddFace(current_length + 3, current_length + 1, current_length + 4, current_length + 6)
                    
                    if not edge.male_created:
                        edge.male_created = True

                elif joint_type == 2:
                    
                    p0 = rg.Vector3d(m.Vertices[original_indexes[0]])
                    p1 = rg.Vector3d(m.Vertices[original_indexes[1]])
                    p2 = rg.Vector3d(m.Vertices[original_indexes[2]])
                    p3 = rg.Vector3d(m.Vertices[original_indexes[3]])
                    
                    vec1 = rg.Vector3d(p1) - rg.Vector3d(p0)
                    vec2 = rg.Vector3d(p3) - rg.Vector3d(p0)
                    vec3 = rg.Vector3d(p2) - rg.Vector3d(p1)
                    
                    normal = rg.Vector3d.CrossProduct(vec1, vec2)
                    normal.Unitize()

                    if not edge.male_created:
                        
                        alpha = joint_length * 0.9
                        beta  = joint_width * 0.45
                        gamma = joint_width * 0.45
                        zeta  = joint_width * 0.45
                        
                    else:
                        
                        alpha = -joint_length
                        beta  = joint_width * 0.5
                        gamma = joint_width * 0.5
                        zeta  = joint_width * 0.5
                        
                    
                    unit_vec1 = rg.Vector3d(vec1)
                    unit_vec1.Unitize()
                    unit_vec1 *= 1

                    unit_vec2 = rg.Vector3d(vec2)
                    unit_vec2.Unitize()
                    unit_vec2 *= 1
                    
                    unit_vec3 = rg.Vector3d(vec3)
                    unit_vec3.Unitize()
                    unit_vec3 *= 1
                    
                    normal *= alpha
                    
                    horizontal_middle = vec1 * 0.5
                    vertical_middle = (vec2 + vec3) * 0.25
                    vertical_unit = (unit_vec2 + unit_vec3) * 0.5
                    vertical_unit.Unitize()
                    p4 = p0 + horizontal_middle - unit_vec1 * beta + vertical_middle - vertical_unit * gamma
                    p5 = p0 + horizontal_middle + unit_vec1 * beta + vertical_middle - vertical_unit * gamma
                    p7 = p0 + horizontal_middle - unit_vec1 * beta + vertical_middle + vertical_unit * zeta
                    p6 = p0 + horizontal_middle + unit_vec1 * beta + vertical_middle + vertical_unit * zeta
                    p8   = p4 + normal
                    p9   = p5 + normal
                    p10  = p6 + normal
                    p11  = p7 + normal
                    
                    current_length = len(m.Vertices) - 1
                    
                    m.Vertices.Add(p4.X, p4.Y, p4.Z) # 1
                    m.Vertices.Add(p5.X, p5.Y, p5.Z) # 2
                    m.Vertices.Add(p6.X, p6.Y, p6.Z) # 3
                    m.Vertices.Add(p7.X, p7.Y, p7.Z) # 4
                    m.Vertices.Add(p8.X, p8.Y, p8.Z) # 5
                    m.Vertices.Add(p9.X, p9.Y, p9.Z) # 6
                    m.Vertices.Add(p10.X, p10.Y, p10.Z) # 7
                    m.Vertices.Add(p11.X, p11.Y, p11.Z) # 8
                    
                    m.Faces.AddFace(original_indexes[0], original_indexes[1], current_length + 2, current_length + 1)
                    m.Faces.AddFace(original_indexes[0], current_length + 1, current_length + 4, original_indexes[3])
                    m.Faces.AddFace(original_indexes[1], original_indexes[2], current_length + 3, current_length + 2)
                    m.Faces.AddFace(original_indexes[2], original_indexes[3], current_length + 4, current_length + 3)
                    
                    m.Faces.AddFace(current_length + 5, current_length + 6, current_length + 7, current_length + 8)
                    m.Faces.AddFace(current_length + 1, current_length + 2, current_length + 6, current_length + 5)
                    m.Faces.AddFace(current_length + 2, current_length + 3, current_length + 7, current_length + 6)
                    m.Faces.AddFace(current_length + 3, current_length + 4, current_length + 8, current_length + 7)
                    m.Faces.AddFace(current_length + 4, current_length + 1, current_length + 5, current_length + 8)

                    if not edge.male_created:
                        edge.male_created = True
                    
                else:
                    
                    raise ValueError('No Joint type!')

            m2 = rg.Mesh()
            rg.Mesh.CopyFrom(m2, m)
            rigid_faces.append(m)
            oriented_rigid_faces.append(m2)
            
            # get a base plane
            
            outer_points += [outer_points[0]]
            outer_curve = rg.PolylineCurve(outer_points)

            inner_vec1 = rg.Vector3d(inner_points[1]) - rg.Vector3d(inner_points[0])
            inner_vec2 = rg.Vector3d(inner_points[2]) - rg.Vector3d(inner_points[0])
            
            outer_vec1 = rg.Vector3d(outer_points[1]) - rg.Vector3d(outer_points[0])
            outer_vec2 = rg.Vector3d(outer_points[2]) - rg.Vector3d(outer_points[0])
            
            inner_area = rg.Vector3d.CrossProduct(inner_vec1, inner_vec2).Length
            outer_area = rg.Vector3d.CrossProduct(outer_vec1, outer_vec2).Length

            if inner_area > outer_area:

                use_inner = True
                points = inner_points
            
            else:
                
                use_inner = False
                points = outer_points[:-1]
            
            cx, cy, cz = 0, 0, 0
            
            for p in points:
                
                cx += p.X
                cy += p.Y
                cz += p.Z
            
            cx /= 3
            cy /= 3
            cz /= 3
            
            center = rg.Point3d(cx, cy, cz)
            
            vector1 = rg.Vector3d(points[1]) - rg.Vector3d(points[0])
            vector2 = rg.Vector3d(points[2]) - rg.Vector3d(points[0])

            vector1.Unitize()
            vector2.Unitize()
            
            normal = rg.Vector3d.CrossProduct(vector1, vector2)
            normal.Unitize()
            
            base_plane = rg.Plane(center, normal)
            base_planes.append(base_plane)

            if not use_inner:
                base_plane.Flip()
    
        i = 0

        rows = math.sqrt(len(oriented_rigid_faces))

        for base_plane, face in zip(base_planes, oriented_rigid_faces):
            
            transform = rg.Transform.PlaneToPlane(base_plane, rg.Plane.WorldXY)
            
            face.Transform(transform)
            
            dx = interval * (i % rows)
            dy = interval * (i / rows)
            
            transform = rg.Transform.Translation(origin.X + dx, origin.Y + dy, origin.Z)
            
            face.Transform(transform)
            
            i += 1

        return rigid_faces, oriented_rigid_faces
