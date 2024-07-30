import numpy as np
from Geometry3D import Point, Segment, ConvexPolygon, ConvexPolyhedron, intersection, Renderer

from utils.transform import rotation_matrix


def create_convex_polygon(plane):
    points = [Point(p) for p in plane]
    return ConvexPolygon(points)


def create_convex_polyhedron(planes):
    convex_polygons = [create_convex_polygon(plane) for plane in planes]
    return ConvexPolyhedron(convex_polygons)


def is_plane_visible(convex_polyhedron, plane):
    origin = Point(0, 0, 0)
    for point in plane:
        seg = Segment(origin, Point(point))
        inter = intersection(convex_polyhedron, seg)
        if not isinstance(inter, Point):
            return False
    return True


def get_planes(corners):
    face_x_p = corners[[0, 1, 4, 5]]
    face_x_n = corners[[2, 3, 6, 7]]
    face_y_p = corners[[0, 3, 4, 7]]
    face_y_n = corners[[1, 2, 5, 6]]
    face_z_p = corners[[0, 1, 2, 3]]
    face_z_n = corners[[4, 5, 6, 7]]
    return face_x_p, face_x_n, face_y_p, face_y_n, face_z_p, face_z_n


def get_potential_visible_planes(corners):
    planes = get_planes(corners)
    convex_polyhedron = create_convex_polyhedron(planes)


def get_grid_points(plane):
    xmin = min(plane[:, 0])
    xmax = max(plane[:, 0])
    ymin = min(plane[:, 1])
    ymax = max(plane[:, 1])
    zmin = min(plane[:, 2])
    zmax = max(plane[:, 2])
    xs = np.linspace(xmin, xmax, 10)
    ys = np.linspace(ymin, ymax, 10)
    zs = np.linspace(zmin, zmax, 10)
    # Xs, Ys, Zs = np.meshgrid(xs, ys, zs)
    Ys, Zs = np.meshgrid(ys, zs)
    Xs = np.ones_like(Ys) * xmin
    voxels = np.dstack([Xs, Ys, Zs])


length = 5
width = 3
height = 5
x = 5
y = 10
z = -1
yaw = 0
rot_matrix = rotation_matrix(yaw)

x_corners = length / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
y_corners = width / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
z_corners = height / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
corners = np.vstack((x_corners, y_corners, z_corners))
corners = np.dot(rot_matrix, corners)
corners[0, :] = corners[0, :] + x
corners[1, :] = corners[1, :] + y
corners[2, :] = corners[2, :] + z
corners = corners.transpose()

for idx, corner in enumerate(corners):
    print("corner_{}: {}".format(idx+1, corner))

faces = get_surfaces(corners)
polyhedron = create_convex_polyhedron(faces)
r = Renderer()
r.add((Point(0, 0, 0), 'g', 40), normal_length=0)
r.add((polyhedron, 'r', 2), normal_length=0)

polygon_added = False
for idx, face in enumerate(faces):
    if is_surface_visible(polyhedron, face):
        print("face_{} is visible".format(idx+1))
        for point in face:
            r.add((Point(point), 'b', 30), normal_length=0)
        if not polygon_added:
            # polygons = get_frustum(face)
            # for polygon in polygons:
            #     r.add((polygon, 'g', 2), normal_length=0)
            polygon_added = True

sample_face = faces[0]
xmin = min(sample_face[:, 0])
xmax = max(sample_face[:, 0])
ymin = min(sample_face[:, 1])
ymax = max(sample_face[:, 1])
zmin = min(sample_face[:, 2])
zmax = max(sample_face[:, 2])
xs = np.linspace(xmin, xmax, 10)
ys = np.linspace(ymin, ymax, 10)
zs = np.linspace(zmin, zmax, 10)
# Xs, Ys, Zs = np.meshgrid(xs, ys, zs)
Ys, Zs = np.meshgrid(ys, zs)
Xs = np.ones_like(Ys) * xmin
voxels = np.dstack([Xs, Ys, Zs])

print("sample face: {}".format(sample_face))
for face in voxels:
    for voxel in face:
        r.add((Point(voxel), 'b', 10))
r.show()

# a = Point(0, 0, 0)
# b = Point(0, 1, 0)
# c = Point(1, 1, 0)
# d = Point(1, 0, 0)
#
# e = Point(0, 0, 1)
# f = Point(0, 1, 1)
# g = Point(1, 1, 1)
# h = Point(1, 0, 1)
#
# face1 = ConvexPolygon((a, b, c, d))
# face2 = ConvexPolygon((a, b, e, f))
# face3 = ConvexPolygon((c, d, g, h))
# face4 = ConvexPolygon((e, f, g, h))
# face5 = ConvexPolygon((a, d, e, h))
# face6 = ConvexPolygon((b, c, f, g))
#
# poly1 = ConvexPolyhedron((face1, face2, face3, face4, face5, face6))
#
# a1 = Point(-1, -1, -1)
# b1 = Point(2, 2, 4.27)
# line2 = Segment(a1, b1)
#
# inter = intersection(poly1, line2)
#
# print(inter) # results I needed
#
#
#
# # visualize planes
# r = Renderer()
# r.add((poly1,'r',2),normal_length = 0)
# r.add((line2,'g',2),normal_length = 0)
# if inter is not None:
#     for point in inter:
#         r.add((point, 'b', 10), normal_length=0)
# r.show()