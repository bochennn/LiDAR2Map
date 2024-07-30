import numpy as np
from shapely.geometry import Point, LineString


def projection_distance(line, point):
    return LineString(line).distance(Point(point))


def points_distance(point1, point2):
    if not isinstance(point1, np.ndarray):
        return np.linalg.norm(np.array(point1) - np.array(point2))
    else:
        return np.linalg.norm(point1 - point2)
