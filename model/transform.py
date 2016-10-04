
import numpy as np
import math

deg_to_rad = 3.141592/180.0


def rot_matrix(roll, pitch, yaw):

    roll *= deg_to_rad
    pitch *= deg_to_rad
    yaw *= deg_to_rad

    R_x = np.array([[1, 0,           0],
              [0, math.cos(roll), -math.sin(roll)],
              [0, math.sin(roll), math.cos(roll)]])

    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]])

    R_z = np.array([[math.cos(yaw), math.sin(yaw), 0],
                    [-math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1]])

    return np.dot(R_z, R_y).dot(R_x)


def angles_from_rot(R):
    yaw = math.atan2(R[1, 0], R[0, 0])
    pitch = math.atan2(-R[2, 0], math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    roll = math.atan2(R[2, 1], R[2, 2])

    yaw /= deg_to_rad
    pitch /= deg_to_rad
    roll /= deg_to_rad

    return roll, pitch, yaw


def relative_angles(base, angles):
    Rb = rot_matrix(base[0], base[1], base[2])
    Ri = rot_matrix(angles[0], angles[1], angles[2])

    R = Rb.dot(Ri.T)
    return angles_from_rot(R)
