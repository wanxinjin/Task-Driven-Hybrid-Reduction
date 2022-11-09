import numpy as np
import math
from casadi import *


#########################################
#########################################

# converter to quaternion from (radian angle, direction)
def angle_dir_to_quat(angle, dir):
    if type(dir) == list:
        dir = np.array(dir)
    dir = dir / np.linalg.norm(dir)
    quat = np.zeros(4)
    quat[0] = math.cos(angle / 2)
    quat[1:] = math.sin(angle / 2) * dir
    return quat


# alternative form
def axisangle2quat(axisangle):
    dir = axisangle[0:3]
    angle = axisangle[3]
    dir = dir / np.linalg.norm(dir)
    quat = np.zeros(4)
    quat[0] = math.cos(angle / 2)
    quat[1:] = math.sin(angle / 2) * dir
    return quat


#########################################
#########################################

# conjugate quaternion matrix (casadi function)
# https://www.ashwinnarayan.com/post/how-to-integrate-quaternions/
# https://opensource.docs.anymal.com/doxygen/kindr/master/cheatsheet_latest.pdf
q = SX.sym('q', 4)
cqm = vertcat(
    horzcat(q[0], -q[1], -q[2], -q[3]),
    horzcat(q[1], q[0], q[3], -q[2]),
    horzcat(q[2], -q[3], q[0], q[1]),
    horzcat(q[3], q[2], -q[1], q[0]),
)
csd_conjquatmat_fn = Function('csd_conjquatmat_fn', [q], [cqm])

wb = SX.sym('wb', 3)
wb_cqm = vertcat(
    horzcat(0, -wb[0], -wb[1], -wb[2]),
    horzcat(wb[0], 0, wb[2], -wb[1]),
    horzcat(wb[1], -wb[2], 0, wb[0]),
    horzcat(wb[2], wb[1], -wb[0], 0),
)
csd_conjquatmat_wb_fn = Function('csd_conjquatmat_wb_fn', [wb], [wb_cqm])

#########################################
#########################################

# quaternion to dcm (casadi function)
q = SX.sym('q', 4)
dcm = casadi.vertcat(
    casadi.horzcat(
        1 - 2 * (q[2] ** 2 + q[3] ** 2),
        2 * (q[1] * q[2] - q[0] * q[3]),
        2 * (q[1] * q[3] + q[0] * q[2]),
    ),
    casadi.horzcat(
        2 * (q[1] * q[2] + q[0] * q[3]),
        1 - 2 * (q[1] ** 2 + q[3] ** 2),
        2 * (q[2] * q[3] - q[0] * q[1]),
    ),
    casadi.horzcat(
        2 * (q[1] * q[3] - q[0] * q[2]),
        2 * (q[2] * q[3] + q[0] * q[1]),
        1 - 2 * (q[1] ** 2 + q[2] ** 2),
    ),
)
csd_quat2dcm_fn = Function('csd_quat2dcm', [q], [dcm])


#########################################
#########################################

# https://github.com/Khrylx/Mujoco-modeler/blob/master/transformation.py
# quaternion multiplication
def quaternion_multiply(quaternion1, quaternion0):
    """Return multiplication of two quaternions.
    """
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return numpy.array([
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=numpy.float64)


def quaternion_mat(q):
    return np.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1], q[0], -q[3], q[2]],
        [q[2], q[3], q[0], -q[1]],
        [q[3], -q[2], q[1], q[0]],
    ], dtype=numpy.float64)


def quaternion_mul(q1, q2):
    return quaternion_mat(q1) @ q2


def quaternion_conjugate(quaternion):
    """Return conjugate of quaternion.
    """
    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    numpy.negative(q[1:], q[1:])
    return q


def quaternion_inverse(quaternion):
    """Return inverse of quaternion.
    """
    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    numpy.negative(q[1:], q[1:])
    return q / numpy.dot(q, q)


def quaternion_real(quaternion):
    """Return real part of quaternion.
    """
    return float(quaternion[0])


def quaternion_imag(quaternion):
    """Return imaginary part of quaternion.
    """
    return numpy.array(quaternion[1:4], dtype=numpy.float64, copy=True)


def quaternion_slerp(quat0, quat1, N):
    """Return spherical linear interpolation between two quaternions.
    """
    _EPS = 1e-6

    q0 = quat0 / np.linalg.norm(quat0)
    q1 = quat1 / np.linalg.norm(quat1)

    d = numpy.dot(q0, q1)

    if abs(abs(d) - 1.0) < _EPS:
        return np.tile(q0, (N, 1))

    angle = math.acos(d)
    isin = 1.0 / math.sin(angle)

    fractions = np.linspace(0, 1, N)

    q = []
    for frac in fractions:
        q.append(math.sin((1.0 - frac) * angle) * isin * q0 +
                 math.sin(frac * angle) * isin * q1)

    return np.array(q)


def random_quaternion(rand=None):
    """Return uniform random unit quaternion.
    rand: array like or None
        Three independent random variables that are uniformly distributed
        between 0 and 1.
    """
    if rand is None:
        rand = numpy.random.rand(3)
    else:
        assert len(rand) == 3
    r1 = numpy.sqrt(1.0 - rand[0])
    r2 = numpy.sqrt(rand[0])
    pi2 = math.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return numpy.array([numpy.cos(t2) * r2, numpy.sin(t1) * r1,
                        numpy.cos(t1) * r1, numpy.sin(t2) * r2])


#########################################
#########################################

def quat2rotmat(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix


#########################################
# below is problematic
#########################################

def quat2angle(q):
    return 2.0 * math.acos(q[0]) * np.sign(q[-1])


#########################################
#########################################

def angle2mat(angle):
    mat = np.array(
        [[math.cos(angle), -math.sin(angle)],
         [math.sin(angle), math.cos(angle)]
         ]
    )
    return mat
