import numpy as np
from scipy.spatial.transform import Rotation as R


def pack_to_2d(pos: np.ndarray, rot: np.ndarray, grasp: float, size=32) -> np.ndarray:
    """
    Hacky packing of pos and quat into a 2D array/tensor so we can concat such into with images.
    pos ~ [x, y, z]
    rot ~ quaternion [qx, qy, qz, qw] or 3x3 rotation matrix
    grasp ~ bool
    """
    if size % 4 != 0:
        raise ValueError("size must be divisible by 4")
    scale = size // 4

    # use 3x3 rotation matrix
    if not rot.shape == (3, 3):
        rot = R.from_quat(rot).as_matrix()
    rot = rot.flatten()

    # pack into a (4x4) 2D array with rot on the edges and pos in the middle
    packed = np.zeros((4, 4))
    packed[0] = rot[:4]
    packed[1:-1, -1] = rot[4:6]
    packed[-1, 1:] = rot[6:]
    packed[1:, 0] = pos
    packed[1, 1:3] = pos[:2]
    packed[2, 1] = pos[2]
    packed[2, 2] = grasp

    # scale 16x16 to size x size
    packed = packed.repeat(scale, axis=0).repeat(scale, axis=1)

    return packed


def unpack_to_1d(packed: np.array) -> [np.ndarray, np.ndarray]:
    """
    Unpacks a 2D array/tensor into pos and quat and grasp.
    """
    if packed.ndim != 2:
        raise ValueError("packed must be 2D")
    if packed.shape[0] != packed.shape[1]:
        raise ValueError("packed must be square")
    size = packed.shape[0]
    if size % 4 != 0:
        raise ValueError("size must be divisible by 4")

    scale = size // 4

    # average pooling repeated entries
    packed = packed.reshape(4, scale, 4, scale).mean(axis=(1, 3))
    rot = np.zeros(9)
    rot[:4] = packed[0]
    rot[4:6] = packed[1:-1, -1]
    rot[6:] = packed[-1, 1:]
    pos = (packed[1:, 0] + np.concatenate([packed[1, 1:3], [packed[2, 1]]])) / 2
    grasp = packed[2, 2] > 0.5

    # project rotation matrix to SO(3)
    U, S, Vh = np.linalg.svd(rot.reshape(3, 3))
    rot = U @ Vh
    quat = R.from_matrix(rot).as_quat()

    return pos, quat, grasp


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pos = np.array([0.1, 0.2, 0.3])
    quat = np.array([0.3826834, 0.9238795, 0, 0])
    grasp = 1.0
    packed = pack_to_2d(pos, quat, grasp, size=32)
    pos2, quat2, grasp2 = unpack_to_1d(packed)
    print(pos - pos2)
    print(quat - quat2)
    print(grasp - grasp2)
