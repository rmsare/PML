import numpy as np

def deform_point_cloud(point_cloud, strike, dip, d = 0.01, u_ss = 1.0, u_ds = 1.0, x0 = None, y0 = None):

    from utils import get_xyz_from_pdal, put_xyz_to_pdal
    from pml import read_file

    xyz = get_xyz_from_pdal(read_file(point_cloud)) if isinstance(point_cloud, str) else \
        get_xyz_from_pdal(point_cloud[0]) if isinstance(point_cloud, list) else point_cloud if point_cloud.dtype.fields \
            is None else get_xyz_from_pdal(point_cloud)

    x = xyz[:,0]
    y = xyz[:,1]

    x0 = np.mean(x) if x0 is None else x0
    y0 = np.mean(y) if y0 is None else y0

    xt = x-x0
    yt = y-y0

    deltarad = np.deg2rad(dip)
    thetarad = np.deg2rad(strike)
    X1p = xt * np.cos(np.pi - thetarad) + yt * np.sin(np.pi - thetarad)
    Zeta = (X1p / d) - (1 / np.tan(deltarad))
    u1 = (u_ds / np.pi) * (np.cos(deltarad) * np.arctan(Zeta) + (np.sin(deltarad) - Zeta * np.cos(deltarad)) / (
                1 + np.power(Zeta, 2)))
    u3 = (-u_ds / np.pi) * (np.sin(deltarad) * np.arctan(Zeta) + (np.cos(deltarad) + Zeta * np.sin(deltarad)) / (
                1 + np.power(Zeta, 2)))
    u2 = (u_ss / np.pi) * (
                np.arctan2(Zeta * np.power(np.sin(deltarad), 2), (1 - Zeta * np.sin(deltarad) * np.cos(deltarad))) + (
                    deltarad - np.sign(deltarad) * np.pi / 2.0))

    u1p = u1 * np.cos(thetarad - np.pi) + u2 * np.sin(thetarad - np.pi)
    u2p = -u1 * np.sin(thetarad - np.pi) + u2 * np.cos(thetarad - np.pi)

    deformed_points = xyz + np.vstack((u1p, u2p, u3)).T

    return put_xyz_to_pdal(read_file(point_cloud), deformed_points) if isinstance(point_cloud, str) else \
        list(put_xyz_to_pdal(point_cloud[0])) if isinstance(point_cloud, list) \
            else put_xyz_to_pdal(point_cloud) if point_cloud.dtype.fields is not None else deformed_points


