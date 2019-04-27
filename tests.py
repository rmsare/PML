import pml
import numpy as np

def test_icp_recursive():
    fixed = pml.read_file('clip_small.txt')
    transform = np.array([[1.0, 0.0, 0.0, -3.2],
                          [0.0, 1.0, 0.0, 4.1],
                          [0.0, 0.0, 1.0, 0.5],
                          [0.0, 0.0, 0.0, 1.0]]);
    moving = pml.transform(fixed, transform)

    transformed_matrix, (transform, residual) = pml.icp(fixed, moving)

def test_icp_scale(dx = 10, dy = 10, max_scale = 4, buffer_fraction = 0.2):
    fixed = pml.read_file('clip_small.txt')
    moving = pml.read_file('test_dataset.las')
    bounds = pml.get_bounds(fixed)
    x = np.arange(bounds[0][0], bounds[0][1], dx)
    y = np.arange(bounds[1][0], bounds[1][1], dy)
    ux, uy, uz = pml.icp_scale(fixed, moving, x, y, max_scale = max_scale, buffer_fraction = buffer_fraction)

    return x, y, ux, uy, uz



def create_test_dataset(in_pt_cloud_filename = None, x_a = 1E-3, x_b = 0.5E-3, x_c = -1.2E-3, x_d = -0.8E-1, x_e = -1.0E-1, x_f = 1.0,
                        y_a = -1.5E-4, y_b = -1.0E-4, y_c = 0.72E-3, y_d = 0.75E-1, y_e = 1.4E-1, y_f = -1.0,
                        z_a = 1E-4, z_b = -4E-4, z_c = -6E-4, z_d = -1E-1, z_e = 1E-1, z_f = 1.0):

    from utils import get_xyz_from_pdal, put_xyz_to_pdal
    from pml import read_file

    if in_pt_cloud_filename is None:
        in_pt_cloud = read_file('clip_small.txt')
    else:
        in_pt_cloud = read_file(in_pt_cloud_filename)

    xyz = get_xyz_from_pdal(in_pt_cloud)
    xyzm = np.mean(xyz, axis = 0)
    xyz -= xyzm

    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]

    dx = x_a * np.power(x,2) + x_b * x * y + x_c * np.power(y,2) + x_d * x + x_e * y + x_f
    dy = y_a * np.power(x,2) + y_b * x * y + y_c * np.power(y,2) + y_d * x + y_e * y + y_f
    dz = z_a * np.power(x,2) + z_b * x * y + z_c * np.power(y,2) + z_d * x + z_e * y + z_f

    xyz[:,0] += dx
    xyz[:,1] += dy
    xyz[:,2] += dz

    return put_xyz_to_pdal(in_pt_cloud, xyz+xyzm)

def create_test_grid(in_pt_cloud, dx = 10, dy = 10, x_a = 1E-3, x_b = 0.5E-3, x_c = -1.2E-3, x_d = -0.8E-1, x_e = -1.0E-1, x_f = 1.0,
                        y_a = -1.5E-4, y_b = -1.0E-4, y_c = 0.72E-3, y_d = 0.75E-1, y_e = 1.4E-1, y_f = -1.0,
                        z_a = 1E-4, z_b = -4E-4, z_c = -6E-4, z_d = -1E-1, z_e = 1E-1, z_f = 1.0):

    from pml import get_bounds
    from utils import get_xyz_from_pdal

    bounds = get_bounds(in_pt_cloud)

    xyz = get_xyz_from_pdal(in_pt_cloud)
    xyzm = np.mean(xyz, axis = 0)

    x = np.arange(bounds[0][0], bounds[0][1], dx) - xyzm[0]
    y = np.arange(bounds[1][0], bounds[1][1], dy) - xyzm[1]

    [x, y] = np.meshgrid(x,y)

    ux = x_a * np.power(x,2) + x_b * x * y + x_c * np.power(y,2) + x_d * x + x_e * y + x_f
    uy = y_a * np.power(x,2) + y_b * x * y + y_c * np.power(y,2) + y_d * x + y_e * y + y_f
    uz = z_a * np.power(x,2) + z_b * x * y + z_c * np.power(y,2) + z_d * x + z_e * y + z_f

    return x, y, ux, uy, uz

def test_one_rotation(fixed, moving):

    from pml import icp, read_file
    from utils import get_xyz_from_pdal

    fixed_array = read_file(fixed)
    moving_array = read_file(moving)

    transformed_array, (transform_matrix, center, residual) = icp(fixed_array, moving_array)

    # manually rotate moving points back:

    xyz_transformed = get_xyz_from_pdal(transformed_array)
    xyz_moving = get_xyz_from_pdal(moving_array) - center

    xyz_moving = np.concatenate((xyz_moving, np.ones((xyz_moving.shape[0], 1))), axis=1)

    xyz_manual_transform = np.matmul(xyz_moving, transform_matrix.T)[:,0:3] + center

    return xyz_transformed, xyz_manual_transform

def test_two_rotations(fixed, moving):

    from pml import icp, read_file, transform_pdal_array
    from utils import get_xyz_from_pdal

    fixed_array = read_file(fixed)
    moving_array = read_file(moving)

    transformed_array, (transform_matrix, center, residual) = icp(fixed_array, moving_array)

    # perform second transformation:

    center_2 = center - 50.0

    transformed_array = transform_pdal_array(transformed_array, transform_matrix, center=center_2)

    # manually rotate moving points back:

    xyz_transformed = get_xyz_from_pdal(transformed_array)
    xyz_moving = get_xyz_from_pdal(moving_array) - center

    xyz_moving = np.concatenate((xyz_moving, np.ones((xyz_moving.shape[0], 1))), axis=1)

    xyz_manual_transform = np.matmul(xyz_moving, transform_matrix.T)[:, 0:3] + center
    xyz_manual_transform = np.concatenate((xyz_manual_transform - center_2, np.ones((xyz_manual_transform.shape[0],1))), axis=1)
    xyz_manual_transform = np.matmul(xyz_manual_transform, transform_matrix.T)[:, 0:3] + center_2

    return xyz_transformed, xyz_manual_transform

