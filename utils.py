import numpy as np

def tiles_for_bounds(bounds, buffer = None, buffer_fraction = None):

    assert not (buffer is not None and buffer_fraction is not None), "Keyword arguments buffer and buffer_fraction cannot be simultaneously specified"

    dx = np.round((bounds[0][1] - bounds[0][0])*1000)/1000.0
    dy = np.round((bounds[1][1] - bounds[1][0])*1000)/1000.0

    if dx >= dy:

        num_y = 2
        d_tile = dy / 2.0
        offset_y = bounds[1][0]

        num_x = int(np.ceil(dx / d_tile))
        offset_x = bounds[0][0] - 0.5*(d_tile*num_x - dx)

    else:

        num_x = 2
        d_tile = dx / 2.0
        offset_x = bounds[0][0]

        num_y = int(np.ceil(dy / d_tile))
        offset_y = bounds[1][0] - 0.5*(d_tile*num_y - dy)

    tiles = []
    for y in range(0,num_y):
        for x in range(0,num_x):
            tiles += [((d_tile*x+offset_x, d_tile*(x+1)+offset_x),(d_tile*y+offset_y, d_tile*(y+1)+offset_y))]
    tiles = tuple(tiles)
    if buffer is None and buffer_fraction is None:
        buffered_tiles = None
    else:
        if buffer is None:
            buffer_pixels = d_tile * buffer_fraction
        else:
            buffer_pixels = buffer
        buffered_tiles = []
        for tile in tiles:
            buffered_tiles += [((tile[0][0] - buffer_pixels,
                                tile[0][1] + buffer_pixels),
                                (tile[1][0] - buffer_pixels,
                                tile[1][1] + buffer_pixels))]
        buffered_tiles = tuple(buffered_tiles)

    return (tiles, buffered_tiles)

def get_xyz_from_pdal(pdal_array):

    if isinstance(pdal_array, list):
        pdarray = pdal_array[0]
    else:
        pdarray = pdal_array
    x = pdarray.dtype.fields['X']
    y = pdarray.dtype.fields['Y']
    z = pdarray.dtype.fields['Z']
    xyz = np.zeros((pdarray.size, 3), dtype=np.float64)
    xyz[:,0] = pdarray.getfield(x[0], offset = x[1]).astype(np.float64)
    xyz[:,1] = pdarray.getfield(y[0], offset = y[1]).astype(np.float64)
    xyz[:,2] = pdarray.getfield(z[0], offset = z[1]).astype(np.float64)

    return np.ascontiguousarray(xyz)

def put_xyz_to_pdal(pdal_array, xyz):

    from copy import deepcopy
    output_pdal_array = deepcopy(pdal_array)
    if isinstance(pdal_array, list):
        x = pdal_array[0].dtype.fields['X']
        y = pdal_array[0].dtype.fields['Y']
        z = pdal_array[0].dtype.fields['Z']
        output_pdal_array[0].setfield(xyz[:,0].astype(x[0]), x[0], offset = x[1])
        output_pdal_array[0].setfield(xyz[:,1].astype(y[0]), y[0], offset = y[1])
        output_pdal_array[0].setfield(xyz[:,2].astype(z[0]), z[0], offset = z[1])
    else:
        x = pdal_array.dtype.fields['X']
        y = pdal_array.dtype.fields['Y']
        z = pdal_array.dtype.fields['Z']
        output_pdal_array.setfield(xyz[:, 0].astype(x[0]), x[0], offset=x[1])
        output_pdal_array.setfield(xyz[:, 1].astype(y[0]), y[0], offset=y[1])
        output_pdal_array.setfield(xyz[:, 2].astype(z[0]), z[0], offset=z[1])
    return output_pdal_array


def transform_xyz(xyz, transform, center = np.array([0.0, 0.0, 0.0])):

    xyzb = np.zeros((xyz.shape[0],4))
    xyzb[:,0:3] = xyz - center
    xyzb[:,3] = 1.0

    return np.matmul(xyzb, transform.T)[:,0:3] + center
