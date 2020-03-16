import pdal
import numpy as np
from laspy.file import File

def read_file(filename, bounds = None):

    if bounds is None:
        json = u"""
                {
                    "pipeline": [
                     \"""" + filename + """\"
                    ]
                }"""
    else:
        json = u"""
                {
                  "pipeline": [
                    \"""" + filename + """\",
                    {
                        "type":"filters.crop",
                        "bounds":"([""" + str(bounds[0][0]) + """,""" \
                                        + str(bounds[0][1]) + """],[""" \
                                        + str(bounds[1][0]) + """,""" \
                                        + str(bounds[1][1]) + """])"
                    }
                  ]
                }"""
    pipeline = pdal.Pipeline(json)
    pipeline.validate()
    pipeline.loglevel = 8
    pipeline.execute()
    return pipeline.arrays

def crop_to_tiles(pointcloud, x, y, dx_window, dy_window, buffer_fraction = 0.0, client = None):

    from .utils import get_xyz_from_pdal

    pointcloud = get_xyz_from_pdal(read_file(pointcloud)) if isinstance(pointcloud,str) else \
        get_xyz_from_pdal(pointcloud[0]) if isinstance(pointcloud, list) else pointcloud if pointcloud.dtype.fields \
        is None else get_xyz_from_pdal(pointcloud)

    base_dx = np.mean(np.diff(x)) if dx_window is None else dx_window
    base_dy = np.mean(np.diff(y)) if dy_window is None else dy_window

    (dx_half, dy_half) = (base_dx*(1 + buffer_fraction)/2.0, base_dy*(1+buffer_fraction)/2.0)

    X, Y = np.meshgrid(x, y)

    (ny, nx) = X.shape

    data = tuple()

    for i in range(ny):
        row_point_cloud = pointcloud[np.where((pointcloud[:,1] >= (Y[i,0] - dy_half)) & \
            (pointcloud[:,1] <= (Y[i,0] + dy_half)))]
        for j in range(nx):
            if client is not None:
                data += (client.scatter(row_point_cloud[np.where((row_point_cloud[:,0] >= (X[0,j] - dx_half)) & \
                    (row_point_cloud[:,0] <= (X[0,j]) + dx_half))]), )
            else:
                data += (row_point_cloud[np.where((row_point_cloud[:, 0] >= (X[0, j] - dx_half)) & \
                                                                 (row_point_cloud[:, 0] <= (X[0, j]) + dx_half))],)

    tiles = tuple();
    for i in range(ny):
        for j in range(nx):
            tiles += ((i,j),)

    return tiles, data

def write_file(filename, arrays):

    json = u"""
                {
                    "pipeline": [
                      {
                        "type":"writers.las",
                        "filename":\"""" + filename + """\"
                      }
                    ]
                }"""


    pipeline = pdal.Pipeline(json, arrays = arrays)
    pipeline.validate()
    pipeline.loglevel = 8
    pipeline.execute()

def get_bounds(arg):

    if isinstance(arg, str):

        '''
                        json = u"""
        {
            "pipeline": [
                \"""" + arg + """\",
                {
                    "type":"filters.stats",
                    "dimensions":"X,Y,Z"
                }
            ]
        }
        pipeline = pdal.Pipeline(json)
        '''

        f = File(arg, mode='r')
        min_values = f.header.min
        max_values = f.header.max
        return((min_values[0],max_values[0]),(min_values[1],max_values[1]))

    else:
        json = u"""
        {
            "pipeline": [
                {
                    "type":"filters.stats",
                    "dimensions":"X,Y,Z"
                }
            ]
        }
        """
        pipeline = pdal.Pipeline(json, arrays=arg)
        pipeline.validate()
        pipeline.loglevel = 8
        pipeline.execute()
        import json as j
        metadata = j.loads(pipeline.metadata)
        metadata['metadata']['filters.stats'][0]['bbox']['native']['bbox']['minx']
        return ((metadata['metadata']['filters.stats'][0]['bbox']['native']['bbox']['minx'],
                metadata['metadata']['filters.stats'][0]['bbox']['native']['bbox']['maxx']),
                (metadata['metadata']['filters.stats'][0]['bbox']['native']['bbox']['miny'],
                metadata['metadata']['filters.stats'][0]['bbox']['native']['bbox']['maxy']))

def crop(arrays, bounds):

    json = u"""
                    {
                      "pipeline": [
                        {
                            "type":"filters.crop",
                            "bounds":"([""" + str(bounds[0][0]) + """,""" \
                                            + str(bounds[0][1]) + """],[""" \
                                            + str(bounds[1][0]) + """,""" \
                                            + str(bounds[1][1]) + """])"
                        }
                      ]
                    }"""
    pipeline = pdal.Pipeline(json, arrays = arrays)
    pipeline.validate()
    pipeline.loglevel = 8
    pipeline.execute()
    return pipeline.arrays

def transform(arg, transformation_matrix):

    matrix_string = " ".join([str(e) for trans_row in transformation_matrix for e in trans_row])

    if isinstance(arg, str):
        json = u"""
                        {
                          "pipeline": [
                            \"""" + arg + """\",
                            {
                                "type":"filters.transformation",
                                "matrix":\"""" + matrix_string + """\"
                            }
                          ]
                        }"""
        pipeline = pdal.Pipeline(json)
    else:
        json = u"""
                        {
                            "pipeline": [
                              {
                                    "type":"filters.transformation",
                                    "matrix":\"""" + matrix_string + """\"
                              }
                            ]
                        }"""
        pipeline = pdal.Pipeline(json, arrays = arg)

    pipeline.validate()
    pipeline.loglevel = 8
    pipeline.execute()
    return pipeline.arrays

def transform_pdal_array(pdal_array, transform_matrix, center = np.array([0.0, 0.0, 0.0])):

    from .utils import get_xyz_from_pdal, transform_xyz, put_xyz_to_pdal
    xyz = get_xyz_from_pdal(pdal_array)
    xyz_transformed = transform_xyz(xyz, transform_matrix, center = center)
    return put_xyz_to_pdal(pdal_array, xyz_transformed)

def icp(fixed, moving, center = None):

    from .utils import get_xyz_from_pdal

    fixed_array = read_file(fixed) if isinstance(fixed, str) else fixed
    moving_array = read_file(moving) if isinstance(moving, str) else moving


    XYZ_fixed = fixed_array if fixed_array.dtype.fields is None else get_xyz_from_pdal(fixed_array)
    XYZ_moving = moving_array if moving_array.dtype.fields is None else get_xyz_from_pdal(moving_array)

    if XYZ_fixed.shape[0] > 5 and XYZ_moving.shape[0] > 5:
        if center is None:
            center = np.mean(XYZ_fixed, axis=0)

        import pyicp

        transform_matrix, residual = pyicp.icp(XYZ_fixed - center, XYZ_moving - center)

        transformed_array = transform_pdal_array(moving_array, transform_matrix, center = center)


    else:
        transformed_array, (transform_matrix, center, residual) = ([np.array([])], (np.array([[1.0, 0.0, 0.0, 0.0],
                                                                                              [0.0, 1.0, 0.0, 0.0],
                                                                                              [0.0, 0.0, 1.0, 0.0],
                                                                                              [0.0, 0.0, 0.0,
                                                                                               1.0]]),
                                                                                    center, None))
    return transformed_array, (transform_matrix, center, residual)

def icp_calc_displacement(fixed_tile, moving_tile, center, min_num_points = 15):

    if fixed_tile.shape[0] > min_num_points and moving_tile.shape[0] > min_num_points:
        if center is None:
            center = np.mean(fixed_tile, axis=0)

        import pyicp

        transform_matrix, residual = pyicp.icp(fixed_tile - center, moving_tile - center)

    else:
        transform_matrix, residual = ((np.array([[1.0, 0.0, 0.0, 0.0],
                                                 [0.0, 1.0, 0.0, 0.0],
                                                 [0.0, 0.0, 1.0, 0.0],
                                                 [0.0, 0.0, 0.0, 1.0]])), None)

    from numpy.linalg import inv
    return inv(transform_matrix).T[-1,0:3]

def icp_tile(fixed, moving, x, y, buffer_fraction = 0.5, dx_window = None, dy_window = None, use_dask = True, \
             distributed = False, min_num_points = 15):
    from .utils import get_xyz_from_pdal

    def calc_u_tile(fixed_tile, moving_tile, ij, xyc):
        fixed_tile = np.unique(fixed_tile, axis = 0)
        moving_tile = np.unique(moving_tile, axis = 0)
        if fixed_tile.shape[0] <= min_num_points or moving_tile.shape[0] <= min_num_points:
            displacements = np.array([0,0,0])
        else:
            mean_z = np.mean(fixed_tile, axis=0)[2]
            (xc, yc) = xyc
            position = np.array([xc, yc, mean_z])
            displacements = icp_calc_displacement(fixed_tile, moving_tile, position)
        print('done with', ij, flush = True)
        return displacements, ij

    if use_dask and distributed:
        from dask.distributed import Client
        client = Client()
    elif use_dask:
        from dask.delayed import delayed
        from dask.dataframe import compute
        client = None
    else:
        client = None

    X, Y = np.meshgrid(x, y)
    UX = np.zeros_like(X)
    UY = np.zeros_like(X)
    UZ = np.zeros_like(X)

    print('Loading acquisition 1.', flush=True)
    (ij, fixed_tiles) = crop_to_tiles(fixed, x, y, dx_window=dx_window, dy_window=dy_window, buffer_fraction=0.0, client = client)
    print('Done', flush=True)
    print('Loading acqusition 2.', flush = True)
    (ij_m, moving_tiles) = crop_to_tiles(moving, x, y, dx_window=dx_window, dy_window=dy_window, buffer_fraction=buffer_fraction, client = client)
    print('Done', flush = True)

    tasks = []

    for i in range(len(ij)):
        tasks.append(client.submit(calc_u_tile, fixed_tiles[i], moving_tiles[i], ij[i], (X[ij[i][0],ij[i][1]], \
            Y[ij[i][0],ij[i][1]])) if use_dask and distributed else delayed(calc_u_tile)(fixed_tiles[i], \
            moving_tiles[i], ij[i], (X[ij[i][0],ij[i][1]], Y[ij[i][0],ij[i][1]])) if use_dask else \
            calc_u_tile(fixed_tiles[i], moving_tiles[i], ij[i], (X[ij[i][0],ij[i][1]], Y[ij[i][0],ij[i][1]])))

    results = client.gather(tasks) if use_dask and distributed else compute(*tasks) if use_dask else tasks

    for ((ux, uy, uz), (i, j)) in results:
        UX[i, j] = ux
        UY[i, j] = uy
        UZ[i, j] = uz

    return UX, UY, UZ



def icp_scale(fixed, moving, x, y, max_scale = 4, buffer_fraction = 0.5):

    from numpy.linalg import inv

    base_dx = np.mean(np.diff(x))
    base_dy = np.mean(np.diff(y))

    def calc_transform(fixed, moving, x, scale, center):

        dx = base_dx * np.power(2, scale)
        dy = base_dy * np.power(2, scale)

        if len(fixed[0]) == 0 or len(moving[0]) == 0:
            fixed_tile_clip = fixed
            moving_tile_clip = moving
        else:
            bounds = ((x[0] - dx, x[0] + dx), (x[1] - dy, x[1] + dy))
            buffered_bounds = ((x[0] - (1+buffer_fraction), x[0] + (1+buffer_fraction)*dx), (x[1] - (1+buffer_fraction)*dy, x[1] + (1+buffer_fraction)*dy))
            fixed_tile_clip = crop(fixed, bounds)
            moving_tile_clip = crop(moving, buffered_bounds)

        if len(fixed_tile_clip[0]) > 5 and len(moving_tile_clip[0]) > 5:
            transformed_array, (transform_matrix, center, residual) = icp(fixed_tile_clip, moving_tile_clip,
                                                                      center=center)
        else:
            transformed_array, (transform_matrix, center, residual) = ([np.array([])], (np.array([[1.0, 0.0, 0.0, 0.0],
                                                                                                  [0.0, 1.0, 0.0, 0.0],
                                                                                                  [0.0, 0.0, 1.0, 0.0],
                                                                                                  [0.0, 0.0, 0.0,
                                                                                                   1.0]]),
                                                                                        center, None))

        if scale != 0:
            scale -= 1
            newpos = calc_transform(fixed_tile_clip, transformed_array, x, scale, center)
            return np.matmul(np.concatenate((np.array([newpos - center]), np.ones((1,1))), axis = 1), inv(transform_matrix).T)[0,0:3] + center
        else:
            return np.matmul(np.concatenate((np.array([center - center]), np.ones((1,1))), axis = 1), inv(transform_matrix).T)[0,0:3] + center


    fixed_array = read_file(fixed) if isinstance(fixed, str) else fixed
    moving_array = read_file(moving) if isinstance(moving, str) else moving



    X, Y = np.meshgrid(x, y)
    UX = np.zeros_like(X)
    UY = np.zeros_like(X)
    UZ = np.zeros_like(X)

    (ny, nx) = X.shape
    from .utils import get_xyz_from_pdal

    meanz = np.mean(get_xyz_from_pdal(fixed_array), axis = 0)[2]

    def calculate_u(xyc, ij):
        (xc, yc) = xyc
        (i, j) = ij
        position = np.array([xc, yc, meanz])
        u = calc_transform(fixed_array, moving_array, position, max_scale, position) - position
        print('done with: ', (i, j))
        return u, (i, j)

    from dask import compute, delayed
    import dask

    with dask.config.set(scheduler='processes'):
        dask_tasks = [delayed(calculate_u)((X[i,j], Y[i,j]), (i, j)) for i in range(ny) for j in range(nx)]
        results = compute(*dask_tasks)

    for ((ux, uy, uz), (i, j)) in results:
        UX[i,j] = ux
        UY[i,j] = uy
        UZ[i,j] = uz

    return UX, UY, UZ

def icp_recursive(fixed, moving, min_dx = 1.0, min_size_to_thread = 1.5E4, buffer_fraction=0.5):

    def find_center(tile, fixed_array):
        from .utils import get_xyz_from_pdal
        center_xy = ((tile[0][1] + tile[0][0]) / 2.0, (tile[1][1] + tile[1][0]) / 2.0)
        if fixed_array is not None:
            xyz = get_xyz_from_pdal(fixed_array)
            center_z = np.mean(xyz, axis=0)[2]
        else:
            center_z = np.array([0.0]);
        return np.array([center_xy[0], center_xy[1], center_z])


    fixed_array = read_file(fixed) if isinstance(fixed, str) else fixed
    moving_array = read_file(moving) if isinstance(moving, str) else moving

    bounds = get_bounds(fixed)

    center = find_center(bounds, fixed_array)

    from graph import pml_graph

    graph = pml_graph(center)


    def icp_recursive_evaluation(fixed_tile, moving_tile, tile_extent, buffered_tile_extent, graph, parent_node, min_dx,
                                 min_size_to_thread, buffer_fraction=0.5, center = center):

        in_dimensions = (tile_extent[0][1] - tile_extent[0][0], tile_extent[1][1] - tile_extent[1][0])

        if (in_dimensions[0] <= min_dx) or (in_dimensions[1] <= min_dx):
            return

        node_center = find_center(tile_extent, fixed_array)

        def launch_nodask(fixed_tile_clip, graph, parent_node, min_dx, center = center):

            from .utils import tiles_for_bounds

            (tiles, buffered_tiles) = tiles_for_bounds(tile_extent, buffer_fraction=buffer_fraction)
            transformed_array, (transform_matrix, center, residual) = ([np.array([])], (np.array([[1.0, 0.0, 0.0, 0.0],
                                                                                                  [0.0, 1.0, 0.0, 0.0],
                                                                                                  [0.0, 0.0, 1.0, 0.0],
                                                                                                  [0.0, 0.0, 0.0,
                                                                                                   1.0]]),
                                                                                        center, None))
            node = graph.add_node(node_center, transform_matrix, in_dimensions, residual, from_node=parent_node)
            [icp_recursive_evaluation(fixed_tile_clip, transformed_array, tile, buffered_tile, graph, node, min_dx,
                                      min_size_to_thread) for (tile, buffered_tile) in zip(tiles, buffered_tiles)]

        def launch_dask(fixed_tile_clip, moving_tile_clip, graph, parent_node, min_dx, center = center):

            from .utils import tiles_for_bounds
            from dask import compute, delayed

            transformed_array, (transform_matrix, center, residual) = icp(fixed_tile_clip, moving_tile_clip,
                                                                          center=center)
            node = graph.add_node(node_center, transform_matrix, in_dimensions, residual, from_node=parent_node)
            (tiles, buffered_tiles) = tiles_for_bounds(tile_extent, buffer_fraction=buffer_fraction)
            dask_tasks = [
                delayed(icp_recursive_evaluation)(fixed_tile_clip, transformed_array, tile, buffered_tile, graph, node,
                                                  min_dx, min_size_to_thread) for (tile, buffered_tile) in
                zip(tiles, buffered_tiles)]
            compute(*dask_tasks)

        if (len(fixed_tile[0]) > 0) and (len(moving_tile[0]) > 0):

            fixed_tile_clip = crop(fixed_tile, tile_extent)
            moving_tile_clip = crop(moving_tile, buffered_tile_extent)

            if (len(fixed_tile_clip[0]) >= 5) and (len(moving_tile_clip[0]) >= 5) and (
                    len(fixed_tile_clip[0]) + len(moving_tile_clip[0])) > min_size_to_thread:
                launch_dask(fixed_tile_clip, moving_tile_clip, graph, parent_node, min_dx)
            else:
                launch_nodask(fixed_tile_clip, graph, parent_node, min_dx)
        else:
            launch_nodask(fixed_tile, graph, parent_node, min_dx)

    dimensions = (bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0])

    transformed_array, (transform_matrix, center, residual) = icp(fixed_array, moving_array, center = center)
    parent = graph.add_node(center, transform_matrix, dimensions, residual)

    from .utils import tiles_for_bounds

    (tiles, buffered_tiles) = tiles_for_bounds(bounds, buffer_fraction=buffer_fraction)

    from dask import compute, delayed

    dask_tasks = [delayed(icp_recursive_evaluation)(fixed_array, transformed_array, tile, buffered_tile, graph, parent, min_dx, min_size_to_thread) for (tile,buffered_tile) in zip(tiles, buffered_tiles)]

    compute(*dask_tasks)

    return graph
