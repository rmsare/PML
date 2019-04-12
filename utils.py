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
        tile_row = []
        for x in range(0,num_x):
            tile_row += [((d_tile*x+offset_x, d_tile*(x+1)+offset_x),(d_tile*y+offset_y, d_tile*(y+1)+offset_y))]
        tiles += [tuple(tile_row)]
    tiles = tuple(tiles)
    if buffer is None and buffer_fraction is None:
        buffered_tiles = None
    else:
        if buffer is None:
            buffer_pixels = d_tile * buffer_fraction
        else:
            buffer_pixels = buffer
        buffered_tiles = []
        for y in range(0,num_y):
            buffered_tile_row = []
            for x in range(0,num_x):
                buffered_tile_row += [((tiles[y][x][0][0] - buffer_pixels,
                                        tiles[y][x][0][1] + buffer_pixels),
                                        (tiles[y][x][1][0] - buffer_pixels,
                                        tiles[y][x][1][1] + buffer_pixels))]
            buffered_tiles += [tuple(buffered_tile_row)]
        buffered_tiles = tuple(buffered_tiles)

    return (tiles, buffered_tiles)
