import pdal
import numpy as np

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
        """
        pipeline = pdal.Pipeline(json)

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

def icp(fixed, moving):

    if isinstance(fixed, str) and isinstance(moving, str):
        json = u"""
        {
            "pipeline": [
                \"""" + fixed + """\",
                \"""" + moving + """\",
                {
                    "type": "filters.icp"
                }
            ]
        }"""
        pipeline = pdal.Pipeline(json)
    else:
        fixed_array = read_file(fixed) if isinstance(fixed, str) else None
        moving_array = read_file(moving) if isinstance(moving, str) else None
        arrays = [fixed_array[0] if fixed_array is not None else fixed[0],
                  moving_array[0] if moving_array is not None else moving[0]]
        json = u"""
        {
            "pipeline": [
                {
                    "type": "filters.icp"
                }
            ]
        }"""
        pipeline = pdal.Pipeline(json, arrays = arrays)

    pipeline.validate()
    pipeline.loglevel = 8
    pipeline.execute()
    return (pipeline.arrays, pipeline.metadata)


