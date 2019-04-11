import pdal

def read_file(filename):

    json = """
    {
      "pipeline": [
        \"""" + filename + """\"
      ]
    }"""

    pipeline = pdal.Pipeline(json)
    pipeline.validate()
    pipeline.loglevel = 8
    pipeline.execute()
    return (pipeline.arrays, pipeline.metadata, pipeline.log)

def get_bounds(arrays):

    json = """
    {
        "pipeline": [
            {
                "type":"filters.stats",
                "dimensions":"X,Y,Z"
            }
        ]
    }
    """

    pipeline = pdal.Pipeline(json, arrays=arrays)
    pipeline.validate()
    pipeline.loglevel = 8
    pipeline.execute()
    return (pipeline.arrays, pipeline.metadata, pipeline.log)