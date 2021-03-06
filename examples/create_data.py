import sys

sys.path.append('../')

from versionedzarrlib import VersionedData
import numpy as np
import zarr

root_path = "/Users/zouinkhim/Desktop/versioned_data"

dims = (600, 600, 600)
chunk_size = (128, 128, 128)
data = VersionedData(path=root_path, shape=dims, raw_chunk_size=chunk_size)
data.create(overwrite=True)


# dummy_data = np.ones(data.raw_chunk_size, dtype='i8')
