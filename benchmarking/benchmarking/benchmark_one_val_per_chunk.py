import sys

sys.path.append('../../')
sys.path.append('../')
from versionedzarrlib.data import VersionedData
# from config import *
from path_config import *
from utils import *
from benchmark import *
import numpy as np
import dask.array as da
from tqdm import tqdm
import zarr

dims = (1000, 1000, 1000)
raw_chunk_size = (1, 1, 1)
index_chunk_size = (64, 64, 64)
iterations = 100
compress_index = True
branches = ["main", "dev"]
import ClusterWrap
from ClusterWrap.decorator import cluster


@cluster
def distributed_fill(data,elms, from_val, to_val, b_time, cluster=None,kwargs={}):

    dask_data = da.from_array(elms)
    print("Dask created")
    dask_data = dask_data.reshape(dims)
    print("Array reshaped")
    dask_data = dask_data.rechunk(index_chunk_size)
    print("rechunked")
    b_time.start_element(Writing_index_time)
    dest = zarr.open(data._index_dataset_path)
    print("Filling data ..")
    da.store(dask_data, dest)
    # data.fill_index_dataset(data=dask_data)
    b_time.done_element()


def main():
    with ClusterWrap.cluster() as cluster:
        print(cluster.client)
        # client = Client(processes=False)
        # print(client)
        # print(client.dashboard_link)

        data = VersionedData(path=data_path, shape=dims, raw_chunk_size=raw_chunk_size,
                             index_chunk_size=index_chunk_size)
        extra = "file_limit_{}_shape_{}_index_{}_compression_{}".format(iterations,
                                                                        format_tuple(
                                                                            dims),
                                                                        format_tuple(
                                                                            index_chunk_size),
                                                                        compress_index)
        data.create(overwrite=True)
        data.vc.checkout_branch("dev", create=True)
        data.vc.add_all()
        data.vc.commit("inital")
        data.vc.checkout_branch("main", create=True)

        # empty_trash()
        size_benchmark = Benchmarking(
            Benchmarking.create_path(current_folder=benchmark_path, elm_type=Type_size, extra=extra))
        size_benchmark.write_line(SizeBenchmark.get_header())
        time_benchmark = Benchmarking(
            Benchmarking.create_path(current_folder=benchmark_path, elm_type=Type_Time, extra=extra))
        time_benchmark.write_line(TimeBenchmark.get_header())

        total = 1
        for i in dims:
            total = total * i

        for i in tqdm(range(iterations)):
            b_time = TimeBenchmark(i)
            size_b = SizeBenchmark(i)

            size_b.add(Initial_size, data.du_size())

            size_b.add(Git_Size_Before, data.git_size())
            elms = np.arange(start=from_val, stop=to_val, dtype=np.uint64)

            print("Got numpy array..")

            np.random.shuffle(elms)
            print("Array shuffled..")
            distributed_fill(data=data,elms=elms, from_val=total * i, to_val=total * (i + 1), b_time=b_time, cluster=cluster)

            size_b.add(After_Write_Before_GIT, data.du_size())

            b_time.start_element(Commit_time)
            data.vc.add_all()
            data.vc.commit(str(i))
            b_time.done_element()
            size_b.add(Git_Size_After, data.git_size())

            b_time.start_element(GC_time)
            data.vc.gc()
            b_time.done_element()
            size_b.add(Git_After_GC, data.git_size())

            b_time.start_element(Checkout_time)
            data.vc.checkout_branch(branch_name=branches[int(i % 2)])
            b_time.done_element()

            size_benchmark.write_line(size_b.format())
            time_benchmark.write_line(b_time.format())


if __name__ == "__main__":
    main()
