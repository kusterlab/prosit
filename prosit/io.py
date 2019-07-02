from . import utils


def get_array(tensor, keys):
    utils.check_mandatory_keys(tensor, keys)
    return [tensor[key] for key in keys]


def to_hdf5(dictionary, path):
    import h5py

    with h5py.File(path, "w") as f:
        for key, data in dictionary.items():
            f.create_dataset(key, data=data, dtype=data.dtype, compression="gzip")


def from_hdf5(path):
    import h5py

    with h5py.File(path, "r") as f:
        data = {k: f[k][...] for k in f.keys()}
    return data
