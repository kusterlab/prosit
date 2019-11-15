from . import utils


def get_array(tensor, keys):
    utils.check_mandatory_keys(tensor, keys)
    return [tensor[key] for key in keys]


def to_hdf5(dictionary, path):
    import h5py

    with h5py.File(path, "w") as f:
        for key, data in dictionary.items():
            f.create_dataset(key, data=data, dtype=data.dtype, compression="gzip")


def from_hdf5(path, n_samples=None):
    from keras.utils import HDF5Matrix
    import h5py
    
    # Get a list of the keys for the datasets
    with h5py.File(path, 'r') as f:
        dataset_list = list(f.keys())
    
    # Assemble into a dictionary
    data = dict()
    for dataset in dataset_list:
        data[dataset] = HDF5Matrix(path, dataset, start=0, end=n_samples, normalizer=None)
    return data
