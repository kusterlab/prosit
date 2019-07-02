import numpy

from . import tensorize

ACE_RANGE = list(range(18, 40, 1))


def get_alignment_tensor(tensor, subset_size=10000):
    mask_score = (tensor["score"] > 100).reshape(tensor["score"].shape[0])
    mask_decoy = (tensor["reverse"] == False).reshape(tensor["score"].shape[0])
    tm = {key: data[mask_score & mask_decoy] for key, data in tensor.items()}
    if tm["score"].shape[0] < subset_size:
        subset_idx = range(tm["score"].shape[0])
    else:
        idx = list(range(tm["intensities_raw"].shape[0]))
        numpy.random.shuffle(idx)
        subset_idx = idx[:10000]
    alignment_tensors = {}
    for cea in ACE_RANGE:
        tmp = {k: d[subset_idx] for k, d in tm.items()}
        tmp["collision_energy_aligned"] = tmp["collision_energy"] * 0 + cea
        tmp["collision_energy_aligned_normed"] = tmp["collision_energy_aligned"] / 100.0
        alignment_tensors[cea] = tmp
    alignment_tensor = tensorize.stack(alignment_tensors)
    return alignment_tensor


def get_ace_dist(tensor):
    dist = {}
    for ace in ACE_RANGE:
        mask_ace = tensor["collision_energy_aligned_normed"] == ace / 100.0
        mask_ace = mask_ace.reshape(mask_ace.shape[0])
        sa = numpy.median(tensor["spectral_angle"][mask_ace])
        dist[int(ace)] = sa
    return dist


def get_ace(tensor):
    dist = get_ace_dist(tensor)
    return max(dist, key=dist.get)
