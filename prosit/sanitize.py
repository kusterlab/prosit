import numpy
import functools
from .constants import *
from . import losses


def reshape_dims(array):
    n, dims = array.shape
    assert dims == 174
    nlosses = 1
    return array.reshape(
        [array.shape[0], MAX_SEQUENCE - 1, len(ION_TYPES), nlosses, MAX_FRAG_CHARGE]
    )


def reshape_flat(array):
    s = array.shape
    flat_dim = [s[0], functools.reduce(lambda x, y: x * y, s[1:], 1)]
    return array.reshape(flat_dim)


def normalize_base_peak(array):
    # flat
    maxima = array.max(axis=1)
    array = array / maxima[:, numpy.newaxis]
    return array


def mask_outofrange(array, lengths, mask=-1.0):
    # dim
    for i in range(array.shape[0]):
        array[i, lengths[i] - 1 :, :, :, :] = mask
    return array


def cap(array, nlosses=1, z=3):
    return array[:, :, :, :nlosses, :z]


def mask_outofcharge(array, charges, mask=-1.0):
    # dim
    for i in range(array.shape[0]):
        if charges[i] < 3:
            array[i, :, :, :, charges[i] :] = mask
    return array


def get_spectral_angle(true, pred, batch_size=600):
    import tensorflow

    n = true.shape[0]
    sa = numpy.zeros([n])

    def iterate():
        if n > batch_size:
            for i in range(n // batch_size):
                true_sample = true[i * batch_size : (i + 1) * batch_size]
                pred_sample = pred[i * batch_size : (i + 1) * batch_size]
                yield i, true_sample, pred_sample
            i = n // batch_size
            yield i, true[(i) * batch_size :], pred[(i) * batch_size :]
        else:
            yield 0, true, pred

    for i, t_b, p_b in iterate():
        tensorflow.reset_default_graph()
        with tensorflow.Session() as s:
            sa_graph = losses.masked_spectral_distance(t_b, p_b)
            sa_b = 1 - s.run(sa_graph)
            sa[i * batch_size : i * batch_size + sa_b.shape[0]] = sa_b
    sa = numpy.nan_to_num(sa)
    return sa


def prediction(data, batch_size=600):
    assert "sequence_integer" in data
    assert "intensities_pred" in data
    assert "precursor_charge_onehot" in data

    sequence_lengths = numpy.count_nonzero(data["sequence_integer"], axis=1)
    intensities = data["intensities_pred"]
    charges = list(data["precursor_charge_onehot"].argmax(axis=1) + 1)

    intensities[intensities < 0] = 0
    intensities = normalize_base_peak(intensities)
    intensities = reshape_dims(intensities)
    intensities = mask_outofrange(intensities, sequence_lengths)
    intensities = mask_outofcharge(intensities, charges)
    intensities = reshape_flat(intensities)
    data["intensities_pred"] = intensities

    if "intensities_raw" in data:
        data["spectral_angle"] = get_spectral_angle(
            data["intensities_raw"], data["intensities_pred"], batch_size=batch_size
        )
    return data
