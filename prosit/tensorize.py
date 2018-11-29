import collections
import numpy

from . import constants
from . import utils


# helpers


def stack(queue):
    listed = collections.defaultdict(list)
    for t in queue.values():
        if t is not None:
            for k, d in t.items():
                listed[k].append(d)
    stacked = {}
    for k, d in listed.items():
        if isinstance(d[0], list):
            stacked[k] = [item for sublist in d for item in sublist]
        else:
            stacked[k] = numpy.vstack(d)
    return stacked


def get_numbers(vals, dtype=float):
    a = numpy.array(vals).astype(dtype)
    return a.reshape([len(vals), 1])


def get_precursor_charge_onehot(charges):
    array = numpy.zeros([len(charges), max(constants.CHARGES)], dtype=int)
    for i, precursor_charge in enumerate(charges):
        array[i, precursor_charge - 1] = 1
    return array


def get_sequence_integer(sequences):
    array = numpy.zeros([len(sequences), constants.MAX_SEQUENCE], dtype=int)
    for i, sequence in enumerate(sequences):
        for j, s in enumerate(utils.peptide_parser(sequence)):
            array[i, j] = constants.ALPHABET[s]
    return array


# file types


def peptidelist(df):
    df.reset_index(drop=True, inplace=True)
    assert "modified_sequence" in df.columns
    assert "collision_energy" in df.columns
    assert "precursor_charge" in df.columns
    tensor = {
        "collision_energy_aligned_normed": get_numbers(df.collision_energy) / 100.,
        "sequence_integer": get_sequence_integer(df.modified_sequence),
        "precursor_charge_onehot": get_precursor_charge_onehot(df.precursor_charge),
    }
    return tensor


def msms_txt(df):
    # TODO: implement
    pass
