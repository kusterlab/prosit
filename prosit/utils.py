from .constants import MAX_ION, ION_TYPES, ALPHABET_S


def check_mandatory_keys(dictionary, keys):
    for key in keys:
        if key not in dictionary.keys():
            raise KeyError("key {} is missing".format(key))
    return True


def reshape_dims(array, nlosses=1, z=3):
    return array.reshape([array.shape[0], MAX_ION, len(ION_TYPES), nlosses, z])


def get_sequence(sequence):
    d = ALPHABET_S
    return "".join([d[i] if i in d else "" for i in sequence])


def sequence_integer_to_str(array):
    sequences = [get_sequence(array[i]) for i in range(array.shape[0])]
    return sequences


def peptide_parser(p):
    p = p.replace("_", "")
    if p[0] == "(":
        raise ValueError("sequence starts with '('")
    n = len(p)
    i = 0
    while i < n:
        if i < n - 3 and p[i + 1] == "(":
            j = p[i + 2 :].index(")")
            offset = i + j + 3
            yield p[i:offset]
            i = offset
        else:
            yield p[i]
            i += 1
