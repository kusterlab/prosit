import numpy

from . import annotate
from . import constants


def read_attribute(row, attribute):
    if " " not in str(row[attribute]):
        return []
    else:
        return [float(m) for m in row[attribute].split(" ")]


def peptide_parser(p):
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


def get_forward_backward(peptide):
    amino_acids = peptide_parser(peptide)
    masses = [constants.AMINO_ACID[a] for a in amino_acids]
    forward = numpy.cumsum(masses)
    backward = numpy.cumsum(list(reversed(masses)))
    return forward, backward


def get_tolerance(theoretical, mass_analyzer):
    if mass_analyzer in constants.TOLERANCE:
        tolerance, unit = constants.TOLERANCE[mass_analyzer]
        if unit == "ppm":
            return theoretical * float(tolerance) / 10 ** 6
        elif unit == "da":
            return float(tolerance)
        else:
            raise ValueError("unit {} not implemented".format(unit))
    else:
        raise ValueError("no tolerance implemented for {}".format(mass_analyzer))


def is_in_tolerance(theoretical, observed, mass_analyzer):
    mz_tolerance = get_tolerance(theoretical, mass_analyzer)
    lower = observed - mz_tolerance
    upper = observed + mz_tolerance
    return theoretical >= lower and theoretical <= upper


def binarysearch(masses_raw, theoretical, mass_analyzer):
    lo, hi = 0, len(masses_raw) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if is_in_tolerance(theoretical, masses_raw[mid], mass_analyzer):
            return mid
        elif masses_raw[mid] < theoretical:
            lo = mid + 1
        elif theoretical < masses_raw[mid]:
            hi = mid - 1
    return None


def match(row, ion_types, max_charge=constants.DEFAULT_MAX_CHARGE):
    masses_observed = read_attribute(row, "masses_raw")
    intensities_observed = read_attribute(row, "intensities_raw")
    forward_sum, backward_sum = get_forward_backward(row.modified_sequence[1:-1])
    _max_charge = row.charge if row.charge <= max_charge else max_charge
    matches = []
    for charge_index in range(_max_charge):
        d = {
            "masses_raw": [],
            "masses_theoretical": [],
            "intensities_raw": [],
            "matches": [],
        }
        charge = charge_index + 1
        annotations = annotate.get_annotation(
            forward_sum, backward_sum, charge, ion_types
        )
        for annotation, mass_t in annotations.items():
            index = binarysearch(masses_observed, mass_t, row.mass_analyzer)
            if index is not None:
                d["masses_raw"].append(masses_observed[index])
                d["intensities_raw"].append(intensities_observed[index])
                d["masses_theoretical"].append(mass_t)
                d["matches"].append(annotation)
        matches.append(d)
    return matches


def c_lambda(matches, charge, attr):
    def mapping(i):
        charge_index = int(charge - 1)
        m = matches[i]
        if charge_index < len(m):
            try:
                s = ";".join(map(str, m[charge_index][attr]))
            except:
                raise ValueError(m[charge_index][attr])
        else:
            s = ""
        return s

    return mapping


def augment(df, ion_types, charge_max):
    matches = {}
    for i, row in df.iterrows():
        matches[i] = match(row, ion_types, charge_max)

    # augment dataframe and write
    for charge in range(1, charge_max + 1):
        df["matches_charge{}".format(charge)] = df.index.map(
            c_lambda(matches, charge, "matches")
        )
        df["masses_the_charge{}".format(charge)] = df.index.map(
            c_lambda(matches, charge, "masses_theoretical")
        )
        df["masses_raw_charge{}".format(charge)] = df.index.map(
            c_lambda(matches, charge, "masses_raw")
        )
        df["intensities_raw_charge{}".format(charge)] = df.index.map(
            c_lambda(matches, charge, "intensities_raw")
        )

    return df
