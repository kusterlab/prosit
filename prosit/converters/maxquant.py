import pandas

import os

from .. import constants
from .. import match
from .. import annotate
from .. import utils

COL_SEP = "\t"


def rename_column(attribute):
    lower_no_spaces = attribute.lower().replace(" ", "_")
    bad_chars = ".[]()/"
    return lower_no_spaces.translate({ord(c): None for c in bad_chars})


def read(filepath, low_memory=False):
    COL_SEP = "\t"
    CONVERTERS = {"Reverse": lambda r: True if r == "+" else False}
    TYPES = {
        "Type": "object",
        "Masses": "object",
        "Intensities": "object",
        "Matches": "object",
        "Sequence": "object",
        "Modifications": "object",
        "Modified sequence": "object",
        "Raw file": "object",
        "Score": float,
        "Precursor Intensity": float,
        "Mass": float,
        "Mass Error [ppm]": float,
        "Delta score": float,
        "Peptide ID": int,
        "Scan event number": int,
        "Scan number": int,
        "Charge": int,
        "Reverse": bool,
    }

    # fix different maxquant formats
    header = pandas.read_csv(filepath, nrows=1, sep=COL_SEP).columns
    if "Mass Error [ppm]" not in header and "Mass error [ppm]" in header:
        TYPES["Mass error [ppm]"] = TYPES.pop("Mass Error [ppm]")

    df = pandas.read_csv(
        filepath,
        header="infer",
        sep=COL_SEP,
        usecols=TYPES.keys(),
        dtype=TYPES,
        converters=CONVERTERS,
        low_memory=low_memory,
    )
    df.columns = list(map(rename_column, df.columns))
    df = df[df.type.map(lambda x: x != "MULTI-SECPEP")]
    df = df.set_index("scan_number", drop=False)
    return df


def write(df, filepath):
    df.to_csv(filepath, sep=COL_SEP, index=False)


def convert_prediction(tensor):
    assert "intensities_pred" in tensor
    assert "sequence_integer" in tensor
    assert "precursor_charge_onehot" in tensor
    intensities_pred = utils.reshape_dims(tensor["intensities_pred"])
    modified_sequences = utils.sequence_integer_to_str(tensor["sequence_integer"])
    natural_losses_max = 0

    def convert_row(i):
        modified_sequence = modified_sequences[i]
        fw, bw = match.get_forward_backward(modified_sequence)
        mzs = []
        ions = []
        intes = []
        for fz in range(constants.MAX_FRAG_CHARGE):
            ann = annotate.get_annotation(fw, bw, fz + 1, "yb")
            for fty_i, fty in enumerate(constants.ION_TYPES):
                for fi in range(constants.MAX_ION):
                    ion = fty + str(fi + 1)
                    inte = intensities_pred[i, fi, fty_i, natural_losses_max, fz]
                    if inte > 0:
                        mz = ann[ion]
                        if fz > 0:
                            ion += "({}+)".format(fz + 1)
                        mzs.append(mz)
                        ions.append(ion)
                        intes.append(inte)
                    else:
                        continue

        mzs_s = ";".join(map(str, mzs))
        matches_s = ";".join(ions)
        ints_s = ";".join(map(str, intes))
        return mzs_s, matches_s, ints_s

    masses_c = []
    matches_c = []
    ints_c = []
    for i in range(len(modified_sequences)):
        mzs, matches, ints = convert_row(i)
        masses_c.append(mzs)
        matches_c.append(matches)
        ints_c.append(ints)
    df = pandas.DataFrame(
        {
            "Matches": matches_c,
            "Masses": masses_c,
            "Intensities": ints_c,
            "Modified Sequence": modified_sequences,
        }
    )
    df["Charge"] = tensor["precursor_charge_onehot"].argmax(1) + 1
    return df
