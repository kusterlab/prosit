import pandas as pd
import numpy as np
import multiprocessing as mp
import pyteomics.mass

from ..constants import MAX_ION, ION_TYPES, MAX_FRAG_CHARGE
from .. import utils


aa_comp = dict(pyteomics.mass.std_aa_comp)
aa_comp["o"] = pyteomics.mass.Composition({"O": 1})
translate2spectronaut = {"C": "C[Carbamidomethyl (C)]", "M(ox)": "M[Oxidation (M)]"}
shape = [MAX_ION, len(ION_TYPES), MAX_FRAG_CHARGE]
FragmentNumber = np.zeros(shape, dtype=int)
FragmentType = np.zeros(shape, dtype="object")
FragmentCharge = np.zeros(shape, dtype=int)

for z in range(MAX_FRAG_CHARGE):
    for j in range(MAX_ION):
        for tyi, ty in enumerate(ION_TYPES):
            FragmentNumber[j, tyi, z] = j + 1
            FragmentType[j, tyi, z] = ty
            FragmentCharge[j, tyi, z] = z + 1

FragmentNumber = FragmentNumber.flatten()
FragmentType = FragmentType.flatten()
FragmentCharge = FragmentCharge.flatten()


def convert_spectrum(data):
    df = pd.DataFrame(
        {
            "RelativeIntensity": data["intensities_pred"],
            "FragmentMz": data["masses_pred"],
            "idx": list(range(174)),
        }
    )
    spectrum = df[df.RelativeIntensity > 0].reset_index(drop=True)
    idx = list(spectrum.idx)
    sequence = utils.get_sequence(data["sequence_integer"])
    charge = int(data["precursor_charge_onehot"].argmax() + 1)
    irt = float(data["iRT"])
    precursor_mz = pyteomics.mass.calculate_mass(
        sequence=sequence.replace("M(ox)", "oM"), charge=charge, aa_comp=aa_comp
    )

    spectrum["ModifiedPeptide"] = sequence
    spectrum["LabeledPeptide"] = sequence
    spectrum["StrippedPeptide"] = spectrum.LabeledPeptide.map(
        lambda p: p.replace("M(ox)", "M")
    )
    spectrum["PrecursorCharge"] = charge
    spectrum["PrecursorMz"] = precursor_mz
    spectrum["iRT"] = irt
    spectrum["FragmentNumber"] = FragmentNumber[idx]
    spectrum["FragmentType"] = FragmentType[idx]
    spectrum["FragmentCharge"] = FragmentCharge[idx]
    spectrum["FragmentLossType"] = "noloss"
    for source, target in translate2spectronaut.items():
        spectrum["ModifiedPeptide"] = spectrum.ModifiedPeptide.map(
            lambda s: s.replace(source, target)
        )
    spectrum["ModifiedPeptide"] = spectrum.ModifiedPeptide.map(lambda s: "_" + s + "_")
    del spectrum["idx"]
    return spectrum


class Converter:
    def __init__(self, data, out_path, maxsize=256, batch_size=32):
        self.data = data
        self.out_path = out_path
        self.queue = mp.Manager().Queue(maxsize)
        self.batch_size = batch_size
        self.cores = mp.cpu_count()

    def batch(self, iterable):
        l = len(iterable)
        for ndx in range(0, l, self.batch_size):
            yield iterable[ndx : min(ndx + self.batch_size, l)]

    def slice_data(self, i):
        return {k: d[i] for k, d in self.data.items()}

    def fill_queue(self, pool):
        n = self.data["sequence_integer"].shape[0]
        indeces = list(range(n))

        for b in self.batch(indeces):
            spectra = pool.map(convert_spectrum, [self.slice_data(i) for i in b])
            for s in spectra:
                self.queue.put(s)

        # Stop writing process
        self.queue.put(None)

    def get_converted(self):
        while True:
            x = self.queue.get()
            if x is None:
                break
            else:
                yield x

    def to_csv(self):
        # keeps file open
        with open(self.out_path, "w") as _file:
            converted = self.get_converted()
            spectrum = next(converted)
            spectrum.to_csv(_file, index=False)
            for spectrum in converted:
                spectrum.to_csv(_file, header=False, index=False)

    def convert(self):
        io_process = mp.Process(target=self.to_csv)
        io_process.daemon = True
        io_process.start()
        with mp.Pool(processes=self.cores * 2) as pool:
            self.fill_queue(pool)
        io_process.join()


if __name__ == "__main__":

    #data = pwyll.tensorize.read(HDF5_PATH)
    conv = ConverterSP(data, to_spectronaut, OUT_PATH)
    io_process = mp.Process(target=conv.to_csv)
    io_process.daemon = True
    io_process.start()
    with mp.Pool(processes=N_CORES * 2) as pool:
        conv.fill_queue(pool)
    io_process.join()
