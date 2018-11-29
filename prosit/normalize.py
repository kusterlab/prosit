import numpy


def base_peak(spectral):
    max_int = spectral.max(1)
    spectral = spectral / max_int[:, numpy.newaxis]
    spectral = numpy.nan_to_num(spectral)
    return spectral
