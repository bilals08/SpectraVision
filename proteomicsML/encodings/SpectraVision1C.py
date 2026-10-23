# One-channel MS/MS image encoding.
import numpy as np
import math
from proteomicsML.encodings.base import SpectrumImageBase


class SpectraVision1C(SpectrumImageBase):

    def __init__(self,
                 max_mz: int = 5000,
                 num_peaks: int = 200,
                 bin_multiply_factor: int = 1,
                 ):
        super().__init__(bin_multiply_factor=bin_multiply_factor, num_peaks=num_peaks, max_mz=max_mz)

        self.img_size = self._get_img_size(max_mz * bin_multiply_factor)


    def _init_canvas(self):
        bins = self.img_size * self.img_size
        return np.zeros((bins), dtype=np.float32)

    def spectrum_to_vector(self, spectrum):
        spectrum = np.asarray(spectrum, dtype=np.float32)
        mz, intensity = self._get_spectrum_peaks(spectrum)

        mz = np.where(mz >= self.max_mz, self.max_mz - 1, mz)
        intensity = intensity[:len(mz)]

        total_bins = self.img_size * self.img_size
        # Map m/z into image-bin coordinates.
        idx = (mz / self.max_mz * total_bins).astype(np.int64)
        idx = np.where(idx > total_bins, total_bins - 1, idx)
        valid = (idx >= 0)
        idx = idx[valid]
        intensity = intensity[valid]


        canvas = self._init_canvas()
        ch_intensity = canvas


        # Channel 0 stores the strongest peak intensity in each bin.
        np.maximum.at(ch_intensity, idx, intensity)
        ch0 = ch_intensity

        ch0_u8 = (ch0 * 255.0).astype(np.uint8)
        ch0_u8 = ch0_u8.reshape(1, self.img_size * self.img_size)
        return ch0_u8

    def vector_to_encoding(self, vector2):
        bins = vector2.shape[1]
        total = self.img_size * self.img_size
        return vector2.reshape(1, self.img_size, self.img_size)

    def spectrum_encoding(self, spectrum):
        vec = self.spectrum_to_vector(spectrum)
        return self.vector_to_encoding(vec)

    def compute_cost(self):
        return {
            'img_size': self.img_size,
            'channels': 1,
            'pixels': self.img_size * self.img_size,
            'max_mz': self.max_mz,
            'bin_multiply_factor': self.bin_multiply_factor,
            'num_peaks': self.num_peaks,
        }
