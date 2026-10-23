# Three-channel MS/MS image encoding.
import numpy as np
from proteomicsML.encodings.base import SpectrumImageBase


class SpectraVision3C(SpectrumImageBase):

    def __init__(self,
                 max_mz: int = 5000,
                 num_peaks: int = 200,
                 bin_multiply_factor: int = 1,
                 ):
        super().__init__(bin_multiply_factor=bin_multiply_factor, num_peaks=num_peaks, max_mz=max_mz)

        self.img_size = self._get_img_size(max_mz * bin_multiply_factor)


    def _init_canvas(self):
        bins = self.img_size * self.img_size
        return np.zeros((3, bins), dtype=np.float32)

    def spectrum_to_vector(self, spectrum):
        spectrum = np.asarray(spectrum, dtype=np.float32)
        mz, intensity = self._get_spectrum_peaks(spectrum)

        mz = np.where(mz >= self.max_mz, self.max_mz - 1, mz)
        intensity = intensity[:len(mz)]

        total_bins = self.img_size * self.img_size

        # Map m/z into image-bin coordinates.
        idx = (mz / self.max_mz * total_bins).astype(np.int64)
        idx = np.where(idx >= total_bins, total_bins - 1, idx)

        valid = idx >= 0
        idx = idx[valid]
        mz_valid = mz[valid]
        intensity = intensity[valid]

        canvas = self._init_canvas()
        ch_mz_strength, ch_intensity, _ = canvas

        # Channel 0 stores the mean m/z of peaks in each bin.
        counts = np.zeros_like(ch_mz_strength)

        np.add.at(ch_mz_strength, idx, mz_valid)
        np.add.at(counts, idx, 1)

        nonzero = counts > 0
        ch_mz_strength[nonzero] /= counts[nonzero]

        if ch_mz_strength.max() > 0:
            ch0 = ch_mz_strength / ch_mz_strength.max()
        else:
            ch0 = ch_mz_strength

        # Channel 1 stores the strongest peak intensity in each bin.
        np.maximum.at(ch_intensity, idx, intensity)
        ch1 = ch_intensity

        # Channel 2 stores the adjacent-bin intensity difference.
        ch_grad = np.abs(np.diff(ch_intensity, prepend=0))

        ch2 = ch_grad

        ch0_u8 = (ch0 * 255.0).astype(np.uint8)
        ch1_u8 = (ch1 * 255.0).astype(np.uint8)
        ch2_u8 = (ch2 * 255.0).astype(np.uint8)

        return np.stack([ch0_u8, ch1_u8, ch2_u8], axis=0)


    def vector_to_encoding(self, vector2):
        bins = vector2.shape[1]
        total = self.img_size * self.img_size
        return vector2.reshape(3, self.img_size, self.img_size)

    def spectrum_encoding(self, spectrum):
        vec = self.spectrum_to_vector(spectrum)
        return self.vector_to_encoding(vec)

    def compute_cost(self):
        return {
            'img_size': self.img_size,
            'channels': 3,
            'pixels': self.img_size * self.img_size * 3,
            'max_mz': self.max_mz,
            'bin_multiply_factor': self.bin_multiply_factor,
            'num_peaks': self.num_peaks,
        }
