# Shared peak processing for image encoders.

import math
import numpy as np

from proteomicsML.base.spectrum_theoretical import SpectrumTheoreticalProcessor
from proteomicsML.base.sequence import SequenceProcessor

class SpectrumImageBase:
    # Share peak preparation and image encoding across models.
    def __init__(self,
            num_peaks: int = 200,
            max_mz=5000,
            bin_multiply_factor=10):

            self.num_peaks = num_peaks
            self.bin_multiply_factor = bin_multiply_factor
            self.max_mz = max_mz

    def _get_img_size(self, length_of_vector):
        size = int(math.ceil(math.sqrt(length_of_vector)))
        if size % 2 == 1:
            size += 1
        return size

    def _get_spectrum_peaks(self, spectrum):
        # Select valid peaks and normalize intensity once.
        spectrum = np.asarray(spectrum, dtype=np.float32).reshape(-1, 2)
        if spectrum.size == 0:
            empty = np.zeros(0, dtype=np.float32)
            return empty, empty

        valid = (
            np.isfinite(spectrum).all(axis=1)
            & (spectrum[:, 0] > 0)
            & (spectrum[:, 0] < self.max_mz)
            & (spectrum[:, 1] > 0)
        )
        spectrum = spectrum[valid]
        if spectrum.size == 0:
            empty = np.zeros(0, dtype=np.float32)
            return empty, empty

        mz, intensity = spectrum.T

        # Retain the strongest peaks before normalizing intensities.
        if intensity.size > self.num_peaks:
            top_idx = np.argpartition(intensity, -self.num_peaks)[-self.num_peaks:]
            mz = mz[top_idx]
            intensity = intensity[top_idx]

        intensity = np.sqrt(intensity)

        min_intensity = 0.0
        max_intensity = intensity.max() if intensity.size else min_intensity
        intensity_range = max_intensity - min_intensity
        if intensity_range > 0:
            intensity = (intensity - min_intensity) / intensity_range

        return mz, intensity

    def _get_padded_peak_sequence(self, spectrum):
        # Sort and pad normalized peaks for sequence models.
        mz, intensity = self._get_spectrum_peaks(spectrum)
        peaks = np.column_stack((mz, intensity))

        if len(peaks):
            peaks = peaks[np.argsort(peaks[:, 0])]

        encoded = np.zeros((self.num_peaks, 2), dtype=np.float32)
        encoded[: len(peaks)] = peaks
        return encoded

    def bin_spectrum(self, spectrum):
        max_bin = self.bin_multiply_factor * self.max_mz
        binned = np.zeros(max_bin, dtype=np.float32)
        for mz, intensity in spectrum:
            idx = int(min(mz * self.bin_multiply_factor, max_bin - 1))
            binned[idx] = max(binned[idx], intensity)
        return binned

    def spectrum_to_vector(self, spectrum):
        pass
    def vector_to_encoding(self, vector):
        pass


    def spectrum_encoding(self, spectrum):
        raise NotImplementedError("spectrum_encoding must be implemented in subclasses")


    def spectrum_encoding_theoretical(self, sequence, meta=None):
        # Encode the theoretical spectrum of a peptide.
        source_meta = meta
        theo, meta = SpectrumTheoreticalProcessor.generate_theoretical_spectrum(
            sequence, meta=meta, scale=None, normalize=False
        )
        if getattr(self, "requires_precursor_metadata", False):
            return self.spectrum_encoding(theo, meta=source_meta), meta
        return self.spectrum_encoding(theo), meta

    def spectrum_encoding_decoy(self, sequence, meta=None):
        # Encode a decoy spectrum for a peptide.
        decoy_seq = SequenceProcessor.get_decoy(sequence)
        source_meta = meta
        decoy, decoy_meta = SpectrumTheoreticalProcessor.generate_theoretical_spectrum(
            decoy_seq, meta=meta, scale=None, normalize=False
        )
        if getattr(self, "requires_precursor_metadata", False):
            return self.spectrum_encoding(decoy, meta=source_meta), decoy_meta
        return self.spectrum_encoding(decoy), decoy_meta
