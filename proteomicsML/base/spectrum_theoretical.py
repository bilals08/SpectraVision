# Theoretical peptide ions and spectra.
from proteomicsML.base.constants import nist_atomic_mono_masses, aa_mono_masses, fragment_ion_composition, PROTON_MASS, WATER_MASS
from proteomicsML.base.sequence import SequenceProcessor
import numpy as np

class SpectrumTheoreticalProcessor:
    # Calculate ion m/z values and weighted theoretical spectra.


    @staticmethod
    def get_peptide_mz(peptide, ion="M", charge=1, modifications=None):
        # Compute monoisotopic m/z for a fragment and its modifications.
        mass = sum(aa_mono_masses[aa] for aa in peptide)

        # Fragment callers pass only modifications retained by this ion.
        if modifications:
            for residue, pos, mass_shift in modifications:
                if pos == -1:
                    mass += mass_shift
                elif 0 <= pos < len(peptide):
                    mass += mass_shift

        ion_str = str(ion)
        ion_base = ion_str.split('-')[0] if ion_str else "M"

        # y ions include water; b/y neutral losses are applied only once.
        if ion_base in ("b", "y"):
            if ion_base == "y":
                mass += WATER_MASS

            if "-H2O" in ion_str:
                mass -= WATER_MASS
            if "-NH3" in ion_str:
                mass -= (nist_atomic_mono_masses["N"] + 3 * nist_atomic_mono_masses["H"])
        else:
            # Other ions use the neutral peptide mass as their baseline.
            base_mass = mass + WATER_MASS
            if ion_str in fragment_ion_composition:
                for atom, delta in fragment_ion_composition[ion_str].items():
                    base_mass += delta * nist_atomic_mono_masses[atom]
            mass = base_mass

        mz = (mass + (charge * PROTON_MASS)) / charge
        return mz

    @staticmethod
    def fragments(peptide, types=('b', 'y', 'b-H2O', 'y-H2O', 'b-NH3', 'y-NH3'), maxcharge=2):

        clean_peptide, modifications = SequenceProcessor.parse_modifications(peptide)

        for i in range(1, len(clean_peptide)):
            for ion_type in types:
                for charge in range(1, maxcharge + 1):
                    if ion_type[0].startswith(('b', 'a', 'c')):
                        fragment_seq = clean_peptide[:i]
                        # N-terminal modifications contribute to every N-terminal fragment.
                        fragment_mods = []
                        for residue, pos, mass_shift in modifications:
                            if pos == -1:
                                fragment_mods.append((residue, pos, mass_shift))
                            elif 0 <= pos < i:
                                fragment_mods.append((residue, pos, mass_shift))

                        mz = SpectrumTheoreticalProcessor.get_peptide_mz(
                            fragment_seq, ion=ion_type, charge=charge, modifications=fragment_mods
                        )
                    else:
                        fragment_seq = clean_peptide[i:]
                        # C-terminal fragments use modification positions relative to the cut.
                        fragment_mods = []
                        for residue, pos, mass_shift in modifications:
                            if pos >= i:
                                adjusted_pos = pos - i
                                fragment_mods.append((residue, adjusted_pos, mass_shift))

                        mz = SpectrumTheoreticalProcessor.get_peptide_mz(
                            fragment_seq, ion=ion_type, charge=charge, modifications=fragment_mods
                        )

                    yield mz, ion_type[0], charge

    @staticmethod
    def generate_theoretical_spectrum(
        peptide,
        meta=None,
        types=('b','y'),
        maxcharge=2,
        ion_weights=None,
        charge_gamma=0.8,
        scale='sqrt',
        normalize=True,
    ):
        # Generate weighted fragment peaks and precursor metadata.
        # Match theoretical fragment charge states to the observed precursor charge.
        if meta and isinstance(meta, dict):
            if 'charge' in meta and isinstance(meta['charge'], (int, np.integer)) and meta['charge'] > 0:
                maxcharge = int(meta['charge'])

        # Include all residue and modification shifts in precursor m/z.
        full_pep, full_mods = SequenceProcessor.parse_modifications(peptide)
        prec_mass = sum(aa_mono_masses[aa] for aa in full_pep)
        prec_mass += sum(mass_shift for _, _, mass_shift in full_mods)
        prec_mass += WATER_MASS
        z = max(maxcharge, 1)
        precursor_mz = (prec_mass + z * PROTON_MASS) / z
        meta = {
            'precursor_mz': float(precursor_mz),
            'precursor_charge': z,
            'sequence': peptide,
        }

        if ion_weights is None:
            ion_weights = {'y': 1.0, 'b': 0.85, 'a': 0.6, 'x': 0.75, 'c': 0.65, 'z': 0.65}

        frags = list(SpectrumTheoreticalProcessor.fragments(peptide, types=types, maxcharge=maxcharge))
        if not frags:
            return [], meta
        mzs = np.array([f[0] for f in frags], dtype=np.float64)
        ions= np.array([f[1] for f in frags])
        ch  = np.array([f[2] for f in frags], dtype=np.int32)

        w_ion = np.vectorize(lambda t: ion_weights.get(t, 0.5))(ions).astype(np.float32)
        # Higher fragment charges receive lower intensity weights.
        w_charge = (1.0 / np.power(ch, float(charge_gamma))).astype(np.float32)

        w = w_ion * w_charge

        if scale == 'sqrt':
            w = np.sqrt(np.maximum(w, 0.0))
        elif scale == 'log1p':
            w = np.log1p(np.maximum(w, 0.0))
        else:
            w = np.maximum(w, 0.0)

        if normalize:
            maxw = w.max() if w.size else 0.0
            if maxw > 0:
                w = w / maxw
            else:
                w = np.zeros_like(w, dtype=np.float32)
            intensities = np.clip(np.rint(w * 255.0), 0, 255).astype(np.uint8)
        else:
            intensities = w.astype(np.float32, copy=False)


        merged = {}
        for mz, inten in zip(mzs, intensities):
            if (mz in merged) and (inten <= merged[mz]):
                continue
            merged[mz] = int(inten) if normalize else float(inten)

        theoretical_spectrum = sorted(merged.items(), key=lambda x: x[0])
        return theoretical_spectrum, meta
