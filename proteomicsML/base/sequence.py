# Peptide modification parsing and decoy generation.
from proteomicsML.base.constants import mod_dict

class SequenceProcessor:
    # Handle peptide modification codes and decoy sequences.
        
    @staticmethod
    def get_decoy(pep):
        # Reverse middle residues while preserving modification positions.
        clean_seq, modifications = SequenceProcessor.parse_modifications(pep)
        
        if len(clean_seq) <= 2:
            return pep
        
        # Keep terminal residues fixed while reversing the middle.
        middle = clean_seq[1:-1]
        decoy_clean = clean_seq[0] + middle[::-1] + clean_seq[-1]
        
        decoy_mods = []
        for residue, pos, mass_shift in modifications:
            if pos == -1:
                decoy_mods.append((residue, pos, mass_shift))
            else:
                new_pos = SequenceProcessor.map_decoy_position_to_original(pos, len(clean_seq), reverse=True)
                decoy_mods.append((residue, new_pos, mass_shift))
        
        decoy_with_mods = SequenceProcessor.reconstruct_sequence_with_modifications(decoy_clean, decoy_mods)
        return decoy_with_mods
    
    @staticmethod
    def map_decoy_position_to_original(pos, seq_len, reverse=False):
        # Map terminal and reversed internal residue positions.
        if seq_len <= 2:
            return pos
            
        if pos == 0 or pos == seq_len - 1:
            return pos
        
        if reverse:
            return seq_len - 1 - pos
        else:
            return seq_len - 1 - pos
    
    @staticmethod
    def reconstruct_sequence_with_modifications(clean_seq, modifications):
        # Place N-terminal codes before residues and site codes after residues.
        mass_to_letter = dict(mod_dict)

        result = ""

        # N-terminal modification codes precede the first residue.
        nterm_mods = [mod for mod in modifications if mod[1] == -1]
        for _, _, mass_shift in nterm_mods:
            key = f"{mass_shift:.3f}"
            letter = mass_to_letter.get(key)
            if letter is None:
                raise ValueError(f"Unknown modification mass shift {mass_shift!r}")
            result += letter

        # Residue modifications follow their residue.
        for i, residue in enumerate(clean_seq):
            result += residue
            residue_mods = [mod for mod in modifications if mod[1] == i]
            for _, _, mass_shift in residue_mods:
                key = f"{mass_shift:.3f}"
                letter = mass_to_letter.get(key)
                if letter is None:
                    raise ValueError(f"Unknown modification mass shift {mass_shift!r}")
                result += letter
        return result
    
    @staticmethod
    def parse_modifications(peptide_with_mods):
        # Separate uppercase residues from lowercase modification mass shifts.
        letter_to_mass = {letter: float(mass) for mass, letter in mod_dict.items()}

        modifications = []
        clean_chars = []
        last_res_pos = -1

        for index, ch in enumerate(peptide_with_mods):
            if 'A' <= ch <= 'Z':
                clean_chars.append(ch)
                last_res_pos += 1
            elif 'a' <= ch <= 'z':
                if ch not in letter_to_mass:
                    raise ValueError(
                        f"Unknown modification code {ch!r} at position {index} "
                        f"in peptide {peptide_with_mods!r}"
                    )
                mass_shift = letter_to_mass[ch]
                if last_res_pos == -1:
                    modifications.append((None, -1, mass_shift))
                else:
                    residue = clean_chars[last_res_pos]
                    modifications.append((residue, last_res_pos, mass_shift))
            else:
                raise ValueError(
                    f"Unsupported character {ch!r} at position {index} "
                    f"in peptide {peptide_with_mods!r}"
                )

        clean_sequence = ''.join(clean_chars)
        return clean_sequence, modifications
