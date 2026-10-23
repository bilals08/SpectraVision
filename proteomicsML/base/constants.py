# Peptide masses, ion compositions, and modification codes.
from collections import defaultdict

nist_atomic_mono_masses = {
    "H": 1.00782503223,   # Hydrogen
    "C": 12.00000000000,  # Carbon (defined standard)
    "N": 14.00307400443,  # Nitrogen
    "O": 15.99491461957,  # Oxygen
    "S": 31.97207117440,  # Sulfur
    "Se": 79.91652180000  # Selenium
}

aa_mono_masses = defaultdict(float, {'G': 57.02146372057, 'A': 71.03711378471, 'S': 87.03202840427001, 'P': 97.05276384885, 'V': 99.06841391299, 'T': 101.04767846841, 'C': 103.00918478471, 'L': 113.08406397713001, 'I': 113.08406397713001, 'N': 114.04292744114001, 'D': 115.02694302383001, 'Q': 128.05857750527997, 'K': 128.09496301399997, 'E': 129.04259308796998, 'M': 131.04048491299, 'H': 137.05891185845002, 'F': 147.06841391298997, 'U': 150.95363508471, 'R': 156.10111102359997, 'Y': 163.06332853254997, 'W': 186.07931294985997, 'O': 237.14772686284996})

fragment_ion_composition = {
    "M": {},
    "M-H2O": {"H": -2, "O": -1},
    "M-NH3": {"N": -1, "H": -3},
    "a": {"H": -2, "O": -2, "C": -1},
    "a-H2O": {"H": -4, "O": -3, "C": -1},
    "a-NH3": {"H": -5, "O": -2, "C": -1, "N": -1},
    "b": {"H": -2, "O": -1},
    "b-H2O": {"H": -4, "O": -2},
    "b-NH3": {"H": -5, "O": -1, "N": -1},
    "c": {"H": 1, "O": -1, "N": 1},
    "c-1": {"O": -1, "N": 1},
    "c-dot": {"H": 2, "O": -1, "N": 1},
    "c+1": {"H": 2, "O": -1, "N": 1},
    "c+2": {"H": 3, "O": -1, "N": 1},
    "c-H2O": {"H": -1, "O": -2, "N": 1},
    "c-NH3": {"H": -2, "O": -1},
    "x": {"H": -2, "O": 1, "C": 1},
    "x-H2O": {"H": -4, "C": 1},
    "x-NH3": {"H": -5, "O": 1, "C": 1, "N": -1},
    "y": {},
    "y-H2O": {"H": -2, "O": -1},
    "y-NH3": {"N": -1, "H": -3},
    "z": {"H": -3, "N": -1},
    "z-dot": {"H": -2, "N": -1},
    "z+1": {"H": -1, "N": -1},
    "z+2": {"N": -1},
    "z+3": {"H": 1, "N": -1},
    "z-H2O": {"H": -5, "N": -1, "O": -1},
    "z-NH3": {"H": -6, "N": -2}
}


PROTON_MASS = 1.007276466812  # for m/z calculation
WATER_MASS = 18.01056468403  # H2O mass for peptide mass calculation


mod_dict = {
    "57.021": "c",
    "15.995": "o",
    "43.006": "k",
    "0.984": "d",
    "42.011": "a",
    "-17.027": "q",
    "79.966": "p",
}

letter_to_mass = {v: k for k, v in mod_dict.items()}

letter_to_mod_name = {
    "c": "Carbamidomethylation",
    "o": "Oxidation",
    "k": "Carbamylation",
    "d": "Deamidation",
    "a": "Acetylation",
    "q": "Pyroglutamate (N-terminal Gln)",
    "p": "Phosphorylation",
}