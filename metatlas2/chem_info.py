"""
metatlas2/chem_info.py

Cheminformatics utilities for the metatlas2 standards-annotation workflow.

Provides:
  - Adduct loading and filtering via matchms
  - Precursor m/z calculation from exact mass + adduct
  - Exact mass calculation from SMILES or InChI via RDKit
  - InChI / InChIKey / SMILES interconversion via RDKit
  - Molecule validation helpers
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np

import metatlas2.logging_config as lcf

logger = lcf.get_logger("chem_info")


# ---------------------------------------------------------------------------
# Adduct loading (matchms)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _load_known_adducts_df():
    """Return the matchms known-adducts DataFrame (cached).

    The DataFrame has columns: ``adduct``, ``ionmode``, ``mass_multiplier``,
    ``correction_mass``, ``charge``.

    Raises:
        ImportError: if matchms is not installed.
    """
    from matchms.filtering.filter_utils.load_known_adducts import load_known_adducts
    return load_known_adducts()


def load_adducts_dict() -> Dict[str, Dict[str, Union[str, float]]]:
    """Return a dict of all known adducts keyed by adduct string.

    Each value is a dict with keys: ``ionmode``, ``mass_multiplier``,
    ``correction_mass``, ``charge``.

    Returns:
        Dict mapping adduct name → property dict.
    """
    return _load_known_adducts_df().set_index("adduct").to_dict(orient="index")


def load_selected_adducts(
    include_adducts: List[str],
) -> Dict[str, Dict[str, Union[str, float]]]:
    """Filter the known-adducts dict to only the requested adducts.

    Args:
        include_adducts: List of adduct strings to include.

    Returns:
        Dict mapping adduct name → property dict for the requested adducts.

    Raises:
        KeyError: if any adduct in *include_adducts* is not known to matchms.
    """
    all_adducts = load_adducts_dict()
    missing = [a for a in include_adducts if a not in all_adducts]
    if missing:
        logger.warning(f"Unknown adducts (not in matchms): {missing}")
    return {a: all_adducts[a] for a in include_adducts if a in all_adducts}


def separate_adducts_dict(
    adducts: Dict[str, Dict[str, Union[str, float]]],
) -> Tuple[Dict[str, Dict[str, Union[str, float]]], Dict[str, Dict[str, Union[str, float]]]]:
    """Split an adducts dict into positive and negative ion-mode sub-dicts.

    Args:
        adducts: Dict of adduct name → property dict (as returned by
                 :func:`load_adducts_dict` or :func:`load_selected_adducts`).

    Returns:
        Tuple of ``(pos_adducts, neg_adducts)`` dicts.
    """
    pos: Dict[str, Dict[str, Union[str, float]]] = {}
    neg: Dict[str, Dict[str, Union[str, float]]] = {}
    for adduct, data in adducts.items():
        if data.get("ionmode") == "positive":
            pos[adduct] = data
        elif data.get("ionmode") == "negative":
            neg[adduct] = data
    return pos, neg


def valid_adduct(value: str) -> bool:
    """Return True if *value* is a matchms-recognised adduct string.

    Args:
        value: Adduct string to test, e.g. ``"[M+H]+"``.

    Returns:
        ``True`` if the adduct is in the matchms known-adducts table.
    """
    try:
        return value in _load_known_adducts_df()["adduct"].tolist()
    except Exception:
        return False


def is_positive_mode(adduct: str) -> bool:
    """Return True if *adduct* is a positive-ion-mode adduct.

    Args:
        adduct: Adduct string, e.g. ``"[M+H]+"``.

    Returns:
        ``True`` for positive mode, ``False`` for negative mode.

    Raises:
        KeyError: if the adduct is not recognised by matchms.
    """
    df = _load_known_adducts_df()
    if adduct not in df["adduct"].values:
        raise KeyError(f"Adduct '{adduct}' is not supported by matchms")
    return df.loc[df["adduct"] == adduct, "ionmode"].values[0] == "positive"


# ---------------------------------------------------------------------------
# Precursor m/z calculation
# ---------------------------------------------------------------------------

DEFAULT_ADDUCTS: List[str] = [
    "[M+H]+", "[M+Na]+", "[M-H2O+H]+", "[M+K]+", "[M+NH4]+",
    "[M]+", "[M+2H]2+",
    "[M-H]-", "[M+Cl]-", "[M-H2O-H]-", "[M]-", "[M-2H]2-",
]


def get_precursor_mz(exact_mass: float, adduct: str) -> float:
    """Calculate the precursor m/z for a given exact mass and adduct.

    Uses the matchms adduct table (``mass_multiplier`` and
    ``correction_mass``) to compute:

        ``precursor_mz = exact_mass * mass_multiplier + correction_mass``

    Args:
        exact_mass: Neutral monoisotopic mass of the molecule.
        adduct:     Adduct string, e.g. ``"[M+H]+"``.

    Returns:
        Precursor m/z value, or ``float('nan')`` if the adduct is unknown.
    """
    try:
        df = _load_known_adducts_df()
        row = df[df["adduct"] == adduct]
        if row.empty:
            logger.warning(f"Unknown adduct '{adduct}'; returning NaN for precursor m/z")
            return float("nan")
        multiplier = float(row["mass_multiplier"].values[0])
        correction = float(row["correction_mass"].values[0])
        return exact_mass * multiplier + correction
    except Exception as exc:
        logger.warning(f"Error computing precursor m/z for adduct '{adduct}': {exc}")
        return float("nan")


def get_parent_mass(precursor_mz: float, adduct: str) -> float:
    """Return the neutral parent mass that would produce *precursor_mz* for *adduct*.

    Inverts the matchms formula:
        ``parent_mass = (precursor_mz - correction_mass) / mass_multiplier``

    Args:
        precursor_mz: Observed precursor m/z.
        adduct:       Adduct string, e.g. ``"[M+H]+"``.

    Returns:
        Neutral parent mass, or ``float('nan')`` if the adduct is unknown.
    """
    try:
        df = _load_known_adducts_df()
        row = df[df["adduct"] == adduct]
        if row.empty:
            logger.warning(f"Unknown adduct '{adduct}'; returning NaN for parent mass")
            return float("nan")
        multiplier = float(row["mass_multiplier"].values[0])
        correction = float(row["correction_mass"].values[0])
        if multiplier == 0:
            return float("nan")
        return (precursor_mz - correction) / multiplier
    except Exception as exc:
        logger.warning(f"Error computing parent mass for adduct '{adduct}': {exc}")
        return float("nan")


def calc_all_adducts(
    exact_mass: float,
    polarity: str,
    include_adducts: List[str],
) -> List[Tuple[str, float]]:
    """Calculate precursor m/z for all selected adducts matching *polarity*.

    Args:
        exact_mass:      Neutral monoisotopic mass of the compound.
        polarity:        ``"POS"`` or ``"NEG"``.
        include_adducts: List of adduct strings to consider.

    Returns:
        List of ``(adduct, precursor_mz)`` tuples for adducts matching
        the requested polarity.
    """
    assert polarity in ("POS", "NEG"), "polarity must be 'POS' or 'NEG'"
    selected = load_selected_adducts(include_adducts)
    pos_adducts, neg_adducts = separate_adducts_dict(selected)
    pool = pos_adducts if polarity == "POS" else neg_adducts
    return [(adduct, get_precursor_mz(exact_mass, adduct)) for adduct in pool]


# ---------------------------------------------------------------------------
# Exact mass from SMILES / InChI
# ---------------------------------------------------------------------------

def calc_exact_mass_from_smiles_or_inchi(molecule_id: str) -> float:
    """Return the exact monoisotopic mass from a SMILES or InChI string.

    Uses :func:`rdkit.Chem.Descriptors.ExactMolWt`.

    Args:
        molecule_id: SMILES or InChI string.

    Returns:
        Exact molecular mass, or ``float('nan')`` on failure.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Descriptors import ExactMolWt

        mol = inchi_or_smiles_to_molecule(molecule_id)
        return ExactMolWt(mol)
    except Exception as exc:
        logger.warning(f"Error computing exact mass for '{str(molecule_id)[:60]}': {exc}")
        return float("nan")


# alias used in the user-provided reference code
inchi_or_smiles_to_mass = calc_exact_mass_from_smiles_or_inchi


# ---------------------------------------------------------------------------
# Molecule conversion helpers (RDKit)
# ---------------------------------------------------------------------------

def inchi_or_smiles_to_molecule(molecule_id: str):
    """Convert an InChI or SMILES string to an RDKit ``Mol`` object.

    Tries InChI first (detected by the ``InChI=`` prefix), then SMILES.

    Args:
        molecule_id: InChI or SMILES string.

    Returns:
        ``rdkit.Chem.rdchem.Mol`` object.

    Raises:
        ValueError: if the string cannot be parsed as either InChI or SMILES.
    """
    from rdkit import Chem

    mol = None
    if str(molecule_id).startswith("InChI="):
        try:
            mol = Chem.inchi.MolFromInchi(molecule_id)
        except Exception:
            pass
    if mol is None:
        mol = Chem.MolFromSmiles(molecule_id)
    if mol is None:
        raise ValueError(f"'{str(molecule_id)[:80]}' is not a valid InChI or SMILES")
    return mol


def smiles_to_inchi(smiles: str) -> str:
    """Convert a SMILES string to an InChI string.

    Args:
        smiles: SMILES string.

    Returns:
        InChI string.

    Raises:
        ValueError: if *smiles* cannot be parsed or converted.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"'{smiles[:80]}' is not a valid SMILES")
    out = Chem.inchi.MolToInchi(mol)
    if out is None:
        raise ValueError(f"Could not convert SMILES to InChI: '{smiles[:80]}'")
    return out


def inchi_to_smiles(inchi: str) -> str:
    """Convert an InChI string to a canonical SMILES string.

    Args:
        inchi: InChI string.

    Returns:
        Canonical SMILES string.

    Raises:
        ValueError: if *inchi* cannot be parsed or converted.
    """
    from rdkit import Chem

    mol = Chem.inchi.MolFromInchi(inchi)
    if mol is None:
        raise ValueError(f"'{inchi[:80]}' is not a valid InChI")
    out = Chem.MolToSmiles(mol)
    if out is None:
        raise ValueError(f"Could not convert InChI to SMILES: '{inchi[:80]}'")
    return out


def inchi_or_smiles_to_inchi(molecule_id: str) -> str:
    """Convert an InChI or SMILES string to a canonical InChI string.

    Args:
        molecule_id: InChI or SMILES string.

    Returns:
        InChI string.

    Raises:
        ValueError: if the string cannot be parsed or converted.
    """
    from rdkit import Chem

    mol = inchi_or_smiles_to_molecule(molecule_id)
    out = Chem.inchi.MolToInchi(mol)
    if out is None:
        raise ValueError(f"Could not convert to InChI: '{str(molecule_id)[:80]}'")
    return out


def inchi_or_smiles_to_smiles(molecule_id: str) -> str:
    """Convert an InChI or SMILES string to a canonical SMILES string.

    Args:
        molecule_id: InChI or SMILES string.

    Returns:
        Canonical SMILES string.

    Raises:
        ValueError: if the string cannot be parsed or converted.
    """
    from rdkit import Chem

    mol = inchi_or_smiles_to_molecule(molecule_id)
    out = Chem.MolToSmiles(mol)
    if out is None:
        raise ValueError(f"Could not convert to SMILES: '{str(molecule_id)[:80]}'")
    return out


def smiles_to_inchi_key(smiles: str) -> str:
    """Derive an InChIKey from a SMILES string.

    Args:
        smiles: SMILES string.

    Returns:
        InChIKey string, or ``""`` on failure.
    """
    try:
        inchi = smiles_to_inchi(smiles)
        return inchi_to_inchi_key(inchi)
    except Exception as exc:
        logger.warning(f"Could not derive InChIKey from SMILES '{smiles[:60]}': {exc}")
        return ""


def inchi_to_inchi_key(inchi: str) -> str:
    """Derive an InChIKey from an InChI string.

    Args:
        inchi: InChI string.

    Returns:
        InChIKey string, or ``""`` on failure.
    """
    try:
        from rdkit.Chem.inchi import InchiToInchiKey
        result = InchiToInchiKey(inchi)
        return result or ""
    except Exception as exc:
        logger.warning(f"Could not derive InChIKey from InChI '{inchi[:60]}': {exc}")
        return ""


def resolve_inchi_and_inchi_key(
    smiles: str,
    inchi: str,
    inchi_key: str,
) -> Tuple[str, str, str]:
    """Fill in missing InChI / InChIKey from SMILES or InChI using RDKit.

    Given the three chemical identifiers (any may be blank), attempts to
    derive the missing ones:
      - If *smiles* is present and *inchi* is blank → derive InChI from SMILES.
      - If *inchi* is present and *inchi_key* is blank → derive InChIKey from InChI.
      - If *smiles* is present and *inchi_key* is still blank → derive via SMILES→InChI→InChIKey.

    Args:
        smiles:    SMILES string (may be empty).
        inchi:     InChI string (may be empty).
        inchi_key: InChIKey string (may be empty).

    Returns:
        Tuple of ``(smiles, inchi, inchi_key)`` with any derivable fields filled in.
    """
    # SMILES → InChI
    if smiles and not inchi:
        try:
            inchi = smiles_to_inchi(smiles)
        except Exception:
            pass

    # InChI → InChIKey
    if inchi and not inchi_key:
        inchi_key = inchi_to_inchi_key(inchi)

    # SMILES → InChI → InChIKey (fallback if InChI derivation above failed)
    if smiles and not inchi_key:
        try:
            _inchi = smiles_to_inchi(smiles)
            inchi_key = inchi_to_inchi_key(_inchi)
            if not inchi:
                inchi = _inchi
        except Exception:
            pass

    return smiles, inchi, inchi_key


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def is_valid_inchi_pair(inchi: str, inchi_key: str) -> bool:
    """Return True if *inchi* hashes to *inchi_key*.

    Args:
        inchi:     InChI string.
        inchi_key: Expected InChIKey.

    Returns:
        ``True`` if the InChI is valid and its key matches *inchi_key*.
    """
    try:
        derived = inchi_to_inchi_key(inchi)
        return bool(derived) and derived == inchi_key
    except Exception:
        return False


def is_valid_inchi_smiles_pair(inchi: str, smiles: str) -> bool:
    """Return True if *inchi* and *smiles* represent the same structure.

    Uses RDKit substructure matching (both directions) as the equality test.

    Args:
        inchi:   InChI string.
        smiles:  SMILES string.

    Returns:
        ``True`` if both parse successfully and are mutual substructures.
    """
    try:
        from rdkit import Chem

        mol_inchi = Chem.inchi.MolFromInchi(inchi)
        mol_smiles = Chem.MolFromSmiles(smiles)
        if mol_inchi is None or mol_smiles is None:
            return False
        return (
            mol_inchi.HasSubstructMatch(mol_smiles)
            and mol_smiles.HasSubstructMatch(mol_inchi)
        )
    except Exception:
        return False
