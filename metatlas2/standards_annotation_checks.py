"""Structure and selected-peak checks for standards annotation."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.optimize import linear_sum_assignment

import metatlas2.logging_config as lcf

logger = lcf.get_logger("standards_annotation_checks")

_NEUTRALIZATION_PATTERNS = (
    ("[n+;H]", "n"),
    ("[N+;!H0]", "N"),
    ("[$([O-]);!$([O-][#7])]", "O"),
    ("[S-;X1]", "S"),
    ("[$([N-;X2]S(=O)=O)]", "N"),
    ("[$([N-;X2][C,N]=C)]", "N"),
    ("[n-]", "[nH]"),
    ("[$([S-]=O)]", "S"),
    ("[$([N-]C=O)]", "N"),
)


def validate_neutral_inchi_key(compound_name: str, inchi: str, inchi_key: str) -> None:
    """Compare the supplied or derived key with the historical desalted, neutralized key."""
    if not inchi or not inchi_key:
        raise ValueError(
            f"Cannot check neutralization for '{compound_name}': a valid InChI and "
            "InChIKey are required. Supply a valid structure or check the PubChem lookup."
        )
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        raise ValueError(f"Cannot check neutralization for '{compound_name}': invalid InChI {inchi!r}")
    structure_key = Chem.InchiToInchiKey(inchi)
    if structure_key != inchi_key:
        raise ValueError(
            f"Compound '{compound_name}' has InChIKey {inchi_key}, but its input structure "
            f"has InChIKey {structure_key}. Provide matching structure identifiers."
        )

    # Match the historical desalter: round-trip each fragment and keep the largest.
    Chem.SanitizeMol(mol)
    if len(Chem.GetMolFrags(mol)) > 1:
        fragments = []
        for smiles in Chem.MolToSmiles(mol, isomericSmiles=True).split("."):
            fragment = Chem.MolFromSmiles(smiles, sanitize=False)
            fragment_inchi = Chem.MolToInchi(fragment) if fragment is not None else ""
            fragment = Chem.MolFromInchi(fragment_inchi) if fragment_inchi else None
            if fragment is not None:
                fragments.append(fragment)
        if not fragments:
            raise ValueError(f"Cannot desalt the structure for '{compound_name}'")
        mol = max(fragments, key=lambda fragment: fragment.GetNumAtoms())

    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() == 1 and atom.GetFormalCharge() == 1:
            atom.SetNumRadicalElectrons(0)
            atom.SetFormalCharge(0)
    for pattern, replacement in _NEUTRALIZATION_PATTERNS:
        reactant = Chem.MolFromSmarts(pattern)
        product = Chem.MolFromSmiles(replacement)
        while mol.HasSubstructMatch(reactant):
            mol = Chem.ReplaceSubstructs(mol, reactant, product, replaceAll=True)[0]
    neutral_inchi = Chem.MolToInchi(mol)
    neutral_key = Chem.InchiToInchiKey(neutral_inchi) if neutral_inchi else ""
    if not neutral_key:
        raise ValueError(f"Could not calculate a neutralized InChIKey for '{compound_name}'")
    if inchi_key != neutral_key:
        raise ValueError(
            f"Compound '{compound_name}' has InChIKey {inchi_key}, but its desalted and "
            f"neutralized structure has InChIKey {neutral_key}. Provide a matching "
            "neutral, desalted structure and InChIKey."
        )


def _percent_difference(a: float, b: float) -> float:
    denominator = (abs(a) + abs(b)) / 2
    return 0.0 if denominator == 0 else abs(a - b) / denominator * 100


def _match_nearest(left: list[float], right: list[float]) -> list[tuple[int, int]]:
    if not left or not right:
        return []
    costs = np.abs(np.subtract.outer(left, right))
    rows, columns = linear_sum_assignment(costs)
    return list(zip(rows.tolist(), columns.tolist()))


def _rt_buckets(group: pd.DataFrame) -> list[list[tuple[str, float]]]:
    """Assign each adduct's peaks to distinct nearest-RT buckets."""
    by_adduct = [
        (str(adduct), sorted(rows["rt_peak"].astype(float)))
        for adduct, rows in group.groupby("adduct", sort=True)
    ]
    by_adduct.sort(key=lambda item: (-len(item[1]), item[0]))
    buckets: list[list[tuple[str, float]]] = []
    for adduct, rts in by_adduct:
        centers = [float(np.median([rt for _, rt in bucket])) for bucket in buckets]
        pairs = _match_nearest(centers, rts)
        matched = set()
        for bucket_index, rt_index in pairs:
            buckets[bucket_index].append((adduct, rts[rt_index]))
            matched.add(rt_index)
        buckets.extend([[(adduct, rt)] for i, rt in enumerate(rts) if i not in matched])
    return buckets


def check_isomer_adducts(selections: pd.DataFrame, targets: pd.DataFrame) -> None:
    """Compare selected adducts for candidate isomers within each chrom/polarity."""
    errors = []
    for (chrom, pol), group in selections.groupby(["chromatography", "polarity"], sort=True):
        relevant_targets = targets[
            (targets["chromatography"] == chrom) & (targets["polarity"] == pol)
        ]
        compounds = {str(name): rows for name, rows in group.groupby("compound_name", sort=True)}
        for (name_a, rows_a), (name_b, rows_b) in combinations(compounds.items(), 2):
            keys_a = {str(key).split("-")[0] for key in rows_a["inchi_key"].dropna() if str(key)}
            keys_b = {str(key).split("-")[0] for key in rows_b["inchi_key"].dropna() if str(key)}
            mass_a = pd.to_numeric(
                relevant_targets.loc[relevant_targets["compound_name"] == name_a, "mz_theoretical"],
                errors="coerce",
            ).dropna().to_numpy()
            mass_b = pd.to_numeric(
                relevant_targets.loc[relevant_targets["compound_name"] == name_b, "mz_theoretical"],
                errors="coerce",
            ).dropna().to_numpy()
            masses_match = any(
                a > 0 and b > 0 and _percent_difference(a, b) <= 0.0005
                for a in mass_a for b in mass_b
            )
            if not (keys_a & keys_b or masses_match):
                continue
            adducts_a = set(rows_a["adduct"])
            adducts_b = set(rows_b["adduct"])
            context = f"{chrom}/{pol}: candidate isomers '{name_a}' and '{name_b}'"
            if not adducts_a & adducts_b:
                errors.append(
                    f"{context} have no selected adduct in common: "
                    f"{name_a}={sorted(adducts_a)}, {name_b}={sorted(adducts_b)}"
                )
                continue
            for name, extras, other in (
                (name_a, adducts_a - adducts_b, name_b),
                (name_b, adducts_b - adducts_a, name_a),
            ):
                if extras:
                    logger.warning("%s: '%s' has extra selected adducts %s not in '%s'",
                                   context, name, sorted(extras), other)
    if errors:
        raise ValueError("Isomer adduct selections disagree:\n" + "\n".join(errors))


def check_retention_times(selections: pd.DataFrame) -> None:
    """Warn about inconsistent RTs across adducts or polarities in matched buckets."""
    valid = selections.copy()
    valid["rt_peak"] = pd.to_numeric(valid["rt_peak"], errors="coerce")
    invalid = ~np.isfinite(valid["rt_peak"]) | (valid["rt_peak"] < 0)
    if invalid.any():
        logger.warning("Cannot compare RTs for selected peaks with invalid RT values: %s",
                       valid.loc[invalid, "peak_uid"].tolist())
    valid = valid.loc[~invalid]
    for (chrom, compound), group in valid.groupby(["chromatography", "compound_name"], sort=True):
        polarity_buckets = {}
        for pol, pol_group in group.groupby("polarity", sort=True):
            buckets = _rt_buckets(pol_group)
            polarity_buckets[pol] = buckets
            for bucket in buckets:
                rts = [rt for _, rt in bucket]
                if len(rts) > 1 and _percent_difference(min(rts), max(rts)) > 5:
                    logger.warning(
                        "%s/%s/%s: selected adduct RT peaks differ by more than 5%%: %s",
                        chrom, pol, compound, bucket,
                    )
        for (pol_a, buckets_a), (pol_b, buckets_b) in combinations(polarity_buckets.items(), 2):
            medians_a = [float(np.median([rt for _, rt in bucket])) for bucket in buckets_a]
            medians_b = [float(np.median([rt for _, rt in bucket])) for bucket in buckets_b]
            for i, j in _match_nearest(medians_a, medians_b):
                if _percent_difference(medians_a[i], medians_b[j]) > 5:
                    logger.warning(
                        "%s/%s: median RT peaks differ by more than 5%% between %s (%.4f) "
                        "and %s (%.4f); buckets %s and %s",
                        chrom, compound, pol_a, medians_a[i], pol_b, medians_b[j],
                        buckets_a[i], buckets_b[j],
                    )
