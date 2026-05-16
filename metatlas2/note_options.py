import numpy as np


def get_notes_opts(owner: str = "jgi") -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Return default MS2/MS1/Other note hotkey maps by owner."""
    JGI_DEFAULT_MS2_HOTKEYS = {
        "no selection": "0",
        "-1.0, poor match, should remove": "w",
        "0.0, no match or no MSMS collected": "e",
        "0.5, partial or putative match of fragments": "r",
        "1.0, good match": "t",
        "0.5, co-isolated precursor, partial match": "y",
        "1.0, co-isolated precursor, good match": "u",
        "0.5, single ion match, no evidence": "i",
        "1.0, single ion match, ISTD/ref evidence": "o",
    }
    JGI_DEFAULT_MS1_HOTKEYS = {
        "keep": "p",
        "remove": "q",
    }
    JGI_DEFAULT_OTHER_HOTKEYS = {
        "unresolvable isomers": "1",
        "poor peak shape": "2",
        "potential rt shifting": "3",
        "high ppm diff": "4",
        "noisy or high background": "5",
        "needs review": "6",
        "contains refstd files": "7",
    }
    EGSB_DEFAULT_MS2_HOTKEYS = {
        "-1.0, poor match, should remove": "w",
        "0.0, no match or no MSMS collected": "e",
        "0.5, partial or putative match of fragments": "r",
        "1.0, good match": "t",
        "0.5, co-isolated precursor, partial match": "y",
        "1.0, co-isolated precursor, good match": "u",
        "0.5, single ion match, no evidence": "i",
        "1.0, single ion match, ISTD/ref evidence": "o",
    }
    EGSB_DEFAULT_MS1_HOTKEYS = {
        "no selection": "1",
        "OK - single peak at correct RT": "2",
        "OK - single peak at shifted RT": "3",
        "OK - multiple peaks, unresolvable": "4",
        "OK - multiple peaks, resolvable": "5",
        "Remove - background level/noise": "6",
        "Remove - signal in ExCtrl >= sample": "7",
        "Remove - bad MSMS": "8",
        "Remove - ND": "9",
        "Remove - Not Evaluated": "0",
        "Remove - duplicate": "[",
        "Remove - evidence of contamination or incorrect ID": "]",
    }
    EGSB_DEFAULT_OTHER_HOTKEYS = {
        "unresolvable isomers": "z",
        "poor peak shape": "x",
        "potential rt shifting": "c",
        "high ppm diff": "v",
        "noisy or high background": "b",
        "needs review": "g",
        "contains refstd files": "h",
    }

    if (owner or "jgi").lower() == "egsb":
        return EGSB_DEFAULT_MS2_HOTKEYS, EGSB_DEFAULT_MS1_HOTKEYS, EGSB_DEFAULT_OTHER_HOTKEYS
    return JGI_DEFAULT_MS2_HOTKEYS, JGI_DEFAULT_MS1_HOTKEYS, JGI_DEFAULT_OTHER_HOTKEYS


def get_note_options_and_hotkeys(override_dict, default_hotkeys):
    if isinstance(override_dict, dict) and len(override_dict) > 0:
        hotkeys = dict(override_dict)
    else:
        hotkeys = dict(default_hotkeys)
    options = list(hotkeys.keys())
    return options, hotkeys


def _is_missing_note_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def normalize_note_value(value, options: list[str]) -> str:
    """Normalize DB-loaded note value to a valid option, defaulting to options[0]."""
    if not options:
        return ""
    if _is_missing_note_value(value):
        return options[0]
    value_str = str(value)
    return value_str if value_str in options else options[0]


def should_require_note_selection(note_value: str, options: list[str]) -> bool:
    """True when note remains at default first option that requires explicit change.

    The default option "keep" is treated as an explicit valid default and does not
    require selection.
    """
    if not options:
        return False
    default_option = options[0]
    if isinstance(default_option, str) and default_option.strip().lower() == "keep":
        return False
    normalized = normalize_note_value(note_value, options)
    return normalized == default_option
