import pandas as pd
import pubchempy as pcp
import numpy as np
import re
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Any, Tuple

def get_pubchem_data(inchi_key: str, timestamp: str) -> Dict[str, Any]:
    """Get comprehensive compound data from PubChem using InChI key."""
    try:

        # Get CID from InChI key
        cid_result = pcp.get_compounds(inchi_key, namespace='inchikey', 
                                        as_dataframe=True, listkey_count=5)

        if cid_result.empty:
            return None
            
        cid_result = cid_result.reset_index()
        cid = cid_result['cid'].to_string(index=False)
        
        # Handle multiple CIDs
        if "\n" in cid:
            cid = (cid.rstrip().split('\n'))[-1]
        
        # Extract SMILES if available due to broken API response
        try:
            cid_result_subset = cid_result[cid_result['cid'] == int(cid)]
            cid_result_subset_dict = cid_result_subset.to_dict()
            if "record" in cid_result_subset_dict:
                if "props" in cid_result_subset_dict["record"][0]:
                    cid_list = [i for i in cid_result_subset_dict["record"][0]["props"] if i["urn"]["label"] == "SMILES"]
                    if cid_list:
                        smiles = str(cid_list[0]['value']['sval'])
        except:
            smiles = ""

        # Get detailed compound information
        compound = pcp.Compound.from_cid(cid)
        
        # Extract all available properties
        compound_data = {
            "pubchem_cid": str(compound.cid) if compound.cid else "",
            "iupac_name": compound.iupac_name or "",
            "synonyms": compound.synonyms or [],
            "inchi": compound.inchi or "",
            "inchi_key": compound.inchikey or inchi_key,
            "smiles": smiles if smiles else "",
            "formula": compound.molecular_formula or "",
            "mono_isotopic_molecular_weight": str(compound.monoisotopic_mass) if compound.monoisotopic_mass else "",
            "molecular_weight": str(compound.molecular_weight) if compound.molecular_weight else "",
            "cas_number": "",
            "pubchem_retrieval_date": timestamp,
            "pubchem_compound_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{compound.cid}" if compound.cid else ""
        }
        
        # Extract CAS number from synonyms
        if compound.synonyms:
            for synonym in compound.synonyms:
                if not compound_data["cas_number"] and '-' in synonym and len(synonym.split('-')) == 3:
                    parts = synonym.split('-')
                    if all(part.isdigit() for part in parts):
                        compound_data["cas_number"] = synonym
                        break
        
        return compound_data
        
    except Exception as e:
        print(f"Error retrieving PubChem data for {inchi_key}: {e}")
        return None

def filter_synonym_list(synonyms: List[str]) -> str:
    """Filter synonym list to find the best name."""
    if not synonyms or synonyms == ["Undefined"]:
        return "Undefined"
    
    if isinstance(synonyms, str):
        synonyms = [synonyms]
    
    problematic_prefixes = (
        "nan", "Untitled", "Oprea", "Opera", "AKO", "CHEMBL", "SR-", "SCHEM", 
        "EU-", "MLS", "NSC", "ChemDiv", "ST0", "TimTec", "HMS", "BIM", "CB", 
        "CCG-", "Cambridge", "SMR", "AB0", "BRD-", "NCG", "BDBM", "CBKinase", 
        "BAS ", "ZINC", "GNF", "SQX", "CDS", "STK", "NCI", "TNP", "Boc-Tyr-OH", 
        "PD", "UNM", "BSP", "CCRIS", "MFCD", "IDI", "ST5", "AC1", "WAY-", "KUC", 
        "DTXSID", "MixCom", "CK-", "ASN ", "MMV", "SKI-", "VU", "SMSF", "Bio2", 
        "REGID", "SDCC", "BCBc", "SMP", "TCMDC", "cid_", "BCP", "AST ", "SY0", 
        "AM-", "IFLab", "Cream"
    )

    # Remove problematic prefixes
    filtered_synonyms = [x for x in synonyms if not x.startswith(problematic_prefixes)]
    filtered_synonyms = [x for x in filtered_synonyms if not "cream" in x.lower()]
    
    # Remove entries that are mostly digits or codes
    filtered_synonyms = [x for x in filtered_synonyms 
                        if not x.replace("-", "").replace(re.compile('^A-Z').pattern, "").isdigit()]
    
    if not filtered_synonyms:
        return "Undefined"
    
    # Return the shortest remaining synonym
    return min(filtered_synonyms, key=len)

def load_or_create_pubchem_cache(cache_filename: str, use_cache: bool = True) -> Dict[str, Dict]:
    """Load existing global PubChem cache or create new one"""
    cache_file = Path(cache_filename)
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            print(f"Loaded global PubChem cache with {len(cache)} entries from {cache_file}")
            return cache
        except Exception as e:
            print(f"Error loading cache: {e}. Creating new cache.")
    
    print("Creating new global PubChem cache")
    return {}

def save_pubchem_cache(cache: Dict[str, Dict], cache_filename: str) -> None:
    """Save global PubChem cache to file"""
    cache_file = Path(cache_filename)
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        print(f"Saved global PubChem cache with {len(cache)} entries to {cache_file}")
    except Exception as e:
        print(f"Error saving cache: {e}")