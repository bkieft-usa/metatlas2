import pandas as pd
import pubchempy as pcp
import pickle
import time
import re
import sys
from pathlib import Path
import sys
from typing import Dict, Any, List
from tqdm.notebook import tqdm

sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')
import load_tools as ldt
import logging_config as lcf

# Initialize logger properly at module level
logger = lcf.get_logger('pubchem_retrieval')

def fetch_pubchem_entry(inchi_key: str, timestamp: str) -> Dict[str, Any]:
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
        smiles = ""
        try:
            cid_result_subset = cid_result[cid_result['cid'] == int(cid)]
            cid_result_subset_dict = cid_result_subset.to_dict()
            if "record" in cid_result_subset_dict:
                if "props" in cid_result_subset_dict["record"][0]:
                    smiles = cid_result_subset_dict["record"][0]["props"][0]["value"]["sval"]
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
            "cas_number": "",
            "pubchem_retrieval_date": timestamp,
            "pubchem_compound_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{compound.cid}" if compound.cid else ""
        }
        
        # Extract CAS number from synonyms
        if compound.synonyms:
            for synonym in compound.synonyms:
                if not compound_data["cas_number"] and '-' in synonym and len(synonym.split('-')) == 3:
                    compound_data["cas_number"] = synonym
                    break
        
        return compound_data
        
    except Exception as e:
        logger.error(f"Error retrieving PubChem data for {inchi_key}: {e}")
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
            logger.info(f"Loaded global PubChem cache with {len(cache)} entries from {cache_file}")
            return cache
        except Exception as e:
            logger.error(f"Error loading cache: {e}. Creating new cache.")
    
    logger.info("Creating new global PubChem cache")
    return {}

def save_pubchem_cache(cache: Dict[str, Dict], cache_filename: str) -> None:
    """Save global PubChem cache to file"""
    cache_file = Path(cache_filename)
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        logger.info(f"Saved global PubChem cache with {len(cache)} entries to {cache_file}")
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

def retrieve_pubchem_info(compounds: pd.DataFrame, pubchem_cache_path: str, use_pubchem_cache: bool = True) -> None:
    """Retrieve PubChem information for compounds and update global cache."""

    # Load existing global cache
    pubchem_cache = load_or_create_pubchem_cache(pubchem_cache_path)

    prov = ldt.get_provenance()

    # Determine which compounds need to be fetched
    unique_inchi_keys = compounds['inchi_key'].dropna().unique()

    if use_pubchem_cache is False:
        compounds_to_fetch = list(unique_inchi_keys)
        compounds_in_cache = []
        logger.info(f"Force update enabled - will query all {len(compounds_to_fetch)} compounds")
    else:
        # Normal mode: only query compounds not in cache
        compounds_to_fetch = [key for key in unique_inchi_keys if key not in pubchem_cache]
        compounds_in_cache = [key for key in unique_inchi_keys if key in pubchem_cache]
        logger.info(f"Compounds already in cache: {len(compounds_in_cache)}")
        logger.info(f"Compounds needing PubChem lookup: {len(compounds_to_fetch)}")

    if compounds_to_fetch:
        logger.info(f"Fetching PubChem data for {len(compounds_to_fetch)} compounds...")
        logger.info("This may take several minutes depending on the number of compounds.")
        
        # Track how many were actually updated
        new_entries = 0
        updated_entries = 0
        
        # Fetch data for compounds
        for inchi_key in tqdm(compounds_to_fetch, desc="Fetching PubChem data"):
            was_in_cache = inchi_key in pubchem_cache
            
            # Get PubChem data
            pubchem_data = fetch_pubchem_entry(inchi_key, prov['timestamp'])

            if pubchem_data:
                pubchem_cache[inchi_key] = pubchem_data
                
                if was_in_cache:
                    updated_entries += 1
                else:
                    new_entries += 1
            else:
                logger.warning(f"No PubChem data found for {inchi_key}")

            # Be respectful to PubChem API
            time.sleep(0.25)
        
        # Save updated cache
        save_pubchem_cache(pubchem_cache, pubchem_cache_path)
        
        # Report what was done
        logger.info(f"Cache update completed: {new_entries} new entries added")
        
    else:
        logger.info("All compounds already in cache!")

    successful_retrievals = [k for k, v in pubchem_cache.items() 
                            if v.get('pubchem_cid') and v.get('pubchem_cid') != '']
    failed_retrievals = [k for k, v in pubchem_cache.items() 
                        if not v.get('pubchem_cid') or v.get('pubchem_cid') == '']
                        
    if failed_retrievals:
        logger.warning(f"Some compounds not found in PubChem: {failed_retrievals}...")
        logger.warning("These will be created with minimal information from the input table.")

    logger.info(f"PubChem data retrieval complete!")
    logger.info(f"    Total compounds in global cache: {len(pubchem_cache)}")
    logger.info(f"    Successful PubChem retrievals in cache: {len(successful_retrievals)}")
    logger.info(f"    Failed retrievals in cache: {len(failed_retrievals)}")