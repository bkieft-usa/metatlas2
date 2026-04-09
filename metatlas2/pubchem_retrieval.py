import pandas as pd
import time
import re
import sys
import getpass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import pubchempy as pcp

sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')
import load_tools as ldt
import logging_config as lcf

# Initialize logger properly at module level
logger = lcf.get_logger('pubchem_retrieval')

def get_provenance():
    """Get provenance information for database records."""
    return {
        "analyst": getpass.getuser(),
        "timestamp": datetime.now().isoformat()
    }

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
        
        # Filter synonyms to remove some common problematic entries
        filtered_synonyms = _filter_synonym_list(compound.synonyms) if compound.synonyms else []
        
        # Extract all available properties
        compound_data = {
            "pubchem_cid": str(compound.cid) if compound.cid else "",
            "iupac_name": compound.iupac_name or "",
            "synonyms": filtered_synonyms,
            "inchi": compound.inchi or "",
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
        logger.warning(f"Error retrieving PubChem data for {inchi_key}: {e}")
        return None

def _filter_synonym_list(synonyms: List[str]) -> str:
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

def load_or_create_pubchem_cache(pubchem_cache_path: str, use_cache: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Load existing PubChem cache or create new one.
    
    Args:
        pubchem_cache_path: Path to cache parquet file
        use_cache: If True, load existing cache; if False, return empty cache
        
    Returns:
        Dictionary mapping inchi_keys to PubChem data
    """
    if not use_cache:
        logger.info("Cache disabled - starting with empty cache")
        return {}
    
    try:
        cache_path = Path(pubchem_cache_path)
        if cache_path.exists():
            cache_df = pd.read_parquet(pubchem_cache_path)
            pubchem_cache = cache_df.set_index('inchi_key').to_dict('index')
            logger.info(f"Loaded PubChem cache with {len(pubchem_cache)} entries from {pubchem_cache_path}")
            return pubchem_cache
        else:
            logger.info(f"Cache file not found at {pubchem_cache_path} - creating new cache")
            return {}
    except Exception as e:
        logger.error(f"Error loading PubChem cache: {e}")
        logger.info("Starting with empty cache")
        return {}

def save_pubchem_cache(cache: Dict[str, Dict], cache_filename: str) -> None:
    """Save global PubChem cache to Parquet file"""
    cache_file = Path(cache_filename)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Convert dict of dicts to DataFrame
        df = pd.DataFrame.from_dict(cache, orient='index')
        df.index.name = 'inchi_key'
        df = df.reset_index()
        
        # Remove any duplicate inchi_keys
        df = df.drop_duplicates(subset='inchi_key', keep='last')
        
        # Convert list columns to strings for Parquet compatibility
        if 'synonyms' in df.columns:
            df['synonyms'] = df['synonyms'].apply(lambda x: str(x) if isinstance(x, list) else x)
        
        # Save to Parquet
        df.to_parquet(cache_file, engine='pyarrow', compression='snappy', index=False)
        logger.info(f"Saved global PubChem cache with {len(df)} unique entries to {cache_file}")
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

def retrieve_pubchem_info(compounds: pd.DataFrame, pubchem_cache_path: str, 
                         use_pubchem_cache: bool = True, update_pubchem_cache: bool = False) -> pd.DataFrame:
    """
    Retrieve PubChem information for compounds and optionally update global cache.
    
    Args:
        compounds: DataFrame with 'inchi_key' column
        pubchem_cache_path: Path to cache file
        use_pubchem_cache: If True, try to use existing cache entries
        update_pubchem_cache: If True, force API lookup and update cache for all compounds
        
    Returns:
        DataFrame with PubChem data merged into compounds
    """

    # Load existing global cache
    pubchem_cache = load_or_create_pubchem_cache(pubchem_cache_path, use_cache=use_pubchem_cache)

    prov = get_provenance()
    unique_inchi_keys = compounds['inchi_key'].dropna().unique()

    # Determine which compounds need API lookup
    if update_pubchem_cache:
        # Force update mode: lookup all compounds via API
        compounds_to_fetch = list(unique_inchi_keys)
        compounds_in_cache = []
        logger.info(f"Force update enabled - will query all {len(compounds_to_fetch)} compounds via PubChem API")
    elif use_pubchem_cache:
        # Normal mode: only lookup compounds not in cache
        compounds_to_fetch = [key for key in unique_inchi_keys if key not in pubchem_cache]
        compounds_in_cache = [key for key in unique_inchi_keys if key in pubchem_cache]
        logger.info(f"Compounds already in cache: {len(compounds_in_cache)}")
        logger.info(f"Compounds needing PubChem lookup: {len(compounds_to_fetch)}")
    else:
        # No cache mode: lookup all compounds but don't necessarily update cache
        compounds_to_fetch = list(unique_inchi_keys)
        compounds_in_cache = []
        logger.info(f"Cache disabled - will query all {len(compounds_to_fetch)} compounds via PubChem API")

    # Fetch data from PubChem API if needed
    if compounds_to_fetch:
        logger.info(f"Fetching PubChem data for {len(compounds_to_fetch)} compounds...")
        logger.info("This may take several minutes depending on the number of compounds.")
        
        new_entries = 0
        updated_entries = 0
        
        for inchi_key in compounds_to_fetch:
            was_in_cache = inchi_key in pubchem_cache
            
            # Get PubChem data via API
            pubchem_data = fetch_pubchem_entry(inchi_key, prov['timestamp'])

            if pubchem_data:
                pubchem_cache[inchi_key] = pubchem_data
                
                if was_in_cache:
                    updated_entries += 1
                else:
                    new_entries += 1

            # Be respectful to PubChem API
            time.sleep(0.5)
        
        # Save cache IMMEDIATELY after fetching to persist for next file
        if use_pubchem_cache or update_pubchem_cache:
            save_pubchem_cache(pubchem_cache, pubchem_cache_path)
            logger.info(f"Cache update completed: {new_entries} new entries added, {updated_entries} entries updated")
        else:
            logger.info(f"Retrieved {new_entries} compounds from PubChem (cache not updated)")
        
    else:
        logger.info("All compounds already in cache!")

    # Merge PubChem data back into compounds DataFrame
    pubchem_df = pd.DataFrame.from_dict(pubchem_cache, orient='index')
    pubchem_df.index.name = 'inchi_key'
    pubchem_df = pubchem_df.reset_index()
    
    # Merge PubChem data with compounds
    compounds = compounds.merge(pubchem_df, on='inchi_key', how='left', suffixes=('', '_pubchem'))
    
    # Update columns with PubChem data if not already present
    for col in pubchem_df.columns:
        if col != 'inchi_key' and col in compounds.columns:
            # Fill missing values with PubChem data
            pubchem_col = f"{col}_pubchem"
            if pubchem_col in compounds.columns:
                compounds[col] = compounds[col].fillna(compounds[pubchem_col])
                compounds = compounds.drop(columns=[pubchem_col])

    # Report statistics
    successful_retrievals = [k for k, v in pubchem_cache.items() 
                            if v.get('pubchem_cid') and v.get('pubchem_cid') != '']
    failed_retrievals = [k for k, v in pubchem_cache.items() 
                        if not v.get('pubchem_cid') or v.get('pubchem_cid') == '']
                        
    if failed_retrievals:
        logger.warning(f"Some compounds not found in PubChem: {failed_retrievals[:5]}...")
        logger.warning("These will be created with minimal information from the input table.")

    logger.info(f"PubChem data retrieval complete!")
    logger.info(f"    Total compounds in global cache: {len(pubchem_cache)}")
    logger.info(f"    Successful PubChem retrievals in cache: {len(successful_retrievals)}")
    logger.info(f"    Failed retrievals in cache: {len(failed_retrievals)}")
    
    return