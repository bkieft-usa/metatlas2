from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import sys
from pathlib import Path

sys.path.append('/Users/BKieft/Metabolomics/metatlas2/metatlas2')
import database_interact as dbi
import logging_config as lcf

logger = lcf.get_logger('atlas_classes')

@dataclass
class CompoundReference:
    """
    Immutable reference data for a compound in an atlas.
    This represents the "truth" from the database/atlas.
    """
    
    # Core identifiers
    compound_uid: str
    inchi_key: str
    compound_name: str
    
    # Chemical properties
    formula: str = ""
    mz: float = 0.0
    adduct: str = ""
    polarity: str = ""
    chromatography: str = ""
    mz_tolerance: float = 5.0
    
    # RT reference data
    rt_peak: float = 0.0
    rt_min: float = 0.0
    rt_max: float = 0.0
    
    # Database references
    mz_rt_reference_uid: str = ""
    
    # Optional metadata
    confidence: str = ""
    source: str = ""
    
    @classmethod
    def from_atlas_row(cls, row: pd.Series) -> 'CompoundReference':
        """Create CompoundReference from atlas DataFrame row."""
        return cls(
            compound_uid=row.get('compound_uid', ''),
            inchi_key=row.get('inchi_key', ''),
            compound_name=row.get('compound_name', row.get('label', '')),
            formula=row.get('formula', ''),
            mz=row.get('mz', 0.0),
            adduct=row.get('adduct', ''),
            polarity=row.get('polarity', ''),
            chromatography=row.get('chromatography', ''),
            mz_tolerance=row.get('mz_tolerance', 5.0),
            rt_peak=row.get('rt_peak', 0.0),
            rt_min=row.get('rt_min', 0.0),
            rt_max=row.get('rt_max', 0.0),
            mz_rt_reference_uid=row.get('mz_rt_reference_uid', ''),
            confidence=row.get('confidence', ''),
            source=row.get('source', '')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'compound_uid': self.compound_uid,
            'inchi_key': self.inchi_key,
            'compound_name': self.compound_name,
            'formula': self.formula,
            'mz': self.mz,
            'adduct': self.adduct,
            'polarity': self.polarity,
            'chromatography': self.chromatography,
            'mz_tolerance': self.mz_tolerance,
            'rt_peak': self.rt_peak,
            'rt_min': self.rt_min,
            'rt_max': self.rt_max,
            'mz_rt_reference_uid': self.mz_rt_reference_uid,
            'confidence': self.confidence,
            'source': self.source
        }

@dataclass
class Atlas:
    """
    Represents a collection of reference compounds with RT/MZ data.
    This is the "atlas" concept made into a proper class.
    """
    
    # Core metadata
    atlas_uid: str
    atlas_name: str
    atlas_description: str
    chromatography: str
    polarity: str
    
    # Compound references (immutable reference data)
    compounds: Dict[str, CompoundReference] = field(default_factory=dict)
    
    # Atlas metadata
    created_by: str = ""
    last_modified: str = ""
    is_rt_corrected: bool = False
    source_atlas_uid: Optional[str] = None
    
    @classmethod
    def from_database(cls, project_db_path: str, atlas_uid: str, 
                     main_db_path: str = None) -> 'Atlas':
        """
        Load atlas from database using existing database functions.
        
        Args:
            project_db_path: Path to project database
            atlas_uid: UID of atlas to load
            main_db_path: Path to main database for compound metadata
        
        Returns:
            Atlas object with loaded compounds
        """
        logger.info(f"Loading atlas {atlas_uid} from database...")
        
        # Get atlas metadata
        atlas_metadata_df = dbi.get_atlas_from_db(project_db_path, atlas_uid)
        if atlas_metadata_df.empty:
            raise ValueError(f"Atlas {atlas_uid} not found in database")
        
        atlas_row = atlas_metadata_df.iloc[0]
        
        # Get compounds with metadata
        atlas_compounds_df = dbi.get_atlas_compounds_with_metadata(
            project_db_path=project_db_path,
            main_db_path=main_db_path,
            atlas_uid=atlas_uid
        )
        
        if atlas_compounds_df.empty:
            logger.warning(f"No compounds found for atlas {atlas_uid}")
        
        # Create atlas object
        atlas = cls(
            atlas_uid=atlas_uid,
            atlas_name=atlas_row.get('atlas_name', ''),
            atlas_description=atlas_row.get('atlas_description', ''),
            chromatography=atlas_row.get('chromatography', ''),
            polarity=atlas_row.get('polarity', ''),
            created_by=atlas_row.get('created_by', ''),
            last_modified=atlas_row.get('last_modified', ''),
            is_rt_corrected=atlas_compounds_df.get('rt_correction_applied', False).any() if not atlas_compounds_df.empty else False
        )
        
        # Load compounds
        for _, row in atlas_compounds_df.iterrows():
            compound_ref = CompoundReference.from_atlas_row(row)
            atlas.compounds[compound_ref.inchi_key] = compound_ref
        
        logger.info(f"Loaded atlas '{atlas.atlas_name}' with {len(atlas.compounds)} compounds")
        
        return atlas
    
    @classmethod
    def from_dataframe(cls, atlas_df: pd.DataFrame, atlas_uid: str = None, 
                      atlas_name: str = None) -> 'Atlas':
        """
        Create atlas from DataFrame (for compatibility with existing code).
        
        Args:
            atlas_df: DataFrame with atlas compound data
            atlas_uid: Optional atlas UID (will use from DataFrame if not provided)
            atlas_name: Optional atlas name (will use from DataFrame if not provided)
        
        Returns:
            Atlas object
        """
        if atlas_df.empty:
            raise ValueError("Cannot create atlas from empty DataFrame")
        
        # Extract metadata from first row
        first_row = atlas_df.iloc[0]
        
        atlas = cls(
            atlas_uid=atlas_uid or first_row.get('atlas_uid', 'unknown'),
            atlas_name=atlas_name or first_row.get('atlas_name', 'Unknown Atlas'),
            atlas_description=first_row.get('atlas_description', ''),
            chromatography=first_row.get('chromatography', ''),
            polarity=first_row.get('polarity', ''),
            created_by=first_row.get('created_by', ''),
            last_modified=first_row.get('last_modified', ''),
            is_rt_corrected=atlas_df.get('rt_correction_applied', False).any()
        )
        
        # Load compounds
        for _, row in atlas_df.iterrows():
            compound_ref = CompoundReference.from_atlas_row(row)
            atlas.compounds[compound_ref.inchi_key] = compound_ref
        
        return atlas
    
    def get_compound_by_inchi_key(self, inchi_key: str) -> Optional[CompoundReference]:
        """Get compound reference by InChI key."""
        return self.compounds.get(inchi_key)
    
    def get_compound_by_uid(self, compound_uid: str) -> Optional[CompoundReference]:
        """Get compound reference by compound UID."""
        for compound in self.compounds.values():
            if compound.compound_uid == compound_uid:
                return compound
        return None
    
    def filter_by_chromatography(self, chromatography: str) -> 'Atlas':
        """Create filtered atlas copy with only compounds matching chromatography."""
        filtered_compounds = {
            inchi_key: compound for inchi_key, compound in self.compounds.items()
            if compound.chromatography == chromatography
        }
        
        filtered_atlas = Atlas(
            atlas_uid=f"{self.atlas_uid}_filtered_{chromatography}",
            atlas_name=f"{self.atlas_name} ({chromatography})",
            atlas_description=f"Filtered version of {self.atlas_name} for {chromatography}",
            chromatography=chromatography,
            polarity=self.polarity,
            compounds=filtered_compounds,
            created_by=self.created_by,
            last_modified=self.last_modified,
            is_rt_corrected=self.is_rt_corrected,
            source_atlas_uid=self.atlas_uid
        )
        
        return filtered_atlas
    
    def filter_by_polarity(self, polarity: str) -> 'Atlas':
        """Create filtered atlas copy with only compounds matching polarity."""
        filtered_compounds = {
            inchi_key: compound for inchi_key, compound in self.compounds.items()
            if compound.polarity == polarity
        }
        
        filtered_atlas = Atlas(
            atlas_uid=f"{self.atlas_uid}_filtered_{polarity}",
            atlas_name=f"{self.atlas_name} ({polarity})",
            atlas_description=f"Filtered version of {self.atlas_name} for {polarity}",
            chromatography=self.chromatography,
            polarity=polarity,
            compounds=filtered_compounds,
            created_by=self.created_by,
            last_modified=self.last_modified,
            is_rt_corrected=self.is_rt_corrected,
            source_atlas_uid=self.atlas_uid
        )
        
        return filtered_atlas
    
    def validate(self) -> List[str]:
        """
        Validate atlas data and return list of issues found.
        
        Returns:
            List of validation error messages
        """
        issues = []
        
        # Check basic metadata
        if not self.atlas_uid:
            issues.append("Atlas UID is missing")
        if not self.atlas_name:
            issues.append("Atlas name is missing")
        if not self.chromatography:
            issues.append("Chromatography is missing")
        if not self.polarity:
            issues.append("Polarity is missing")
        
        # Check compounds
        if not self.compounds:
            issues.append("No compounds in atlas")
        
        # Check for duplicate compound UIDs
        compound_uids = [c.compound_uid for c in self.compounds.values()]
        if len(compound_uids) != len(set(compound_uids)):
            issues.append("Duplicate compound UIDs found")
        
        # Check individual compounds
        for inchi_key, compound in self.compounds.items():
            if not compound.compound_uid:
                issues.append(f"Compound {inchi_key} missing compound_uid")
            if not compound.compound_name:
                issues.append(f"Compound {inchi_key} missing name")
            if compound.mz <= 0:
                issues.append(f"Compound {inchi_key} has invalid m/z: {compound.mz}")
            if compound.rt_peak <= 0:
                issues.append(f"Compound {inchi_key} has invalid RT peak: {compound.rt_peak}")
            if compound.rt_min >= compound.rt_max:
                issues.append(f"Compound {inchi_key} has invalid RT bounds: {compound.rt_min} >= {compound.rt_max}")
        
        return issues
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert atlas back to DataFrame format for compatibility.
        
        Returns:
            DataFrame with atlas compound data
        """
        if not self.compounds:
            return pd.DataFrame()
        
        rows = []
        for compound in self.compounds.values():
            row = compound.to_dict()
            # Add atlas metadata to each row
            row.update({
                'atlas_uid': self.atlas_uid,
                'atlas_name': self.atlas_name,
                'atlas_description': self.atlas_description,
                'label': compound.compound_name,  # For compatibility
                'rt_correction_applied': self.is_rt_corrected
            })
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for this atlas."""
        chromatographies = set(c.chromatography for c in self.compounds.values())
        polarities = set(c.polarity for c in self.compounds.values())
        adducts = set(c.adduct for c in self.compounds.values())
        
        return {
            'atlas_uid': self.atlas_uid,
            'atlas_name': self.atlas_name,
            'total_compounds': len(self.compounds),
            'chromatographies': list(chromatographies),
            'polarities': list(polarities),
            'adducts': list(adducts),
            'is_rt_corrected': self.is_rt_corrected,
            'source_atlas_uid': self.source_atlas_uid
        }
    
    def __len__(self) -> int:
        """Return number of compounds in atlas."""
        return len(self.compounds)
    
    def __iter__(self):
        """Iterate over compounds."""
        return iter(self.compounds.values())