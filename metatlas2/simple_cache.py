import pickle
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import sys

sys.path.append('/Users/BKieft/Metabolomics/metatlas2/metatlas2')
import logging_config as lcf

logger = lcf.get_logger('simple_cache')

def save_data_cache(atlas_df: pd.DataFrame, 
                   project_data: Any, 
                   plot_data: Dict[str, Any], 
                   project_dir: str, 
                   atlas_uid: str,
                   timestamp: Optional[str] = None) -> str:
    """Save data cache (atlas_dataframe, project_data, plot_data) to cache directory."""
    
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    # Clean timestamp for filesystem
    clean_timestamp = timestamp.replace(':', '-').replace('.', '-')
    
    # Create cache directory
    cache_dir = Path(project_dir) / "cache" / "data_cache" / clean_timestamp
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save the three main data products
        with open(cache_dir / "atlas_dataframe.pkl", 'wb') as f:
            pickle.dump(atlas_df, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(cache_dir / "project_data.pkl", 'wb') as f:
            pickle.dump(project_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(cache_dir / "plot_data.pkl", 'wb') as f:
            pickle.dump(plot_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'atlas_uid': atlas_uid,
            'total_compounds': len(atlas_df),
            'cache_type': 'data',
            'version': '1.0'
        }
        
        with open(cache_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Data cache saved to {cache_dir}")
        return timestamp
        
    except Exception as e:
        logger.error(f"Failed to save data cache: {e}")
        raise

def load_data_cache(project_dir: str, 
                   use_cache_setting: Any, 
                   atlas_uid: str) -> Optional[Tuple[pd.DataFrame, Any, Dict[str, Any]]]:
    """Load data cache based on use_cache_setting."""
    
    cache_base_dir = Path(project_dir) / "cache" / "data_cache"
    
    if not cache_base_dir.exists():
        logger.info("No data cache directory found")
        return None
    
    # Determine which cache to load
    target_timestamp = None
    if use_cache_setting is True:
        # Load most recent
        timestamps = list_available_data_cache_timestamps(project_dir, atlas_uid)
        if timestamps:
            target_timestamp = timestamps[-1]  # Most recent
    elif isinstance(use_cache_setting, str):
        # Load specific timestamp
        target_timestamp = use_cache_setting
    else:
        return None
    
    if not target_timestamp:
        logger.info("No suitable data cache found")
        return None
    
    # Clean timestamp for filesystem
    clean_timestamp = target_timestamp.replace(':', '-').replace('.', '-')
    cache_dir = cache_base_dir / clean_timestamp
    
    if not cache_dir.exists():
        logger.warning(f"Data cache directory not found: {cache_dir}")
        return None
    
    try:
        # Validate metadata
        with open(cache_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        if metadata.get('atlas_uid') != atlas_uid:
            logger.warning(f"Cache atlas UID mismatch: expected {atlas_uid}, got {metadata.get('atlas_uid')}")
            return None
        
        # Load data
        with open(cache_dir / "atlas_dataframe.pkl", 'rb') as f:
            atlas_df = pickle.load(f)
        
        with open(cache_dir / "project_data.pkl", 'rb') as f:
            project_data = pickle.load(f)
        
        with open(cache_dir / "plot_data.pkl", 'rb') as f:
            plot_data = pickle.load(f)
        
        logger.info(f"Loaded data cache from {target_timestamp}")
        return atlas_df, project_data, plot_data
        
    except Exception as e:
        logger.error(f"Failed to load data cache: {e}")
        return None

def save_gui_cache(gui_obj: Any, 
                  project_dir: str, 
                  cache_type: str = "progress",
                  timestamp: Optional[str] = None) -> str:
    """Save essential GUI data (AnalystModifications) to cache directory as JSON."""
    
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    # Clean timestamp for filesystem
    clean_timestamp = timestamp.replace(':', '-').replace('.', '-')
    
    # Create cache directory
    cache_dir = Path(project_dir) / "cache" / "gui_cache" / cache_type / clean_timestamp
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Extract AnalystModifications from GUI object
        if hasattr(gui_obj, 'get_modifications'):
            modifications = gui_obj.get_modifications()
        else:
            raise ValueError("GUI object does not have get_modifications() method")
        
        # Convert AnalystModifications to JSON-serializable format
        modifications_data = {
            'rt_modifications': modifications.rt_modifications,
            'annotation_modifications': modifications.annotation_modifications,
            'modified_compounds': list(modifications.modified_compounds)
        }
        
        # Save modifications as JSON
        with open(cache_dir / "analyst_modifications.json", 'w') as f:
            json.dump(modifications_data, f, indent=2)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'cache_type': cache_type,
            'version': '1.0',
            'modified_compounds_count': len(modifications.modified_compounds)
        }
        
        with open(cache_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"GUI cache ({cache_type}) saved to {cache_dir}")
        return timestamp
        
    except Exception as e:
        logger.error(f"Failed to save GUI cache: {e}")
        raise

def load_gui_cache(project_dir: str, 
                  use_cache_setting: Any, 
                  cache_type: str = "complete") -> Optional[Any]:
    """Load AnalystModifications from cache and return a simple container."""
    
    cache_base_dir = Path(project_dir) / "cache" / "gui_cache" / cache_type
    
    if not cache_base_dir.exists():
        logger.info(f"No GUI cache directory found for {cache_type}")
        return None
    
    # Determine which cache to load
    target_timestamp = None
    if use_cache_setting is True:
        # Load most recent
        timestamps = list_available_gui_cache_timestamps(project_dir, cache_type)
        if timestamps:
            target_timestamp = timestamps[-1]  # Most recent
    elif isinstance(use_cache_setting, str):
        # Load specific timestamp
        target_timestamp = use_cache_setting
    else:
        return None
    
    if not target_timestamp:
        logger.info(f"No suitable GUI cache found for {cache_type}")
        return None
    
    # Clean timestamp for filesystem
    clean_timestamp = target_timestamp.replace(':', '-').replace('.', '-')
    cache_dir = cache_base_dir / clean_timestamp
    
    if not cache_dir.exists():
        logger.warning(f"GUI cache directory not found: {cache_dir}")
        return None
    
    try:
        # Load AnalystModifications data
        with open(cache_dir / "analyst_modifications.json", 'r') as f:
            modifications_data = json.load(f)
        
        # Create a simple container with the cached modifications
        gui_container = ModificationsContainer(modifications_data)
        
        logger.info(f"Loaded GUI cache ({cache_type}) from {target_timestamp}")
        return gui_container
        
    except Exception as e:
        logger.error(f"Failed to load GUI cache: {e}")
        return None

def list_available_data_cache_timestamps(project_dir: str, atlas_uid: str) -> List[str]:
    """List available data cache timestamps for a specific atlas."""
    
    cache_base_dir = Path(project_dir) / "cache" / "data_cache"
    
    if not cache_base_dir.exists():
        return []
    
    timestamps = []
    for cache_dir in cache_base_dir.iterdir():
        if cache_dir.is_dir():
            try:
                with open(cache_dir / "metadata.json", 'r') as f:
                    metadata = json.load(f)
                
                if metadata.get('atlas_uid') == atlas_uid:
                    timestamps.append(metadata['timestamp'])
            except:
                continue
    
    timestamps.sort()
    return timestamps

def list_available_gui_cache_timestamps(project_dir: str, cache_type: str) -> List[str]:
    """List available GUI cache timestamps for a specific cache type."""
    
    cache_base_dir = Path(project_dir) / "cache" / "gui_cache" / cache_type
    
    if not cache_base_dir.exists():
        return []
    
    timestamps = []
    for cache_dir in cache_base_dir.iterdir():
        if cache_dir.is_dir():
            try:
                with open(cache_dir / "metadata.json", 'r') as f:
                    metadata = json.load(f)
                
                timestamps.append(metadata['timestamp'])
            except:
                continue
    
    timestamps.sort()
    return timestamps

def cleanup_old_caches(project_dir: str, keep_last_n: int = 5):
    """Clean up old cache directories, keeping only the most recent N."""
    
    cache_dirs = [
        Path(project_dir) / "cache" / "data_cache",
        Path(project_dir) / "cache" / "gui_cache" / "progress",
        Path(project_dir) / "cache" / "gui_cache" / "complete"
    ]
    
    for cache_dir in cache_dirs:
        if not cache_dir.exists():
            continue
        
        # Get all subdirectories sorted by modification time
        subdirs = [d for d in cache_dir.iterdir() if d.is_dir()]
        subdirs.sort(key=lambda d: d.stat().st_mtime)
        
        # Delete old ones
        to_delete = subdirs[:-keep_last_n] if len(subdirs) > keep_last_n else []
        
        for old_dir in to_delete:
            try:
                import shutil
                shutil.rmtree(old_dir)
                logger.info(f"Deleted old cache: {old_dir}")
            except Exception as e:
                logger.error(f"Failed to delete old cache {old_dir}: {e}")

def has_existing_gui_cache(project_dir: str, cache_type: str = "progress") -> bool:
    """Check if any GUI cache exists for the given cache type."""
    cache_base_dir = Path(project_dir) / "cache" / "gui_cache" / cache_type
    
    if not cache_base_dir.exists():
        return False
    
    # Check if any valid cache directories exist
    timestamps = list_available_gui_cache_timestamps(project_dir, cache_type)
    return len(timestamps) > 0

class ModificationsContainer:
    """Simple container for AnalystModifications."""
    
    def __init__(self, modifications_data: Dict[str, Any]):
        # Import here to avoid circular imports
        from metatlas2.data_classes import AnalystModifications
        
        # Reconstruct AnalystModifications object from JSON data
        self._modifications = AnalystModifications()
        self._modifications.rt_modifications = modifications_data.get('rt_modifications', {})
        self._modifications.annotation_modifications = modifications_data.get('annotation_modifications', {})
        self._modifications.modified_compounds = set(modifications_data.get('modified_compounds', []))
        
        # Store empty metadata (can be set externally if needed)
        self.metadata = {}
    
    def get_modifications(self):
        """Return the AnalystModifications object."""
        return self._modifications
    
    def get_plot_data(self):
        """Generate plot data format from modifications (requires metadata to be set)."""
        if not self.metadata:
            raise ValueError("Metadata must be set before calling get_plot_data()")
        return self._modifications.to_plot_data_format(self.metadata)

def validate_cached_modifications(modifications: Any, compound_metadata: Dict) -> bool:
    """Validate that cached modifications are compatible with current compound metadata."""
    try:
        cached_compounds = set(modifications.modified_compounds)
        current_compounds = set(compound_metadata.keys())
        
        # Check if all cached modifications refer to compounds that still exist
        invalid_compounds = cached_compounds - current_compounds
        if invalid_compounds:
            logger.warning(f"Cached modifications contain {len(invalid_compounds)} compounds not in current dataset")
            return False
        
        logger.info(f"Cached modifications validated: {len(cached_compounds)} modified compounds")
        return True
        
    except Exception as e:
        logger.error(f"Failed to validate cached modifications: {e}")
        return False
