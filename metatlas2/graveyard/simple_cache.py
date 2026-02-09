"""
Simplified cache module for workflow-level data caching.
Basic utilities for AnalysisProject caching - most workflow caching is now handled by CacheManager.
"""
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

def save_analysis_cache(analysis_project, project_dir: str, atlas_uid: str) -> str:
    """
    Save AnalysisProject object to cache with comprehensive metadata.
    This is for per-atlas analysis caching, not workflow stage caching.
    
    Args:
        analysis_project: AnalysisProject object containing all analysis data
        project_dir: Project directory path
        atlas_uid: Atlas UID for organizing caches
    
    Returns:
        str: timestamp of saved cache
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create organized cache directory structure
    cache_dir = Path(project_dir) / "cache" / "analysis" / atlas_uid
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"analysis_project_{timestamp}.pkl"
    metadata_file = cache_dir / f"metadata_{timestamp}.json"
    
    try:
        # Save AnalysisProject object
        with open(cache_file, 'wb') as f:
            pickle.dump(analysis_project, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Generate comprehensive metadata
        metadata = _generate_analysis_metadata(analysis_project, timestamp, atlas_uid)
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Create/update latest symlink for easy access
        latest_file = cache_dir / "analysis_project_latest.pkl"
        latest_metadata = cache_dir / "metadata_latest.json"
        
        if latest_file.exists():
            latest_file.unlink()
        if latest_metadata.exists():
            latest_metadata.unlink()
            
        latest_file.symlink_to(cache_file.name)
        latest_metadata.symlink_to(metadata_file.name)
        
        logger.info(f"Analysis cache saved: {cache_file}")
        logger.info(f"  Compounds: {metadata['total_compounds']}")
        logger.info(f"  EIC data: {metadata['compounds_with_eic']}")
        logger.info(f"  MS2 data: {metadata['compounds_with_ms2']}")
        logger.info(f"  Modified: {metadata['modified_compounds']}")
        
        return timestamp
        
    except Exception as e:
        logger.error(f"Failed to save analysis cache: {e}")
        raise

def load_analysis_cache(project_dir: str, use_cache, atlas_uid: str):
    """
    Load AnalysisProject object from cache.
    
    Args:
        project_dir: Project directory path
        use_cache: Cache setting (True for latest, timestamp string for specific, False to skip)
        atlas_uid: Atlas UID for organizing caches
    
    Returns:
        AnalysisProject object or None if not found/failed
    """
    if use_cache is False:
        return None
    
    cache_dir = Path(project_dir) / "cache" / "analysis" / atlas_uid
    
    if not cache_dir.exists():
        logger.info(f"No analysis cache directory found for atlas {atlas_uid}")
        return None
    
    # Determine which cache to load
    cache_file = None
    metadata_file = None
    
    if use_cache is True:
        # Load latest cache
        cache_file = cache_dir / "analysis_project_latest.pkl"
        metadata_file = cache_dir / "metadata_latest.json"
        
        if not cache_file.exists():
            logger.info("No latest analysis cache found")
            return None
            
    elif isinstance(use_cache, str):
        # Load specific timestamp
        cache_file = cache_dir / f"analysis_project_{use_cache}.pkl"
        metadata_file = cache_dir / f"metadata_{use_cache}.json"
        
        if not cache_file.exists():
            logger.warning(f"Analysis cache not found for timestamp: {use_cache}")
            return None
    else:
        return None
    
    try:
        # Validate metadata first
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Loading analysis cache from {metadata['timestamp']}")
            logger.info(f"  Cache contains {metadata['total_compounds']} compounds")
            logger.info(f"  Analysis stage: {metadata.get('analysis_stage', 'unknown')}")
        
        # Load AnalysisProject object
        with open(cache_file, 'rb') as f:
            analysis_project = pickle.load(f)
        
        logger.info("Analysis cache loaded successfully")
        return analysis_project
        
    except Exception as e:
        logger.error(f"Failed to load analysis cache: {e}")
        return None

def _generate_analysis_metadata(analysis_project, timestamp: str, atlas_uid: str) -> Dict[str, Any]:
    """Generate comprehensive metadata for AnalysisProject cache."""
    
    # Count compounds with different types of data
    compounds_with_eic = sum(1 for c in analysis_project.compounds.values() if c.eic_data_files)
    compounds_with_ms2 = sum(1 for c in analysis_project.compounds.values() 
                           if c.ms2_data_files)  # Simple check: any MS2 files exist
    modified_compounds = sum(1 for c in analysis_project.compounds.values() 
                           if c.is_rt_modified or c.is_annotation_modified)
    
    # Determine analysis stage based on data completeness
    analysis_stage = "initialized"
    if compounds_with_eic > 0 or compounds_with_ms2 > 0:
        analysis_stage = "data_extracted"
    if modified_compounds > 0:
        analysis_stage = "modified"
    
    # Calculate data statistics
    total_eic_files = sum(len(c.eic_data_files) for c in analysis_project.compounds.values())
    total_ms2_files = sum(len(c.ms2_data_files) for c in analysis_project.compounds.values())
    
    return {
        'timestamp': timestamp,
        'atlas_uid': atlas_uid,
        'project_db_path': analysis_project.project_db_path,
        'total_compounds': len(analysis_project.compounds),
        'compounds_with_eic': compounds_with_eic,
        'compounds_with_ms2': compounds_with_ms2,
        'modified_compounds': modified_compounds,
        'analysis_stage': analysis_stage,
        'total_eic_files': total_eic_files,
        'total_ms2_files': total_ms2_files,
        'cache_version': '2.0',
        'cache_type': 'analysis_project'
    }

def list_analysis_caches(project_dir: str, atlas_uid: str) -> List[Dict[str, Any]]:
    """
    List all available analysis caches for an atlas with their metadata.
    
    Returns:
        List of cache info dictionaries
    """
    cache_dir = Path(project_dir) / "cache" / "analysis" / atlas_uid
    
    if not cache_dir.exists():
        return []
    
    caches = []
    
    # Find all metadata files
    for metadata_file in cache_dir.glob("metadata_*.json"):
        if metadata_file.name == "metadata_latest.json":
            continue  # Skip symlink
            
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if corresponding cache file exists
            timestamp = metadata['timestamp']
            cache_file = cache_dir / f"analysis_project_{timestamp}.pkl"
            
            if cache_file.exists():
                cache_info = {
                    'timestamp': timestamp,
                    'file_size_mb': cache_file.stat().st_size / (1024 * 1024),
                    'total_compounds': metadata.get('total_compounds', 0),
                    'compounds_with_eic': metadata.get('compounds_with_eic', 0),
                    'compounds_with_ms2': metadata.get('compounds_with_ms2', 0),
                    'modified_compounds': metadata.get('modified_compounds', 0),
                    'analysis_stage': metadata.get('analysis_stage', 'unknown'),
                    'cache_file': str(cache_file),
                    'metadata': metadata
                }
                caches.append(cache_info)
                
        except Exception as e:
            logger.warning(f"Failed to read cache metadata from {metadata_file}: {e}")
            continue
    
    # Sort by timestamp (newest first)
    caches.sort(key=lambda x: x['timestamp'], reverse=True)
    return caches

def cleanup_old_caches(project_dir: str, atlas_uid: str = None, keep_last_n: int = 5):
    """
    Clean up old cache files, keeping only the most recent N caches.
    
    Args:
        project_dir: Project directory
        atlas_uid: Specific atlas UID to clean, or None for all atlases
        keep_last_n: Number of recent caches to keep
    """
    cache_base_dir = Path(project_dir) / "cache"
    
    if not cache_base_dir.exists():
        return
    
    # Define cache directories to clean
    cache_dirs = []
    
    if atlas_uid:
        # Clean specific atlas
        cache_dirs.extend([
            cache_base_dir / "analysis" / atlas_uid,
        ])
    else:
        # Clean all atlases
        analysis_dir = cache_base_dir / "analysis"
        
        if analysis_dir.exists():
            cache_dirs.extend([d for d in analysis_dir.iterdir() if d.is_dir()])
    
    total_deleted = 0
    total_size_freed = 0
    
    for cache_dir in cache_dirs:
        if not cache_dir.exists():
            continue
        
        # Get all cache files (not symlinks)
        cache_files = []
        for pkl_file in cache_dir.glob("*.pkl"):
            if not pkl_file.is_symlink():
                cache_files.append(pkl_file)
        
        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda f: f.stat().st_mtime)
        
        # Delete old ones, keeping the most recent N
        to_delete = cache_files[:-keep_last_n] if len(cache_files) > keep_last_n else []
        
        for old_file in to_delete:
            try:
                # Also delete corresponding metadata file
                timestamp = old_file.stem.split('_')[-1]
                metadata_file = cache_dir / f"metadata_{timestamp}.json"
                
                file_size = old_file.stat().st_size
                old_file.unlink()
                total_deleted += 1
                total_size_freed += file_size
                
                if metadata_file.exists():
                    metadata_file.unlink()
                
            except Exception as e:
                logger.error(f"Failed to delete old cache {old_file}: {e}")
    
    if total_deleted > 0:
        logger.info(f"Cleanup complete: deleted {total_deleted} cache files, freed {total_size_freed / (1024*1024):.1f} MB")

def get_cache_status(project_dir: str, atlas_uid: str) -> Dict[str, Any]:
    """
    Get comprehensive cache status for an atlas.
    
    Returns:
        Dictionary with cache information
    """
    cache_info = {
        'atlas_uid': atlas_uid,
        'has_latest_cache': False,
        'total_caches': 0,
        'latest_cache_info': None,
        'cache_size_mb': 0
    }
    
    # Check analysis caches
    analysis_caches = list_analysis_caches(project_dir, atlas_uid)
    cache_info['total_caches'] = len(analysis_caches)
    
    if analysis_caches:
        cache_info['has_latest_cache'] = True
        cache_info['latest_cache_info'] = analysis_caches[0]  # Most recent
        cache_info['cache_size_mb'] = sum(c['file_size_mb'] for c in analysis_caches)
    
    return cache_info
