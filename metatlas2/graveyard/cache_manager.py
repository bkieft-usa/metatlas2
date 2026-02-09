# filepath: /Users/BKieft/Metabolomics/metatlas2/metatlas2/cache_manager.py
import pickle
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd
import shutil
import sys

sys.path.append('/Users/BKieft/Metabolomics/metatlas2/metatlas2')
import logging_config as lcf

logger = lcf.get_logger('cache_manager')

class CacheManager:
    """
    Centralized cache manager for all workflow stages.
    Handles RT Correction, Putative Identifications, and Manual Curation caching.
    """
    
    def __init__(self, project_directory: str):
        """
        Initialize cache manager for a project.
        
        Args:
            project_directory: Path to the project directory
        """
        self.project_directory = Path(project_directory)
        self.cache_base_dir = self.project_directory / "cache"
        
        # Define cache directories for each stage
        self.rt_correction_dir = self.cache_base_dir / "rt_correction"
        self.putative_ids_dir = self.cache_base_dir / "putative_ids"
        self.manual_curation_dir = self.cache_base_dir / "manual_curation"
        
        # Create cache directories
        self._setup_cache_directories()
    
    def _setup_cache_directories(self):
        """Create cache directory structure."""
        for cache_dir in [self.rt_correction_dir, self.putative_ids_dir, self.manual_curation_dir]:
            cache_dir.mkdir(parents=True, exist_ok=True)
    
    # =============================================================================
    # RT CORRECTION CACHING
    # =============================================================================
    
    def save_rt_correction(self, rt_models: Dict, corrected_atlases: Dict) -> str:
        """
        Save RT correction results to cache.
        
        Args:
            rt_models: Dictionary of RT correction models by method
            corrected_atlases: Dictionary of corrected atlas UIDs by type/method
            
        Returns:
            timestamp of saved cache
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        cache_data = {
            'rt_models': rt_models,
            'corrected_atlases': corrected_atlases,
            'timestamp': timestamp,
            'cache_version': '2.0',
            'stage': 'rt_correction'
        }
        
        # Save timestamped version
        cache_file = self.rt_correction_dir / f"rt_correction_{timestamp}.pkl"
        metadata_file = self.rt_correction_dir / f"rt_correction_{timestamp}.json"
        
        try:
            # Save cache data
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save metadata
            metadata = {
                'timestamp': timestamp,
                'stage': 'rt_correction',
                'cache_version': '2.0',
                'rt_models_count': len(rt_models),
                'corrected_atlases_count': sum(
                    len(chroms) for atlas_chroms in corrected_atlases.values()
                    for chroms in atlas_chroms.values()
                ),
                'methods': list(rt_models.keys())
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Update latest symlinks
            self._update_latest_symlinks(self.rt_correction_dir, "rt_correction", timestamp)
            
            logger.info(f"RT correction cache saved: {cache_file}")
            logger.info(f"  Models: {len(rt_models)} methods")
            logger.info(f"  Corrected atlases: {metadata['corrected_atlases_count']}")
            
            return timestamp
            
        except Exception as e:
            logger.error(f"Failed to save RT correction cache: {e}")
            raise
    
    def load_rt_correction(self, timestamp: Optional[str] = None) -> Optional[Dict]:
        """
        Load RT correction results from cache.
        
        Args:
            timestamp: Specific timestamp to load, or None for latest
            
        Returns:
            Dictionary with rt_models and corrected_atlases, or None if not found
        """
        if timestamp:
            cache_file = self.rt_correction_dir / f"rt_correction_{timestamp}.pkl"
        else:
            cache_file = self.rt_correction_dir / "rt_correction_latest.pkl"
        
        if not cache_file.exists():
            logger.info("No RT correction cache found")
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            logger.info(f"Loaded RT correction cache from {cache_data.get('timestamp', 'unknown')}")
            logger.info(f"  Models: {len(cache_data.get('rt_models', {}))}")
            logger.info(f"  Corrected atlases: {cache_data.get('corrected_atlases', {})}")
            
            return cache_data
            
        except Exception as e:
            logger.error(f"Failed to load RT correction cache: {e}")
            return None
    
    def has_rt_correction_cache(self) -> bool:
        """Check if RT correction cache exists."""
        return (self.rt_correction_dir / "rt_correction_latest.pkl").exists()
    
    # =============================================================================
    # PUTATIVE IDENTIFICATIONS CACHING
    # =============================================================================
    
    def save_putative_identifications(self, putative_ids: Dict, summary_stats: Dict = None) -> str:
        """
        Save putative identification results to cache.
        
        Args:
            putative_ids: Dictionary of putative identifications by atlas type and method
            summary_stats: Optional summary statistics
            
        Returns:
            timestamp of saved cache
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        cache_data = {
            'putative_ids': putative_ids,
            'summary_stats': summary_stats,
            'timestamp': timestamp,
            'cache_version': '2.0',
            'stage': 'putative_identifications'
        }
        
        # Save timestamped version
        cache_file = self.putative_ids_dir / f"putative_ids_{timestamp}.pkl"
        metadata_file = self.putative_ids_dir / f"putative_ids_{timestamp}.json"
        
        try:
            # Save cache data
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Calculate metadata
            total_ids = sum(
                len(method_ids) for atlas_methods in putative_ids.values()
                for method_ids in atlas_methods.values()
            )
            
            # Save metadata
            metadata = {
                'timestamp': timestamp,
                'stage': 'putative_identifications',
                'cache_version': '2.0',
                'total_putative_ids': total_ids,
                'by_atlas_type': {
                    atlas_type: sum(len(method_ids) for method_ids in methods.values())
                    for atlas_type, methods in putative_ids.items()
                },
                'atlas_methods': {
                    atlas_type: list(methods.keys())
                    for atlas_type, methods in putative_ids.items()
                }
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Update latest symlinks
            self._update_latest_symlinks(self.putative_ids_dir, "putative_ids", timestamp)
            
            logger.info(f"Putative identifications cache saved: {cache_file}")
            logger.info(f"  Total identifications: {total_ids}")
            logger.info(f"  By atlas type: {metadata['by_atlas_type']}")
            
            return timestamp
            
        except Exception as e:
            logger.error(f"Failed to save putative identifications cache: {e}")
            raise
    
    def load_putative_identifications(self, timestamp: Optional[str] = None) -> Optional[Dict]:
        """
        Load putative identification results from cache.
        
        Args:
            timestamp: Specific timestamp to load, or None for latest
            
        Returns:
            Dictionary with putative_ids and summary_stats, or None if not found
        """
        if timestamp:
            cache_file = self.putative_ids_dir / f"putative_ids_{timestamp}.pkl"
        else:
            cache_file = self.putative_ids_dir / "putative_ids_latest.pkl"
        
        if not cache_file.exists():
            logger.info("No putative identifications cache found")
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            total_ids = sum(
                len(method_ids) for atlas_methods in cache_data.get('putative_ids', {}).values()
                for method_ids in atlas_methods.values()
            )
            
            logger.info(f"Loaded putative identifications cache from {cache_data.get('timestamp', 'unknown')}")
            logger.info(f"  Total identifications: {total_ids}")
            
            return cache_data
            
        except Exception as e:
            logger.error(f"Failed to load putative identifications cache: {e}")
            return None
    
    def has_putative_identifications_cache(self) -> bool:
        """Check if putative identifications cache exists."""
        return (self.putative_ids_dir / "putative_ids_latest.pkl").exists()
    
    # =============================================================================
    # MANUAL CURATION CACHING
    # =============================================================================
    
    def save_manual_curation(self, putative_ids: Dict, curation_progress: Dict = None, 
                           partial_save: bool = False) -> str:
        """
        Save manual curation results to cache.
        Supports partial saves during curation process.
        
        Args:
            putative_ids: Dictionary of curated putative identifications
            curation_progress: Progress tracking information
            partial_save: Whether this is a partial save during curation
            
        Returns:
            timestamp of saved cache
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        cache_data = {
            'putative_ids': putative_ids,
            'curation_progress': curation_progress,
            'timestamp': timestamp,
            'cache_version': '2.0',
            'stage': 'manual_curation',
            'partial_save': partial_save,
            'completed': not partial_save
        }
        
        # Use different naming for partial saves
        if partial_save:
            cache_file = self.manual_curation_dir / f"curation_partial_{timestamp}.pkl"
            metadata_file = self.manual_curation_dir / f"curation_partial_{timestamp}.json"
        else:
            cache_file = self.manual_curation_dir / f"curation_{timestamp}.pkl"
            metadata_file = self.manual_curation_dir / f"curation_{timestamp}.json"
        
        try:
            # Save cache data
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Calculate curation statistics
            all_ids = []
            for atlas_methods in putative_ids.values():
                for method_ids in atlas_methods.values():
                    all_ids.extend(method_ids)
            
            curation_stats = {
                'pending': sum(1 for pid in all_ids if pid.curation_status == 'pending'),
                'reviewed': sum(1 for pid in all_ids if pid.curation_status == 'reviewed'),
                'finalized': sum(1 for pid in all_ids if pid.curation_status == 'finalized'),
                'rt_modified': sum(1 for pid in all_ids if pid.is_rt_modified),
                'annotation_modified': sum(1 for pid in all_ids if pid.is_annotation_modified)
            }
            
            # Save metadata
            metadata = {
                'timestamp': timestamp,
                'stage': 'manual_curation',
                'cache_version': '2.0',
                'partial_save': partial_save,
                'completed': not partial_save,
                'total_identifications': len(all_ids),
                'curation_stats': curation_stats,
                'progress_percent': ((curation_stats['reviewed'] + curation_stats['finalized']) / len(all_ids) * 100) if all_ids else 0
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Update latest symlinks (always update, even for partial saves)
            self._update_latest_symlinks(self.manual_curation_dir, "curation", timestamp)
            
            save_type = "partial" if partial_save else "complete"
            logger.info(f"Manual curation cache saved ({save_type}): {cache_file}")
            logger.info(f"  Progress: {metadata['progress_percent']:.1f}%")
            logger.info(f"  Curation stats: {curation_stats}")
            
            return timestamp
            
        except Exception as e:
            logger.error(f"Failed to save manual curation cache: {e}")
            raise
    
    def load_manual_curation(self, timestamp: Optional[str] = None, 
                           prefer_partial: bool = True) -> Optional[Dict]:
        """
        Load manual curation results from cache.
        
        Args:
            timestamp: Specific timestamp to load, or None for latest
            prefer_partial: If True, prefer partial saves over complete ones when loading latest
            
        Returns:
            Dictionary with putative_ids and curation_progress, or None if not found
        """
        if timestamp:
            # Try both partial and complete versions for specific timestamp
            cache_file = self.manual_curation_dir / f"curation_{timestamp}.pkl"
            if not cache_file.exists():
                cache_file = self.manual_curation_dir / f"curation_partial_{timestamp}.pkl"
        else:
            # Load latest - check for partial saves first if preferred
            if prefer_partial:
                # Find latest partial save
                partial_files = list(self.manual_curation_dir.glob("curation_partial_*.pkl"))
                complete_files = list(self.manual_curation_dir.glob("curation_[0-9]*.pkl"))
                
                all_files = partial_files + complete_files
                if all_files:
                    # Sort by timestamp in filename and get most recent
                    latest_file = max(all_files, key=lambda f: f.stem.split('_')[-1])
                    cache_file = latest_file
                else:
                    cache_file = self.manual_curation_dir / "curation_latest.pkl"
            else:
                cache_file = self.manual_curation_dir / "curation_latest.pkl"
        
        if not cache_file.exists():
            logger.info("No manual curation cache found")
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Calculate current progress
            all_ids = []
            for atlas_methods in cache_data.get('putative_ids', {}).values():
                for method_ids in atlas_methods.values():
                    all_ids.extend(method_ids)
            
            reviewed_count = sum(1 for pid in all_ids if pid.curation_status in ['reviewed', 'finalized'])
            progress_percent = (reviewed_count / len(all_ids) * 100) if all_ids else 0
            
            save_type = "partial" if cache_data.get('partial_save', False) else "complete"
            logger.info(f"Loaded manual curation cache ({save_type}) from {cache_data.get('timestamp', 'unknown')}")
            logger.info(f"  Total identifications: {len(all_ids)}")
            logger.info(f"  Progress: {progress_percent:.1f}%")
            
            return cache_data
            
        except Exception as e:
            logger.error(f"Failed to load manual curation cache: {e}")
            return None
    
    def has_manual_curation_cache(self) -> bool:
        """Check if manual curation cache exists (including partial saves)."""
        latest_exists = (self.manual_curation_dir / "curation_latest.pkl").exists()
        partial_exists = len(list(self.manual_curation_dir.glob("curation_partial_*.pkl"))) > 0
        return latest_exists or partial_exists
    
    def auto_save_curation_progress(self, putative_ids: Dict, interval_minutes: int = 5) -> None:
        """
        Auto-save curation progress at regular intervals.
        This would be called periodically during the GUI curation process.
        
        Args:
            putative_ids: Current state of putative identifications
            interval_minutes: How often to auto-save
        """
        # Check if enough time has passed since last auto-save
        last_autosave_file = self.manual_curation_dir / ".last_autosave"
        
        should_save = True
        if last_autosave_file.exists():
            try:
                last_save_time = datetime.fromisoformat(last_autosave_file.read_text().strip())
                time_since_save = (datetime.now() - last_save_time).total_seconds() / 60
                should_save = time_since_save >= interval_minutes
            except Exception:
                should_save = True
        
        if should_save:
            try:
                self.save_manual_curation(putative_ids, partial_save=True)
                last_autosave_file.write_text(datetime.now().isoformat())
                logger.debug("Auto-saved curation progress")
            except Exception as e:
                logger.error(f"Auto-save failed: {e}")
    
    # =============================================================================
    # GENERAL CACHE MANAGEMENT
    # =============================================================================
    
    def _update_latest_symlinks(self, cache_dir: Path, prefix: str, timestamp: str) -> None:
        """Update latest symlinks for a cache type."""
        latest_pkl = cache_dir / f"{prefix}_latest.pkl"
        latest_json = cache_dir / f"{prefix}_latest.json"
        
        target_pkl = f"{prefix}_{timestamp}.pkl"
        target_json = f"{prefix}_{timestamp}.json"
        
        # Handle partial saves for manual curation
        if prefix == "curation" and (cache_dir / f"curation_partial_{timestamp}.pkl").exists():
            target_pkl = f"curation_partial_{timestamp}.pkl"
            target_json = f"curation_partial_{timestamp}.json"
        
        # Update symlinks
        for latest_file, target_file in [(latest_pkl, target_pkl), (latest_json, target_json)]:
            if latest_file.exists():
                latest_file.unlink()
            if (cache_dir / target_file).exists():
                latest_file.symlink_to(target_file)
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get comprehensive cache status for all stages."""
        status = {
            'cache_directory': str(self.cache_base_dir),
            'stages': {}
        }
        
        # RT Correction status
        rt_cache = self.rt_correction_dir / "rt_correction_latest.pkl"
        status['stages']['rt_correction'] = {
            'has_cache': rt_cache.exists(),
            'cache_file': str(rt_cache) if rt_cache.exists() else None,
            'available_timestamps': [
                f.stem.replace('rt_correction_', '') 
                for f in self.rt_correction_dir.glob('rt_correction_*.pkl')
                if not f.is_symlink()
            ]
        }
        
        # Putative Identifications status
        putative_cache = self.putative_ids_dir / "putative_ids_latest.pkl"
        status['stages']['putative_identifications'] = {
            'has_cache': putative_cache.exists(),
            'cache_file': str(putative_cache) if putative_cache.exists() else None,
            'available_timestamps': [
                f.stem.replace('putative_ids_', '') 
                for f in self.putative_ids_dir.glob('putative_ids_*.pkl')
                if not f.is_symlink()
            ]
        }
        
        # Manual Curation status
        curation_cache = self.manual_curation_dir / "curation_latest.pkl"
        partial_saves = list(self.manual_curation_dir.glob("curation_partial_*.pkl"))
        complete_saves = [f for f in self.manual_curation_dir.glob("curation_*.pkl") 
                         if not f.is_symlink() and 'partial' not in f.name]
        
        status['stages']['manual_curation'] = {
            'has_cache': curation_cache.exists() or len(partial_saves) > 0,
            'cache_file': str(curation_cache) if curation_cache.exists() else None,
            'partial_saves_count': len(partial_saves),
            'complete_saves_count': len(complete_saves),
            'latest_partial': max(partial_saves, key=lambda f: f.stat().st_mtime).name if partial_saves else None,
            'available_timestamps': [
                f.stem.replace('curation_', '').replace('partial_', '') 
                for f in (partial_saves + complete_saves)
            ]
        }
        
        return status
    
    def list_available_caches(self, stage: str) -> List[Dict[str, Any]]:
        """
        List all available caches for a specific stage with metadata.
        
        Args:
            stage: Stage name ('rt_correction', 'putative_identifications', 'manual_curation')
            
        Returns:
            List of cache info dictionaries
        """
        caches = []
        
        if stage == 'rt_correction':
            cache_dir = self.rt_correction_dir
            pattern = 'rt_correction_*.json'
        elif stage == 'putative_identifications':
            cache_dir = self.putative_ids_dir
            pattern = 'putative_ids_*.json'
        elif stage == 'manual_curation':
            cache_dir = self.manual_curation_dir
            pattern = 'curation*.json'
        else:
            logger.warning(f"Unknown stage: {stage}")
            return caches
        
        # Find all metadata files
        for metadata_file in cache_dir.glob(pattern):
            if 'latest' in metadata_file.name:
                continue  # Skip symlinks
            
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Check if corresponding cache file exists
                pkl_file = metadata_file.with_suffix('.pkl')
                if pkl_file.exists():
                    cache_info = {
                        'timestamp': metadata['timestamp'],
                        'file_size_mb': pkl_file.stat().st_size / (1024 * 1024),
                        'cache_file': str(pkl_file),
                        'metadata_file': str(metadata_file),
                        'metadata': metadata
                    }
                    
                    # Add stage-specific info
                    if stage == 'manual_curation':
                        cache_info['partial_save'] = metadata.get('partial_save', False)
                        cache_info['progress_percent'] = metadata.get('progress_percent', 0)
                    
                    caches.append(cache_info)
                    
            except Exception as e:
                logger.warning(f"Failed to read cache metadata from {metadata_file}: {e}")
                continue
        
        # Sort by timestamp (newest first)
        caches.sort(key=lambda x: x['timestamp'], reverse=True)
        return caches
    
    def clear_stage_cache(self, stage: str, keep_last_n: int = 0) -> int:
        """
        Clear cache for a specific stage.
        
        Args:
            stage: Stage name to clear
            keep_last_n: Number of recent caches to keep (0 = delete all)
            
        Returns:
            Number of cache files deleted
        """
        deleted_count = 0
        
        if stage == 'rt_correction':
            cache_dir = self.rt_correction_dir
            pattern = 'rt_correction_*.pkl'
        elif stage == 'putative_identifications':
            cache_dir = self.putative_ids_dir
            pattern = 'putative_ids_*.pkl'
        elif stage == 'manual_curation':
            cache_dir = self.manual_curation_dir
            pattern = 'curation*.pkl'
        else:
            logger.warning(f"Unknown stage: {stage}")
            return 0
        
        # Get all cache files (not symlinks)
        cache_files = [f for f in cache_dir.glob(pattern) if not f.is_symlink()]
        
        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda f: f.stat().st_mtime)
        
        # Delete old ones, keeping the most recent N
        to_delete = cache_files[:-keep_last_n] if keep_last_n > 0 else cache_files
        
        for cache_file in to_delete:
            try:
                # Also delete corresponding metadata file
                metadata_file = cache_file.with_suffix('.json')
                
                cache_file.unlink()
                deleted_count += 1
                
                if metadata_file.exists():
                    metadata_file.unlink()
                
                logger.debug(f"Deleted cache file: {cache_file}")
                
            except Exception as e:
                logger.error(f"Failed to delete cache file {cache_file}: {e}")
        
        # Update symlinks if we deleted the latest
        if deleted_count > 0 and keep_last_n > 0:
            remaining_files = [f for f in cache_dir.glob(pattern) if not f.is_symlink()]
            if remaining_files:
                latest_file = max(remaining_files, key=lambda f: f.stat().st_mtime)
                timestamp = latest_file.stem.split('_')[-1]
                prefix = stage.replace('_', '_') if stage != 'manual_curation' else 'curation'
                self._update_latest_symlinks(cache_dir, prefix, timestamp)
        
        if deleted_count > 0:
            logger.info(f"Cleared {deleted_count} cache files for {stage}")
        
        return deleted_count
    
    def clear_all_cache(self) -> int:
        """
        Clear all caches for the project.
        
        Returns:
            Total number of cache files deleted
        """
        total_deleted = 0
        
        for stage in ['rt_correction', 'putative_identifications', 'manual_curation']:
            deleted = self.clear_stage_cache(stage, keep_last_n=0)
            total_deleted += deleted
        
        logger.info(f"Cleared all caches: {total_deleted} files deleted")
        return total_deleted
    
    def get_cache_size_info(self) -> Dict[str, Any]:
        """Get cache size information for all stages."""
        size_info = {
            'total_size_mb': 0,
            'by_stage': {}
        }
        
        for stage, cache_dir in [
            ('rt_correction', self.rt_correction_dir),
            ('putative_identifications', self.putative_ids_dir),
            ('manual_curation', self.manual_curation_dir)
        ]:
            stage_size = 0
            file_count = 0
            
            if cache_dir.exists():
                for cache_file in cache_dir.glob('*.pkl'):
                    if not cache_file.is_symlink():
                        stage_size += cache_file.stat().st_size
                        file_count += 1
            
            stage_size_mb = stage_size / (1024 * 1024)
            size_info['by_stage'][stage] = {
                'size_mb': stage_size_mb,
                'file_count': file_count
            }
            size_info['total_size_mb'] += stage_size_mb
        
        return size_info
    
    # Example usage of the simplified caching architecture
    """
    Example usage of the new CacheManager with simplified workflow caching.

    # Create a workflow with automatic caching
    workflow = wfo.TargetedMetabolomicsWorkflow(
        config=config,
        project_db_path="/path/to/project.duckdb",
        project_directory="/path/to/project"
    )

    # The CacheManager is automatically initialized and available
    cache_manager = workflow.cache_manager

    # Get cache status for all stages
    cache_status = cache_manager.get_cache_status()
    print("Current cache status:", cache_status)

    # Run workflow with caching enabled (default behavior)
    workflow.run_complete_workflow(
        stop_at_stage=wfo.WorkflowStage.MANUAL_CURATION,
        use_cache=True  # Uses cache for all stages where available
    )

    # Manual curation with automatic progress saving
    # This launches the GUI and auto-saves progress every 5 minutes
    gui_result, compounds = workflow._create_curation_gui(config)

    # Check curation progress
    progress = workflow.get_curation_progress()
    print(f"Curation progress: {progress['progress_percent']:.1f}%")

    # Force regeneration of a specific stage
    workflow.force_regenerate_stage(wfo.WorkflowStage.PUTATIVE_IDENTIFICATION)

    # Clean up old caches, keeping only the 3 most recent
    workflow.cache_manager.clear_stage_cache('manual_curation', keep_last_n=3)

    # Get comprehensive cache summary
    cache_summary = workflow.get_cache_summary()
    print("Cache summary:", cache_summary)
    """