import pickle
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
from dataclasses import asdict
import metatlas2.data_classes as dcl
import metatlas2.logging_config as lcf

logger = lcf.get_logger('checkpoint_manager')

class CheckpointManager:
    """Manages real-time saving and loading of targeted analysis sessions."""
    
    def __init__(self, project_db_path: str, analysis_atlas_uid: str):
        self.project_db_path = project_db_path
        self.analysis_atlas_uid = analysis_atlas_uid
        
        # Create checkpoint directory
        project_dir = Path(project_db_path).parent
        self.checkpoint_dir = project_dir / "checkpoints" / analysis_atlas_uid
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.project_data_file = self.checkpoint_dir / "project_data.pkl"
        self.atlas_df_file = self.checkpoint_dir / "atlas_df.pkl"
        self.plot_data_file = self.checkpoint_dir / "plot_data.pkl"
        self.modifications_file = self.checkpoint_dir / "modifications.json"
        self.metadata_file = self.checkpoint_dir / "session_metadata.json"
        
        self.auto_save_enabled = True
        self.last_save_time = 0
        self.save_interval = 30  # seconds between auto-saves
    
    def save_session(self, project_data: dcl.ProjectDataCollection, 
                    atlas_df: pd.DataFrame,
                    plot_data: Dict[str, Any],
                    modifications: dcl.AnalystModifications,
                    timestamp: Optional[str] = None) -> str:
        """Save complete session state to disk."""
        
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        try:
            # Save project data (the heavy data)
            with open(self.project_data_file, 'wb') as f:
                pickle.dump(project_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save atlas dataframe
            with open(self.atlas_df_file, 'wb') as f:
                pickle.dump(atlas_df, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save plot data
            with open(self.plot_data_file, 'wb') as f:
                pickle.dump(plot_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save modifications as JSON (more human-readable)
            modifications_dict = {
                'rt_modifications': modifications.rt_modifications,
                'annotation_modifications': modifications.annotation_modifications,
                'modified_compounds': list(modifications.modified_compounds)
            }
            
            with open(self.modifications_file, 'w') as f:
                json.dump(modifications_dict, f, indent=2)
            
            # Save session metadata
            metadata = {
                'timestamp': timestamp,
                'project_db_path': self.project_db_path,
                'analysis_atlas_uid': self.analysis_atlas_uid,
                'total_compounds': len(atlas_df),
                'modified_compounds': len(modifications.modified_compounds),
                'version': '1.0'
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.last_save_time = time.time()
            logger.info(f"Session saved successfully at {timestamp}")
            return timestamp
            
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            raise
    
    def load_session(self) -> tuple:
        """Load the most recent session state."""
        
        if not self.has_checkpoint():
            raise FileNotFoundError("No checkpoint found")
        
        try:
            # Load project data
            with open(self.project_data_file, 'rb') as f:
                project_data = pickle.load(f)
            
            # Load atlas dataframe
            with open(self.atlas_df_file, 'rb') as f:
                atlas_df = pickle.load(f)
            
            # Load plot data
            with open(self.plot_data_file, 'rb') as f:
                plot_data = pickle.load(f)
            
            # Load modifications
            with open(self.modifications_file, 'r') as f:
                modifications_dict = json.load(f)
            
            # Reconstruct AnalystModifications object
            modifications = dcl.AnalystModifications()
            modifications.rt_modifications = modifications_dict['rt_modifications']
            modifications.annotation_modifications = modifications_dict['annotation_modifications']
            modifications.modified_compounds = set(modifications_dict['modified_compounds'])
            
            # Load metadata
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Session loaded from {metadata['timestamp']}")
            return project_data, atlas_df, plot_data, modifications, metadata
            
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            raise
    
    def has_checkpoint(self) -> bool:
        """Check if a valid checkpoint exists."""
        required_files = [
            self.project_data_file,
            self.atlas_df_file,
            self.plot_data_file,
            self.modifications_file,
            self.metadata_file
        ]
        return all(f.exists() for f in required_files)
    
    def should_auto_save(self) -> bool:
        """Check if enough time has passed for auto-save."""
        return (self.auto_save_enabled and 
                time.time() - self.last_save_time > self.save_interval)
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about the current checkpoint."""
        if not self.has_checkpoint():
            return {"exists": False}
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Add file sizes
            sizes = {}
            for name, path in [
                ("project_data", self.project_data_file),
                ("atlas_df", self.atlas_df_file),
                ("plot_data", self.plot_data_file),
                ("modifications", self.modifications_file)
            ]:
                sizes[f"{name}_size_mb"] = path.stat().st_size / (1024 * 1024)
            
            return {
                "exists": True,
                "metadata": metadata,
                "file_sizes": sizes,
                "checkpoint_dir": str(self.checkpoint_dir)
            }
        except Exception as e:
            logger.error(f"Failed to get checkpoint info: {e}")
            return {"exists": False, "error": str(e)}
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """Clean up old checkpoint directories (if you implement versioned checkpoints)."""
        # This could be extended to keep multiple timestamped versions
        pass
    
    def list_available_timestamps(self) -> List[str]:
        """List all available checkpoint timestamps."""
        timestamps = []
        if self.has_checkpoint():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                timestamps.append(metadata['timestamp'])
            except Exception as e:
                logger.error(f"Failed to read checkpoint metadata: {e}")
        return timestamps
    
    def get_timestamp_info(self, timestamp: str) -> Dict[str, Any]:
        """Get information about a specific timestamp checkpoint."""
        if not self.has_checkpoint():
            return {"exists": False}
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if metadata['timestamp'] == timestamp:
                return {
                    "exists": True,
                    "metadata": metadata,
                    "matches": True
                }
            else:
                return {
                    "exists": True,
                    "metadata": metadata,
                    "matches": False,
                    "available_timestamp": metadata['timestamp']
                }
        except Exception as e:
            logger.error(f"Failed to get timestamp info: {e}")
            return {"exists": False, "error": str(e)}
