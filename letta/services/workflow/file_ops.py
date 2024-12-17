"""File operations module for workflow management."""

import os
import json
import aiofiles
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path

class FileOperations:
    """Handles asynchronous file operations with versioning support."""
    
    def __init__(self, base_path: str):
        """Initialize file operations with a base path.
        
        Args:
            base_path: Base directory for all file operations
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    async def read_file(self, filepath: str) -> str:
        """Read file contents asynchronously.
        
        Args:
            filepath: Path to file relative to base_path
            
        Returns:
            str: File contents
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        full_path = self.base_path / filepath
        async with aiofiles.open(full_path, 'r') as f:
            return await f.read()
            
    async def write_file(self, filepath: str, content: str) -> None:
        """Write content to file asynchronously.
        
        Args:
            filepath: Path to file relative to base_path
            content: Content to write
        """
        full_path = self.base_path / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(full_path, 'w') as f:
            await f.write(content)
            
    async def append_file(self, filepath: str, content: str) -> None:
        """Append content to file asynchronously.
        
        Args:
            filepath: Path to file relative to base_path
            content: Content to append
        """
        full_path = self.base_path / filepath
        async with aiofiles.open(full_path, 'a') as f:
            await f.write(content)
            
    async def save_version(self, filepath: str, metadata: Optional[Dict] = None) -> str:
        """Save a versioned copy of a file.
        
        Args:
            filepath: Path to file relative to base_path
            metadata: Optional metadata to store with version
            
        Returns:
            str: Path to the versioned file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_path = f"{filepath}.{timestamp}"
        
        # Save file content
        content = await self.read_file(filepath)
        await self.write_file(version_path, content)
        
        # Save metadata if provided
        if metadata:
            meta_path = f"{version_path}.meta.json"
            await self.write_file(meta_path, json.dumps(metadata))
            
        return version_path
        
    async def list_versions(self, filepath: str) -> List[str]:
        """List all versions of a file.
        
        Args:
            filepath: Path to file relative to base_path
            
        Returns:
            List[str]: List of version file paths
        """
        base_name = self.base_path / filepath
        pattern = f"{base_name}.*"
        versions = [str(p) for p in Path(base_name.parent).glob(pattern)
                   if not p.name.endswith('.meta.json')]
        return sorted(versions)
        
    async def get_latest_version(self, filepath: str) -> Optional[str]:
        """Get the latest version of a file.
        
        Args:
            filepath: Path to file relative to base_path
            
        Returns:
            Optional[str]: Path to latest version or None if no versions exist
        """
        versions = await self.list_versions(filepath)
        return versions[-1] if versions else None 