"""Tests for file operations module."""

import pytest
import json
from pathlib import Path
from datetime import datetime
from letta.services.workflow.file_ops import FileOperations

@pytest.fixture
async def file_ops(tmp_path):
    """Create a FileOperations instance with a temporary directory."""
    return FileOperations(str(tmp_path))

@pytest.mark.asyncio
async def test_write_and_read_file(file_ops):
    """Test writing and reading a file."""
    content = "Test content"
    await file_ops.write_file("test.txt", content)
    
    result = await file_ops.read_file("test.txt")
    assert result == content

@pytest.mark.asyncio
async def test_append_file(file_ops):
    """Test appending to a file."""
    initial = "Initial content\n"
    append = "Appended content"
    
    await file_ops.write_file("append.txt", initial)
    await file_ops.append_file("append.txt", append)
    
    result = await file_ops.read_file("append.txt")
    assert result == initial + append

@pytest.mark.asyncio
async def test_save_version(file_ops):
    """Test saving file versions with metadata."""
    content = "Version 1"
    metadata = {"version": 1, "author": "test"}
    
    # Write initial file
    await file_ops.write_file("versioned.txt", content)
    
    # Save version
    version_path = await file_ops.save_version(
        "versioned.txt",
        metadata=metadata
    )
    
    # Check version content
    version_content = await file_ops.read_file(version_path)
    assert version_content == content
    
    # Check metadata
    meta_path = f"{version_path}.meta.json"
    meta_content = await file_ops.read_file(meta_path)
    assert json.loads(meta_content) == metadata

@pytest.mark.asyncio
async def test_list_versions(file_ops):
    """Test listing file versions."""
    content = "Test content"
    await file_ops.write_file("multi_version.txt", content)
    
    # Create multiple versions
    version1 = await file_ops.save_version("multi_version.txt")
    version2 = await file_ops.save_version("multi_version.txt")
    
    versions = await file_ops.list_versions("multi_version.txt")
    assert len(versions) == 3  # Original + 2 versions
    assert version1 in versions
    assert version2 in versions

@pytest.mark.asyncio
async def test_get_latest_version(file_ops):
    """Test getting the latest version of a file."""
    content = "Original"
    await file_ops.write_file("latest.txt", content)
    
    # Create versions
    await file_ops.save_version("latest.txt")
    latest = await file_ops.save_version("latest.txt")
    
    result = await file_ops.get_latest_version("latest.txt")
    assert result == latest

@pytest.mark.asyncio
async def test_file_not_found(file_ops):
    """Test handling of non-existent files."""
    with pytest.raises(FileNotFoundError):
        await file_ops.read_file("nonexistent.txt")

@pytest.mark.asyncio
async def test_nested_directories(file_ops):
    """Test handling nested directory creation."""
    content = "Nested content"
    nested_path = "deep/nested/dir/test.txt"
    
    await file_ops.write_file(nested_path, content)
    result = await file_ops.read_file(nested_path)
    
    assert result == content
    assert (file_ops.base_path / "deep/nested/dir").exists() 