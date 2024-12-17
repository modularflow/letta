"""Tests for workflow memory module."""

import pytest
import json
from pathlib import Path
from datetime import datetime
from letta.services.workflow.memory import WorkflowMemory
from letta.services.workflow.file_ops import FileOperations

@pytest.fixture
async def memory(tmp_path):
    """Create a WorkflowMemory instance with a temporary directory."""
    file_ops = FileOperations(str(tmp_path))
    return WorkflowMemory(
        role="test_agent",
        persona="Test persona",
        human="Test human",
        file_ops=file_ops,
        context={"test": True}
    )

@pytest.mark.asyncio
async def test_save_and_load_state(memory):
    """Test saving and loading agent state."""
    # Update state
    test_state = {"key": "value", "nested": {"data": 123}}
    await memory.update_state(test_state)
    
    # Create new memory instance with same file_ops
    new_memory = WorkflowMemory(
        role="test_agent",
        persona="Test persona",
        human="Test human",
        file_ops=memory.file_ops
    )
    
    # Load state in new instance
    await new_memory.load_state()
    
    assert new_memory.state == test_state
    assert new_memory.context.get("test") == True

@pytest.mark.asyncio
async def test_state_history(memory):
    """Test retrieving state history."""
    # Create multiple state versions
    states = [
        {"version": 1, "data": "first"},
        {"version": 2, "data": "second"},
        {"version": 3, "data": "third"}
    ]
    
    for state in states:
        await memory.update_state(state)
        await asyncio.sleep(0.1)  # Ensure different timestamps
    
    history = await memory.get_state_history()
    assert len(history) == len(states)
    
    # Check that states are in chronological order
    for i, hist_entry in enumerate(history):
        assert hist_entry["state"]["version"] == states[i]["version"]

@pytest.mark.asyncio
async def test_share_memory(memory):
    """Test sharing memory between agents."""
    shared_data = {"key": "value"}
    target_role = "other_agent"
    
    await memory.share_memory(target_role, shared_data)
    
    # Create another memory instance for the target agent
    target_memory = WorkflowMemory(
        role=target_role,
        persona="Other persona",
        human="Test human",
        file_ops=memory.file_ops
    )
    
    # Get shared memory
    received = await target_memory.get_shared_memory(memory.role)
    assert received["data"] == shared_data
    assert received["from_role"] == memory.role
    assert received["to_role"] == target_role

@pytest.mark.asyncio
async def test_get_nonexistent_shared_memory(memory):
    """Test getting shared memory that doesn't exist."""
    result = await memory.get_shared_memory("nonexistent_agent")
    assert result is None

@pytest.mark.asyncio
async def test_update_state_persistence(memory):
    """Test that state updates are persisted."""
    updates = [
        {"key1": "value1"},
        {"key2": "value2"},
        {"key1": "updated"}
    ]
    
    for update in updates:
        await memory.update_state(update)
    
    # Load state in new instance
    new_memory = WorkflowMemory(
        role="test_agent",
        persona="Test persona",
        human="Test human",
        file_ops=memory.file_ops
    )
    await new_memory.load_state()
    
    # Check final state
    assert new_memory.state["key1"] == "updated"
    assert new_memory.state["key2"] == "value2"

@pytest.mark.asyncio
async def test_context_persistence(memory):
    """Test that context is persisted with state."""
    # Update context through state update
    context_update = {"context_key": "context_value"}
    memory.context.update(context_update)
    await memory.save_state()
    
    # Load in new instance
    new_memory = WorkflowMemory(
        role="test_agent",
        persona="Test persona",
        human="Test human",
        file_ops=memory.file_ops
    )
    await new_memory.load_state()
    
    assert new_memory.context["context_key"] == "context_value"
    assert new_memory.context["test"] == True  # Original context preserved 