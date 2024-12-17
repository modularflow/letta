"""Tests for workflow coordinator module."""

import pytest
import asyncio
from pathlib import Path
from datetime import datetime
from letta.services.workflow.coordinator import (
    WorkflowCoordinator,
    Task,
    TaskStatus
)
from letta.services.workflow.memory import WorkflowMemory
from letta.services.workflow.file_ops import FileOperations

class MockAgent:
    """Mock agent for testing."""
    def __init__(self, role: str):
        self.role = role
        self.executed_tasks = []
        
    async def execute_task(self, task_description: str):
        self.executed_tasks.append(task_description)
        return {"status": "success", "task": task_description}

@pytest.fixture
async def coordinator(tmp_path):
    """Create a WorkflowCoordinator instance with a temporary directory."""
    file_ops = FileOperations(str(tmp_path))
    return WorkflowCoordinator(file_ops)

@pytest.fixture
def mock_memory(tmp_path):
    """Create a mock memory instance."""
    file_ops = FileOperations(str(tmp_path))
    return WorkflowMemory(
        role="test_agent",
        persona="Test persona",
        human="Test human",
        file_ops=file_ops
    )

@pytest.mark.asyncio
async def test_register_agent(coordinator, mock_memory):
    """Test registering an agent."""
    await coordinator.register_agent(mock_memory)
    assert mock_memory.role in coordinator.agents

@pytest.mark.asyncio
async def test_add_task(coordinator, mock_memory):
    """Test adding a task."""
    await coordinator.register_agent(mock_memory)
    
    task = await coordinator.add_task(
        role="test_agent",
        description="Test task",
        task_id="task_001"
    )
    
    assert task.id == "task_001"
    assert task.role == "test_agent"
    assert task.status == TaskStatus.PENDING

@pytest.mark.asyncio
async def test_add_task_with_dependencies(coordinator, mock_memory):
    """Test adding a task with dependencies."""
    await coordinator.register_agent(mock_memory)
    
    # Create first task
    task1 = await coordinator.add_task(
        role="test_agent",
        description="Task 1",
        task_id="task_001"
    )
    
    # Create dependent task
    task2 = await coordinator.add_task(
        role="test_agent",
        description="Task 2",
        task_id="task_002",
        dependencies=[task1.id]
    )
    
    assert task2.status == TaskStatus.BLOCKED
    assert task2.id in coordinator.dependencies
    assert task1.id in coordinator.dependencies[task2.id]

@pytest.mark.asyncio
async def test_process_task(coordinator, mock_memory):
    """Test processing a task."""
    # Register mock agent
    mock_agent = MockAgent("test_agent")
    coordinator.agents["test_agent"] = mock_agent
    
    # Create and process task
    task = Task(
        id="task_001",
        role="test_agent",
        description="Test task",
        dependencies=[]
    )
    
    await coordinator.process_task(task)
    
    assert task.status == TaskStatus.COMPLETED
    assert "Test task" in mock_agent.executed_tasks

@pytest.mark.asyncio
async def test_dependency_resolution(coordinator, mock_memory):
    """Test that dependent tasks are processed in correct order."""
    # Register mock agent
    mock_agent = MockAgent("test_agent")
    coordinator.agents["test_agent"] = mock_agent
    
    # Create tasks with dependencies
    task1 = await coordinator.add_task(
        role="test_agent",
        description="Task 1",
        task_id="task_001"
    )
    
    task2 = await coordinator.add_task(
        role="test_agent",
        description="Task 2",
        task_id="task_002",
        dependencies=[task1.id]
    )
    
    # Run workflow
    await coordinator.run_workflow()
    
    # Check execution order
    assert mock_agent.executed_tasks == ["Task 1", "Task 2"]

@pytest.mark.asyncio
async def test_workflow_status(coordinator, mock_memory):
    """Test getting workflow status."""
    await coordinator.register_agent(mock_memory)
    
    # Add some tasks
    task1 = await coordinator.add_task(
        role="test_agent",
        description="Task 1",
        task_id="task_001"
    )
    
    task2 = await coordinator.add_task(
        role="test_agent",
        description="Task 2",
        task_id="task_002",
        dependencies=[task1.id]
    )
    
    status = await coordinator.get_workflow_status()
    
    assert "task_001" in status["tasks"]
    assert "task_002" in status["tasks"]
    assert status["tasks"]["task_001"]["status"] == TaskStatus.PENDING.value
    assert status["tasks"]["task_002"]["status"] == TaskStatus.BLOCKED.value

@pytest.mark.asyncio
async def test_error_handling(coordinator, mock_memory):
    """Test handling of task execution errors."""
    class ErrorAgent:
        def __init__(self, role: str):
            self.role = role
            
        async def execute_task(self, task_description: str):
            raise ValueError("Test error")
    
    # Register error-producing agent
    coordinator.agents["test_agent"] = ErrorAgent("test_agent")
    
    # Create task
    task = await coordinator.add_task(
        role="test_agent",
        description="Error task",
        task_id="task_001"
    )
    
    # Run workflow
    await coordinator.run_workflow()
    
    # Check task status
    assert task.status == TaskStatus.FAILED
    assert "Test error" in task.error

@pytest.mark.asyncio
async def test_parallel_execution(coordinator, mock_memory):
    """Test that independent tasks can run in parallel."""
    class SlowAgent:
        def __init__(self, role: str, delay: float):
            self.role = role
            self.delay = delay
            self.start_times = {}
            self.end_times = {}
            
        async def execute_task(self, task_description: str):
            self.start_times[task_description] = datetime.now()
            await asyncio.sleep(self.delay)
            self.end_times[task_description] = datetime.now()
            return {"status": "success"}
    
    # Register slow agent
    slow_agent = SlowAgent("test_agent", 0.1)
    coordinator.agents["test_agent"] = slow_agent
    
    # Create independent tasks
    tasks = []
    for i in range(3):
        task = await coordinator.add_task(
            role="test_agent",
            description=f"Task {i}",
            task_id=f"task_00{i}"
        )
        tasks.append(task)
    
    # Run workflow
    await coordinator.run_workflow()
    
    # Check that tasks overlapped in time
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            task_i = f"Task {i}"
            task_j = f"Task {j}"
            
            # Check if task_j started before task_i ended
            assert slow_agent.start_times[task_j] < slow_agent.end_times[task_i] 