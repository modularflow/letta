import pytest
from unittest.mock import Mock, patch
from letta.services.workflow.coordinator import WorkflowCoordinator
from letta.services.workflow.factory import AgentFactory
from letta.services.workflow.file_ops import FileOps
from letta.services.workflow.memory import WorkflowMemory

@pytest.mark.asyncio
async def test_workflow_initialization():
    # Setup mocks
    mock_factory = Mock(spec=AgentFactory)
    mock_file_ops = Mock(spec=FileOps)
    mock_memory = Mock(spec=WorkflowMemory)
    
    coordinator = WorkflowCoordinator(
        agent_factory=mock_factory,
        file_ops=mock_file_ops,
        memory=mock_memory
    )
    
    # Test data
    workflow_config = {
        "name": "test_workflow",
        "agents": ["architect", "developer", "reviewer"]
    }
    
    # Test workflow initialization
    await coordinator.initialize_workflow(workflow_config)
    mock_factory.create_agents.assert_called_once()
    mock_memory.initialize_workflow.assert_called_once()

@pytest.mark.asyncio
async def test_agent_task_execution():
    mock_factory = Mock(spec=AgentFactory)
    mock_file_ops = Mock(spec=FileOps)
    mock_memory = Mock(spec=WorkflowMemory)
    
    coordinator = WorkflowCoordinator(
        agent_factory=mock_factory,
        file_ops=mock_file_ops,
        memory=mock_memory
    )
    
    # Test data
    task_data = {
        "agent": "developer",
        "task": "implement_feature",
        "context": {"feature": "test_feature"}
    }
    
    # Mock agent
    mock_agent = Mock()
    mock_agent.execute_task.return_value = {"status": "success", "output": "Task completed"}
    mock_factory.get_agent.return_value = mock_agent
    
    # Test task execution
    result = await coordinator.execute_task(task_data)
    
    mock_agent.execute_task.assert_called_once()
    mock_memory.store_results.assert_called_once()
    assert result["status"] == "success"

@pytest.mark.asyncio
async def test_file_operations():
    mock_file_ops = Mock(spec=FileOps)
    
    # Test data
    file_data = {
        "path": "test/file.py",
        "content": "print('Hello, World!')"
    }
    
    # Test file operations
    await mock_file_ops.write_file(file_data["path"], file_data["content"])
    mock_file_ops.write_file.assert_called_once_with(
        file_data["path"],
        file_data["content"]
    )
    
    await mock_file_ops.read_file(file_data["path"])
    mock_file_ops.read_file.assert_called_once_with(file_data["path"])

@pytest.mark.asyncio
async def test_complete_workflow():
    # Setup mocks
    mock_factory = Mock(spec=AgentFactory)
    mock_file_ops = Mock(spec=FileOps)
    mock_memory = Mock(spec=WorkflowMemory)
    
    coordinator = WorkflowCoordinator(
        agent_factory=mock_factory,
        file_ops=mock_file_ops,
        memory=mock_memory
    )
    
    # Test data
    workflow_config = {
        "name": "feature_development",
        "agents": ["architect", "developer", "reviewer"],
        "tasks": [
            {"agent": "architect", "task": "design"},
            {"agent": "developer", "task": "implement"},
            {"agent": "reviewer", "task": "review"}
        ]
    }
    
    # Mock agent responses
    mock_agents = {
        "architect": Mock(),
        "developer": Mock(),
        "reviewer": Mock()
    }
    for agent in mock_agents.values():
        agent.execute_task.return_value = {"status": "success", "output": "Task completed"}
    
    mock_factory.get_agent.side_effect = lambda name: mock_agents[name]
    
    # Test complete workflow execution
    results = await coordinator.execute_workflow(workflow_config)
    
    # Verify workflow execution
    assert mock_factory.create_agents.called
    assert len(mock_memory.store_results.mock_calls) == len(workflow_config["tasks"])
    assert all(result["status"] == "success" for result in results)

@pytest.mark.asyncio
async def test_workflow_memory_operations():
    mock_memory = Mock(spec=WorkflowMemory)
    
    # Test data
    workflow_data = {
        "workflow_id": "workflow_123",
        "step": "design",
        "output": {"design": "test design"}
    }
    
    # Test memory operations
    await mock_memory.store_results(
        workflow_data["workflow_id"],
        workflow_data["step"],
        workflow_data["output"]
    )
    
    mock_memory.store_results.assert_called_once_with(
        workflow_data["workflow_id"],
        workflow_data["step"],
        workflow_data["output"]
    )
    
    await mock_memory.get_workflow_results(workflow_data["workflow_id"])
    mock_memory.get_workflow_results.assert_called_once_with(
        workflow_data["workflow_id"]
    ) 