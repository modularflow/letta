"""Integration tests for workflow system."""

import pytest
import asyncio
import aiohttp
import json
from pathlib import Path
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker
)
from letta.services.workflow import (
    WorkflowCoordinator,
    FileOperations,
    WorkflowMemory
)
from letta.services.workflow.factory import WorkflowAgentFactory
from letta.orm.sqlalchemy_base import Base

# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///test.db"

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def db_engine():
    """Create a test database engine."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=True)
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

@pytest.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async_session = async_sessionmaker(
        db_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    async with async_session() as session:
        yield session

@pytest.fixture
async def workflow_factory(tmp_path, db_session):
    """Create a workflow factory with test configuration."""
    return WorkflowAgentFactory(
        workspace_path=str(tmp_path),
        session=db_session
    )

@pytest.fixture
async def coordinator(workflow_factory):
    """Create a workflow coordinator for testing."""
    return WorkflowCoordinator(
        file_ops=workflow_factory.file_ops,
        session=workflow_factory.session
    )

@pytest.mark.asyncio
async def test_create_and_persist_agent(workflow_factory, db_session):
    """Test creating and persisting an agent."""
    # Create agent
    agent = await workflow_factory.create_agent(
        role="test_agent",
        name="test_name",
        persona="Test persona",
        human="Test human",
        context={"test": True}
    )
    
    # Save to database
    db_session.add(agent)
    await db_session.commit()
    
    # Retrieve from database
    stmt = select(WorkflowAgent).where(WorkflowAgent.name == "test_name")
    result = await db_session.execute(stmt)
    loaded_agent = result.scalar_one()
    
    assert loaded_agent.role == "test_agent"
    assert loaded_agent.name == "test_name"
    assert loaded_agent.context["test"] == True

@pytest.mark.asyncio
async def test_workflow_state_persistence(coordinator, db_session):
    """Test workflow state persistence in database."""
    # Create and register agent
    agent = await coordinator.workflow_factory.create_agent(
        role="test_agent",
        name="test_name",
        persona="Test persona",
        human="Test human"
    )
    await coordinator.register_agent(agent.memory)
    
    # Add task
    task = await coordinator.add_task(
        role="test_agent",
        description="Test task",
        task_id="task_001"
    )
    
    # Save state
    await coordinator.save_workflow_state()
    
    # Create new coordinator and load state
    new_coordinator = WorkflowCoordinator(
        file_ops=coordinator.file_ops,
        session=db_session
    )
    await new_coordinator.load_workflow_state()
    
    # Verify state
    loaded_task = new_coordinator.tasks.get("task_001")
    assert loaded_task is not None
    assert loaded_task.description == "Test task"
    assert loaded_task.role == "test_agent"

@pytest.mark.asyncio
async def test_concurrent_task_processing(coordinator, db_session):
    """Test concurrent task processing with database operations."""
    # Create slow agent that uses database
    class SlowDbAgent:
        def __init__(self, role: str, session: AsyncSession):
            self.role = role
            self.session = session
            
        async def execute_task(self, task_description: str):
            # Simulate database operation
            await asyncio.sleep(0.1)
            result = {"status": "success", "task": task_description}
            
            # Save result to database
            task_result = TaskResult(
                task_id=task_description,
                result=json.dumps(result)
            )
            self.session.add(task_result)
            await self.session.commit()
            
            return result
    
    # Register agent
    coordinator.agents["test_agent"] = SlowDbAgent("test_agent", db_session)
    
    # Create multiple tasks
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
    
    # Verify results in database
    stmt = select(TaskResult)
    results = await db_session.execute(stmt)
    saved_results = results.scalars().all()
    
    assert len(saved_results) == 3
    for result in saved_results:
        data = json.loads(result.result)
        assert data["status"] == "success"

@pytest.mark.asyncio
async def test_webserver_integration(aiohttp_client):
    """Test integration with web server."""
    # Create app with workflow routes
    app = web.Application()
    app.router.add_post("/workflow/task", handle_task)
    app.router.add_get("/workflow/status/{task_id}", get_task_status)
    
    # Create test client
    client = await aiohttp_client(app)
    
    # Test creating task
    task_data = {
        "role": "test_agent",
        "description": "Test task",
        "task_id": "task_001"
    }
    resp = await client.post("/workflow/task", json=task_data)
    assert resp.status == 200
    result = await resp.json()
    assert result["task_id"] == "task_001"
    
    # Test getting task status
    resp = await client.get("/workflow/status/task_001")
    assert resp.status == 200
    status = await resp.json()
    assert status["status"] == "pending"

@pytest.mark.asyncio
async def test_websocket_communication(aiohttp_client):
    """Test WebSocket communication for real-time updates."""
    # Create app with WebSocket route
    app = web.Application()
    app.router.add_get("/ws", websocket_handler)
    
    # Create test client
    client = await aiohttp_client(app)
    
    # Connect to WebSocket
    ws = await client.ws_connect("/ws")
    
    # Send task creation message
    await ws.send_json({
        "type": "create_task",
        "data": {
            "role": "test_agent",
            "description": "Test task",
            "task_id": "task_001"
        }
    })
    
    # Receive confirmation
    msg = await ws.receive_json()
    assert msg["type"] == "task_created"
    assert msg["data"]["task_id"] == "task_001"
    
    # Receive status updates
    msg = await ws.receive_json()
    assert msg["type"] == "task_status"
    assert msg["data"]["status"] == "pending"
    
    await ws.close() 