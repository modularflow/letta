import pytest
from unittest.mock import Mock, patch
from letta.services.tool_manager import ToolManager
from letta.orm.sqlalchemy_base import AsyncSession

@pytest.mark.asyncio
async def test_tool_creation_flow():
    # Setup mocks
    mock_client = Mock()
    mock_server = Mock()
    mock_tool_manager = Mock()
    mock_orm = Mock()
    mock_db = Mock()
    
    # Test data
    tool_data = {
        "name": "test_tool",
        "description": "A test tool",
        "schema": {"type": "object"}
    }
    actor = {"id": "test_actor", "org_id": "test_org"}
    
    # Test new tool creation
    with patch('letta.services.tool_manager.get_tool_by_name', return_value=None):
        result = await mock_tool_manager.create_or_update_tool(tool_data, actor)
        assert result is not None
        mock_orm.create_tool.assert_called_once()

    # Test existing tool update
    existing_tool = {"id": "existing_id", "name": "test_tool"}
    with patch('letta.services.tool_manager.get_tool_by_name', return_value=existing_tool):
        result = await mock_tool_manager.create_or_update_tool(tool_data, actor)
        assert result is not None
        mock_orm.update_tool_by_id.assert_called_once()

@pytest.mark.asyncio
async def test_tool_database_operations():
    async with AsyncSession() as session:
        tool_manager = ToolManager(session)
        
        # Test tool creation in database
        tool_data = {
            "name": "db_test_tool",
            "description": "Database test tool",
            "schema": {"type": "object"}
        }
        actor = {"id": "test_actor", "org_id": "test_org"}
        
        created_tool = await tool_manager.create_or_update_tool(tool_data, actor)
        assert created_tool.name == tool_data["name"]
        
        # Test tool retrieval
        retrieved_tool = await tool_manager.get_tool_by_name(tool_data["name"], actor)
        assert retrieved_tool is not None
        assert retrieved_tool.name == tool_data["name"] 