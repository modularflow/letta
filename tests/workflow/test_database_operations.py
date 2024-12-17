import pytest
from unittest.mock import Mock, patch
from sqlalchemy.ext.asyncio import AsyncSession
from letta.orm.sqlalchemy_base import Base

@pytest.mark.asyncio
async def test_basic_crud_operations():
    # Setup mocks
    mock_session = Mock(spec=AsyncSession)
    mock_sqlalchemy = Mock()
    
    # Test data
    test_entity = {
        "id": "test_123",
        "name": "Test Entity",
        "description": "Test Description"
    }
    
    # Test create operation
    with patch('sqlalchemy.ext.asyncio.AsyncSession.execute') as mock_execute:
        mock_execute.return_value.scalar.return_value = test_entity
        result = await mock_session.execute(
            "INSERT INTO test_table (name, description) VALUES (:name, :description)",
            test_entity
        )
        assert mock_execute.called
        assert result.scalar().get("id") == test_entity["id"]

@pytest.mark.asyncio
async def test_complex_queries():
    mock_session = Mock(spec=AsyncSession)
    
    # Test data
    test_query = """
        SELECT t1.*, t2.name as related_name 
        FROM test_table t1 
        JOIN related_table t2 ON t1.related_id = t2.id
        WHERE t1.status = :status
    """
    query_params = {"status": "active"}
    
    # Test query execution
    with patch('sqlalchemy.ext.asyncio.AsyncSession.execute') as mock_execute:
        mock_execute.return_value.fetchall.return_value = [
            {"id": "1", "name": "Test 1", "related_name": "Related 1"},
            {"id": "2", "name": "Test 2", "related_name": "Related 2"}
        ]
        
        result = await mock_session.execute(test_query, query_params)
        results = result.fetchall()
        
        assert mock_execute.called
        assert len(results) == 2

@pytest.mark.asyncio
async def test_transaction_management():
    mock_session = Mock(spec=AsyncSession)
    
    # Test data
    test_operations = [
        ("INSERT INTO table1 VALUES (:id, :name)", {"id": 1, "name": "Test 1"}),
        ("UPDATE table2 SET status = :status WHERE id = :id", {"id": 1, "status": "updated"})
    ]
    
    # Test transaction
    async with mock_session.begin() as transaction:
        for query, params in test_operations:
            await mock_session.execute(query, params)
        
        # Test transaction commit
        await transaction.commit()
    
    assert mock_session.begin.called
    assert mock_session.execute.call_count == len(test_operations)

@pytest.mark.asyncio
async def test_orm_mappings():
    # Test ORM class
    class TestEntity(Base):
        __tablename__ = "test_entities"
        id = "id"
        name = "name"
        description = "description"
    
    mock_session = Mock(spec=AsyncSession)
    
    # Test data
    test_entity = TestEntity(
        name="Test Entity",
        description="Test Description"
    )
    
    # Test ORM operations
    with patch('sqlalchemy.ext.asyncio.AsyncSession.add') as mock_add:
        await mock_session.add(test_entity)
        mock_add.assert_called_once_with(test_entity)
        
        await mock_session.commit()
        assert mock_session.commit.called

@pytest.mark.asyncio
async def test_error_handling():
    mock_session = Mock(spec=AsyncSession)
    
    # Test data
    invalid_query = "INSERT INTO non_existent_table VALUES (:value)"
    
    # Test error handling
    with pytest.raises(Exception):
        with patch('sqlalchemy.ext.asyncio.AsyncSession.execute', side_effect=Exception("Database error")):
            await mock_session.execute(invalid_query, {"value": "test"})
    
    # Verify rollback was called
    assert mock_session.rollback.called

@pytest.mark.asyncio
async def test_connection_management():
    mock_session = Mock(spec=AsyncSession)
    
    # Test connection lifecycle
    async with mock_session as session:
        # Test that session is active
        assert not session.closed
        
        # Perform some operation
        await session.execute("SELECT 1")
        
        # Test that session is still active
        assert not session.closed
    
    # Test that session is closed after context
    assert mock_session.close.called 