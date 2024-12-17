import pytest
from unittest.mock import Mock, patch
from letta.memory import MemoryManager, Storage

@pytest.mark.asyncio
async def test_archival_memory_insert():
    # Setup mocks
    mock_storage = Mock(spec=Storage)
    mock_db = Mock()
    memory_manager = MemoryManager(storage=mock_storage)
    
    # Test data
    test_data = {
        "content": "Test memory content",
        "metadata": {"type": "test", "tags": ["test"]}
    }
    
    # Mock storage response
    mock_storage.store_embedding.return_value = "memory_123"
    
    # Test memory insertion
    memory_id = await memory_manager.insert(test_data)
    
    # Verify flow
    mock_storage.store_embedding.assert_called_once()
    assert memory_id == "memory_123"

@pytest.mark.asyncio
async def test_memory_retrieval():
    mock_storage = Mock(spec=Storage)
    memory_manager = MemoryManager(storage=mock_storage)
    
    # Setup test data
    test_query = "test query"
    mock_results = [
        {"id": "mem1", "content": "Result 1", "score": 0.9},
        {"id": "mem2", "content": "Result 2", "score": 0.8}
    ]
    
    mock_storage.search_memories.return_value = mock_results
    
    # Test memory search
    results = await memory_manager.search(test_query)
    
    # Verify search flow
    mock_storage.search_memories.assert_called_once_with(test_query)
    assert len(results) == 2
    assert results[0]["score"] == 0.9

@pytest.mark.asyncio
async def test_memory_storage_operations():
    mock_db = Mock()
    storage = Storage(database=mock_db)
    
    # Test data
    memory_data = {
        "content": "Test content",
        "embedding": [0.1, 0.2, 0.3],
        "metadata": {"type": "test"}
    }
    
    # Test storage operations
    with patch('letta.memory.Storage.save_memory') as mock_save:
        mock_save.return_value = "memory_456"
        memory_id = await storage.store_embedding(memory_data)
        
        mock_save.assert_called_once()
        assert memory_id == "memory_456"

@pytest.mark.asyncio
async def test_full_memory_flow():
    # Setup components
    mock_client = Mock()
    mock_server = Mock()
    mock_storage = Mock(spec=Storage)
    memory_manager = MemoryManager(storage=mock_storage)
    
    # Test data
    memory_data = {
        "content": "Important information to remember",
        "metadata": {"source": "test", "timestamp": "2023-01-01"}
    }
    
    # Mock responses
    mock_storage.store_embedding.return_value = "memory_789"
    
    # Test full flow
    with patch('letta.server.server.MemoryManager', return_value=memory_manager):
        # Insert memory
        memory_id = await memory_manager.insert(memory_data)
        
        # Verify storage operations
        mock_storage.store_embedding.assert_called_once()
        assert memory_id == "memory_789"
        
        # Test retrieval
        mock_storage.get_memory.return_value = {**memory_data, "id": memory_id}
        retrieved_memory = await memory_manager.get(memory_id)
        
        assert retrieved_memory["id"] == memory_id
        assert retrieved_memory["content"] == memory_data["content"] 