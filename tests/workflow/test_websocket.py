import pytest
import asyncio
from unittest.mock import Mock, patch
from letta.server.ws_api.interface import WSInterface

@pytest.mark.asyncio
async def test_websocket_connection():
    mock_websocket = Mock()
    mock_ws_interface = WSInterface()
    
    # Test client registration
    await mock_ws_interface.register_client(mock_websocket)
    assert mock_websocket in mock_ws_interface.clients
    
    # Test client message handling
    test_message = {
        "user_id": "test_user",
        "agent_id": "test_agent",
        "content": "Hello, world!"
    }
    
    with patch('letta.server.server.process_message') as mock_process:
        await mock_ws_interface.handle_message(mock_websocket, test_message)
        mock_process.assert_called_once_with(
            test_message["user_id"],
            test_message["agent_id"],
            test_message["content"]
        )

@pytest.mark.asyncio
async def test_message_broadcast():
    mock_ws_interface = WSInterface()
    mock_clients = [Mock() for _ in range(3)]
    
    # Register multiple clients
    for client in mock_clients:
        await mock_ws_interface.register_client(client)
    
    # Test broadcasting
    test_response = {"type": "response", "content": "Test broadcast"}
    await mock_ws_interface.broadcast_message(test_response)
    
    # Verify each client received the message
    for client in mock_clients:
        client.send_json.assert_called_once_with(test_response)

@pytest.mark.asyncio
async def test_client_disconnection():
    mock_websocket = Mock()
    mock_ws_interface = WSInterface()
    
    # Register and then disconnect client
    await mock_ws_interface.register_client(mock_websocket)
    await mock_ws_interface.unregister_client(mock_websocket)
    
    assert mock_websocket not in mock_ws_interface.clients

@pytest.mark.asyncio
async def test_full_message_flow():
    mock_client = Mock()
    mock_server = Mock()
    mock_agent_manager = Mock()
    mock_ws_interface = WSInterface()
    
    # Setup test message
    user_message = {
        "user_id": "test_user",
        "agent_id": "test_agent",
        "content": "Process this message"
    }
    
    # Mock response from agent
    agent_response = {"type": "response", "content": "Processed message"}
    mock_agent_manager.process_message.return_value = agent_response
    
    with patch('letta.server.server.AgentManager', return_value=mock_agent_manager):
        # Register client and send message
        await mock_ws_interface.register_client(mock_client)
        await mock_ws_interface.handle_message(mock_client, user_message)
        
        # Verify message processing
        mock_agent_manager.process_message.assert_called_once()
        mock_client.send_json.assert_called_with(agent_response) 