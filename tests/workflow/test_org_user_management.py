import pytest
from unittest.mock import Mock, patch
from letta.services.organization_manager import OrganizationManager
from letta.services.user_manager import UserManager

@pytest.mark.asyncio
async def test_organization_creation():
    # Setup mocks
    mock_db = Mock()
    org_manager = OrganizationManager(session=mock_db)
    
    # Test data
    org_data = {
        "name": "Test Organization",
        "description": "Test Description",
        "metadata": {"type": "test"}
    }
    
    # Test organization creation
    with patch('letta.services.organization_manager.save_organization') as mock_save:
        mock_save.return_value = {"id": "org_123", **org_data}
        created_org = await org_manager.create_organization(org_data)
        
        mock_save.assert_called_once()
        assert created_org["id"] == "org_123"
        assert created_org["name"] == org_data["name"]

@pytest.mark.asyncio
async def test_user_creation_with_org():
    # Setup mocks
    mock_db = Mock()
    org_manager = OrganizationManager(session=mock_db)
    user_manager = UserManager(session=mock_db)
    
    # Test data
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "org_id": "org_123"
    }
    
    # Mock organization verification
    with patch('letta.services.organization_manager.verify_organization') as mock_verify:
        mock_verify.return_value = True
        
        # Test user creation
        with patch('letta.services.user_manager.save_user') as mock_save_user:
            mock_save_user.return_value = {"id": "user_456", **user_data}
            created_user = await user_manager.create_user(user_data)
            
            mock_verify.assert_called_once_with("org_123")
            mock_save_user.assert_called_once()
            assert created_user["id"] == "user_456"
            assert created_user["org_id"] == user_data["org_id"]

@pytest.mark.asyncio
async def test_user_organization_operations():
    mock_db = Mock()
    org_manager = OrganizationManager(session=mock_db)
    user_manager = UserManager(session=mock_db)
    
    # Test data
    org_data = {"name": "Test Org", "description": "Test"}
    user_data = {
        "username": "testuser",
        "email": "test@example.com"
    }
    
    # Test full flow
    with patch('letta.services.organization_manager.save_organization') as mock_save_org:
        mock_save_org.return_value = {"id": "org_789", **org_data}
        org = await org_manager.create_organization(org_data)
        
        # Add user to organization
        user_data["org_id"] = org["id"]
        with patch('letta.services.user_manager.save_user') as mock_save_user:
            mock_save_user.return_value = {"id": "user_789", **user_data}
            user = await user_manager.create_user(user_data)
            
            assert user["org_id"] == org["id"]
            mock_save_user.assert_called_once()

@pytest.mark.asyncio
async def test_organization_user_listing():
    mock_db = Mock()
    org_manager = OrganizationManager(session=mock_db)
    
    # Test data
    org_id = "org_123"
    mock_users = [
        {"id": "user1", "username": "user1", "org_id": org_id},
        {"id": "user2", "username": "user2", "org_id": org_id}
    ]
    
    # Test user listing
    with patch('letta.services.organization_manager.get_organization_users') as mock_get_users:
        mock_get_users.return_value = mock_users
        users = await org_manager.get_organization_users(org_id)
        
        mock_get_users.assert_called_once_with(org_id)
        assert len(users) == 2
        assert all(user["org_id"] == org_id for user in users) 