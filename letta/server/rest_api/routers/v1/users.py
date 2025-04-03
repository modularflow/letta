from typing import TYPE_CHECKING, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from letta.schemas.api_key import APIKey, APIKeyCreate
from letta.schemas.user import User, UserCreate, UserUpdate
from letta.server.rest_api.utils import get_letta_server

# from letta.server.schemas.users import (
#     CreateAPIKeyRequest,
#     CreateAPIKeyResponse,
#     CreateUserRequest,
#     CreateUserResponse,
#     DeleteAPIKeyResponse,
#     DeleteUserResponse,
#     GetAllUsersResponse,
#     GetAPIKeysResponse,
# )

if TYPE_CHECKING:
    from letta.schemas.user import User
    from letta.server.server import AsyncServer


router = APIRouter(prefix="/users", tags=["users", "admin"])


@router.get("/", tags=["admin"], response_model=List[User], operation_id="list_users")
async def list_users(
    cursor: Optional[str] = Query(None),
    limit: Optional[int] = Query(50),
    server: "AsyncServer" = Depends(get_letta_server),
):
    """
    Get a list of all users in the database
    """
    try:
        next_cursor, users = await server.user_manager.list_users(cursor=cursor, limit=limit)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")
    return users


@router.post("/", tags=["admin"], response_model=User, operation_id="create_user")
async def create_user(
    request: UserCreate = Body(...),
    server: "AsyncServer" = Depends(get_letta_server),
):
    """
    Create a new user in the database
    """
    user = User(**request.model_dump())
    user = await server.user_manager.create_user(user)
    return user


@router.put("/", tags=["admin"], response_model=User, operation_id="update_user")
async def update_user(
    user: UserUpdate = Body(...),
    server: "AsyncServer" = Depends(get_letta_server),
):
    """
    Update a user in the database
    """
    user = await server.user_manager.update_user(user)
    return user


@router.delete("/", tags=["admin"], response_model=User, operation_id="delete_user")
async def delete_user(
    user_id: str = Query(..., description="The user_id key to be deleted."),
    server: "AsyncServer" = Depends(get_letta_server),
):
    # TODO make a soft deletion, instead of a hard deletion
    try:
        user = await server.user_manager.get_user_by_id(user_id=user_id)
        if user is None:
            raise HTTPException(status_code=404, detail=f"User does not exist")
        await server.user_manager.delete_user_by_id(user_id=user_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")
    return user


@router.post("/keys", response_model=APIKey, operation_id="create_api_key")
async def create_new_api_key(
    create_key: APIKeyCreate = Body(...),
    server: "AsyncServer" = Depends(get_letta_server),
):
    """
    Create a new API key for a user
    """
    api_key = await server.create_api_key(create_key)
    return api_key


@router.get("/keys", response_model=List[APIKey], operation_id="list_api_keys")
async def get_api_keys(
    user_id: str = Query(..., description="The unique identifier of the user."),
    server: "AsyncServer" = Depends(get_letta_server),
):
    """
    Get a list of all API keys for a user
    """
    if await server.user_manager.get_user_by_id(user_id=user_id) is None:
        raise HTTPException(status_code=404, detail=f"User does not exist")
    api_keys = await server.ms.get_all_api_keys_for_user(user_id=user_id)
    return api_keys


@router.delete("/keys", response_model=APIKey, operation_id="delete_api_key")
async def delete_api_key(
    api_key: str = Query(..., description="The API key to be deleted."),
    server: "AsyncServer" = Depends(get_letta_server),
):
    return await server.delete_api_key(api_key)
