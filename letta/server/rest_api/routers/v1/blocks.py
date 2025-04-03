from typing import TYPE_CHECKING, List, Optional

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query

from letta.schemas.block import Block, CreateBlock, UpdateBlock
from letta.server.rest_api.utils import get_letta_server
from letta.server.server import AsyncServer

if TYPE_CHECKING:
    pass

router = APIRouter(prefix="/blocks", tags=["blocks"])


@router.get("/", response_model=List[Block], operation_id="list_memory_blocks")
async def list_blocks(
    # query parameters
    label: Optional[str] = Query(None, description="Labels to include (e.g. human, persona)"),
    templates_only: bool = Query(True, description="Whether to include only templates"),
    name: Optional[str] = Query(None, description="Name of the block"),
    server: AsyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    actor = await server.get_user_or_default(user_id=user_id)

    blocks = await server.get_blocks(user_id=actor.id, label=label, template=templates_only)
    if blocks is None:
        return []
    return blocks


@router.post("/", response_model=Block, operation_id="create_memory_block")
async def create_block(
    create_block: CreateBlock = Body(...),
    server: AsyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    actor = await server.get_user_or_default(user_id=user_id)

    create_block.user_id = actor.id
    return await server.create_block(user_id=actor.id, request=create_block)


@router.patch("/{block_id}", response_model=Block, operation_id="update_memory_block")
async def update_block(
    block_id: str,
    updated_block: UpdateBlock = Body(...),
    server: AsyncServer = Depends(get_letta_server),
):
    # actor = server.get_current_user()

    updated_block.id = block_id
    return await server.update_block(request=updated_block)


# TODO: delete should not return anything
@router.delete("/{block_id}", response_model=Block, operation_id="delete_memory_block")
async def delete_block(
    block_id: str,
    server: AsyncServer = Depends(get_letta_server),
):

    return await server.delete_block(block_id=block_id)


@router.get("/{block_id}", response_model=Block, operation_id="get_memory_block")
async def get_block(
    block_id: str,
    server: AsyncServer = Depends(get_letta_server),
):

    block = await server.get_block(block_id=block_id)
    if block is None:
        raise HTTPException(status_code=404, detail="Block not found")
    return block
