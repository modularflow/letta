from typing import TYPE_CHECKING, List, Literal, Optional, Type

from sqlalchemy import String, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from letta.log import get_logger
from letta.orm.base import Base, CommonSqlalchemyMetaMixins
from letta.orm.errors import NoResultFound

if TYPE_CHECKING:
    from pydantic import BaseModel
    from sqlalchemy.orm import Session
    from letta.schemas.user import User

logger = get_logger(__name__)


class SqlalchemyBase(CommonSqlalchemyMetaMixins, Base):
    __abstract__ = True

    __order_by_default__ = "created_at"

    id: Mapped[str] = mapped_column(String, primary_key=True)

    @classmethod
    async def list(
        cls, *, db_session: "AsyncSession", cursor: Optional[str] = None, limit: Optional[int] = 50, **kwargs
    ) -> List[Type["SqlalchemyBase"]]:
        """List records with optional cursor (for pagination) and limit."""
        async with db_session as session:
            # Start with the base query filtered by kwargs
            query = select(cls).filter_by(**kwargs)

            # Add a cursor condition if provided
            if cursor:
                query = query.where(cls.id > cursor)

            # Add a limit to the query if provided
            query = query.order_by(cls.id).limit(limit)

            # Handle soft deletes if the class has the 'is_deleted' attribute
            if hasattr(cls, "is_deleted"):
                query = query.where(cls.is_deleted == False)

            # Execute the query and return the results as a list of model instances
            result = await session.execute(query)
            return list(result.scalars())

    @classmethod
    async def apply_access_predicate(
        cls,
        query: "select",
        actor: "User",
        access: List[Literal["read", "write", "admin"]],
    ) -> "select":
        """applies a WHERE clause restricting results to the given actor and access level
        Args:
            query: The initial sqlalchemy select statement
            actor: The user acting on the query. **Note**: this is called 'actor' to identify the
                   person or system acting. Users can act on users, making naming very sticky otherwise.
            access:
                what mode of access should the query restrict to? This will be used with granular permissions,
                but because of how it will impact every query we want to be explicitly calling access ahead of time.
        Returns:
            the sqlalchemy select statement restricted to the given access.
        """
        del access  # entrypoint for row-level permissions. Defaults to "same org as the actor, all permissions" at the moment
        org_id = getattr(actor, "organization_id", None)
        if not org_id:
            raise ValueError(f"object {actor} has no organization accessor")
        return query.where(cls.organization_id == org_id, cls.is_deleted == False)

    @property
    def __pydantic_model__(self) -> Type["BaseModel"]:
        raise NotImplementedError("Sqlalchemy models must declare a __pydantic_model__ property to be convertable.")

    async def to_pydantic(self) -> Type["BaseModel"]:
        """converts to the basic pydantic model counterpart"""
        return self.__pydantic_model__.model_validate(self)

    async def to_record(self) -> Type["BaseModel"]:
        """Deprecated accessor for to_pydantic"""
        logger.warning("to_record is deprecated, use to_pydantic instead.")
        return await self.to_pydantic()

    async def _set_created_and_updated_by_fields(self, actor_id: str) -> None:
        """Async version of _set_created_and_updated_by_fields."""
        if not self.created_by_id:
            self.created_by_id = actor_id
        # Always set the last_updated_by_id when updating
        self.last_updated_by_id = actor_id

    @classmethod
    async def aread(
        cls,
        db_session: "AsyncSession",
        identifier: Optional[str] = None,
        actor: Optional["User"] = None,
        access: Optional[List[Literal["read", "write", "admin"]]] = ["read"],
        **kwargs,
    ) -> Type["SqlalchemyBase"]:
        """Async version of read."""
        query = select(cls)
        query_conditions = []

        if identifier:
            query = query.where(cls.id == identifier)
            query_conditions.append(f"id='{identifier}'")

        if kwargs:
            query = query.filter_by(**kwargs)
            query_conditions.append(", ".join(f"{key}='{value}'" for key, value in kwargs.items()))

        if actor:
            query = await cls.apply_access_predicate(query, actor, access)
            query_conditions.append(f"access level in {access} for actor='{actor}'")

        if hasattr(cls, "is_deleted"):
            query = query.where(cls.is_deleted == False)
            query_conditions.append("is_deleted=False")

        result = await db_session.execute(query)
        if found := result.scalar():
            return found

        # Construct a detailed error message based on query conditions
        conditions_str = ", ".join(query_conditions) if query_conditions else "no specific conditions"
        raise NoResultFound(f"{cls.__name__} not found with {conditions_str}")

    async def acreate(self, db_session: "AsyncSession", actor: Optional["User"] = None) -> Type["SqlalchemyBase"]:
        """Async version of create."""
        if actor:
            await self._set_created_and_updated_by_fields(actor.id)

        async with db_session as session:
            session.add(self)
            await session.commit()
            await session.refresh(self)
            return self

    async def adelete(self, db_session: "AsyncSession", actor: Optional["User"] = None) -> Type["SqlalchemyBase"]:
        """Async version of delete."""
        if actor:
            await self._set_created_and_updated_by_fields(actor.id)

        self.is_deleted = True
        return await self.aupdate(db_session)

    async def ahard_delete(self, db_session: "AsyncSession", actor: Optional["User"] = None) -> None:
        """Async version of hard_delete."""
        if actor:
            logger.info(f"User {actor.id} requested hard deletion of {self.__class__.__name__} with ID {self.id}")

        async with db_session as session:
            try:
                await session.delete(self)
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.exception(f"Failed to hard delete {self.__class__.__name__} with ID {self.id}")
                raise ValueError(f"Failed to hard delete {self.__class__.__name__} with ID {self.id}: {e}")
            else:
                logger.info(f"{self.__class__.__name__} with ID {self.id} successfully hard deleted")

    async def aupdate(self, db_session: "AsyncSession", actor: Optional["User"] = None) -> Type["SqlalchemyBase"]:
        """Async version of update."""
        if actor:
            await self._set_created_and_updated_by_fields(actor.id)

        async with db_session as session:
            session.add(self)
            await session.commit()
            await session.refresh(self)
            return self
