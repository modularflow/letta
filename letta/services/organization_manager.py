from typing import List, Optional

from letta.orm.errors import NoResultFound
from letta.orm.organization import Organization as OrganizationModel
from letta.schemas.organization import Organization as PydanticOrganization
from letta.utils import enforce_types


class OrganizationManager:
    """Manager class to handle business logic related to Organizations."""

    DEFAULT_ORG_ID = "org-00000000-0000-4000-8000-000000000000"
    DEFAULT_ORG_NAME = "default_org"

    def __init__(self):
        # This is probably horrible but we reuse this technique from metadata.py
        # TODO: Please refactor this out
        # I am currently working on a ORM refactor and would like to make a more minimal set of changes
        # - Matt
        from letta.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    def get_default_organization(self) -> PydanticOrganization:
        """Fetch the default organization."""
        return self.get_organization_by_id(self.DEFAULT_ORG_ID)

    @enforce_types
    def get_organization_by_id(self, org_id: str) -> PydanticOrganization:
        """Fetch an organization by ID."""
        with self.session_maker() as session:
            try:
                organization = OrganizationModel.read(db_session=session, identifier=org_id)
                return organization.to_pydantic()
            except NoResultFound:
                raise ValueError(f"Organization with id {org_id} not found.")

    @enforce_types
    def create_organization(self, pydantic_org: PydanticOrganization) -> PydanticOrganization:
        """Create a new organization. If a name is provided, it is used, otherwise, a random one is generated."""
        with self.session_maker() as session:
            org = OrganizationModel(**pydantic_org.model_dump())
            org.create(session)
            return org.to_pydantic()

    @enforce_types
    def create_default_organization(self) -> PydanticOrganization:
        """Create the default organization."""
        with self.session_maker() as session:
            # Try to get it first
            try:
                org = OrganizationModel.read(db_session=session, identifier=self.DEFAULT_ORG_ID)
            # If it doesn't exist, make it
            except NoResultFound:
                org = OrganizationModel(name=self.DEFAULT_ORG_NAME, id=self.DEFAULT_ORG_ID)
                org.create(session)

            return org.to_pydantic()

    @enforce_types
    def update_organization_name_using_id(self, org_id: str, name: Optional[str] = None) -> PydanticOrganization:
        """Update an organization."""
        with self.session_maker() as session:
            org = OrganizationModel.read(db_session=session, identifier=org_id)
            if name:
                org.name = name
            org.update(session)
            return org.to_pydantic()

    @enforce_types
    def delete_organization_by_id(self, org_id: str):
        """Delete an organization by marking it as deleted."""
        with self.session_maker() as session:
            organization = OrganizationModel.read(db_session=session, identifier=org_id)
            organization.delete(session)

    @enforce_types
    def list_organizations(self, cursor: Optional[str] = None, limit: Optional[int] = 50) -> List[PydanticOrganization]:
        """List organizations with pagination based on cursor (org_id) and limit."""
        with self.session_maker() as session:
            results = OrganizationModel.list(db_session=session, cursor=cursor, limit=limit)
            return [org.to_pydantic() for org in results]

class AsyncOrganizationManager:
    """Async manager class to handle business logic related to Organizations."""

    DEFAULT_ORG_ID = "org-00000000-0000-4000-8000-000000000000"
    DEFAULT_ORG_NAME = "default_org"

    def __init__(self):
        from letta.server.server import async_db_context

        self.session_maker = async_db_context

    @enforce_types
    async def create_default_org(self) -> PydanticOrganization:
        """Create the default organization."""
        async with self.session_maker() as session:
            # Try to retrieve the org
            try:
                organization = await OrganizationModel.read(db_session=session, identifier=self.DEFAULT_ORG_ID)
            except NoResultFound:
                # If it doesn't exist, make it
                organization = OrganizationModel(id=self.DEFAULT_ORG_ID, name=self.DEFAULT_ORG_NAME)
                await organization.create(session)

            return await organization.to_pydantic()

    @enforce_types
    async def get_org_by_id(self, org_id: str) -> PydanticOrganization:
        """Fetch an organization by ID."""
        async with self.session_maker() as session:
            org = await OrganizationModel.read(db_session=session, identifier=org_id)
            return await org.to_pydantic()

    @enforce_types
    async def create_org(self, pydantic_org: PydanticOrganization) -> PydanticOrganization:
        """Create a new organization."""
        async with self.session_maker() as session:
            # Check if an organization with the same name already exists
            try:
                existing_org = await OrganizationModel.read(db_session=session, name=pydantic_org.name)
                # If we get here, an org with this name exists
                return await existing_org.to_pydantic()
            except NoResultFound:
                # Create new organization
                org = OrganizationModel(**pydantic_org.model_dump())
                await org.create(session)
                return await org.to_pydantic()

    @enforce_types
    async def update_org(self, pydantic_org: PydanticOrganization) -> PydanticOrganization:
        """Update an organization."""
        async with self.session_maker() as session:
            org = await OrganizationModel.read(db_session=session, identifier=pydantic_org.id)
            for key, value in pydantic_org.model_dump().items():
                setattr(org, key, value)
            await org.update(session)
            return await org.to_pydantic()

    @enforce_types
    async def list_orgs(self, cursor: Optional[str] = None, limit: Optional[int] = 50) -> List[PydanticOrganization]:
        """List organizations with pagination."""
        async with self.session_maker() as session:
            results = await OrganizationModel.list(db_session=session, cursor=cursor, limit=limit)
            pydantic_orgs = []
            for org in results:
                pydantic_orgs.append(await org.to_pydantic())
            return pydantic_orgs
