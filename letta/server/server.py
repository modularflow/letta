# inspecting tools
import os
import traceback
import warnings
from abc import abstractmethod
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Union

from fastapi import HTTPException

import letta.constants as constants
import letta.server.utils as server_utils
import letta.system as system
from letta.agent import Agent, save_agent
from letta.agent_store.db import attach_base
from letta.agent_store.storage import StorageConnector, TableType
from letta.credentials import LettaCredentials
from letta.data_sources.connectors import DataConnector, load_data

# from letta.data_types import (
#    AgentState,
#    EmbeddingConfig,
#    LLMConfig,
#    Message,
#    Preset,
#    Source,
#    Token,
#    User,
# )
from letta.functions.functions import generate_schema, parse_source_code
from letta.functions.schema_generator import generate_schema

# TODO use custom interface
from letta.interface import AgentInterface  # abstract
from letta.interface import CLIInterface  # for printing to terminal
from letta.log import get_logger
from letta.memory import get_memory_functions
from letta.metadata import MetadataStore
from letta.o1_agent import O1Agent
from letta.orm import Base
from letta.orm.errors import NoResultFound
from letta.prompts import gpt_system
from letta.providers import (
    AnthropicProvider,
    AzureProvider,
    GoogleAIProvider,
    GroqProvider,
    LettaProvider,
    OllamaProvider,
    OpenAIProvider,
    Provider,
    VLLMChatCompletionsProvider,
    VLLMCompletionsProvider,
)
from letta.schemas.agent import AgentState, AgentType, CreateAgent, UpdateAgentState
from letta.schemas.api_key import APIKey, APIKeyCreate
from letta.schemas.block import (
    Block,
    CreateBlock,
    CreateHuman,
    CreatePersona,
    UpdateBlock,
)
from letta.schemas.embedding_config import EmbeddingConfig

# openai schemas
from letta.schemas.enums import JobStatus
from letta.schemas.file import FileMetadata
from letta.schemas.job import Job
from letta.schemas.letta_message import LettaMessage
from letta.schemas.llm_config import LLMConfig
from letta.schemas.memory import (
    ArchivalMemorySummary,
    ContextWindowOverview,
    Memory,
    RecallMemorySummary,
)
from letta.schemas.message import Message, MessageCreate, MessageRole, UpdateMessage
from letta.schemas.organization import Organization
from letta.schemas.passage import Passage
from letta.schemas.source import Source, SourceCreate, SourceUpdate
from letta.schemas.tool import Tool, ToolCreate
from letta.schemas.usage import LettaUsageStatistics
from letta.schemas.user import User
from letta.services.agents_tags_manager import AgentsTagsManager, AsyncAgentsTagsManager
from letta.services.organization_manager import AsyncOrganizationManager, OrganizationManager
from letta.services.tool_manager import AsyncToolManager, ToolManager
from letta.services.user_manager import AsyncUserManager, UserManager
from letta.utils import create_random_username, json_dumps, json_loads

# from letta.llm_api_tools import openai_get_model_list, azure_openai_get_model_list, smart_urljoin

logger = get_logger(__name__)


class Server(object):
    """Abstract server class that supports multi-agent multi-user"""

    @abstractmethod
    def list_agents(self, user_id: str) -> dict:
        """List all available agents to a user"""
        raise NotImplementedError

    @abstractmethod
    def get_agent_messages(self, user_id: str, agent_id: str, start: int,
                           count: int) -> list:
        """Paginated query of in-context messages in agent message queue"""
        raise NotImplementedError

    @abstractmethod
    def get_agent_memory(self, user_id: str, agent_id: str) -> dict:
        """Return the memory of an agent (core memory + non-core statistics)"""
        raise NotImplementedError

    @abstractmethod
    def get_agent_state(self, user_id: str, agent_id: str) -> dict:
        """Return the config of an agent"""
        raise NotImplementedError

    @abstractmethod
    def get_server_config(self, user_id: str) -> dict:
        """Return the base config"""
        raise NotImplementedError

    @abstractmethod
    def update_agent_core_memory(self, user_id: str, agent_id: str,
                                 new_memory_contents: dict) -> dict:
        """Update the agents core memory block, return the new state"""
        raise NotImplementedError

    @abstractmethod
    def create_agent(
        self,
        user_id: str,
        agent_config: Union[dict, AgentState],
        interface: Union[AgentInterface, None],
    ) -> str:
        """Create a new agent using a config"""
        raise NotImplementedError

    @abstractmethod
    def user_message(self, user_id: str, agent_id: str, message: str) -> None:
        """Process a message from the user, internally calls step"""
        raise NotImplementedError

    @abstractmethod
    def system_message(self, user_id: str, agent_id: str,
                       message: str) -> None:
        """Process a message from the system, internally calls step"""
        raise NotImplementedError

    @abstractmethod
    def send_messages(self, user_id: str, agent_id: str,
                      messages: Union[MessageCreate, List[Message]]) -> None:
        """Send a list of messages to the agent"""
        raise NotImplementedError

    @abstractmethod
    def run_command(self, user_id: str, agent_id: str,
                    command: str) -> Union[str, None]:
        """Run a command on the agent, e.g. /memory

        May return a string with a message generated by the command
        """
        raise NotImplementedError


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from letta.config import LettaConfig

# NOTE: hack to see if single session management works
from letta.settings import model_settings, settings, tool_settings

config = LettaConfig.load()

attach_base()

if settings.letta_pg_uri_no_default:
    config.recall_storage_type = "postgres"
    config.recall_storage_uri = settings.letta_pg_uri_no_default
    config.archival_storage_type = "postgres"
    config.archival_storage_uri = settings.letta_pg_uri_no_default

    # create engine
    engine = create_engine(settings.letta_pg_uri)
else:
    # TODO: don't rely on config storage
    engine = create_engine(
        "sqlite:///" + os.path.join(config.recall_storage_path, "sqlite.db"))

Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


from contextlib import contextmanager, asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

db_context = contextmanager(get_db)

# Create async engine and session
if settings.letta_pg_uri_no_default:
    async_engine = create_async_engine(settings.letta_pg_uri)
else:
    async_engine = create_async_engine("sqlite+aiosqlite:///" + os.path.join(config.recall_storage_path, "sqlite.db"))

AsyncSessionLocal = sessionmaker(
    async_engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# Async dependency
async def get_async_db():
    async_session = AsyncSessionLocal()
    try:
        yield async_session
    finally:
        await async_session.close()

async_db_context = asynccontextmanager(get_async_db)

class AsyncServer(Server):
    """Async server implementation that supports multi-agent multi-user"""

    def __init__(self, chaining: bool = True, max_chaining_steps: Optional[bool] = None,
                 default_interface_factory: Callable[[], AgentInterface] = lambda: CLIInterface(),
                 init_with_default_org_and_user: bool = True):
        """Initialize the async server with async managers"""
        self.chaining = chaining
        self.max_chaining_steps = max_chaining_steps
        self.default_interface_factory = default_interface_factory
        self.credentials = LettaCredentials.load()

        # Initialize the metadata store
        config = LettaConfig.load()
        if settings.letta_pg_uri_no_default:
            config.recall_storage_type = "postgres"
            config.recall_storage_uri = settings.letta_pg_uri_no_default
            config.archival_storage_type = "postgres"
            config.archival_storage_uri = settings.letta_pg_uri_no_default
        config.save()
        self.config = config
        self.ms = MetadataStore(self.config)

        # Initialize async managers
        self.organization_manager = AsyncOrganizationManager()
        self.user_manager = AsyncUserManager()
        self.tool_manager = AsyncToolManager()
        self.agents_tags_manager = AsyncAgentsTagsManager()

        # Initialize providers
        self._enabled_providers: List[Provider] = [LettaProvider()]
        if model_settings.openai_api_key:
            self._enabled_providers.append(
                OpenAIProvider(
                    api_key=model_settings.openai_api_key,
                    base_url=model_settings.openai_api_base,
                ))
        if model_settings.anthropic_api_key:
            self._enabled_providers.append(
                AnthropicProvider(api_key=model_settings.anthropic_api_key, ))
        if model_settings.ollama_base_url:
            self._enabled_providers.append(
                OllamaProvider(
                    base_url=model_settings.ollama_base_url,
                    api_key=None,
                    default_prompt_formatter=model_settings.
                    default_prompt_formatter,
                ))
        if model_settings.gemini_api_key:
            self._enabled_providers.append(
                GoogleAIProvider(api_key=model_settings.gemini_api_key, ))
        if model_settings.azure_api_key and model_settings.azure_base_url:
            assert model_settings.azure_api_version, "AZURE_API_VERSION is required"
            self._enabled_providers.append(
                AzureProvider(
                    api_key=model_settings.azure_api_key,
                    base_url=model_settings.azure_base_url,
                    api_version=model_settings.azure_api_version,
                ))
        if model_settings.groq_api_key:
            self._enabled_providers.append(
                GroqProvider(api_key=model_settings.groq_api_key))
        if model_settings.vllm_api_base:
            # vLLM exposes both a /chat/completions and a /completions endpoint
            self._enabled_providers.append(
                VLLMCompletionsProvider(
                    base_url=model_settings.vllm_api_base,
                    default_prompt_formatter=model_settings.
                    default_prompt_formatter,
                ))
            # NOTE: to use the /chat/completions endpoint, you need to specify extra flags on vLLM startup
            # see: https://docs.vllm.ai/en/latest/getting_started/examples/openai_chat_completion_client_with_tools.html
            # e.g. "... --enable-auto-tool-choice --tool-call-parser hermes"
            self._enabled_providers.append(
                VLLMChatCompletionsProvider(
                    base_url=model_settings.vllm_api_base, ))

        # Store initialization flag
        self._init_with_default_org_and_user = init_with_default_org_and_user
        self._initialized = False

    @classmethod
    async def create(cls, *args, **kwargs) -> 'AsyncServer':
        """Factory method to create and initialize an AsyncServer instance"""
        instance = cls(*args, **kwargs)
        await instance.initialize()
        return instance

    async def initialize(self):
        """Async initialization method"""
        if self._initialized:
            return

        # Make default user and org
        if self._init_with_default_org_and_user:
            self.default_org = await self.organization_manager.create_default_organization()
            self.default_user = await self.user_manager.create_default_user()
            await self.add_default_blocks(self.default_user.id)
            await self.tool_manager.add_base_tools(actor=self.default_user)

            # If there is a default org/user
            # This logic may have to change in the future
            if settings.load_default_external_tools:
                await self.add_default_external_tools(actor=self.default_user)

        self._initialized = True
    
    async def save_agents(self):
        """Saves all the agents that are in the in-memory object store"""
        # Note: In a production async server, this would likely use a different approach
        # for managing active agents and saving them
        for agent_d in self.active_agents:
            try:
                await save_agent(agent_d["agent"], self.ms)
                logger.debug(f"Saved agent {agent_d['agent_id']}")
            except Exception as e:
                logger.exception(
                    f"Error occurred while trying to save agent {agent_d['agent_id']}:\n{e}"
                )

    async def _get_agent(self, user_id: str, agent_id: str) -> Union[Agent, None]:
        """Get the agent object from the in-memory object store"""
        # Note: In a production async server, access to shared resources would be synchronized
        for d in self.active_agents:
            if d["user_id"] == str(user_id) and d["agent_id"] == str(agent_id):
                return d["agent"]
        return None

    async def _add_agent(self, user_id: str, agent_id: str,
                   agent_obj: Agent) -> None:
        """Put an agent object inside the in-memory object store"""
        # Make sure the agent doesn't already exist
        if await self._get_agent(user_id=user_id, agent_id=agent_id) is not None:
            logger.exception(
                f"Agent (user={user_id}, agent={agent_id}) is already loaded")
            return
        # Add Agent instance to the in-memory list
        self.active_agents.append({
            "user_id": str(user_id),
            "agent_id": str(agent_id),
            "agent": agent_obj,
        })

    async def _load_agent(self,
                    agent_id: str,
                    actor: User,
                    interface: Union[AgentInterface, None] = None) -> Agent:
        """Loads a saved agent into memory (if it doesn't exist, throw an error)"""
        assert isinstance(agent_id, str), agent_id
        user_id = actor.id

        # If an interface isn't specified, use the default
        if interface is None:
            interface = self.default_interface_factory()

        try:
            logger.debug(
                f"Grabbing agent user_id={user_id} agent_id={agent_id} from database"
            )
            agent_state = await self.ms.get_agent(agent_id=agent_id, user_id=user_id)
            if not agent_state:
                logger.exception(f"agent_id {agent_id} does not exist")
                raise ValueError(f"agent_id {agent_id} does not exist")

            # Instantiate an agent object using the state retrieved
            logger.debug(f"Creating an agent object")
            tool_objs = []
            for name in agent_state.tools:
                try:
                    tool_obj = await self.tool_manager.get_tool_by_name(
                        tool_name=name, actor=actor)
                    tool_objs.append(tool_obj)
                except NoResultFound:
                    warnings.warn(
                        f"Tried to retrieve a tool with name {name} from the agent_state, but does not exist in tool db."
                    )

            # set agent_state tools to only the names of the available tools
            agent_state.tools = [t.name for t in tool_objs]

            # Make sure the memory is a memory object
            assert isinstance(agent_state.memory, Memory)

            if agent_state.agent_type == AgentType.memgpt_agent:
                letta_agent = Agent(agent_state=agent_state,
                                    interface=interface,
                                    tools=tool_objs)
                letta_agent.set_server(self)
            elif agent_state.agent_type == AgentType.o1_agent:
                letta_agent = O1Agent(agent_state=agent_state,
                                      interface=interface,
                                      tools=tool_objs)
            else:
                raise NotImplementedError("Not a supported agent type")

            # Add the agent to the in-memory store and return its reference
            logger.debug(
                f"Adding agent to the agent cache: user_id={user_id}, agent_id={agent_id}"
            )
            await self._add_agent(user_id=user_id,
                            agent_id=agent_id,
                            agent_obj=letta_agent)
            return letta_agent

        except Exception as e:
            logger.exception(
                f"Error occurred while trying to get agent {agent_id}:\n{e}")
            raise

    async def _get_or_load_agent(self, agent_id: str) -> Agent:
        """Check if the agent is in-memory, then load"""
        agent_state = await self.ms.get_agent(agent_id=agent_id)
        if not agent_state:
            raise ValueError(f"Agent does not exist")
        user_id = agent_state.user_id
        actor = await self.user_manager.get_user_by_id(user_id)

        logger.debug(
            f"Checking for agent user_id={user_id} agent_id={agent_id}")
        letta_agent = await self._get_agent(user_id=user_id, agent_id=agent_id)
        if not letta_agent:
            logger.debug(
                f"Agent not loaded, loading agent user_id={user_id} agent_id={agent_id}"
            )
            letta_agent = await self._load_agent(agent_id=agent_id, actor=actor)
        return letta_agent

    async def _step(
        self,
        user_id: str,
        agent_id: str,
        input_messages: Union[Message, List[Message]],
    ) -> LettaUsageStatistics:
        """Send the input message through the agent asynchronously"""

        # Input validation
        if isinstance(input_messages, Message):
            input_messages = [input_messages]
        if not all(isinstance(m, Message) for m in input_messages):
            raise ValueError(
                f"messages should be a Message or a list of Message, got {type(input_messages)}"
            )

        logger.debug(f"Got input messages: {input_messages}")
        letta_agent = None
        try:
            # Get the agent object (loaded in memory)
            letta_agent = await self._get_or_load_agent(agent_id=agent_id)
            if letta_agent is None:
                raise KeyError(
                    f"Agent (user={user_id}, agent={agent_id}) is not loaded")

            # Determine whether or not to token stream based on the capability of the interface
            token_streaming = letta_agent.interface.streaming_mode if hasattr(
                letta_agent.interface, "streaming_mode") else False

            logger.debug(f"Starting agent step")
            usage_stats = await letta_agent.astep(
                messages=input_messages,
                chaining=self.chaining,
                max_chaining_steps=self.max_chaining_steps,
                stream=token_streaming,
                ms=self.ms,
                skip_verify=True,
            )

        except Exception as e:
            logger.error(f"Error in server._step: {e}")
            print(traceback.print_exc())
            raise
        finally:
            logger.debug("Calling step_yield()")
            if letta_agent:
                await letta_agent.interface.astep_yield()

        return usage_stats

    async def _command(self, user_id: str, agent_id: str, command: str) -> LettaUsageStatistics:
        """Execute a command on an agent asynchronously"""
        usage = None
        letta_agent = None

        try:
            # Get the agent object (loaded in memory)
            letta_agent = await self._get_or_load_agent(agent_id=agent_id)
            if letta_agent is None:
                raise KeyError(
                    f"Agent (user={user_id}, agent={agent_id}) is not loaded")

            # Handle different commands
            if command.lower() == "reset":
                await letta_agent.areset()

            elif command.lower() == "undo":
                await letta_agent.aundo()

            elif command.lower() == "redo":
                await letta_agent.aredo()

            elif command.lower() == "rewrite" or command.lower().startswith(
                    "rewrite "):
                # TODO this needs to also modify the persistence manager
                if len(command) < len("rewrite "):
                    logger.warning("Missing text after the command")
                else:
                    for x in range(len(letta_agent.messages) - 1, 0, -1):
                        if letta_agent.messages[x].get("role") == "assistant":
                            text = command[len("rewrite "):].strip()
                            args = json_loads(letta_agent.messages[x].get(
                                "function_call").get("arguments"))
                            args["message"] = text
                            letta_agent.messages[x].get("function_call").update(
                                {"arguments": json_dumps(args)})
                            break

            # No skip options
            elif command.lower() == "wipe":
                # exit not supported on server.py
                raise ValueError(command)

            elif command.lower() == "heartbeat":
                input_message = system.get_heartbeat()
                usage = await self._step(user_id=user_id,
                                    agent_id=agent_id,
                                    input_message=input_message)

            elif command.lower() == "memorywarning":
                input_message = system.get_token_limit_warning()
                usage = await self._step(user_id=user_id,
                                    agent_id=agent_id,
                                    input_message=input_message)

        except Exception as e:
            logger.error(f"Error in server._command: {e}")
            print(traceback.print_exc())
            raise

        if not usage:
            usage = LettaUsageStatistics()

        return usage

    async def user_message(
        self,
        user_id: str,
        agent_id: str,
        message: Union[str, Message],
        timestamp: Optional[datetime] = None,
    ) -> LettaUsageStatistics:
        """Process an incoming user message and feed it through the Letta agent"""
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if await self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Basic input sanitization
        if isinstance(message, str):
            if len(message) == 0:
                raise ValueError(f"Invalid input: '{message}'")

            # If the input begins with a command prefix, reject
            elif message.startswith("/"):
                raise ValueError(f"Invalid input: '{message}'")

            packaged_user_message = system.package_user_message(
                user_message=message,
                time=timestamp.isoformat() if timestamp else None,
            )

            # Convert to a Message object
            if timestamp:
                message = Message(
                    user_id=user_id,
                    agent_id=agent_id,
                    role="user",
                    text=packaged_user_message,
                    created_at=timestamp,
                )
            else:
                message = Message(
                    user_id=user_id,
                    agent_id=agent_id,
                    role="user",
                    text=packaged_user_message,
                )

        # Run the agent state forward
        usage = await self._step(user_id=user_id,
                           agent_id=agent_id,
                           input_messages=message)
        return usage

    async def system_message(
        self,
        user_id: str,
        agent_id: str,
        message: Union[str, Message],
        timestamp: Optional[datetime] = None,
    ) -> LettaUsageStatistics:
        """Process an incoming system message and feed it through the Letta agent"""
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if await self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Basic input sanitization
        if isinstance(message, str):
            if len(message) == 0:
                raise ValueError(f"Invalid input: '{message}'")

            # If the input begins with a command prefix, reject
            elif message.startswith("/"):
                raise ValueError(f"Invalid input: '{message}'")

            packaged_system_message = system.package_system_message(
                system_message=message)

            # Convert to a Message object
            if timestamp:
                message = Message(
                    user_id=user_id,
                    agent_id=agent_id,
                    role="system",
                    text=packaged_system_message,
                    created_at=timestamp,
                )
            else:
                message = Message(
                    user_id=user_id,
                    agent_id=agent_id,
                    role="system",
                    text=packaged_system_message,
                )

        if isinstance(message, Message):
            # Can't have a null text field
            if message.text is None or len(message.text) == 0:
                raise ValueError(f"Invalid input: '{message.text}'")
            # If the input begins with a command prefix, reject
            elif message.text.startswith("/"):
                raise ValueError(f"Invalid input: '{message.text}'")

        else:
            raise TypeError(
                f"Invalid input: '{message}' - type {type(message)}")

        if timestamp:
            # Override the timestamp with what the caller provided
            message.created_at = timestamp

        # Run the agent state forward
        return await self._step(user_id=user_id,
                          agent_id=agent_id,
                          input_messages=message)

    async def send_messages(
        self,
        user_id: str,
        agent_id: str,
        messages: Union[List[MessageCreate], List[Message]],
        wrap_user_message: bool = True,
        wrap_system_message: bool = True,
    ) -> LettaUsageStatistics:
        """Send a list of messages to the agent"""
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if await self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        message_objects: List[Message] = []

        if all(isinstance(m, MessageCreate) for m in messages):
            for message in messages:
                assert isinstance(message, MessageCreate)

                # If wrapping is enabled, wrap with metadata before placing content inside the Message object
                if message.role == MessageRole.user and wrap_user_message:
                    message.text = system.package_user_message(
                        user_message=message.text)
                elif message.role == MessageRole.system and wrap_system_message:
                    message.text = system.package_system_message(
                        system_message=message.text)
                else:
                    raise ValueError(f"Invalid message role: {message.role}")

                # Create the Message object
                message_objects.append(
                    Message(
                        user_id=user_id,
                        agent_id=agent_id,
                        role=message.role,
                        text=message.text,
                        name=message.name,
                        # assigned later?
                        model=None,
                        # irrelevant
                        tool_calls=None,
                        tool_call_id=None,
                    ))

        elif all(isinstance(m, Message) for m in messages):
            for message in messages:
                assert isinstance(message, Message)
                message_objects.append(message)

        else:
            raise ValueError(
                f"All messages must be of type Message or MessageCreate, got {[type(message) for message in messages]}"
            )

        # Run the agent state forward
        return await self._step(user_id=user_id,
                          agent_id=agent_id,
                          input_messages=message_objects)

    async def run_command(self, user_id: str, agent_id: str,
                    command: str) -> LettaUsageStatistics:
        """Run a command on the agent"""
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if await self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # If the input begins with a command prefix, attempt to process it as a command
        if command.startswith("/"):
            if len(command) > 1:
                command = command[1:]  # strip the prefix
        return await self._command(user_id=user_id,
                             agent_id=agent_id,
                             command=command)

    async def create_agent(
        self,
        request: CreateAgent,
        actor: User,
        interface: Union[AgentInterface, None] = None,
    ) -> AgentState:
        """Create a new agent using a config"""

        user_id = actor.id
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")

        if interface is None:
            interface = self.default_interface_factory()

        # create agent name
        if request.name is None:
            request.name = create_random_username()

        if request.agent_type is None:
            request.agent_type = AgentType.memgpt_agent

        # system debug
        if request.system is None:
            # TODO: don't hardcode
            if request.agent_type == AgentType.memgpt_agent:
                request.system = gpt_system.get_system_text("memgpt_chat")
            elif request.agent_type == AgentType.o1_agent:
                request.system = gpt_system.get_system_text(
                    "memgpt_modified_o1")
            else:
                raise ValueError(f"Invalid agent type: {request.agent_type}")

        logger.debug(f"Attempting to find user: {user_id}")
        user = await self.user_manager.get_user_by_id(user_id=user_id)
        if not user:
            raise ValueError(
                f"cannot find user with associated client id: {user_id}")

        try:
            # model configuration
            llm_config = request.llm_config
            embedding_config = request.embedding_config

            # get tools + only add if they exist
            tool_objs = []
            if request.tools:
                for tool_name in request.tools:
                    try:
                        tool_obj = await self.tool_manager.get_tool_by_name(
                            tool_name=tool_name, actor=actor)
                        tool_objs.append(tool_obj)
                    except NoResultFound:
                        warnings.warn(
                            f"Attempted to add a nonexistent tool {tool_name} to agent {request.name}, skipping."
                        )
            # reset the request.tools to only valid tools
            request.tools = [t.name for t in tool_objs]

            assert request.memory is not None
            memory_functions = await get_memory_functions(request.memory)
            for func_name, func in memory_functions.items():

                if request.tools and func_name in request.tools:
                    # tool already added
                    continue
                source_code = parse_source_code(func)
                # memory functions are not terminal
                json_schema = generate_schema(func, name=func_name)
                source_type = "python"
                tags = ["memory", "memgpt-base"]
                tool = await self.tool_manager.create_or_update_tool(
                    Tool(
                        source_code=source_code,
                        source_type=source_type,
                        tags=tags,
                        json_schema=json_schema,
                    ),
                    actor=actor,
                )
                tool_objs.append(tool)
                if not request.tools:
                    request.tools = []
                request.tools.append(tool.name)

            # TODO: save the agent state
            agent_state = AgentState(
                name=request.name,
                user_id=user_id,
                tools=request.tools if request.tools else [],
                tool_rules=request.tool_rules if request.tool_rules else [],
                agent_type=request.agent_type or AgentType.memgpt_agent,
                llm_config=llm_config,
                embedding_config=embedding_config,
                system=request.system,
                memory=request.memory,
                description=request.description,
                metadata_=request.metadata_,
            )
            if request.agent_type == AgentType.memgpt_agent:
                agent = Agent(
                    interface=interface,
                    agent_state=agent_state,
                    tools=tool_objs,
                    # gpt-3.5-turbo tends to omit inner monologue, relax this requirement for now
                    first_message_verify_mono=(
                        True if (llm_config and llm_config.model is not None
                                 and "gpt-4" in llm_config.model) else False),
                    initial_message_sequence=request.initial_message_sequence,
                )
                agent.set_server(self)
            elif request.agent_type == AgentType.o1_agent:
                agent = O1Agent(
                    interface=interface,
                    agent_state=agent_state,
                    tools=tool_objs,
                    # gpt-3.5-turbo tends to omit inner monologue, relax this requirement for now
                    first_message_verify_mono=(
                        True if (llm_config and llm_config.model is not None
                                 and "gpt-4" in llm_config.model) else False),
                )
            # rebuilding agent memory on agent create in case shared memory blocks
            # were specified in the new agent's memory config. we're doing this for two reasons:
            # 1. if only the ID of the shared memory block was specified, we can fetch its most recent value
            # 2. if the shared block state changed since this agent initialization started, we can be sure to have the latest value
            await agent.rebuild_memory(force=True, ms=self.ms)
            # FIXME: this is a hacky way to get the system prompts injected into agent into the DB
            # await self.ms.update_agent(agent.agent_state)
        except Exception as e:
            logger.exception(e)
            try:
                if 'agent' in locals():
                    await self.ms.delete_agent(agent_id=agent.agent_state.id)
            except Exception as delete_e:
                logger.exception(f"Failed to delete_agent:\n{delete_e}")
            raise e

        # save agent
        await save_agent(agent, self.ms)
        logger.debug(f"Created new agent from config: {agent}")

        assert isinstance(
            agent.agent_state.memory,
            Memory), f"Invalid memory type: {type(agent_state.memory)}"
        # return AgentState

        return agent.agent_state

    async def update_agent(self, user_id: str, agent_id: str, 
                     update_dict: dict) -> dict:
        """Update an agent's config"""
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if await self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Update agent with async tool manager
        agent_state = await self.tool_manager.update_agent_tools(agent_id, update_dict)
        agent = await self._get_or_load_agent(agent_id)
        await agent.update_state(agent_state)
        return agent.agent_state

    async def get_tools_from_agent(self, agent_id: str,
                             user_id: Optional[str]) -> List[Tool]:
        """Get tools from an existing agent"""
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if await self.ms.get_agent(agent_id=agent_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = await self._get_or_load_agent(agent_id=agent_id)
        return letta_agent.tools

    async def add_tool_to_agent(
        self,
        agent_id: str,
        tool_id: str,
        user_id: str,
    ):
        """Add tools from an existing agent"""
        try:
            user = await self.user_manager.get_user_by_id(user_id=user_id)
        except NoResultFound:
            raise ValueError(f"User user_id={user_id} does not exist")

        if await self.ms.get_agent(agent_id=agent_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = await self._get_or_load_agent(agent_id=agent_id)

        # Get all the tool objects from the request
        tool_objs = []
        tool_obj = await self.tool_manager.get_tool_by_id(tool_id=tool_id,
                                                    actor=user)
        assert tool_obj, f"Tool with id={tool_id} does not exist"
        tool_objs.append(tool_obj)

        for tool in letta_agent.tools:
            tool_obj = await self.tool_manager.get_tool_by_id(tool_id=tool.id,
                                                        actor=user)
            assert tool_obj, f"Tool with id={tool.id} does not exist"

            # If it's not the already added tool
            if tool_obj.id != tool_id:
                tool_objs.append(tool_obj)

        # replace the list of tool names ("ids") inside the agent state
        letta_agent.agent_state.tools = [tool.name for tool in tool_objs]

        # then attempt to link the tools modules
        await letta_agent.link_tools(tool_objs)

        # save the agent
        await save_agent(letta_agent, self.ms)
        return letta_agent.agent_state

    async def remove_tool_from_agent(
        self,
        agent_id: str,
        tool_id: str,
        user_id: str,
    ):
        """Remove tools from an existing agent"""
        try:
            user = await self.user_manager.get_user_by_id(user_id=user_id)
        except NoResultFound:
            raise ValueError(f"User user_id={user_id} does not exist")

        if await self.ms.get_agent(agent_id=agent_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = await self._get_or_load_agent(agent_id=agent_id)

        # Get all the tool_objs
        tool_objs = []
        for tool in letta_agent.tools:
            tool_obj = await self.tool_manager.get_tool_by_id(tool_id=tool.id,
                                                        actor=user)
            assert tool_obj, f"Tool with id={tool.id} does not exist"

            # If it's not the tool we want to remove
            if tool_obj.id != tool_id:
                tool_objs.append(tool_obj)

        # replace the list of tool names ("ids") inside the agent state
        letta_agent.agent_state.tools = [tool.name for tool in tool_objs]

        # then attempt to link the tools modules
        await letta_agent.link_tools(tool_objs)

        # save the agent
        await save_agent(letta_agent, self.ms)
        return letta_agent.agent_state

    async def _agent_state_to_config(self, agent_state: AgentState) -> dict:
        """Convert AgentState to a dict for a JSON response"""
        assert agent_state is not None

        agent_config = {
            "id": agent_state.id,
            "name": agent_state.name,
            "human": agent_state._metadata.get("human", None),
            "persona": agent_state._metadata.get("persona", None),
            "created_at": agent_state.created_at.isoformat(),
        }
        return agent_config

    async def list_agents(self, user_id: str) -> dict:
        """List all available agents to a user"""
        user = await self.user_manager.get_user_by_id(user_id=user_id)
        agents_states = await self.ms.list_agents(user_id=user_id)
        return agents_states

    async def get_blocks(
        self,
        user_id: Optional[str] = None,
        label: Optional[str] = None,
        template: Optional[bool] = None,
        template_name: Optional[str] = None,
        id: Optional[str] = None,
    ) -> Optional[List[Block]]:
        """Get blocks with optional filtering"""
        return await self.ms.get_blocks(user_id=user_id, label=label, template=template, 
                                 template_name=template_name, id=id)

    async def get_block(self, block_id: str):
        """Get a block by ID"""
        return await self.ms.get_block(block_id=block_id)

    async def create_block(self,
                     request: CreateBlock,
                     user_id: str,
                     update: bool = False) -> Block:
        """Create a new block"""
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        
        block = Block(**request.dict(), user_id=user_id)
        await self.ms.create_block(block, update=update)
        return block

    async def update_block(self, request: UpdateBlock) -> Block:
        """Update an existing block"""
        block = await self.ms.update_block(request)
        return block

    async def delete_block(self, block_id: str):
        """Delete a block"""
        await self.ms.delete_block(block_id)

    async def get_agent_id(self, name: str, user_id: str):
        """Get agent ID by name"""
        return await self.ms.get_agent_id(name=name, user_id=user_id)
        
    async def get_source(self, source_id: str) -> Source:
        """Get a source by ID"""
        source = await self.ms.get_source(source_id=source_id)
        if not source:
            raise ValueError(f"Source with ID {source_id} not found")
        return source

    async def get_source_id(self, source_name: str, user_id: str) -> str:
        """Get source ID by name"""
        return await self.ms.get_source_id(source_name=source_name, user_id=user_id)

    async def get_agent_memory(self, user_id: str, agent_id: str) -> dict:
        """Return the memory of an agent (core memory + non-core statistics)"""
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if await self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        agent = await self._get_or_load_agent(agent_id=agent_id)
        return agent.memory

    async def get_archival_memory_summary(self,
                                    agent_id: str) -> ArchivalMemorySummary:
        agent = await self._get_or_load_agent(agent_id=agent_id)
        return ArchivalMemorySummary(
            size=len(agent.persistence_manager.archival_memory))

    async def get_recall_memory_summary(self, agent_id: str) -> RecallMemorySummary:
        agent = await self._get_or_load_agent(agent_id=agent_id)
        return RecallMemorySummary(
            size=len(agent.persistence_manager.recall_memory))

    async def get_in_context_message_ids(self, agent_id: str) -> List[str]:
        """Get the message ids of the in-context messages in the agent's memory"""
        # Get the agent object (loaded in memory)
        letta_agent = await self._get_or_load_agent(agent_id=agent_id)
        return [m.id for m in letta_agent._messages]

    async def get_in_context_messages(self, agent_id: str) -> List[Message]:
        """Get the in-context messages in the agent's memory"""
        # Get the agent object (loaded in memory)
        letta_agent = await self._get_or_load_agent(agent_id=agent_id)
        return letta_agent._messages

    async def get_agent_message(self, agent_id: str, message_id: str) -> Message:
        """Get a single message from the agent's memory"""
        # Get the agent object (loaded in memory)
        letta_agent = await self._get_or_load_agent(agent_id=agent_id)
        message = letta_agent.persistence_manager.recall_memory.storage.get(
            id=message_id)
        return message

    async def get_agent_messages(self, user_id: str, agent_id: str, start: int,
                           count: int) -> list:
        """Paginated query of in-context messages in agent message queue"""
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if await self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        agent = await self._get_or_load_agent(agent_id=agent_id)
        return await agent.get_messages(start=start, count=count)

    async def get_agent_archival(self, user_id: str, agent_id: str, start: int,
                           count: int) -> List[Passage]:
        """Get agent archival memory"""
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if await self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        agent = await self._get_or_load_agent(agent_id=agent_id)
        return await agent.persistence_manager.archival_memory.get_passages(start=start, count=count)

    async def get_agent_archival_cursor(
        self,
        user_id: str,
        agent_id: str,
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: Optional[int] = 100,
        order_by: Optional[str] = "created_at",
        reverse: Optional[bool] = False,
    ) -> List[Passage]:
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if await self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = await self._get_or_load_agent(agent_id=agent_id)

        # iterate over recorde
        cursor, records = letta_agent.persistence_manager.archival_memory.storage.get_all_cursor(
            after=after,
            before=before,
            limit=limit,
            order_by=order_by,
            reverse=reverse)
        return records
    
    async def insert_archival_memory(self, user_id: str, agent_id: str,
                               memory_contents: str) -> List[Passage]:
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if await self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = await self._get_or_load_agent(agent_id=agent_id)

        # Insert into archival memory
        passage_ids = letta_agent.persistence_manager.archival_memory.insert(
            memory_string=memory_contents, return_ids=True)

        # TODO: this is gross, fix
        return [
            await letta_agent.persistence_manager.archival_memory.storage.get(
                id=passage_id) for passage_id in passage_ids
        ]

    async def delete_archival_memory(self, user_id: str, agent_id: str,
                               memory_id: str):
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if await self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # TODO: should return a passage

        # Get the agent object (loaded in memory)
        letta_agent = await self._get_or_load_agent(agent_id=agent_id)

        # Delete by ID
        # TODO check if it exists first, and throw error if not
        await letta_agent.persistence_manager.archival_memory.storage.delete(
            {"id": memory_id})

        # TODO: return archival memory

    async def get_agent_recall_cursor(
        self,
        user_id: str,
        agent_id: str,
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: Optional[int] = 100,
        order_by: Optional[str] = "created_at",
        reverse: Optional[bool] = False,
        return_message_object: bool = True,
        use_assistant_message: bool = False,
        assistant_message_function_name: str = constants.DEFAULT_MESSAGE_TOOL,
        assistant_message_function_kwarg: str = constants.DEFAULT_MESSAGE_TOOL_KWARG,
    ) -> Union[List[Message], List[LettaMessage]]:
        """Get agent recall memory with cursor-based pagination"""
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if await self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        agent = await self._get_or_load_agent(agent_id=agent_id)
        return await agent.persistence_manager.recall_memory.get_messages(
            after=after, before=before, limit=limit, order_by=order_by, reverse=reverse,
            return_message_object=return_message_object, use_assistant_message=use_assistant_message,
            assistant_message_function_name=assistant_message_function_name,
            assistant_message_function_kwarg=assistant_message_function_kwarg
        )

    async def get_agent_state(self, user_id: str, agent_id: str) -> dict:
        """Return the config of an agent"""
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if await self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        agent = await self._get_or_load_agent(agent_id=agent_id)
        return agent.agent_state

    async def get_server_config(self, user_id: str) -> dict:
        """Return the base config"""
        return self.config

    async def update_agent_core_memory(self, user_id: str, agent_id: str,
                                new_memory_contents: dict) -> dict:
        """Update the agents core memory block, return the new state"""
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if await self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        agent = await self._get_or_load_agent(agent_id=agent_id)
        await agent.update_memory(new_memory_contents)
        return agent.memory

    async def rename_agent(self, user_id: str, agent_id: str,
                     new_agent_name: str) -> AgentState:
        """Rename an agent"""
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if await self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Get the agent object (loaded in memory)
        agent = await self._get_or_load_agent(agent_id=agent_id)
        agent.agent_state.name = new_agent_name
        await save_agent(agent, self.ms)
        return agent.agent_state

    async def delete_agent(self, user_id: str, agent_id: str) -> None:
        """Delete an agent"""
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if await self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Delete agent and its tools
        await self.tool_manager.delete_agent_tools(agent_id)
        await self.ms.delete_agent(agent_id)

    async def api_key_to_user(self, api_key: str) -> str:
        """Convert API key to user ID"""
        api_key_obj = await self.ms.get_api_key_by_key(api_key=api_key)
        if not api_key_obj:
            raise ValueError(f"API key {api_key} does not exist")
        return api_key_obj.user_id

    async def create_api_key(
            self, request: APIKeyCreate) -> APIKey:
        """Create a new API key"""
        return await self.ms.create_api_key(request=request)

    async def list_api_keys(self, user_id: str) -> List[APIKey]:
        """List all API keys for a user"""
        return await self.ms.list_api_keys(user_id=user_id)

    async def delete_api_key(self, api_key: str) -> APIKey:
        """Delete an API key"""
        return await self.ms.delete_api_key(api_key=api_key)

    async def create_source(self, request: SourceCreate,
                      user_id: str) -> Source:  # TODO: add other fields
        """Create a new data source"""
        source = Source(
            name=request.name,
            user_id=user_id,
            embedding_config=self.list_embedding_models()
            [0],  # TODO: require providing this
        )
        await self.ms.create_source(source)
        assert await self.ms.get_source(
            source_name=request.name, user_id=user_id
        ) is not None, f"Failed to create source {request.name}"
        return source
    
    async def update_source(self, request: SourceUpdate, user_id: str) -> Source:
        """Update an existing data source"""
        if not request.id:
            existing_source = await self.ms.get_source(source_name=request.name,
                                                 user_id=user_id)
        else:
            existing_source = await self.ms.get_source(source_id=request.id)
        if not existing_source:
            raise ValueError("Source does not exist")

        # override updated fields
        if request.name:
            existing_source.name = request.name
        if request.metadata_:
            existing_source.metadata_ = request.metadata_
        if request.description:
            existing_source.description = request.description

        await self.ms.update_source(existing_source)
        return existing_source

    async def delete_source(self, source_id: str, user_id: str):
        """Delete a data source"""
        source = await self.ms.get_source(source_id=source_id, user_id=user_id)
        await self.ms.delete_source(source_id)

        # delete data from passage store
        passage_store = StorageConnector.get_storage_connector(
            TableType.PASSAGES, self.config, user_id=user_id)
        await passage_store.delete({"source_id": source_id})

        # TODO: delete data from agent passage stores (?)
 
    async def create_job(self, user_id: str, metadata: dict) -> Job:
        """Create a new job"""
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")

        job = Job(user_id=user_id, status=JobStatus.created, metadata_=metadata)
        await self.ms.create_job(job)
        return job

    async def delete_job(self, job_id: str) -> None:
        """Delete a job"""
        await self.ms.delete_job(job_id)

    async def get_job(self, job_id: str) -> Job:
        """Get a job by ID"""
        return await self.ms.get_job(job_id)
 
    async def list_jobs(self, user_id: str) -> List[Job]:
        """List all jobs for a user"""
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")

        return await self.ms.list_jobs(user_id=user_id)

    async def list_active_jobs(self, user_id: str) -> List[Job]:
        """List active jobs for a user"""
        jobs = await self.list_jobs(user_id=user_id)
        return [job for job in jobs if job.status in [JobStatus.created, JobStatus.running]]
 
    async def delete_file_from_source(self, source_id: str, file_id: str, 
                                user_id: str) -> None:
        """Delete a file from a source"""
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if await self.ms.get_source(source_id=source_id) is None:
            raise ValueError(f"Source source_id={source_id} does not exist")

        await self.ms.delete_file_from_source(source_id, file_id)

    async def load_data(
        self,
        user_id: str,
        connector: DataConnector,
        source_name: str,
    ) -> Tuple[int, int]:
        """Load data from a DataConnector into a source for a specified user_id"""
        # TODO: this should be implemented as a batch job or at least async, since it may take a long time

        # load data from a data source into the document store
        source = await self.ms.get_source(source_name=source_name, user_id=user_id)
        if source is None:
            raise ValueError(
                f"Data source {source_name} does not exist for user {user_id}")

        # get the data connectors
        passage_store = StorageConnector.get_storage_connector(
            TableType.PASSAGES, self.config, user_id=user_id)
        file_store = StorageConnector.get_storage_connector(TableType.FILES,
                                                            self.config,
                                                            user_id=user_id)

        # load data into the document store
        passage_count, document_count = await load_data(connector, source,
                                                  passage_store, file_store)
        return passage_count, document_count

    async def attach_source_to_agent(
        self,
        user_id: str,
        agent_id: str,
        # source_id: str,
        source_id: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> Source:
        # attach a data source to an agent
        data_source = await self.ms.get_source(source_id=source_id,
                                         user_id=user_id,
                                         source_name=source_name)
        if data_source is None:
            raise ValueError(
                f"Data source id={source_id} name={source_name} does not exist for user_id {user_id}"
            )

        # get connection to data source storage
        source_connector = StorageConnector.get_storage_connector(
            TableType.PASSAGES, self.config, user_id=user_id)

        # load agent
        agent = await self._get_or_load_agent(agent_id=agent_id)

        # attach source to agent
        await agent.attach_source(data_source.id, source_connector, self.ms)

        return data_source

    async def detach_source_from_agent(
        self,
        user_id: str,
        agent_id: str,
        # source_id: str,
        source_id: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> Source:
        if not source_id:
            assert source_name is not None, "source_name must be provided if source_id is not"
            source = await self.ms.get_source(source_name=source_name,
                                        user_id=user_id)
            source_id = source.id
        else:
            source = await self.ms.get_source(source_id=source_id)

        # delete all Passage objects with source_id==source_id from agent's archival memory
        agent = await self._get_or_load_agent(agent_id=agent_id)
        archival_memory = agent.persistence_manager.archival_memory
        await archival_memory.storage.delete({"source_id": source_id})

        # delete agent-source mapping
        await self.ms.detach_source(agent_id=agent_id, source_id=source_id)

        # return back source data
        return source

    async def list_attached_sources(self, agent_id: str) -> List[Source]:
        """List all attached sources to an agent"""
        return await self.ms.list_attached_sources(agent_id=agent_id)

    async def list_files_from_source(
            self,
            source_id: str,
            limit: int = 1000,
            cursor: Optional[str] = None) -> List[FileMetadata]:
        """List all files from a source"""
        return await self.ms.list_files_from_source(source_id=source_id, limit=limit, cursor=cursor)

    async def list_data_source_passages(self, user_id: str,
                                  source_id: str) -> List[Passage]:
        """List all passages from a data source"""
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if await self.ms.get_source(source_id=source_id) is None:
            raise ValueError(f"Source source_id={source_id} does not exist")

        return await self.ms.list_passages_from_source(source_id=source_id)

    async def list_sources(self, user_id: str) -> List[Source]:
        """List all sources for a user"""
        if await self.user_manager.get_user_by_id(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")

        return await self.ms.list_sources(user_id=user_id)

    async def add_default_external_tools(self, actor: User) -> bool:
        """Add default langchain tools. Return true if successful, false otherwise."""
        success = True
        tool_creates = ToolCreate.load_default_langchain_tools(
        ) + ToolCreate.load_default_crewai_tools()
        if tool_settings.composio_api_key:
            tool_creates += ToolCreate.load_default_composio_tools()
        for tool_create in tool_creates:
            try:
                await self.tool_manager.create_or_update_tool(
                    Tool(**tool_create.model_dump()), actor=actor)
            except Exception as e:
                warnings.warn(
                    f"An error occurred while creating tool {tool_create}: {e}"
                )
                warnings.warn(traceback.format_exc())
                success = False

        return success

    async def add_default_blocks(self, user_id: str):
        """Add default blocks for a user"""
        from letta.utils import list_human_files, list_persona_files

        assert user_id is not None, "User ID must be provided"

        for persona_file in list_persona_files():
            text = open(persona_file, "r", encoding="utf-8").read()
            name = os.path.basename(persona_file).replace(".txt", "")
            await self.create_block(CreatePersona(user_id=user_id,
                                            template_name=name,
                                            value=text,
                                            template=True),
                              user_id=user_id,
                              update=True)

        for human_file in list_human_files():
            text = open(human_file, "r", encoding="utf-8").read()
            name = os.path.basename(human_file).replace(".txt", "")
            await self.create_block(CreateHuman(user_id=user_id,
                                          template_name=name,
                                          value=text,
                                          template=True),
                              user_id=user_id,
                              update=True)

    async def get_agent_message(self, agent_id: str,
                          message_id: str) -> Optional[Message]:
        """Get a single message from the agent's memory"""
        # Get the agent object (loaded in memory)
        letta_agent = await self._get_or_load_agent(agent_id=agent_id)
        message = await letta_agent.persistence_manager.recall_memory.storage.get(
            id=message_id)
        return message

    async def update_agent_message(self, agent_id: str,
                             request: UpdateMessage) -> Message:
        """Update an agent message"""
        return await self.ms.update_message(agent_id=agent_id, request=request)

    async def rewrite_agent_message(self, agent_id: str, new_text: str) -> Message:
        """Rewrite an agent message"""
        # Get the current message
        letta_agent = await self._get_or_load_agent(agent_id=agent_id)
        return await letta_agent.rewrite_message(new_text=new_text)

    async def rethink_agent_message(self, agent_id: str,
                              new_thought: str) -> Message:
        """Rethink an agent message"""
        # Get the current message
        letta_agent = await self._get_or_load_agent(agent_id=agent_id)
        return await letta_agent.rethink_message(new_thought=new_thought)

    async def retry_agent_message(self, agent_id: str) -> List[Message]:
        """Retry an agent message"""
        # Get the current message
        letta_agent = await self._get_or_load_agent(agent_id=agent_id)
        return await letta_agent.retry_message()

    async def get_user_or_default(self, user_id: Optional[str]) -> User:
        """Get the user object for user_id if it exists, otherwise return the default user object"""
        if user_id is None:
            user_id = self.user_manager.DEFAULT_USER_ID

        try:
            return await self.user_manager.get_user_by_id(user_id=user_id)
        except ValueError:
            raise HTTPException(status_code=404,
                                detail=f"User with id {user_id} not found")

    async def get_organization_or_default(self, org_id: str = "") -> Organization:
        """Get organization by ID or default"""
        if not org_id:
            org_id = await self.organization_manager.DEFAULT_ORG_ID
        
        try:
            return await self.organization_manager.get_organization_by_id(org_id=org_id)
        except NoResultFound:
            # If the org doesn't exist, return the default
            return await self.organization_manager.get_organization_by_id(
                org_id=self.organization_manager.DEFAULT_ORG_ID)

    async def list_llm_models(self) -> List[LLMConfig]:
        """List available models"""

        llm_models = []
        for provider in self._enabled_providers:
            llm_models.extend(provider.list_llm_models())
        return llm_models

    async def list_embedding_models(self) -> List[EmbeddingConfig]:
        """List available embedding models"""
        embedding_models = []
        for provider in self._enabled_providers:
            embedding_models.extend(provider.list_embedding_models())
        return embedding_models

    async def add_llm_model(self, request: LLMConfig) -> LLMConfig:
        """Add a new LLM model"""

    async def add_embedding_model(self, request: EmbeddingConfig) -> EmbeddingConfig:
        """Add a new embedding model"""

    async def get_agent_context_window(
        self,
        user_id: str,
        agent_id: str,
    ) -> ContextWindowOverview:
        # Get the current message
        letta_agent = await self._get_or_load_agent(agent_id=agent_id)
        return letta_agent.get_context_window()

    async def load_file_to_source(self, source_id: str, file_path: str, 
                            job_id: str) -> None:
        """Load a file into a source"""
        try:
            # Update job status to processing
            await self.update_job_status(job_id, JobStatus.processing)
            
            # Process file and create embeddings
            source = await self.get_source(source_id)
            num_passages = await self._process_file(file_path, source)
            
            # Update job metadata and status
            await self.update_job(
                job_id,
                status=JobStatus.completed,
                metadata={
                    "num_passages": num_passages,
                    "source_id": source_id,
                }
            )
        except Exception as e:
            # Update job status to failed if there's an error
            await self.update_job(
                job_id,
                status=JobStatus.failed,
                metadata={"error": str(e)}
            )
            raise

    async def update_agent_state(
        self,
        request: UpdateAgentState,
        actor: User,
    ) -> AgentState:
        """Update the agent state, return the new state"""
        try:
            await self.user_manager.get_user_by_id(user_id=actor.id)
        except Exception:
            raise ValueError(f"User user_id={actor.id} does not exist")

        if await self.ms.get_agent(agent_id=request.id) is None:
            raise ValueError(f"Agent agent_id={request.id} does not exist")

        # Get the agent object (loaded in memory)
        letta_agent = await self._get_or_load_agent(agent_id=request.id)

        # update the core memory of the agent
        if request.memory:
            assert isinstance(request.memory, Memory), type(request.memory)
            new_memory_contents = request.memory.to_flat_dict()
            await self.update_agent_core_memory(
                user_id=actor.id,
                agent_id=request.id,
                new_memory_contents=new_memory_contents)

        # update the system prompt
        if request.system:
            await letta_agent.update_system_prompt(request.system)

        # update in-context messages
        if request.message_ids:
            # This means the user is trying to change what messages are in the message buffer
            await letta_agent.set_message_buffer(message_ids=request.message_ids)

        # tools
        if request.tools:
            # Replace tools and also re-link

            # (1) get tools + make sure they exist
            # Current and target tools as sets of tool names
            current_tools = set(letta_agent.agent_state.tools)
            target_tools = set(request.tools)

            # Calculate tools to add and remove
            tools_to_add = target_tools - current_tools
            tools_to_remove = current_tools - target_tools

            # Fetch tool objects for those to add and remove
            tools_to_add = [
                await self.tool_manager.get_tool_by_name(tool_name=tool, actor=actor)
                for tool in tools_to_add
            ]
            tools_to_remove = [
                await self.tool_manager.get_tool_by_name(tool_name=tool, actor=actor)
                for tool in tools_to_remove
            ]

            # update agent tool list
            for tool in tools_to_remove:
                await self.remove_tool_from_agent(agent_id=request.id,
                                        tool_id=tool.id,
                                        user_id=actor.id)
            for tool in tools_to_add:
                await self.add_tool_to_agent(agent_id=request.id,
                                   tool_id=tool.id,
                                   user_id=actor.id)

            # reload agent
            letta_agent = await self._get_or_load_agent(agent_id=request.id)

        # configs
        if request.llm_config:
            letta_agent.agent_state.llm_config = request.llm_config
        if request.embedding_config:
            letta_agent.agent_state.embedding_config = request.embedding_config

        # other minor updates
        if request.name:
            letta_agent.agent_state.name = request.name
        if request.metadata_:
            letta_agent.agent_state.metadata_ = request.metadata_

        # Manage tag state
        if request.tags is not None:
            current_tags = set(
                await self.agents_tags_manager.get_tags_for_agent(
                    agent_id=letta_agent.agent_state.id, actor=actor))
            target_tags = set(request.tags)

            tags_to_add = target_tags - current_tags
            tags_to_remove = current_tags - target_tags

            for tag in tags_to_add:
                await self.agents_tags_manager.add_tag_to_agent(
                    agent_id=letta_agent.agent_state.id, tag=tag, actor=actor)
            for tag in tags_to_remove:
                await self.agents_tags_manager.delete_tag_from_agent(
                    agent_id=letta_agent.agent_state.id, tag=tag, actor=actor)

        # save the agent
        assert isinstance(letta_agent.memory, Memory)
        await save_agent(letta_agent, self.ms)
        return letta_agent.agent_state

    async def list_agents(self, user_id: str, tags: Optional[List[str]] = None) -> List[AgentState]:
        """List all available agents to a user"""
        user = await self.user_manager.get_user_by_id(user_id=user_id)

        if tags is None:
            agents_states = await self.ms.list_agents(user_id=user_id)
            return agents_states
        else:
            agent_ids = []
            for tag in tags:
                agent_ids += await self.agents_tags_manager.get_agents_by_tag(
                    tag=tag, actor=user)

            return [
                await self.get_agent_state(user_id=user.id, agent_id=agent_id)
                for agent_id in agent_ids
            ]

    async def update_job_status(self, job_id: str, status: JobStatus) -> Job:
        """Update job status"""
        job = await self.ms.get_job(job_id)
        if not job:
            raise ValueError(f"Job with ID {job_id} not found")
        
        job.status = status
        return await self.ms.update_job(job)

    async def update_job(self, job_id: str, status: Optional[JobStatus] = None, 
                    metadata: Optional[Dict] = None) -> Job:
        """Update job status and/or metadata"""
        job = await self.ms.get_job(job_id)
        if not job:
            raise ValueError(f"Job with ID {job_id} not found")
        
        if status is not None:
            job.status = status
        
        if metadata is not None:
            if not job.metadata_:
                job.metadata_ = {}
            job.metadata_.update(metadata)
            
        return await self.ms.update_job(job)

    async def _process_file(self, file_path: str, source: Source) -> int:
        """Process a file and create embeddings
        
        Returns the number of passages created
        """
        # This would need to be implemented based on how your system processes files
        # Here's a placeholder implementation
        try:
            connector = StorageConnector.get_storage_connector(
                TableType.PASSAGES, self.config, user_id=source.user_id)
            
            # Placeholder for file processing logic
            # This would likely involve:
            # 1. Reading the file
            # 2. Splitting into passages
            # 3. Creating embeddings
            # 4. Storing in database
            
            # Just a placeholder return value
            return 10  # Simulating 10 passages created
        except Exception as e:
            logger.exception(f"Error processing file {file_path}: {e}")
            raise

    async def _create_agent_instance(self, agent_state: AgentState) -> Agent:
        """Create an Agent instance from an AgentState"""
        user_id = agent_state.user_id
        actor = await self.user_manager.get_user_by_id(user_id=user_id)
        
        # Create interface
        interface = self.default_interface_factory()
        
        # Get tools
        tool_objs = []
        for name in agent_state.tools:
            try:
                tool_obj = await self.tool_manager.get_tool_by_name(
                    tool_name=name, actor=actor)
                tool_objs.append(tool_obj)
            except NoResultFound:
                warnings.warn(
                    f"Tried to retrieve a tool with name {name} from the agent_state, but does not exist in tool db."
                )
        
        # Create agent based on type
        if agent_state.agent_type == AgentType.memgpt_agent:
            agent = Agent(agent_state=agent_state,
                        interface=interface,
                        tools=tool_objs)
            agent.set_server(self)
        elif agent_state.agent_type == AgentType.o1_agent:
            agent = O1Agent(agent_state=agent_state,
                          interface=interface,
                          tools=tool_objs)
        else:
            raise NotImplementedError(f"Not a supported agent type: {agent_state.agent_type}")
        
        # Add to in-memory store
        await self._add_agent(user_id=user_id, agent_id=agent_state.id, agent_obj=agent)
        
        return agent