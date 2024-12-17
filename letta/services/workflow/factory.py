"""Workflow agent factory module."""

from typing import Dict, Any, Optional, Type
from pathlib import Path
from .agents import (
    WorkflowAgent,
    ArchitectAgent,
    UserStoryAgent,
    CodeWriterAgent,
    TestWriterAgent,
    QAAgent,
    SummarizerAgent
)
from .memory import WorkflowMemory
from .file_ops import FileOperations
from letta.schemas.llm_config import LLMConfig
from letta.schemas.embedding_config import EmbeddingConfig

class WorkflowAgentFactory:
    """Factory for creating workflow agents."""
    
    AGENT_TYPES = {
        'architect': ArchitectAgent,
        'user_story': UserStoryAgent,
        'code_writer': CodeWriterAgent,
        'test_writer': TestWriterAgent,
        'qa': QAAgent,
        'summarizer': SummarizerAgent
    }
    
    def __init__(self, workspace_path: str):
        """Initialize the factory.
        
        Args:
            workspace_path: Path to workspace directory
        """
        self.workspace_path = Path(workspace_path)
        self.file_ops = FileOperations(str(self.workspace_path))
        self.agents: Dict[str, WorkflowAgent] = {}
        
    def create_agent(self,
                    role: str,
                    name: str,
                    persona: str,
                    human: str,
                    context: Optional[Dict[str, Any]] = None,
                    llm_config: Optional[LLMConfig] = None,
                    embedding_config: Optional[EmbeddingConfig] = None) -> WorkflowAgent:
        """Create a new workflow agent.
        
        Args:
            role: Agent role (must be one of AGENT_TYPES keys)
            name: Agent name
            persona: Agent persona description
            human: Human context
            context: Optional additional context
            llm_config: Optional LLM configuration
            embedding_config: Optional embedding configuration
            
        Returns:
            WorkflowAgent: Created agent instance
            
        Raises:
            ValueError: If role is not recognized
        """
        if role not in self.AGENT_TYPES:
            raise ValueError(f"Unknown agent role: {role}")
            
        # Create agent memory
        memory = WorkflowMemory(
            role=role,
            persona=persona,
            human=human,
            file_ops=self.file_ops,
            context=context
        )
        
        # Create agent instance
        agent_class = self.AGENT_TYPES[role]
        agent = agent_class(
            name=name,
            role=role,
            memory=memory,
            llm_config=llm_config,
            embedding_config=embedding_config
        )
        
        self.agents[role] = agent
        return agent
        
    def get_agent(self, role: str) -> Optional[WorkflowAgent]:
        """Get an existing agent by role.
        
        Args:
            role: Agent role
            
        Returns:
            Optional[WorkflowAgent]: Agent instance if found
        """
        return self.agents.get(role)
        
    def list_agents(self) -> Dict[str, str]:
        """List all created agents.
        
        Returns:
            Dict[str, str]: Mapping of role to agent name
        """
        return {role: agent.name for role, agent in self.agents.items()}
        
    async def save_agent_states(self) -> None:
        """Save states for all agents."""
        for agent in self.agents.values():
            await agent.memory.save_state()
            
    async def load_agent_states(self) -> None:
        """Load states for all agents."""
        for agent in self.agents.values():
            await agent.memory.load_state()
            
    def get_agent_class(self, role: str) -> Optional[Type[WorkflowAgent]]:
        """Get the agent class for a role.
        
        Args:
            role: Agent role
            
        Returns:
            Optional[Type[WorkflowAgent]]: Agent class if role is recognized
        """
        return self.AGENT_TYPES.get(role) 