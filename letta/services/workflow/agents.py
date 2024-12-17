"""Workflow agent implementations."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from letta.schemas.agent import AgentState
from letta.schemas.llm_config import LLMConfig
from letta.schemas.embedding_config import EmbeddingConfig
from .memory import WorkflowMemory

class WorkflowAgent(AgentState, ABC):
    """Base class for workflow agents."""
    
    def __init__(self,
                 name: str,
                 role: str,
                 memory: WorkflowMemory,
                 llm_config: Optional[LLMConfig] = None,
                 embedding_config: Optional[EmbeddingConfig] = None):
        """Initialize workflow agent.
        
        Args:
            name: Agent name
            role: Agent role
            memory: WorkflowMemory instance
            llm_config: Optional LLM configuration
            embedding_config: Optional embedding configuration
        """
        super().__init__(
            name=name,
            memory=memory,
            llm_config=llm_config or LLMConfig.default_config("sonnet"),
            embedding_config=embedding_config or EmbeddingConfig.default_config("ollama")
        )
        self.role = role
        
    @abstractmethod
    async def execute_task(self, task_description: str) -> Dict[str, Any]:
        """Execute a task.
        
        Args:
            task_description: Description of the task to execute
            
        Returns:
            Dict[str, Any]: Task execution results
        """
        pass

class ArchitectAgent(WorkflowAgent):
    """Agent responsible for system architecture."""
    
    async def execute_task(self, task_description: str) -> Dict[str, Any]:
        """Execute architecture-related tasks.
        
        Args:
            task_description: Task description
            
        Returns:
            Dict[str, Any]: Architecture design and documentation
        """
        # Get any existing architecture from state
        state = await self.memory.get_state()
        existing_arch = state.get('architecture', {})
        
        # Generate or update architecture based on task
        if 'create_architecture' in task_description.lower():
            # Generate new architecture
            arch_result = await self.generate_architecture(task_description)
            await self.memory.update_state({'architecture': arch_result})
            return arch_result
        elif 'update_architecture' in task_description.lower():
            # Update existing architecture
            updates = await self.update_architecture(existing_arch, task_description)
            await self.memory.update_state({'architecture': updates})
            return updates
        else:
            # Handle other architecture tasks
            return await self.handle_architecture_task(task_description)

class UserStoryAgent(WorkflowAgent):
    """Agent responsible for user story creation."""
    
    async def execute_task(self, task_description: str) -> Dict[str, Any]:
        """Execute user story related tasks.
        
        Args:
            task_description: Task description
            
        Returns:
            Dict[str, Any]: User stories and acceptance criteria
        """
        # Get architecture from architect's shared memory
        arch_data = await self.memory.get_shared_memory('architect')
        if not arch_data:
            raise ValueError("Architecture data required but not found")
            
        # Generate user stories based on architecture
        stories = await self.generate_user_stories(arch_data['architecture'])
        
        # Save stories to state
        await self.memory.update_state({'user_stories': stories})
        
        return {'user_stories': stories}

class CodeWriterAgent(WorkflowAgent):
    """Agent responsible for code implementation."""
    
    async def execute_task(self, task_description: str) -> Dict[str, Any]:
        """Execute code writing tasks.
        
        Args:
            task_description: Task description
            
        Returns:
            Dict[str, Any]: Implementation details and file changes
        """
        # Get user stories and pseudo code
        stories = await self.memory.get_shared_memory('user_story_writer')
        pseudo = await self.memory.get_shared_memory('pseudo_writer')
        
        if not stories or not pseudo:
            raise ValueError("User stories and pseudo code required")
            
        # Implement code based on stories and pseudo code
        implementation = await self.implement_code(
            stories['user_stories'],
            pseudo['pseudo_code']
        )
        
        # Save implementation to state
        await self.memory.update_state({'implementation': implementation})
        
        return {'implementation': implementation}

class TestWriterAgent(WorkflowAgent):
    """Agent responsible for test creation."""
    
    async def execute_task(self, task_description: str) -> Dict[str, Any]:
        """Execute test writing tasks.
        
        Args:
            task_description: Task description
            
        Returns:
            Dict[str, Any]: Test cases and implementation
        """
        # Get implementation details
        impl = await self.memory.get_shared_memory('code_writer')
        if not impl:
            raise ValueError("Implementation details required")
            
        # Generate tests based on implementation
        tests = await self.generate_tests(impl['implementation'])
        
        # Save tests to state
        await self.memory.update_state({'tests': tests})
        
        return {'tests': tests}

class QAAgent(WorkflowAgent):
    """Agent responsible for quality assurance."""
    
    async def execute_task(self, task_description: str) -> Dict[str, Any]:
        """Execute QA tasks.
        
        Args:
            task_description: Task description
            
        Returns:
            Dict[str, Any]: QA results and issues
        """
        # Get implementation and tests
        impl = await self.memory.get_shared_memory('code_writer')
        tests = await self.memory.get_shared_memory('test_writer')
        
        if not impl or not tests:
            raise ValueError("Implementation and tests required")
            
        # Perform QA checks
        qa_results = await self.perform_qa(
            impl['implementation'],
            tests['tests']
        )
        
        # Save QA results to state
        await self.memory.update_state({'qa_results': qa_results})
        
        return {'qa_results': qa_results}

class SummarizerAgent(WorkflowAgent):
    """Agent responsible for summarizing work and progress."""
    
    async def execute_task(self, task_description: str) -> Dict[str, Any]:
        """Execute summarization tasks.
        
        Args:
            task_description: Task description
            
        Returns:
            Dict[str, Any]: Summary and recommendations
        """
        # Get all relevant data from other agents
        arch_data = await self.memory.get_shared_memory('architect')
        impl_data = await self.memory.get_shared_memory('code_writer')
        qa_data = await self.memory.get_shared_memory('qa')
        
        # Generate comprehensive summary
        summary = await self.generate_summary(
            architecture=arch_data,
            implementation=impl_data,
            qa_results=qa_data
        )
        
        # Save summary to state
        await self.memory.update_state({'summary': summary})
        
        return {'summary': summary} 