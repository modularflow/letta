"""Workflow memory management module."""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from letta.schemas.memory import ChatMemory
from .file_ops import FileOperations

class WorkflowMemory(ChatMemory):
    """Enhanced memory management for workflow agents."""
    
    def __init__(self, 
                 role: str,
                 persona: str,
                 human: str,
                 file_ops: FileOperations,
                 context: Optional[Dict[str, Any]] = None):
        """Initialize workflow memory.
        
        Args:
            role: Agent's role in the workflow
            persona: Agent's persona description
            human: Human context
            file_ops: FileOperations instance for persistence
            context: Optional additional context
        """
        super().__init__(human=human, persona=persona)
        self.role = role
        self.context = context or {}
        self.file_ops = file_ops
        self.state: Dict[str, Any] = {}
        
    async def save_state(self) -> None:
        """Save current agent state to file."""
        state_path = f"states/{self.role}_state.json"
        await self.file_ops.write_file(
            state_path,
            json.dumps({
                'state': self.state,
                'context': self.context,
                'timestamp': datetime.now().isoformat()
            })
        )
        
    async def load_state(self) -> None:
        """Load agent state from file."""
        try:
            state_path = f"states/{self.role}_state.json"
            content = await self.file_ops.read_file(state_path)
            data = json.loads(content)
            self.state = data.get('state', {})
            self.context.update(data.get('context', {}))
        except FileNotFoundError:
            self.state = {}
            
    async def update_state(self, updates: Dict[str, Any]) -> None:
        """Update agent state with new values.
        
        Args:
            updates: Dictionary of state updates
        """
        self.state.update(updates)
        await self.save_state()
        
    async def get_state_history(self) -> List[Dict[str, Any]]:
        """Get history of state changes.
        
        Returns:
            List[Dict[str, Any]]: List of historical states
        """
        state_path = f"states/{self.role}_state.json"
        versions = await self.file_ops.list_versions(state_path)
        
        history = []
        for version in versions:
            try:
                content = await self.file_ops.read_file(version)
                history.append(json.loads(content))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
                
        return history
        
    async def share_memory(self, target_role: str, data: Dict[str, Any]) -> None:
        """Share memory/state with another agent.
        
        Args:
            target_role: Role of the target agent
            data: Data to share
        """
        shared_path = f"shared/{self.role}_{target_role}_shared.json"
        await self.file_ops.write_file(
            shared_path,
            json.dumps({
                'from_role': self.role,
                'to_role': target_role,
                'data': data,
                'timestamp': datetime.now().isoformat()
            })
        )
        
    async def get_shared_memory(self, from_role: str) -> Optional[Dict[str, Any]]:
        """Get memory shared by another agent.
        
        Args:
            from_role: Role of the agent that shared the memory
            
        Returns:
            Optional[Dict[str, Any]]: Shared data if available
        """
        try:
            shared_path = f"shared/{from_role}_{self.role}_shared.json"
            content = await self.file_ops.read_file(shared_path)
            return json.loads(content)
        except FileNotFoundError:
            return None 