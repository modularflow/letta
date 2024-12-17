"""Workflow coordination module."""

import asyncio
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from .memory import WorkflowMemory
from .file_ops import FileOperations

class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class Task:
    """Task data class."""
    id: str
    role: str
    description: str
    dependencies: List[str]
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = datetime.now()
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class WorkflowCoordinator:
    """Coordinates workflow between multiple agents."""
    
    def __init__(self, file_ops: FileOperations):
        """Initialize workflow coordinator.
        
        Args:
            file_ops: FileOperations instance for persistence
        """
        self.file_ops = file_ops
        self.agents: Dict[str, WorkflowMemory] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_queue: asyncio.Queue[Task] = asyncio.Queue()
        self.dependencies: Dict[str, Set[str]] = {}
        self.running_tasks: Set[str] = set()
        
    async def register_agent(self, memory: WorkflowMemory) -> None:
        """Register an agent with the coordinator.
        
        Args:
            memory: Agent's WorkflowMemory instance
        """
        self.agents[memory.role] = memory
        
    async def add_task(self, 
                      role: str,
                      description: str,
                      task_id: str,
                      dependencies: Optional[List[str]] = None) -> Task:
        """Add a new task to the workflow.
        
        Args:
            role: Role of the agent to execute the task
            description: Task description
            task_id: Unique task identifier
            dependencies: Optional list of task IDs this task depends on
            
        Returns:
            Task: Created task instance
            
        Raises:
            ValueError: If role is not registered
        """
        if role not in self.agents:
            raise ValueError(f"Agent role '{role}' not registered")
            
        task = Task(
            id=task_id,
            role=role,
            description=description,
            dependencies=dependencies or []
        )
        
        self.tasks[task_id] = task
        if dependencies:
            self.dependencies[task_id] = set(dependencies)
            if not all(d in self.tasks for d in dependencies):
                task.status = TaskStatus.BLOCKED
            else:
                await self.task_queue.put(task)
        else:
            await self.task_queue.put(task)
            
        return task
        
    async def process_task(self, task: Task) -> None:
        """Process a single task.
        
        Args:
            task: Task to process
        """
        if task.id in self.running_tasks:
            return
            
        # Check dependencies
        if task.dependencies:
            for dep_id in task.dependencies:
                dep_task = self.tasks.get(dep_id)
                if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                    task.status = TaskStatus.BLOCKED
                    return
                    
        self.running_tasks.add(task.id)
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        try:
            # Get agent memory
            agent = self.agents[task.role]
            
            # Load dependencies' results
            dep_results = {}
            for dep_id in task.dependencies:
                dep_task = self.tasks[dep_id]
                if dep_task.result:
                    dep_results[dep_id] = dep_task.result
                    
            # Update agent state with dependency results
            await agent.update_state({'dependency_results': dep_results})
            
            # Execute task (to be implemented by specific agent types)
            result = await agent.execute_task(task.description)
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            # Unblock dependent tasks
            await self.unblock_dependent_tasks(task.id)
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
        finally:
            self.running_tasks.remove(task.id)
            
    async def unblock_dependent_tasks(self, completed_task_id: str) -> None:
        """Unblock tasks that depend on a completed task.
        
        Args:
            completed_task_id: ID of the completed task
        """
        for task_id, deps in self.dependencies.items():
            if completed_task_id in deps:
                deps.remove(completed_task_id)
                if not deps and self.tasks[task_id].status == TaskStatus.BLOCKED:
                    self.tasks[task_id].status = TaskStatus.PENDING
                    await self.task_queue.put(self.tasks[task_id])
                    
    async def run_workflow(self) -> None:
        """Run the workflow until all tasks are completed."""
        while True:
            if self.task_queue.empty() and not self.running_tasks:
                # Check if any tasks are still pending or blocked
                pending_tasks = [t for t in self.tasks.values()
                               if t.status in (TaskStatus.PENDING, TaskStatus.BLOCKED)]
                if not pending_tasks:
                    break
                    
            try:
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                await self.process_task(task)
            except asyncio.TimeoutError:
                continue
                
    async def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status.
        
        Returns:
            Dict[str, Any]: Workflow status information
        """
        return {
            'tasks': {
                task_id: {
                    'status': task.status.value,
                    'role': task.role,
                    'description': task.description,
                    'created_at': task.created_at.isoformat(),
                    'started_at': task.started_at.isoformat() if task.started_at else None,
                    'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                    'error': task.error
                }
                for task_id, task in self.tasks.items()
            },
            'agents': list(self.agents.keys()),
            'running_tasks': list(self.running_tasks)
        } 