"""Example usage of the Letta workflow system."""

import asyncio
from pathlib import Path
from letta.services.workflow import (
    WorkflowCoordinator,
    FileOperations,
    WorkflowMemory
)
from letta.services.workflow.factory import WorkflowAgentFactory

async def main():
    # Set up workspace
    workspace = Path("workspace")
    workspace.mkdir(exist_ok=True)
    
    # Initialize factory and coordinator
    factory = WorkflowAgentFactory(str(workspace))
    file_ops = FileOperations(str(workspace))
    coordinator = WorkflowCoordinator(file_ops)
    
    # Create agents
    architect = factory.create_agent(
        role='architect',
        name='architect_agent',
        persona='You are a software architect focused on creating clean, maintainable architectures.',
        human='Project stakeholder',
        context={'project_type': 'web_application'}
    )
    
    user_story_writer = factory.create_agent(
        role='user_story',
        name='story_writer_agent',
        persona='You are a user story writer focused on capturing user needs and requirements.',
        human='Project stakeholder',
        context={'target_users': ['developers', 'end_users']}
    )
    
    code_writer = factory.create_agent(
        role='code_writer',
        name='code_writer_agent',
        persona='You are a code writer focused on implementing clean, efficient code.',
        human='Project stakeholder',
        context={'languages': ['python', 'typescript']}
    )
    
    # Register agents with coordinator
    await coordinator.register_agent(architect.memory)
    await coordinator.register_agent(user_story_writer.memory)
    await coordinator.register_agent(code_writer.memory)
    
    # Add tasks
    arch_task = await coordinator.add_task(
        role='architect',
        description='Create initial architecture for a web-based task management system',
        task_id='arch_001'
    )
    
    story_task = await coordinator.add_task(
        role='user_story',
        description='Create user stories for the task management system',
        task_id='story_001',
        dependencies=[arch_task.id]
    )
    
    code_task = await coordinator.add_task(
        role='code_writer',
        description='Implement the core task management functionality',
        task_id='code_001',
        dependencies=[story_task.id]
    )
    
    # Run workflow
    await coordinator.run_workflow()
    
    # Get results
    status = await coordinator.get_workflow_status()
    print("\nWorkflow Status:")
    print(status)
    
    # Save final states
    await factory.save_agent_states()

if __name__ == "__main__":
    asyncio.run(main()) 