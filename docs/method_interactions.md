# Method Interactions in Letta

This document illustrates the key method interactions and data flows in the Letta system using mermaid diagrams.

## Tool Management Flow

The following diagram shows how tools are created and managed in the system:

```mermaid
sequenceDiagram
    participant Client
    participant Server
    participant ToolManager
    participant ORM
    participant Database

    Client->>Server: create_tool(tool_data)
    Server->>ToolManager: create_or_update_tool(pydantic_tool, actor)
    
    alt Tool exists
        ToolManager->>ORM: get_tool_by_name(name, actor)
        ORM->>Database: SELECT query
        Database-->>ORM: Tool data
        ToolManager->>ToolManager: update_tool_by_id()
    else Tool doesn't exist
        ToolManager->>ORM: create_tool()
        ORM->>Database: INSERT query
    end
    
    ToolManager-->>Server: PydanticTool
    Server-->>Client: Tool response
```

## WebSocket Communication Flow

The following diagram shows the WebSocket-based communication between client and server:

```mermaid
sequenceDiagram
    participant Client
    participant WebSocketServer
    participant WSInterface
    participant Server
    participant AgentManager

    Client->>WebSocketServer: connect()
    WebSocketServer->>WSInterface: register_client(websocket)
    
    Client->>WebSocketServer: send(user_message)
    WebSocketServer->>Server: user_message(user_id, agent_id, message)
    Server->>AgentManager: process_message()
    
    AgentManager-->>Server: response
    Server-->>WSInterface: broadcast_message()
    WSInterface-->>Client: send(response)
```

## Memory Operations Flow

The following diagram shows how memory operations are handled:

```mermaid
sequenceDiagram
    participant Client
    participant Server
    participant MemoryManager
    participant Storage
    participant Database

    Client->>Server: archival_memory_insert(data)
    Server->>MemoryManager: insert(data)
    MemoryManager->>Storage: store_embedding()
    Storage->>Database: save_memory()
    
    Database-->>Storage: confirmation
    Storage-->>MemoryManager: success
    MemoryManager-->>Server: memory_id
    Server-->>Client: success response
```

## Organization and User Management Flow

The following diagram shows organization and user management interactions:

```mermaid
sequenceDiagram
    participant Client
    participant Server
    participant OrgManager
    participant UserManager
    participant Database

    Client->>Server: create_organization(org_data)
    Server->>OrgManager: create_organization()
    OrgManager->>Database: save_organization()
    
    Client->>Server: add_user(user_data)
    Server->>UserManager: create_user()
    UserManager->>OrgManager: verify_organization()
    UserManager->>Database: save_user()
    
    Database-->>Server: confirmation
    Server-->>Client: success response
```

## Agent Workflow System

The following diagram shows the agent workflow system:

```mermaid
sequenceDiagram
    participant Client
    participant WorkflowCoordinator
    participant AgentFactory
    participant Agents
    participant FileOps
    participant Memory

    Client->>WorkflowCoordinator: start_workflow()
    WorkflowCoordinator->>AgentFactory: create_agents()
    
    loop For each task
        WorkflowCoordinator->>Agents: execute_task()
        Agents->>FileOps: read/write_files()
        Agents->>Memory: store_results()
        Memory-->>WorkflowCoordinator: task_completion
    end
    
    WorkflowCoordinator-->>Client: workflow_results
```

## Database Operations Flow

The following diagram shows the core database operations:

```mermaid
sequenceDiagram
    participant Service
    participant ORM
    participant SQLAlchemy
    participant Database

    Service->>ORM: create/read/update/delete
    ORM->>SQLAlchemy: execute_query()
    SQLAlchemy->>Database: SQL query
    
    Database-->>SQLAlchemy: result
    SQLAlchemy-->>ORM: mapped objects
    ORM-->>Service: domain objects
```

## Key Implementation Details

### Tool Management
- Tools are managed through the `ToolManager` class which handles CRUD operations
- Each tool is stored with metadata including name, description, source code, and JSON schema
- Tools are organization-scoped, meaning they belong to specific organizations

### WebSocket Communication
- The `WebSocketServer` handles real-time bidirectional communication
- Messages are processed through the `WebSocketInterface` which maintains client connections
- Supports both synchronous and asynchronous operations through `SyncWebSocketInterface` and `AsyncWebSocketInterface`

### Memory System
- Implements both archival and recall memory types
- Uses embeddings for semantic search capabilities
- Supports async operations for better performance

### Database Layer
- Uses SQLAlchemy for ORM operations
- Implements both sync and async database operations
- Maintains data consistency through transactions

## Best Practices

1. **Error Handling**
   - All database operations should be wrapped in try-except blocks
   - WebSocket connections should handle disconnections gracefully
   - Memory operations should validate data before storage

2. **Performance**
   - Use async operations for I/O-bound tasks
   - Implement connection pooling for database operations
   - Cache frequently accessed data

3. **Security**
   - Validate all user input
   - Implement proper authentication and authorization
   - Use secure WebSocket connections when needed

4. **Scalability**
   - Design services to be stateless when possible
   - Use message queues for long-running operations
   - Implement proper connection management

## Common Patterns

1. **Manager Pattern**
   - Each major component has a manager class (ToolManager, UserManager, etc.)
   - Managers handle business logic and coordinate with the ORM layer

2. **Interface Pattern**
   - WebSocket communication is abstracted through interfaces
   - Allows for different implementation strategies (sync/async)

3. **Factory Pattern**
   - Used in agent creation and workflow management
   - Provides flexibility in object creation

4. **Repository Pattern**
   - Implemented through the ORM layer
   - Provides clean separation of data access logic 