import asyncio
import threading
import json

import letta.server.ws_api.protocol as protocol
from letta.interface import AgentInterface


class BaseWebSocketInterface(AgentInterface):
    """Interface for interacting with a Letta agent over a WebSocket"""

    def __init__(self):
        self.clients = set()

    def register_client(self, websocket):
        """Register a new client connection"""
        self.clients.add(websocket)

    def unregister_client(self, websocket):
        """Unregister a client connection"""
        self.clients.remove(websocket)

    def step_yield(self):
        pass


class AsyncWebSocketInterface(BaseWebSocketInterface):
    """Async WebSocket interface implementation"""
    
    def __init__(self):
        super().__init__()
        self.clients = set()
        self._running = True
        
    async def start(self):
        """Start the WebSocket interface"""
        self._running = True
        
    async def stop(self):
        """Stop the WebSocket interface"""
        self._running = False
        # Close all client connections
        if self.clients:
            await asyncio.gather(*(client.close() for client in self.clients))
        self.clients.clear()

    async def add_client(self, websocket):
        """Add a new client connection"""
        self.clients.add(websocket)
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except Exception as e:
            print(f"Error handling client messages: {e}")
        finally:
            self.clients.remove(websocket)
            await websocket.close()

    async def handle_message(self, websocket, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'user_message':
                await self.handle_user_message(websocket, data)
            elif msg_type == 'system_message':
                await self.handle_system_message(websocket, data)
            else:
                await websocket.send_json({
                    'type': 'error',
                    'message': f'Unknown message type: {msg_type}'
                })
        except json.JSONDecodeError:
            await websocket.send_json({
                'type': 'error',
                'message': 'Invalid JSON message'
            })
        except Exception as e:
            await websocket.send_json({
                'type': 'error',
                'message': f'Error processing message: {str(e)}'
            })

    async def handle_user_message(self, websocket, data):
        """Handle user messages"""
        # Process user message and potentially trigger agent response
        response = await self.process_user_message(data)
        await websocket.send_json(response)

    async def handle_system_message(self, websocket, data):
        """Handle system messages"""
        # Process system message
        response = await self.process_system_message(data)
        await websocket.send_json(response)

    async def broadcast(self, message):
        """Broadcast a message to all connected clients"""
        if self.clients:
            await asyncio.gather(*(
                client.send_json(message) for client in self.clients
            ))

    async def process_user_message(self, data):
        """Process a user message and return a response"""
        raise NotImplementedError

    async def process_system_message(self, data):
        """Process a system message and return a response"""
        raise NotImplementedError


class SyncWebSocketInterface(BaseWebSocketInterface):
    def __init__(self):
        super().__init__()
        self.clients = set()
        self.loop = asyncio.new_event_loop()  # Create a new event loop
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()

    def _run_event_loop(self):
        """Run the dedicated event loop and handle its closure."""
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            # Run the cleanup tasks in the event loop
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()

    def _run_async(self, coroutine):
        """Schedule coroutine to be run in the dedicated event loop."""
        if not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    async def _send_to_all_clients(self, clients, msg):
        """Asynchronously sends a message to all clients."""
        if clients:
            await asyncio.gather(*(client.send_text(msg) for client in clients))

    def user_message(self, msg):
        """Handle reception of a user message"""
        # Logic to process the user message and possibly trigger agent's response

    def internal_monologue(self, msg):
        """Handle the agent's internal monologue"""
        print(msg)
        if self.clients:
            self._run_async(self._send_to_all_clients(self.clients, protocol.server_agent_internal_monologue(msg)))

    def assistant_message(self, msg):
        """Handle the agent sending a message"""
        print(msg)
        if self.clients:
            self._run_async(self._send_to_all_clients(self.clients, protocol.server_agent_assistant_message(msg)))

    def function_message(self, msg):
        """Handle the agent calling a function"""
        print(msg)
        if self.clients:
            self._run_async(self._send_to_all_clients(self.clients, protocol.server_agent_function_message(msg)))

    def close(self):
        """Shut down the WebSocket interface and its event loop."""
        self.loop.call_soon_threadsafe(self.loop.stop)  # Signal the loop to stop
        self.thread.join()  # Wait for the thread to finish
