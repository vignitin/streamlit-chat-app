"""
MCP Client for Streamlit Chat App
Handles communication with MCP servers using proper protocol
"""

import httpx
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import asyncio


def extract_mcp_response_text(response: Dict[str, Any]) -> str:
    """Extract text content from MCP response format
    
    Args:
        response: MCP response dictionary
        
    Returns:
        Extracted text content as string
    """
    if isinstance(response, dict) and 'content' in response:
        content_items = response.get('content', [])
        if content_items and isinstance(content_items, list):
            text_contents = [item.get('text', '') for item in content_items if item.get('type') == 'text']
            return '\n'.join(text_contents) if text_contents else str(response)
    return str(response)


from datetime import datetime

@dataclass
class MCPServer:
    """Configuration for an MCP server"""
    id: str  # Unique identifier
    name: str
    url: str
    transport: str = "http"
    auth_type: str = "none"  # none, bearer, session
    session_token: Optional[str] = None
    capabilities: Optional[Dict] = None
    connected: bool = False
    tool_count: int = 0
    
class MCPClient:
    """Client for communicating with MCP servers"""
    
    def __init__(self, server_url: str, transport: str = "http"):
        self.server_url = server_url.rstrip('/')
        self.transport = transport
        self.session_token: Optional[str] = None
        self.server_info: Optional[Dict] = None
        self.tools: Optional[List[Dict]] = None
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize connection with MCP server"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.server_url}/mcp/v1/initialize",
                json={},
                headers=self._get_headers()
            )
            
            if response.status_code == 200:
                result = response.json()
                self.server_info = result.get("serverInfo", {})
                return result
            else:
                raise Exception(f"Failed to initialize: {response.status_code} - {response.text}")
    
    async def login(self, username: str, password: str, auth_server: str, auth_port: str = "443") -> str:
        """Login to get session token (for session-based auth)"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Call the login tool directly
            response = await client.post(
                f"{self.server_url}/tools/login",
                json={
                    "username": username,
                    "password": password,
                    "server": auth_server,
                    "port": auth_port
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    self.session_token = result.get("session_token")
                    return self.session_token
                else:
                    raise Exception(result.get("message", "Login failed"))
            else:
                raise Exception(f"Login failed: {response.status_code} - {response.text}")
    
    async def logout(self) -> bool:
        """Logout and invalidate session"""
        if not self.session_token:
            return True
            
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.server_url}/tools/logout",
                json={"session_token": self.session_token}
            )
            
            if response.status_code == 200:
                self.session_token = None
                return True
            return False
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from MCP server"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.server_url}/mcp/v1/list_tools",
                json={},
                headers=self._get_headers()
            )
            
            if response.status_code == 200:
                result = response.json()
                self.tools = result.get("tools", [])
                return self.tools
            else:
                raise Exception(f"Failed to list tools: {response.status_code} - {response.text}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool with arguments"""
        # Authentication is handled via Authorization header, not arguments
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.server_url}/mcp/v1/call_tool",
                json={
                    "params": {
                        "name": tool_name,
                        "arguments": arguments
                    }
                },
                headers=self._get_headers()
            )
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    return result
                except json.JSONDecodeError as e:
                    raise Exception(f"Invalid JSON response: {e}")
            else:
                raise Exception(f"Tool call failed: {response.status_code} - {response.text}")
    
    async def list_prompts(self) -> List[Dict[str, Any]]:
        """List available prompts from MCP server"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.server_url}/mcp/v1/list_prompts",
                json={},
                headers=self._get_headers()
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("prompts", [])
            else:
                # Some servers may not support prompts
                return []
    
    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources from MCP server"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.server_url}/mcp/v1/list_resources",
                json={},
                headers=self._get_headers()
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("resources", [])
            else:
                # Some servers may not support resources
                return []
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for requests"""
        headers = {"Content-Type": "application/json"}
        
        # Add authentication headers if needed
        if self.session_token:
            headers["Authorization"] = f"Bearer {self.session_token}"
            
        return headers
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to MCP server"""
        try:
            # Try to initialize
            init_result = await self.initialize()
            
            # Try to list tools
            tools = await self.list_tools()
            
            return {
                "status": "success",
                "server_info": self.server_info,
                "tool_count": len(tools) if tools else 0,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Utility functions for Streamlit
def create_mcp_client(server_url: str, transport: str = "http") -> MCPClient:
    """Create an MCP client instance"""
    return MCPClient(server_url, transport)


class MCPServerManager:
    """Manages multiple MCP server connections"""
    
    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}  # server_id -> MCPServer config
        self.clients: Dict[str, MCPClient] = {}  # server_id -> MCPClient
        self.tool_mapping: Dict[str, str] = {}  # full_tool_name -> server_id
        self.server_tools: Dict[str, List[Dict]] = {}  # server_id -> tools list
    
    def generate_server_id(self) -> str:
        """Generate unique server ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    async def add_server(self, name: str, url: str, auth_type: str = "none") -> str:
        """Add a new MCP server (without connecting)"""
        server_id = self.generate_server_id()
        server = MCPServer(
            id=server_id,
            name=name,
            url=url,
            transport="http",
            auth_type=auth_type,
            connected=False,
            tool_count=0
        )
        self.servers[server_id] = server
        return server_id
    
    async def connect_server(self, server_id: str, username: str = None, password: str = None, 
                           auth_server: str = None, auth_port: str = None) -> bool:
        """Connect to a specific MCP server"""
        if server_id not in self.servers:
            raise ValueError(f"Server {server_id} not found")
        
        server = self.servers[server_id]
        client = create_mcp_client(server.url)
        
        try:
            # Handle authentication if needed
            if server.auth_type == "session" and username and password:
                session_token = await client.login(username, password, auth_server, auth_port)
                server.session_token = session_token
            
            # Test connection and get tools
            tools = await client.list_tools()
            
            # Store client and update server info
            self.clients[server_id] = client
            server.connected = True
            server.tool_count = len(tools)
            
            # Store tools and update mapping
            self.server_tools[server_id] = tools
            self._update_tool_mapping(server_id, tools)
            
            return True
            
        except Exception as e:
            # Clean up on failure
            if server_id in self.clients:
                del self.clients[server_id]
            server.connected = False
            server.tool_count = 0
            raise e
    
    async def disconnect_server(self, server_id: str):
        """Disconnect from a specific server"""
        if server_id not in self.servers:
            return
        
        # Logout if authenticated
        if server_id in self.clients:
            client = self.clients[server_id]
            server = self.servers[server_id]
            if server.session_token:
                try:
                    await client.logout()
                except:
                    pass  # Ignore logout errors
            
            # Remove client
            del self.clients[server_id]
        
        # Update server status
        self.servers[server_id].connected = False
        self.servers[server_id].session_token = None
        self.servers[server_id].tool_count = 0
        
        # Remove tools from mapping
        self._remove_from_tool_mapping(server_id)
        if server_id in self.server_tools:
            del self.server_tools[server_id]
    
    async def remove_server(self, server_id: str):
        """Remove a server completely"""
        # Disconnect first
        await self.disconnect_server(server_id)
        
        # Remove server config
        if server_id in self.servers:
            del self.servers[server_id]
    
    async def reauthenticate_server(self, server_id: str, username: str, password: str, 
                                   auth_server: str, auth_port: str = "443") -> bool:
        """Re-authenticate an existing server connection"""
        if server_id not in self.servers:
            raise ValueError(f"Server {server_id} not found")
        
        server = self.servers[server_id]
        if server.auth_type != "session":
            raise ValueError(f"Server {server.name} does not use session authentication")
        
        # Get existing client or create new one
        if server_id in self.clients:
            client = self.clients[server_id]
        else:
            client = create_mcp_client(server.url)
            self.clients[server_id] = client
        
        try:
            # Logout from existing session if any
            if server.session_token:
                try:
                    await client.logout()
                except:
                    pass  # Ignore logout errors
            
            # Authenticate with new credentials
            session_token = await client.login(username, password, auth_server, auth_port)
            server.session_token = session_token
            
            # Refresh tools list
            tools = await client.list_tools()
            server.connected = True
            server.tool_count = len(tools)
            
            # Update tool mappings
            self.server_tools[server_id] = tools
            self._update_tool_mapping(server_id, tools)
            
            return True
            
        except Exception as e:
            # Mark as disconnected on failure but keep the server config
            server.connected = False
            server.session_token = None
            server.tool_count = 0
            raise e
    
    def _update_tool_mapping(self, server_id: str, tools: List[Dict]):
        """Update tool name to server mapping"""
        # Remove old mappings for this server
        self._remove_from_tool_mapping(server_id)
        
        # Add new mappings with server prefix to handle conflicts
        server = self.servers[server_id]
        for tool in tools:
            tool_name = tool.get('name', '')
            # Use server name prefix for uniqueness (underscore for API compatibility)
            full_tool_name = f"{server.name}_{tool_name}"
            self.tool_mapping[full_tool_name] = server_id
    
    def _remove_from_tool_mapping(self, server_id: str):
        """Remove all tool mappings for a server"""
        to_remove = [name for name, sid in self.tool_mapping.items() if sid == server_id]
        for name in to_remove:
            del self.tool_mapping[name]
    
    async def get_all_tools(self) -> List[Dict]:
        """Get aggregated tools from all connected servers"""
        all_tools = []
        
        for server_id, tools in self.server_tools.items():
            if server_id in self.servers and self.servers[server_id].connected:
                server = self.servers[server_id]
                # Add server info to each tool
                for tool in tools:
                    enriched_tool = tool.copy()
                    enriched_tool['server_id'] = server_id
                    enriched_tool['server_name'] = server.name
                    # Prefix tool name with server name (using underscore for API compatibility)
                    enriched_tool['full_name'] = f"{server.name}_{tool.get('name', '')}"
                    enriched_tool['original_name'] = tool.get('name', '')
                    all_tools.append(enriched_tool)
        
        return all_tools
    
    async def call_tool(self, full_tool_name: str, arguments: Dict) -> Any:
        """Route tool call to appropriate server"""
        # Find which server has this tool
        server_id = self.tool_mapping.get(full_tool_name)
        if not server_id or server_id not in self.clients:
            raise ValueError(f"Tool {full_tool_name} not found or server not connected")
        
        # Extract original tool name (remove server prefix)
        original_name = full_tool_name.split('_', 1)[1] if '_' in full_tool_name else full_tool_name
        
        # Call tool on appropriate client
        client = self.clients[server_id]
        return await client.call_tool(original_name, arguments)
    
    def get_connected_servers(self) -> List[MCPServer]:
        """Get list of connected servers"""
        return [server for server in self.servers.values() if server.connected]
    
    def get_all_servers(self) -> List[MCPServer]:
        """Get list of all servers"""
        return list(self.servers.values())

async def test_mcp_server(server_url: str) -> Dict[str, Any]:
    """Test an MCP server connection"""
    client = MCPClient(server_url)
    return await client.test_connection()