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
        self.mcp_session_id: Optional[str] = None  # For StreamableHTTPSessionManager
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize connection with MCP server"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.server_url}/mcp/v1/initialize",
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "id": "init-1",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "Streamlit MCP Client",
                            "version": "1.0.0"
                        }
                    }
                },
                headers=self._get_headers()
            )
            
            if response.status_code == 200:
                result = response.json()
                # Handle JSON-RPC response format
                if "result" in result:
                    init_result = result["result"]
                    self.server_info = init_result.get("serverInfo", {})
                    return init_result
                else:
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
            # Try standard MCP endpoint first (works with Apstra and most servers)
            try:
                response = await client.post(
                    f"{self.server_url}/mcp/v1/list_tools",
                    json={
                        "jsonrpc": "2.0",
                        "method": "tools/list",
                        "id": "tools-list-1",
                        "params": {}
                    },
                    headers=self._get_headers()
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Handle JSON-RPC response format
                    if "result" in result:
                        tools_data = result["result"]
                        self.tools = tools_data.get("tools", []) if isinstance(tools_data, dict) else tools_data
                    else:
                        self.tools = result.get("tools", [])
                    return self.tools
                elif response.status_code == 400 and "Missing session ID" in response.text:
                    # This indicates a StreamableHTTPSessionManager server (like Junos)
                    return await self._list_tools_streamable_http(client)
                else:
                    raise Exception(f"Failed to list tools: {response.status_code} - {response.text}")
            except Exception as e:
                if "Missing session ID" in str(e):
                    # Fallback to streamable HTTP protocol
                    return await self._list_tools_streamable_http(client)
                raise e
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool with arguments"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Try standard HTTP first, fallback to StreamableHTTP if needed
            try:
                response = await client.post(
                    f"{self.server_url}/mcp/v1/call_tool",
                    json={
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "id": f"tool-call-{tool_name}",
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
                        # Handle JSON-RPC response format
                        if "result" in result:
                            return result["result"]
                        else:
                            return result
                    except json.JSONDecodeError as e:
                        raise Exception(f"Invalid JSON response: {e}")
                elif response.status_code == 400 and "Missing session ID" in response.text:
                    # This is a StreamableHTTP server, use that protocol
                    return await self._call_tool_streamable_http(client, tool_name, arguments)
                else:
                    raise Exception(f"Tool call failed: {response.status_code} - {response.text}")
            except Exception as e:
                if "Missing session ID" in str(e):
                    # Fallback to streamable HTTP protocol
                    return await self._call_tool_streamable_http(client, tool_name, arguments)
                raise e
    
    async def list_prompts(self) -> List[Dict[str, Any]]:
        """List available prompts from MCP server"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.server_url}/mcp/v1/list_prompts",
                json={
                    "jsonrpc": "2.0",
                    "method": "prompts/list",
                    "id": "prompts-list-1",
                    "params": {}
                },
                headers=self._get_headers()
            )
            
            if response.status_code == 200:
                result = response.json()
                # Handle JSON-RPC response format
                if "result" in result:
                    prompts_data = result["result"]
                    return prompts_data.get("prompts", []) if isinstance(prompts_data, dict) else prompts_data
                else:
                    return result.get("prompts", [])
            else:
                # Some servers may not support prompts
                return []
    
    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources from MCP server"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.server_url}/mcp/v1/list_resources",
                json={
                    "jsonrpc": "2.0",
                    "method": "resources/list",
                    "id": "resources-list-1",
                    "params": {}
                },
                headers=self._get_headers()
            )
            
            if response.status_code == 200:
                result = response.json()
                # Handle JSON-RPC response format
                if "result" in result:
                    resources_data = result["result"]
                    return resources_data.get("resources", []) if isinstance(resources_data, dict) else resources_data
                else:
                    return result.get("resources", [])
            else:
                # Some servers may not support resources
                return []
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for requests"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        # Add authentication headers if needed
        if self.session_token:
            headers["Authorization"] = f"Bearer {self.session_token}"
            
        return headers
    
    def _parse_sse_response(self, sse_text: str) -> Dict[str, Any]:
        """Parse Server-Sent Events format response"""
        lines = sse_text.strip().split('\n')
        data_line = None
        
        for line in lines:
            if line.startswith('data: '):
                data_line = line[6:]  # Remove 'data: ' prefix
                break
        
        if data_line:
            return json.loads(data_line)
        else:
            raise Exception(f"No data line found in SSE response: {sse_text}")
    
    async def _list_tools_streamable_http(self, client: httpx.AsyncClient) -> List[Dict[str, Any]]:
        """Handle tools listing for StreamableHTTPSessionManager servers (like Junos)"""
        # StreamableHTTPSessionManager protocol requires proper session initialization
        # Try with trailing slash first (Starlette Mount pattern)
        endpoint = f"{self.server_url}/mcp/"
        
        try:
            # Step 1: Initialize session (if not already done)
            if not self.mcp_session_id:
                init_response = await client.post(
                    endpoint,
                    json={
                        "jsonrpc": "2.0",
                        "method": "initialize",
                        "id": "init-streamable",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {
                                "name": "Streamlit MCP Client",
                                "version": "1.0.0"
                            }
                        }
                    },
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream"
                    },
                    follow_redirects=True  # Handle 307 redirects
                )
                
                if init_response.status_code != 200:
                    raise Exception(f"Initialization failed: {init_response.status_code} - {init_response.text}")
                
                # Extract session ID from response headers
                self.mcp_session_id = init_response.headers.get("Mcp-Session-Id")
                if not self.mcp_session_id:
                    raise Exception("Server did not provide Mcp-Session-Id header during initialization")
                
                # Handle initialization response - StreamableHTTP returns SSE format
                try:
                    if init_response.text.strip():
                        # Parse SSE format: "event: message\ndata: {json}"
                        init_result = self._parse_sse_response(init_response.text)
                        
                        # Check for initialization errors
                        if "error" in init_result:
                            raise Exception(f"Initialization failed: {init_result['error']}")
                        
                        if "result" in init_result:
                            self.server_info = init_result["result"].get("serverInfo", {})
                        else:
                            self.server_info = init_result.get("serverInfo", {})
                            
                        # Send initialized notification to complete handshake
                        notification_response = await client.post(
                            endpoint,
                            json={
                                "jsonrpc": "2.0",
                                "method": "notifications/initialized",
                                "params": {}
                            },
                            headers={
                                "Content-Type": "application/json",
                                "Accept": "application/json, text/event-stream",
                                "Mcp-Session-Id": self.mcp_session_id
                            }
                        )
                        
                        # Wait for initialization to complete
                        import asyncio
                        await asyncio.sleep(2.0)
                    else:
                        # Empty response is OK for streamable HTTP initialization
                        self.server_info = {"name": "StreamableHTTP Server", "version": "unknown"}
                except Exception as json_error:
                    # If JSON parsing fails, log the response but continue
                    raise Exception(f"JSON parse error in init response: {json_error}. Response text: '{init_response.text}'")
            
            # Step 2: List tools using established session
            tools_response = await client.post(
                endpoint,
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "id": "tools-list-streamable",
                    "params": {}
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Mcp-Session-Id": self.mcp_session_id
                }
            )
            
            if tools_response.status_code == 200:
                try:
                    if tools_response.text.strip():
                        # Parse SSE format response
                        result = self._parse_sse_response(tools_response.text)
                        
                        # Check for JSON-RPC errors
                        if "error" in result:
                            error_msg = result["error"].get("message", "Unknown error")
                            if "before initialization" in error_msg or "Invalid request parameters" in error_msg:
                                # Wait longer and retry with same session
                                import asyncio
                                await asyncio.sleep(2.0)  # Longer wait
                                
                                # Retry tools list with same session (don't reset)
                                retry_response = await client.post(
                                    endpoint,
                                    json={
                                        "jsonrpc": "2.0",
                                        "method": "tools/list",
                                        "id": "tools-list-retry",
                                        "params": {}
                                    },
                                    headers={
                                        "Content-Type": "application/json",
                                        "Accept": "application/json, text/event-stream",
                                        "Mcp-Session-Id": self.mcp_session_id
                                    }
                                )
                                
                                if retry_response.status_code == 200:
                                    retry_result = self._parse_sse_response(retry_response.text)
                                    if "result" in retry_result:
                                        tools_data = retry_result["result"]
                                        self.tools = tools_data.get("tools", []) if isinstance(tools_data, dict) else tools_data
                                        return self.tools
                                    
                                # If retry still fails, reset session for next attempt
                                self.mcp_session_id = None
                                raise Exception(f"Server initialization timeout: {result['error']}")
                            else:
                                raise Exception(f"Server error: {result['error']}")
                        
                        if "result" in result:
                            tools_data = result["result"]
                            self.tools = tools_data.get("tools", []) if isinstance(tools_data, dict) else tools_data
                        else:
                            self.tools = result.get("tools", [])
                        
                        return self.tools
                    else:
                        # Empty response - no tools available
                        self.tools = []
                        return self.tools
                except Exception as json_error:
                    raise Exception(f"JSON parse error in tools response: {json_error}. Response text: '{tools_response.text}'")
            elif tools_response.status_code == 404:
                # Session expired, reset and retry once
                self.mcp_session_id = None
                return await self._list_tools_streamable_http(client)
            else:
                raise Exception(f"Tools list failed: {tools_response.status_code} - {tools_response.text}")
                
        except Exception as e:
            # Reset session on any error
            self.mcp_session_id = None
            raise Exception(f"StreamableHTTP protocol error: {str(e)}")
    
    async def _call_tool_streamable_http(self, client: httpx.AsyncClient, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool calls for StreamableHTTPSessionManager servers"""
        endpoint = f"{self.server_url}/mcp/"
        
        # Ensure we have a session ID
        if not self.mcp_session_id:
            raise Exception("No session ID available for StreamableHTTP tool call")
        
        try:
            response = await client.post(
                endpoint,
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "id": f"tool-call-streamable-{tool_name}",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments
                    }
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Mcp-Session-Id": self.mcp_session_id
                }
            )
            
            if response.status_code == 200:
                # Parse SSE format response
                result = self._parse_sse_response(response.text)
                
                # Check for JSON-RPC errors
                if "error" in result:
                    raise Exception(f"Tool call error: {result['error']}")
                
                if "result" in result:
                    return result["result"]
                else:
                    return result
            else:
                raise Exception(f"StreamableHTTP tool call failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"StreamableHTTP tool call error: {str(e)}")
    
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