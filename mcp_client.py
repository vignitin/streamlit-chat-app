"""
MCP Client for Streamlit Chat App
Handles communication with MCP servers using proper protocol
"""

import httpx
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import asyncio
from datetime import datetime

@dataclass
class MCPServer:
    """Configuration for an MCP server"""
    name: str
    url: str
    transport: str = "http"
    auth_type: str = "none"  # none, bearer, session
    session_token: Optional[str] = None
    capabilities: Optional[Dict] = None
    
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
        async with httpx.AsyncClient() as client:
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
    
    async def login(self, username: str, password: str, apstra_server: str, apstra_port: str = "443") -> str:
        """Login to get session token (for session-based auth)"""
        async with httpx.AsyncClient() as client:
            # Call the login tool directly
            response = await client.post(
                f"{self.server_url}/tools/login",
                json={
                    "apstra_username": username,
                    "apstra_password": password,
                    "apstra_server": apstra_server,
                    "apstra_port": apstra_port
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
            
        async with httpx.AsyncClient() as client:
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
        async with httpx.AsyncClient() as client:
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
        # Add session token to arguments if we have one
        if self.session_token:
            arguments["session_token"] = self.session_token
            
        async with httpx.AsyncClient() as client:
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
                return response.json()
            else:
                raise Exception(f"Tool call failed: {response.status_code} - {response.text}")
    
    async def list_prompts(self) -> List[Dict[str, Any]]:
        """List available prompts from MCP server"""
        async with httpx.AsyncClient() as client:
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
        async with httpx.AsyncClient() as client:
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

async def test_mcp_server(server_url: str) -> Dict[str, Any]:
    """Test an MCP server connection"""
    client = MCPClient(server_url)
    return await client.test_connection()