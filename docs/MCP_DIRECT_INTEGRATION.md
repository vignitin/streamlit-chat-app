# Direct MCP Server Integration (No Separate API Needed)

## Recommendation: Use apstra_mcp.py with HTTP Transport

The apstra-mcp-server already has everything needed for proper RBAC authentication. No separate HTTP API is required.

### Current Capabilities of apstra_mcp.py:

1. **Multiple Transport Support**:
   - stdio: For Claude Desktop (environment variable auth)
   - HTTP: For remote clients with streamable transport
   - SSE: Legacy support for backward compatibility

2. **Built-in RBAC**:
   - Session-based authentication for HTTP/SSE
   - Per-user credentials validation against Apstra
   - Session management with timeout
   - All tools accept session_token parameter

3. **MCP Protocol Compliance**:
   - Tools are properly registered
   - FastMCP handles protocol negotiation
   - Built-in error handling

### How to Use for Streamlit Integration:

```bash
# Start the MCP server with HTTP transport
python apstra_mcp.py -t http -H 0.0.0.0 -p 8080
```

### Required MCP Protocol Endpoints

For proper MCP compliance, the client needs these endpoints:

1. **Initialize** - Protocol handshake
   ```
   POST /mcp/v1/initialize
   ```

2. **List Tools** - Discover available tools
   ```
   POST /mcp/v1/list_tools
   ```

3. **Call Tool** - Execute a specific tool
   ```
   POST /mcp/v1/call_tool
   ```

4. **List Prompts** - Get available prompts (optional)
   ```
   POST /mcp/v1/list_prompts
   ```

5. **List Resources** - Get available resources (optional)
   ```
   POST /mcp/v1/list_resources
   ```

### Implementation Strategy:

1. **Update Streamlit App** to communicate with MCP server:
   - Use standard MCP protocol endpoints
   - Include session token in requests
   - Handle authentication flow

2. **Authentication Flow**:
   ```python
   # 1. Login to get session token
   response = httpx.post(f"{server_url}/tools/login", json={
       "apstra_username": username,
       "apstra_password": password,
       "apstra_server": apstra_server,
       "apstra_port": "443"
   })
   session_token = response.json()["session_token"]
   
   # 2. Use session token in subsequent calls
   response = httpx.post(f"{server_url}/tools/get_bp", json={
       "session_token": session_token
   })
   ```

3. **Benefits**:
   - Single codebase to maintain
   - True RBAC with user identity
   - No duplication of logic
   - Full MCP protocol support
   - Works with any MCP client

### Why Not simple_http_api.py?

1. **Duplication**: It duplicates functionality already in apstra_mcp.py
2. **No RBAC**: Currently has no authentication
3. **Not MCP Compliant**: Missing protocol endpoints
4. **Maintenance Burden**: Two codebases to maintain
5. **Limited Features**: Doesn't support all MCP capabilities

### Next Steps:

1. Remove dependency on simple_http_api.py
2. Update streamlit app to use MCP protocol
3. Add MCP client library or implement protocol
4. Use session-based auth from apstra_mcp.py