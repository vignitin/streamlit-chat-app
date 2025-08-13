# Streamlit Chat App with MCP Support

A streamlit-based chat application with MCP (Model Context Protocol) server integration and multiple LLM support.

## Current Status

### Step 1: MCP Inspector for RBAC Testing ✅
- Streamlit app with proper MCP client implementation
- Connects directly to apstra_mcp.py HTTP transport (no separate API needed)
- Session-based authentication with Apstra credentials
- Tests all MCP protocol endpoints: initialize, list_tools, call_tool, list_prompts, list_resources
- Full MCP protocol compliance with proper RBAC authentication
- Validates that simple_http_api.py is not required

## Quick Start

### Using Docker (Recommended)

```bash
# Build and run with docker-compose
docker-compose up --build

# Access the app at http://localhost:8501
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Testing MCP RBAC Authentication

1. Start the apstra-mcp-server with HTTP transport:
   ```bash
   cd ../apstra-mcp-server
   python apstra_mcp.py -t http -H 0.0.0.0 -p 8080
   ```

2. Open the MCP Inspector at http://localhost:8501

3. Test the authentication flow:
   - Login with your Apstra credentials
   - Run tests to verify proper MCP protocol and RBAC
   - Expected behavior:
     - Without login → Tools fail with auth error
     - With login → Tools work with user's identity
     - Session-based auth maintains state across calls

## Key Findings

✅ **simple_http_api.py is NOT required** - The apstra_mcp.py server already provides:
- Full MCP protocol support with HTTP transport
- Session-based RBAC authentication
- All necessary endpoints for tool discovery and execution
- Proper user identity validation against Apstra

## Next Steps

- [x] ~~Review and improve apstra-mcp-server HTTP API for RBAC~~ (Not needed - use MCP server directly)
- [ ] Build proper chat interface with sidebar
- [ ] Add LLM connectors (OpenAI, Anthropic, Ollama)
- [ ] Implement proper MCP context flow to LLM