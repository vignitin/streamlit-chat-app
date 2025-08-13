# Streamlit Chat App with MCP Support

A streamlit-based chat application with MCP (Model Context Protocol) server integration and multiple LLM support.

## Current Status

### Step 1: MCP Inspector for RBAC Testing ✅
- Simple Streamlit app to test MCP server endpoints
- Supports Bearer token and Basic authentication
- Tests all major MCP endpoints: list_tools, call_tool, list_prompts, list_resources
- Shows response status codes and detailed responses
- Helps verify RBAC authentication is working correctly

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

## Testing RBAC Authentication

1. Start your MCP server (e.g., apstra-mcp-server)
2. Open the MCP Inspector at http://localhost:8501
3. Configure the server URL and authentication method
4. Run tests to verify RBAC is working:
   - Without auth → Should get 401
   - With invalid token → Should get 401
   - With valid token → Should get 200

## Next Steps

- [ ] Review and improve apstra-mcp-server HTTP API for RBAC
- [ ] Build proper chat interface with sidebar
- [ ] Add LLM connectors (OpenAI, Anthropic, Ollama)
- [ ] Implement proper MCP context flow to LLM