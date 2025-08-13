import streamlit as st
import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from mcp_client import MCPClient, create_mcp_client

st.set_page_config(
    page_title="MCP Inspector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 MCP Inspector - Model Context Protocol Testing")

# Initialize session state
if 'test_results' not in st.session_state:
    st.session_state.test_results = []
if 'mcp_client' not in st.session_state:
    st.session_state.mcp_client = None
if 'session_token' not in st.session_state:
    st.session_state.session_token = None

# Sidebar for configuration
with st.sidebar:
    st.header("MCP Server Configuration")
    
    server_url = st.text_input(
        "MCP Server URL",
        value="http://host.docker.internal:8080",
        help="Base URL of your MCP server. Use host.docker.internal for Docker containers."
    )
    
    st.subheader("Authentication")
    auth_method = st.selectbox(
        "Authentication Method",
        ["Session-based (MCP)", "None"]
    )
    
    # MCP Session-based auth
    if auth_method == "Session-based (MCP)":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        auth_server = st.text_input("Authentication Server", value="server.company.com")
        auth_port = st.text_input("Authentication Port", value="443")
        
        # Login/Logout controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔐 Login", disabled=not (username and password and auth_server)):
                async def login():
                    try:
                        client = create_mcp_client(server_url)
                        session_token = await client.login(username, password, auth_server, auth_port)
                        st.session_state.mcp_client = client
                        st.session_state.session_token = session_token
                        st.success("✅ Login successful!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Login failed: {str(e)}")
                
                asyncio.run(login())
        
        with col2:
            if st.button("🔓 Logout", disabled=not st.session_state.session_token):
                async def logout():
                    if st.session_state.mcp_client:
                        await st.session_state.mcp_client.logout()
                    st.session_state.mcp_client = None
                    st.session_state.session_token = None
                    st.success("Logged out successfully!")
                    st.rerun()
                
                asyncio.run(logout())
        
        # Show login status
        if st.session_state.session_token:
            st.success(f"🟢 Logged in as: {username}")
        else:
            st.warning("🔴 Not logged in")
    
    st.divider()
    
    # Test endpoints
    st.subheader("Test Endpoints")
    test_connection = st.checkbox("Test Connection", value=True)
    test_list_tools = st.checkbox("List Tools", value=True)
    test_call_tool = st.checkbox("Call Tool", value=True)
    test_list_prompts = st.checkbox("List Prompts", value=True)
    test_list_resources = st.checkbox("List Resources", value=True)

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Test Controls")
    
    if test_call_tool:
        tool_name = st.text_input("Tool Name", value="get_bp", help="Name of the tool to call")
        tool_args_str = st.text_area(
            "Tool Arguments (JSON)", 
            value='{}',
            help="Arguments to pass to the tool in JSON format. Use {} for no arguments."
        )
    
    if st.button("🚀 Run Tests", type="primary"):
        # Clear previous results
        st.session_state.test_results = []
        
        async def run_tests():
            try:
                # Create or use existing MCP client
                if st.session_state.mcp_client:
                    client = st.session_state.mcp_client
                else:
                    client = create_mcp_client(server_url)
                
                # Test connection
                if test_connection:
                    try:
                        result = await client.test_connection()
                        st.session_state.test_results.append({
                            "endpoint": "Test Connection",
                            "status": "Success" if result["status"] == "success" else "Error",
                            "response": result,
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception as e:
                        st.session_state.test_results.append({
                            "endpoint": "Test Connection",
                            "status": "Error",
                            "response": str(e),
                            "timestamp": datetime.now().isoformat()
                        })
                
                # Test list tools
                if test_list_tools:
                    try:
                        tools = await client.list_tools()
                        st.session_state.test_results.append({
                            "endpoint": "List Tools",
                            "status": "Success",
                            "response": {"tools": tools, "count": len(tools)},
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception as e:
                        st.session_state.test_results.append({
                            "endpoint": "List Tools",
                            "status": "Error",
                            "response": str(e),
                            "timestamp": datetime.now().isoformat()
                        })
                
                # Test call tool
                if test_call_tool and tool_name:
                    try:
                        tool_args = json.loads(tool_args_str)
                        result = await client.call_tool(tool_name, tool_args)
                        
                        # Extract text content from MCP response format
                        response_content = result
                        if isinstance(result, dict) and 'content' in result:
                            # MCP format: {"content": [{"type": "text", "text": "..."}]}
                            content_items = result.get('content', [])
                            if content_items and isinstance(content_items, list):
                                text_contents = [item.get('text', '') for item in content_items if item.get('type') == 'text']
                                response_content = '\n'.join(text_contents) if text_contents else result
                        
                        st.session_state.test_results.append({
                            "endpoint": f"Call Tool: {tool_name}",
                            "status": "Success",
                            "response": response_content,
                            "timestamp": datetime.now().isoformat()
                        })
                    except json.JSONDecodeError as e:
                        st.session_state.test_results.append({
                            "endpoint": f"Call Tool: {tool_name}",
                            "status": "Error",
                            "response": f"JSON parsing error: {str(e)}",
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception as e:
                        st.session_state.test_results.append({
                            "endpoint": f"Call Tool: {tool_name}",
                            "status": "Error",
                            "response": str(e),
                            "timestamp": datetime.now().isoformat()
                        })
                
                # Test list prompts
                if test_list_prompts:
                    try:
                        prompts = await client.list_prompts()
                        st.session_state.test_results.append({
                            "endpoint": "List Prompts",
                            "status": "Success",
                            "response": {"prompts": prompts, "count": len(prompts)},
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception as e:
                        st.session_state.test_results.append({
                            "endpoint": "List Prompts",
                            "status": "Error",
                            "response": str(e),
                            "timestamp": datetime.now().isoformat()
                        })
                
                # Test list resources
                if test_list_resources:
                    try:
                        resources = await client.list_resources()
                        st.session_state.test_results.append({
                            "endpoint": "List Resources",
                            "status": "Success",
                            "response": {"resources": resources, "count": len(resources)},
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception as e:
                        st.session_state.test_results.append({
                            "endpoint": "List Resources",
                            "status": "Error",
                            "response": str(e),
                            "timestamp": datetime.now().isoformat()
                        })
            
            except Exception as e:
                st.session_state.test_results.append({
                    "endpoint": "General Error",
                    "status": "Error", 
                    "response": f"Test execution failed: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
        
        # Run async tests
        asyncio.run(run_tests())
        st.rerun()

with col2:
    st.header("Test Results")
    
    if st.session_state.test_results:
        for result in st.session_state.test_results:
            with st.expander(f"{result['endpoint']} - Status: {result['status']}", expanded=True):
                st.text(f"Timestamp: {result['timestamp']}")
                
                if result['status'] == "Success":
                    st.success(f"Status: {result['status']} ✅")
                elif result['status'] == "Error":
                    st.error(f"Status: {result['status']} ❌")
                else:
                    st.warning(f"Status: {result['status']} ⚠️")
                
                st.subheader("Response:")
                try:
                    if isinstance(result['response'], dict):
                        st.json(result['response'])
                    elif isinstance(result['response'], str):
                        # Try to parse if it's a JSON string
                        try:
                            parsed = json.loads(result['response'])
                            st.json(parsed)
                        except:
                            st.code(result['response'])
                    else:
                        st.code(str(result['response']))
                except Exception as e:
                    st.error(f"Error displaying response: {e}")
                    st.code(str(result['response']))
    else:
        st.info("No test results yet. Configure your server and run tests to see results.")

# Footer
st.divider()
st.markdown("### How to Use")
st.markdown("""
1. **Start the MCP Server**: Start your MCP server (e.g., `python mcp_server.py -t http -H 0.0.0.0 -p 8080`)
2. **Configure Server URL**: Set the MCP server URL (default: http://localhost:8080)
3. **Login**: Use session-based authentication with your server credentials (if required)
4. **Test**: Select endpoints to test and click 'Run Tests'
5. **Review**: Check results to ensure proper MCP protocol behavior
""")

st.markdown("### Expected Authentication Behavior")
st.markdown("""
- **Without login**: Tools should fail with authentication error (if auth required)
- **With invalid credentials**: Login should fail
- **With valid login**: All tools should work with user's identity
- **Session timeout**: Tools should fail after session expires

### MCP Protocol Endpoints Tested:
- **Test Connection**: Validates server availability and protocol
- **List Tools**: Discovers available MCP tools
- **Call Tool**: Executes specific tools with arguments
- **List Prompts**: Gets available prompts (if supported)
- **List Resources**: Gets available resources (if supported)
""")