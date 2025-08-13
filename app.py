import streamlit as st
import json
import httpx
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime

st.set_page_config(
    page_title="MCP Inspector - RBAC Testing",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 MCP Inspector - RBAC Authentication Testing")

# Initialize session state
if 'test_results' not in st.session_state:
    st.session_state.test_results = []

# Sidebar for configuration
with st.sidebar:
    st.header("MCP Server Configuration")
    
    server_url = st.text_input(
        "MCP Server URL",
        value="http://localhost:8000",
        help="Base URL of your MCP server"
    )
    
    st.subheader("Authentication")
    auth_method = st.selectbox(
        "Authentication Method",
        ["None", "Bearer Token", "Basic Auth"]
    )
    
    auth_token = None
    username = None
    password = None
    
    if auth_method == "Bearer Token":
        auth_token = st.text_input("Bearer Token", type="password")
    elif auth_method == "Basic Auth":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
    
    st.divider()
    
    # Test endpoints
    st.subheader("Test Endpoints")
    test_list_tools = st.checkbox("List Tools", value=True)
    test_call_tool = st.checkbox("Call Tool", value=True)
    test_list_prompts = st.checkbox("List Prompts", value=True)
    test_list_resources = st.checkbox("List Resources", value=True)

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Test Controls")
    
    if test_call_tool:
        tool_name = st.text_input("Tool Name", value="", help="Name of the tool to call")
        tool_args_str = st.text_area(
            "Tool Arguments (JSON)", 
            value='{}',
            help="Arguments to pass to the tool in JSON format"
        )
    
    if st.button("🚀 Run Tests", type="primary"):
        # Prepare headers
        headers = {"Content-Type": "application/json"}
        
        if auth_method == "Bearer Token" and auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        
        # Clear previous results
        st.session_state.test_results = []
        
        async def run_tests():
            async with httpx.AsyncClient() as client:
                # Test list tools
                if test_list_tools:
                    try:
                        response = await client.post(
                            f"{server_url}/mcp/v1/list_tools",
                            headers=headers,
                            auth=(username, password) if auth_method == "Basic Auth" else None,
                            json={}
                        )
                        st.session_state.test_results.append({
                            "endpoint": "List Tools",
                            "status": response.status_code,
                            "response": response.json() if response.status_code == 200 else response.text,
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
                        response = await client.post(
                            f"{server_url}/mcp/v1/call_tool",
                            headers=headers,
                            auth=(username, password) if auth_method == "Basic Auth" else None,
                            json={
                                "params": {
                                    "name": tool_name,
                                    "arguments": tool_args
                                }
                            }
                        )
                        st.session_state.test_results.append({
                            "endpoint": f"Call Tool: {tool_name}",
                            "status": response.status_code,
                            "response": response.json() if response.status_code == 200 else response.text,
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
                        response = await client.post(
                            f"{server_url}/mcp/v1/list_prompts",
                            headers=headers,
                            auth=(username, password) if auth_method == "Basic Auth" else None,
                            json={}
                        )
                        st.session_state.test_results.append({
                            "endpoint": "List Prompts",
                            "status": response.status_code,
                            "response": response.json() if response.status_code == 200 else response.text,
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
                        response = await client.post(
                            f"{server_url}/mcp/v1/list_resources",
                            headers=headers,
                            auth=(username, password) if auth_method == "Basic Auth" else None,
                            json={}
                        )
                        st.session_state.test_results.append({
                            "endpoint": "List Resources",
                            "status": response.status_code,
                            "response": response.json() if response.status_code == 200 else response.text,
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception as e:
                        st.session_state.test_results.append({
                            "endpoint": "List Resources",
                            "status": "Error",
                            "response": str(e),
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
                
                if isinstance(result['status'], int):
                    if result['status'] == 200:
                        st.success(f"Status: {result['status']} ✅")
                    elif result['status'] == 401:
                        st.error(f"Status: {result['status']} - Unauthorized ❌")
                    elif result['status'] == 403:
                        st.error(f"Status: {result['status']} - Forbidden ❌")
                    else:
                        st.warning(f"Status: {result['status']} ⚠️")
                else:
                    st.error(f"Status: {result['status']} ❌")
                
                st.subheader("Response:")
                if isinstance(result['response'], dict):
                    st.json(result['response'])
                else:
                    st.code(result['response'])
    else:
        st.info("No test results yet. Configure your server and run tests to see results.")

# Footer
st.divider()
st.markdown("### How to Use")
st.markdown("""
1. Configure your MCP server URL in the sidebar
2. Set up authentication (Bearer token or Basic auth)
3. Select which endpoints to test
4. Click 'Run Tests' to execute the tests
5. Review the results to ensure RBAC is working correctly
""")

st.markdown("### Expected RBAC Behavior")
st.markdown("""
- **Without authentication**: Should return 401 Unauthorized
- **With invalid token**: Should return 401 Unauthorized
- **With valid token but insufficient permissions**: Should return 403 Forbidden
- **With valid token and proper permissions**: Should return 200 OK with data
""")