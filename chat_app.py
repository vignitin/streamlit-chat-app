import streamlit as st
import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from mcp_client import MCPClient, create_mcp_client

st.set_page_config(
    page_title="MCP Chat Interface",
    page_icon="💬",
    layout="wide"
)

# Initialize session state for chat
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'mcp_client' not in st.session_state:
    st.session_state.mcp_client = None
if 'session_token' not in st.session_state:
    st.session_state.session_token = None
if 'available_tools' not in st.session_state:
    st.session_state.available_tools = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Chat"

# Sidebar for configuration and navigation
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Page Navigation
    st.subheader("Navigation")
    page = st.selectbox(
        "Select Page",
        ["Chat", "MCP Inspector"],
        index=0 if st.session_state.current_page == "Chat" else 1
    )
    st.session_state.current_page = page
    
    st.divider()
    
    # MCP Server Configuration
    st.subheader("MCP Server")
    server_url = st.text_input(
        "Server URL",
        value="http://host.docker.internal:8080",
        help="MCP server endpoint"
    )
    
    # Authentication
    st.subheader("Authentication")
    auth_method = st.selectbox(
        "Method",
        ["None", "Session-based"]
    )
    
    if auth_method == "Session-based":
        with st.expander("Login Credentials", expanded=True):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            auth_server = st.text_input("Auth Server", value="server.company.com")
            auth_port = st.text_input("Port", value="443")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔐 Login", disabled=not (username and password and auth_server)):
                    async def login():
                        try:
                            client = create_mcp_client(server_url)
                            session_token = await client.login(username, password, auth_server, auth_port)
                            st.session_state.mcp_client = client
                            st.session_state.session_token = session_token
                            
                            # Load available tools
                            tools = await client.list_tools()
                            st.session_state.available_tools = tools
                            
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
                        st.session_state.available_tools = []
                        st.success("Logged out!")
                        st.rerun()
                    
                    asyncio.run(logout())
    
    # Connection Status
    st.divider()
    st.subheader("Status")
    if st.session_state.session_token:
        st.success("🟢 Connected & Authenticated")
        if st.session_state.available_tools:
            st.info(f"🔧 {len(st.session_state.available_tools)} tools available")
    else:
        st.warning("🔴 Not authenticated")
    
    # Available Tools
    if st.session_state.available_tools:
        st.subheader("Available Tools")
        with st.expander("MCP Tools", expanded=False):
            for tool in st.session_state.available_tools:
                st.text(f"• {tool.get('name', 'Unknown')}")

# Main content area
if st.session_state.current_page == "Chat":
    st.title("💬 MCP Chat Interface")
    
    # Chat messages container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for i, message in enumerate(st.session_state.chat_messages):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Show timestamp for each message
                st.caption(f"🕐 {message['timestamp']}")
                
                # Show tool calls if any
                if "tool_calls" in message and message["tool_calls"]:
                    with st.expander("🔧 Tool Calls", expanded=False):
                        for tool_call in message["tool_calls"]:
                            st.json(tool_call)
    
    # Chat input
    st.divider()
    
    # Message input
    user_input = st.chat_input(
        "Type your message here...",
        disabled=not st.session_state.mcp_client,
        key="chat_input"
    )
    
    if user_input and st.session_state.mcp_client:
        # Add user message to chat
        user_message = {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.chat_messages.append(user_message)
        
        # TODO: Process message with LLM and MCP tools
        # For now, just echo back
        assistant_message = {
            "role": "assistant", 
            "content": f"Echo: {user_input}\n\n*Note: LLM integration coming in Step 3*",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.chat_messages.append(assistant_message)
        
        st.rerun()
    
    # Clear chat button
    if st.session_state.chat_messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_messages = []
            st.rerun()

elif st.session_state.current_page == "MCP Inspector":
    # Import and run the inspector functionality
    st.title("🔍 MCP Inspector - Protocol Testing")
    
    # Initialize session state for inspector if needed
    if 'test_results' not in st.session_state:
        st.session_state.test_results = []
    
    # Test endpoints selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Test Controls")
        
        test_connection = st.checkbox("Test Connection", value=True)
        test_list_tools = st.checkbox("List Tools", value=True)
        test_call_tool = st.checkbox("Call Tool", value=True)
        test_list_prompts = st.checkbox("List Prompts", value=False)
        test_list_resources = st.checkbox("List Resources", value=False)
        
        if test_call_tool:
            tool_name = st.text_input("Tool Name", value="get_bp", help="Name of the tool to call")
            tool_args_str = st.text_area(
                "Tool Arguments (JSON)", 
                value='{}',
                help="Arguments to pass to the tool in JSON format"
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
st.markdown("### MCP Chat Interface")
st.markdown("""
**Current Features:**
- 💬 Chat interface with message history
- 🔧 MCP server configuration in sidebar  
- 🔐 Authentication support
- 🔍 MCP Inspector for protocol testing

**Coming Next:**
- 🤖 LLM integration (OpenAI, Anthropic, Ollama)
- 🔄 Automatic MCP tool calling
- 🎯 Context-aware responses
""")