import streamlit as st
import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from mcp_client import MCPClient, create_mcp_client, extract_mcp_response_text
from llm_connector import create_llm_connector

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
if 'llm_connector' not in st.session_state:
    st.session_state.llm_connector = None
if 'processing_steps' not in st.session_state:
    st.session_state.processing_steps = []
if 'current_user_input' not in st.session_state:
    st.session_state.current_user_input = None

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
    
    # LLM Configuration
    st.divider()
    st.subheader("LLM Configuration")
    
    llm_provider = st.selectbox(
        "LLM Provider",
        ["None", "Anthropic", "OpenAI", "Ollama"],
        help="Select your preferred LLM provider"
    )
    
    if llm_provider == "Anthropic":
        with st.expander("Anthropic Settings", expanded=True):
            anthropic_api_key = st.text_input(
                "Anthropic API Key", 
                type="password",
                help="Your Anthropic API key"
            )
            anthropic_model = st.selectbox(
                "Model",
                ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307", "claude-3-opus-20240229"],
                help="Select Claude model"
            )
            
            if st.button("🤖 Connect Anthropic") and anthropic_api_key:
                try:
                    llm_connector = create_llm_connector(
                        provider="anthropic",
                        api_key=anthropic_api_key,
                        model=anthropic_model
                    )
                    st.session_state.llm_connector = llm_connector
                    st.success("✅ Anthropic connected!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Anthropic connection failed: {str(e)}")
    
    elif llm_provider == "OpenAI":
        with st.expander("OpenAI Settings", expanded=True):
            openai_api_key = st.text_input(
                "OpenAI API Key", 
                type="password",
                help="Your OpenAI API key"
            )
            openai_model = st.selectbox(
                "Model",
                ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                help="Select OpenAI model"
            )
            
            openai_base_url = st.text_input(
                "Base URL (Advanced)",
                value="https://api.openai.com/v1",
                help="OpenAI API base URL - change for Azure OpenAI or compatible endpoints"
            )
            
            if st.button("🤖 Connect OpenAI") and openai_api_key:
                try:
                    llm_connector = create_llm_connector(
                        provider="openai",
                        api_key=openai_api_key,
                        model=openai_model,
                        base_url=openai_base_url
                    )
                    st.session_state.llm_connector = llm_connector
                    st.success("✅ OpenAI connected!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ OpenAI connection failed: {str(e)}")
    
    elif llm_provider == "Ollama":
        with st.expander("Ollama Settings", expanded=True):
            st.info("🏠 Ollama runs locally - no API key required")
            
            ollama_model = st.text_input(
                "Model",
                value="llama3.1:8b",
                help="Ollama model name (e.g., llama3.1:8b, codellama:7b, mistral:7b)"
            )
            
            # Default to host.docker.internal when likely running in Docker
            import os
            default_ollama_url = "http://host.docker.internal:11434" if os.path.exists("/.dockerenv") else "http://localhost:11434"
            
            ollama_base_url = st.text_input(
                "Base URL (Advanced)",
                value=default_ollama_url,
                help="Ollama server URL - defaults to host.docker.internal:11434 in Docker, localhost:11434 otherwise"
            )
            
            if st.button("🤖 Connect Ollama"):
                try:
                    llm_connector = create_llm_connector(
                        provider="ollama",
                        api_key="ollama",  # Not used for Ollama
                        model=ollama_model,
                        base_url=ollama_base_url
                    )
                    st.session_state.llm_connector = llm_connector
                    st.success("✅ Ollama connected!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ollama connection failed: {str(e)}")
    
    # LLM Status
    if st.session_state.llm_connector:
        # Try to determine LLM provider from connector type
        connector_type = type(st.session_state.llm_connector).__name__
        provider_name = connector_type.replace("Connector", "") if "Connector" in connector_type else "LLM"
        st.success(f"🤖 {provider_name} Connected")
    else:
        st.warning("🤖 No LLM connected")
    
    # Available Tools
    if st.session_state.available_tools:
        st.divider()
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
                
                # Show processing steps if any (step-by-step tool calling)
                if "processing_steps" in message and message["processing_steps"]:
                    with st.expander("🔍 Processing Steps", expanded=False):
                        for step in message["processing_steps"]:
                            if step["type"] == "tool_start":
                                st.info(f"⚡ {step['timestamp']} - {step['message']}")
                            elif step["type"] == "tool_complete":
                                st.success(f"✅ {step['timestamp']} - {step['message']}")
                            elif step["type"] == "reasoning":
                                st.write(f"💭 {step['timestamp']} - {step['message']}")
                
                # Show tool calls if any (legacy format)
                if "tool_calls" in message and message["tool_calls"]:
                    with st.expander("🔧 Tool Calls", expanded=False):
                        for tool_call in message["tool_calls"]:
                            st.json(tool_call)
    
    # Chat input
    st.divider()
    
    # Message input (allow LLM-only mode)
    input_disabled = not st.session_state.llm_connector
    help_text = "Connect an LLM provider first" if input_disabled else "Type your message here..."
    
    user_input = st.chat_input(
        help_text,
        disabled=input_disabled,
        key="chat_input"
    )
    
    if user_input and st.session_state.llm_connector:
        # Add user message to chat immediately
        user_message = {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.chat_messages.append(user_message)
        st.session_state.current_user_input = user_input
        st.session_state.processing_steps = []
        
        # Force rerun to show user message immediately
        st.rerun()
        
    # Process LLM request if we have user input that hasn't been processed yet
    if (st.session_state.current_user_input and 
        st.session_state.llm_connector and 
        len(st.session_state.chat_messages) > 0 and 
        st.session_state.chat_messages[-1]["role"] == "user" and 
        st.session_state.chat_messages[-1]["content"] == st.session_state.current_user_input):
        
        # Create containers for step-by-step display
        processing_container = st.container()
        
        with processing_container:
            st.write("🤖 **Processing your request...**")
            steps_placeholder = st.empty()
        
        # Define step callback for real-time updates
        async def step_callback(message: str, step_type: str, data: dict):
            step_info = {
                "message": message,
                "type": step_type,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "data": data
            }
            st.session_state.processing_steps.append(step_info)
            
            # Update the display with current steps
            with steps_placeholder.container():
                for step in st.session_state.processing_steps:
                    if step["type"] == "tool_start":
                        st.info(f"⚡ {step['timestamp']} - {step['message']}")
                    elif step["type"] == "tool_complete":
                        st.success(f"✅ {step['timestamp']} - {step['message']}")
                    elif step["type"] == "reasoning":
                        st.write(f"💭 {step['timestamp']} - {step['message']}")
        
        async def process_with_llm():
            try:
                # Generate response using LLM with optional MCP tools
                response_text, tool_calls = await st.session_state.llm_connector.generate_response(
                    messages=st.session_state.chat_messages,
                    tools=st.session_state.available_tools if st.session_state.mcp_client else None,
                    mcp_client=st.session_state.mcp_client,
                    step_callback=step_callback
                )
                
                # Create assistant message
                assistant_message = {
                    "role": "assistant",
                    "content": response_text,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "tool_calls": tool_calls if tool_calls else [],
                    "processing_steps": st.session_state.processing_steps.copy()
                }
                
                st.session_state.chat_messages.append(assistant_message)
                st.session_state.current_user_input = None  # Clear processed input
                st.session_state.processing_steps = []  # Clear steps
                
            except Exception as e:
                error_message = {
                    "role": "assistant",
                    "content": f"❌ Error processing request: {str(e)}",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.chat_messages.append(error_message)
                st.session_state.current_user_input = None  # Clear processed input
                st.session_state.processing_steps = []  # Clear steps
        
        # Run async LLM processing
        asyncio.run(process_with_llm())
        st.rerun()
    
    # Help text based on connection status
    if not st.session_state.llm_connector:
        st.info("🤖 Please configure and connect an LLM provider to start chatting")
    elif not st.session_state.mcp_client:
        st.warning("🔐 No MCP server connected - you can chat with the LLM, but network management tools are not available")
    else:
        st.success("✅ Ready for AI-powered network management!")
    
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
                            response_content = extract_mcp_response_text(result)
                            
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
- 💬 AI-powered chat with optional network infrastructure context
- 🤖 Multiple LLM providers: Anthropic Claude, OpenAI GPT, Ollama (local)
- 🔧 Step-by-step tool calling visualization with real-time updates
- 🔐 Optional MCP server configuration and authentication
- 🔍 MCP Inspector for protocol testing
- 📋 Real-time processing steps display
- 📨 Immediate user message display

**Coming Next:**
- 🎯 Enhanced context management and conversation memory
- 📊 Advanced network analysis workflows
- 🔄 Multi-step workflow automation
- 🌐 Additional LLM providers and local model support
""")