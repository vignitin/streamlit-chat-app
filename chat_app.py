import streamlit as st
import json
import asyncio
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from dotenv import load_dotenv
from mcp_client import MCPClient, create_mcp_client, extract_mcp_response_text, MCPServerManager
from llm_connector import create_llm_connector

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="MCP Chat Interface",
    page_icon="💬",
    layout="wide"
)

# Initialize session state for chat
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'mcp_client' not in st.session_state:
    st.session_state.mcp_client = None  # Legacy single server support
if 'mcp_manager' not in st.session_state:
    st.session_state.mcp_manager = MCPServerManager()  # Multi-server support
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
if 'show_add_server' not in st.session_state:
    st.session_state.show_add_server = False

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
    
    # Multi-MCP Server Configuration
    st.subheader("🔧 MCP Servers")
    
    # Add/Connect to MCP Server (single form)
    if st.button("➕ Add MCP Server"):
        st.session_state.show_add_server = not st.session_state.show_add_server
    
    if st.session_state.show_add_server:
        st.write("**Add & Connect to MCP Server**")
        
        # Authentication type selection outside form for immediate reactivity
        new_auth_type = st.selectbox("Authentication", ["none", "session"], key="auth_type_select")
        
        with st.form("add_server_form"):
            new_server_name = st.text_input("Server Name", help="Friendly name for this server")
            new_server_url = st.text_input("Server URL", value="http://host.docker.internal:8080")
            
            # Show authentication fields for session auth
            if new_auth_type == "session":
                st.write("**Authentication Details**")
                new_username = st.text_input("Username", key="add_username")
                new_password = st.text_input("Password", type="password", key="add_password")
                new_auth_server = st.text_input("Auth Server", key="add_auth_server")
                new_auth_port = st.text_input("Auth Port", value="443", key="add_auth_port")
            else:
                new_username = new_password = new_auth_server = new_auth_port = None
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Add & Connect"):
                    if new_server_name and new_server_url:
                        # Validate required fields for session auth
                        if new_auth_type == "session" and (not new_username or not new_password or not new_auth_server):
                            st.error("Please provide username, password, and auth server for session authentication")
                        else:
                            async def add_and_connect_server():
                                try:
                                    # Add server
                                    server_id = await st.session_state.mcp_manager.add_server(
                                        new_server_name, new_server_url, new_auth_type
                                    )
                                    
                                    # Connect immediately
                                    if new_auth_type == "session":
                                        await st.session_state.mcp_manager.connect_server(
                                            server_id, new_username, new_password, new_auth_server, new_auth_port
                                        )
                                    else:
                                        await st.session_state.mcp_manager.connect_server(server_id)
                                    
                                    st.success(f"✅ Added and connected to {new_server_name}!")
                                    st.session_state.show_add_server = False
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Failed to add/connect server: {str(e)}")
                            asyncio.run(add_and_connect_server())
                    else:
                        st.error("Please provide server name and URL")
            
            with col2:
                if st.form_submit_button("Cancel"):
                    st.session_state.show_add_server = False
                    st.rerun()
    
    # Display current servers (simplified)
    servers = st.session_state.mcp_manager.get_all_servers()
    
    if servers:
        st.write("**Connected Servers:**")
        for server in servers:
            if server.connected:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"🟢 **{server.name}** ({server.url}) - {server.tool_count} tools")
                with col2:
                    # Show re-auth button only for session-based auth
                    if server.auth_type == "session":
                        if st.button("🔄", key=f"reauth_{server.id}", help="Re-authenticate"):
                            st.session_state[f"show_reauth_{server.id}"] = True
                            st.rerun()
                with col3:
                    if st.button("🗑️", key=f"remove_{server.id}", help="Remove server"):
                        async def remove_server():
                            try:
                                await st.session_state.mcp_manager.remove_server(server.id)
                                st.success(f"Removed {server.name}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to remove: {str(e)}")
                        asyncio.run(remove_server())
                
                # Show re-authentication form if requested
                if st.session_state.get(f"show_reauth_{server.id}", False):
                    with st.expander("🔐 Re-authenticate Server", expanded=True):
                        with st.form(f"reauth_form_{server.id}"):
                            st.write(f"Re-authenticate to **{server.name}**")
                            reauth_username = st.text_input("Username", key=f"reauth_username_{server.id}")
                            reauth_password = st.text_input("Password", type="password", key=f"reauth_password_{server.id}")
                            reauth_auth_server = st.text_input("Auth Server", key=f"reauth_auth_server_{server.id}")
                            reauth_auth_port = st.text_input("Auth Port", value="443", key=f"reauth_auth_port_{server.id}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.form_submit_button("Re-authenticate"):
                                    if reauth_username and reauth_password and reauth_auth_server:
                                        async def reauthenticate():
                                            try:
                                                await st.session_state.mcp_manager.reauthenticate_server(
                                                    server.id, reauth_username, reauth_password, 
                                                    reauth_auth_server, reauth_auth_port
                                                )
                                                st.success(f"✅ Re-authenticated to {server.name}!")
                                                st.session_state[f"show_reauth_{server.id}"] = False
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"❌ Re-authentication failed: {str(e)}")
                                        asyncio.run(reauthenticate())
                                    else:
                                        st.error("Please provide all required fields")
                            
                            with col2:
                                if st.form_submit_button("Cancel"):
                                    st.session_state[f"show_reauth_{server.id}"] = False
                                    st.rerun()
    
    # Overall MCP Status
    st.divider()
    st.subheader("MCP Status")
    connected_servers = st.session_state.mcp_manager.get_connected_servers()
    total_tools = sum(server.tool_count for server in connected_servers)
    
    if connected_servers:
        st.success(f"🟢 {len(connected_servers)} server(s) connected")
        if total_tools > 0:
            st.info(f"🔧 {total_tools} tools available across all servers")
    else:
        st.warning("🔴 No MCP servers connected")
    
    # LLM Configuration
    st.divider()
    st.subheader("LLM Configuration")
    
    # Show info about environment variables
    if os.path.exists(".env"):
        st.info("💡 API keys are auto-loaded from .env file if available")
    else:
        st.info("💡 Create a .env file from .env.example to auto-load API keys")
    
    llm_provider = st.selectbox(
        "LLM Provider",
        ["None", "Anthropic", "OpenAI", "Ollama"],
        help="Select your preferred LLM provider"
    )
    
    if llm_provider == "Anthropic":
        with st.expander("Anthropic Settings", expanded=True):
            # Auto-load API key from environment if available
            default_anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
            anthropic_api_key = st.text_input(
                "Anthropic API Key", 
                type="password",
                value=default_anthropic_key,
                help="Your Anthropic API key (auto-loaded from ANTHROPIC_API_KEY env var if set)"
            )
            anthropic_model = st.selectbox(
                "Model",
                ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307", "claude-3-opus-20240229"],
                help="Select Claude model"
            )
            
            st.info("ℹ️ Smart loop detection prevents infinite tool calls. No limits on complex infrastructure queries.")
            
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
            # Auto-load API key and base URL from environment if available
            default_openai_key = os.getenv("OPENAI_API_KEY", "")
            default_openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            
            openai_api_key = st.text_input(
                "OpenAI API Key", 
                type="password",
                value=default_openai_key,
                help="Your OpenAI API key (auto-loaded from OPENAI_API_KEY env var if set)"
            )
            openai_model = st.selectbox(
                "Model",
                ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                help="Select OpenAI model"
            )
            
            openai_base_url = st.text_input(
                "Base URL (Advanced)",
                value=default_openai_base_url,
                help="OpenAI API base URL - change for Azure OpenAI or compatible endpoints (auto-loaded from OPENAI_BASE_URL env var if set)"
            )
            
            st.info("ℹ️ Smart loop detection prevents infinite tool calls. No limits on complex infrastructure queries.")
            
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
            
            # Auto-load base URL from environment, with Docker detection fallback
            env_ollama_url = os.getenv("OLLAMA_BASE_URL")
            if env_ollama_url:
                default_ollama_url = env_ollama_url
            else:
                default_ollama_url = "http://host.docker.internal:11434" if os.path.exists("/.dockerenv") else "http://localhost:11434"
            
            ollama_base_url = st.text_input(
                "Base URL (Advanced)",
                value=default_ollama_url,
                help="Ollama server URL - defaults to host.docker.internal:11434 in Docker, localhost:11434 otherwise (auto-loaded from OLLAMA_BASE_URL env var if set)"
            )
            
            st.info("ℹ️ Tool calling works with models like Llama 3.1, Mistral, and Qwen2.5. Smart loop detection prevents infinite tool calls.")
            
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
    
    # Available Tools from all servers
    async def get_all_tools_async():
        return await st.session_state.mcp_manager.get_all_tools()
    
    try:
        all_tools = asyncio.run(get_all_tools_async())
        if all_tools:
            st.divider()
            st.subheader("Available Tools")
            with st.expander(f"All MCP Tools ({len(all_tools)})", expanded=False):
                # Group tools by server
                from collections import defaultdict
                tools_by_server = defaultdict(list)
                for tool in all_tools:
                    server_name = tool.get('server_name', 'Unknown Server')
                    tools_by_server[server_name].append(tool)
                
                for server_name, tools in tools_by_server.items():
                    st.write(f"**{server_name}** ({len(tools)} tools):")
                    for tool in tools:
                        st.text(f"  • {tool.get('original_name', 'Unknown')} → {tool.get('full_name', 'Unknown')}")
    except:
        pass  # Ignore errors in tool display

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
                # Get tools from multi-server manager
                tools = await st.session_state.mcp_manager.get_all_tools()
                
                # Generate response using LLM with MCP tools from all servers
                response_text, tool_calls = await st.session_state.llm_connector.generate_response(
                    messages=st.session_state.chat_messages,
                    tools=tools,
                    mcp_client=None,  # Legacy support
                    mcp_manager=st.session_state.mcp_manager,  # Multi-server support
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
    else:
        connected_servers = st.session_state.mcp_manager.get_connected_servers()
        if not connected_servers:
            st.warning("🔐 No MCP servers connected - you can chat with the LLM, but network management tools are not available")
        else:
            total_tools = sum(server.tool_count for server in connected_servers)
            st.success(f"✅ Ready for AI-powered network management! ({len(connected_servers)} server(s), {total_tools} tools)")
    
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
        
        # Server selection
        connected_servers = st.session_state.mcp_manager.get_connected_servers()
        if not connected_servers:
            st.error("❌ No MCP servers connected. Please connect a server first.")
            st.stop()
        
        server_names = [f"{server.name} ({server.url})" for server in connected_servers]
        selected_server_index = st.selectbox(
            "Select Server to Test",
            range(len(server_names)),
            format_func=lambda x: server_names[x]
        )
        selected_server = connected_servers[selected_server_index]
        
        st.divider()
        
        test_connection = st.checkbox("Test Connection", value=True)
        test_list_tools = st.checkbox("List Tools", value=True)
        test_call_tool = st.checkbox("Call Tool", value=True)
        test_list_prompts = st.checkbox("List Prompts", value=False)
        test_list_resources = st.checkbox("List Resources", value=False)
        
        if test_call_tool:
            # Get available tools for the selected server
            server_tools = st.session_state.mcp_manager.server_tools.get(selected_server.id, [])
            tool_names = [tool.get('name', '') for tool in server_tools] if server_tools else []
            
            if tool_names:
                # Use dropdown for available tools
                selected_tool = st.selectbox("Select Tool to Test", tool_names)
                tool_name = selected_tool
            else:
                # Fallback to text input
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
                    # Get the client for the selected server
                    client = st.session_state.mcp_manager.clients.get(selected_server.id)
                    if not client:
                        st.error(f"No client found for server {selected_server.name}")
                        return
                    
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

