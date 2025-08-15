"""
LLM Connector for MCP Chat Interface
Handles communication with various LLM providers while preserving MCP context
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import anthropic
import httpx
from mcp_client import MCPClient, MCPServerManager

# System messages for LLM configuration
SYSTEM_MESSAGE_WITH_MCP = """You are an AI assistant with access to network infrastructure management tools through MCP (Model Context Protocol). 

IMPORTANT: Formatting Guidelines for Data Presentation
1. Check if any connected MCP server provides formatting guidelines:
   - Look for tools named 'formatting_guidelines', 'get_formatting_guidelines', or similar
   - If available, call this tool FIRST before presenting any data from that server
   - Apply the formatting rules consistently to all data from that server

2. When formatting guidelines are available:
   - Follow them exactly as specified
   - Use any status icons, table formats, or organization rules provided
   - Apply the guidelines consistently across your response

3. If no specific formatting guidelines tool is available:
   - Present information in a clear, well-organized manner
   - Use tables for structured data
   - Apply appropriate status indicators (✅ for good/up, ❌ for bad/down, ⚠️ for warning)
   - Use logical grouping and sorting
   - Include summary sections for complex data

Remember: Different MCP servers may have different formatting requirements. Always check for server-specific formatting guidelines before presenting data."""

SYSTEM_MESSAGE_WITHOUT_MCP = """You are an AI assistant. You can have conversations and answer questions, but you don't currently have access to external tools or network infrastructure management capabilities."""

class LLMConnector:
    """Base class for LLM connectors"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.max_tool_rounds = 50  # High default limit
        self.seen_calls = set()  # For loop detection
    
    def _is_loop_detected(self, tool_name: str, arguments: Dict) -> bool:
        """Detect if the same tool call is being repeated (potential infinite loop)"""
        # Create a unique signature for this tool call
        call_signature = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
        
        if call_signature in self.seen_calls:
            return True  # Same exact call repeated - likely a loop
        
        self.seen_calls.add(call_signature)
        return False
    
    def _reset_loop_detection(self):
        """Reset loop detection for new conversation"""
        self.seen_calls.clear()
    
    async def generate_response(self, messages: List[Dict], tools: List[Dict] = None, mcp_client: MCPClient = None, mcp_manager: Any = None) -> Tuple[str, List[Dict]]:
        """Generate response from LLM with tool calling support"""
        raise NotImplementedError
    
    async def _call_mcp_tool(self, tool_name: str, arguments: Dict, mcp_client: MCPClient = None, mcp_manager: Any = None) -> Dict:
        """Call MCP tool and return response"""
        try:
            # Use manager if available (for multi-server support)
            if mcp_manager:
                response = await mcp_manager.call_tool(tool_name, arguments)
            elif mcp_client:
                response = await mcp_client.call_tool(tool_name, arguments)
            else:
                raise ValueError("No MCP client or manager available")
            
            # Extract text content from MCP response format
            if isinstance(response, dict) and 'content' in response:
                content_items = response.get('content', [])
                if content_items and isinstance(content_items, list):
                    text_contents = [item.get('text', '') for item in content_items if item.get('type') == 'text']
                    return {
                        "type": "text",
                        "text": '\n'.join(text_contents) if text_contents else str(response)
                    }
            
            return {
                "type": "text", 
                "text": str(response)
            }
            
        except Exception as e:
            return {
                "type": "text",
                "text": f"Tool call failed: {str(e)}"
            }

class OpenAIConnector(LLMConnector):
    """OpenAI GPT LLM connector with MCP tool integration"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = "https://api.openai.com/v1"):
        super().__init__(api_key)
        self.model = model
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=60.0
        )
    
    def _convert_mcp_tools_to_openai(self, mcp_tools: List[Dict]) -> List[Dict]:
        """Convert MCP tool definitions to OpenAI function format"""
        openai_tools = []
        
        for tool in mcp_tools:
            # Convert MCP tool schema to OpenAI function format
            # Use full_name for multi-server support, fall back to name for legacy compatibility
            tool_name = tool.get("full_name", tool.get("name", ""))
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                }
            }
            openai_tools.append(openai_tool)
        
        return openai_tools
    
    def _convert_messages_to_openai(self, messages: List[Dict]) -> List[Dict]:
        """Convert chat messages to OpenAI format"""
        openai_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle assistant messages with tool calls
            if role == "assistant" and "tool_calls" in msg and msg["tool_calls"]:
                # Create assistant message with tool calls
                tool_calls = []
                for tool_call in msg["tool_calls"]:
                    tool_calls.append({
                        "id": tool_call.get("id", f"call_{datetime.now().timestamp()}"),
                        "type": "function",
                        "function": {
                            "name": tool_call.get("name", ""),
                            "arguments": json.dumps(tool_call.get("arguments", {}))
                        }
                    })
                
                openai_messages.append({
                    "role": "assistant",
                    "content": content if content.strip() else None,
                    "tool_calls": tool_calls
                })
                
                # Add tool result messages
                for tool_call in msg["tool_calls"]:
                    result_content = ""
                    if "result" in tool_call and "text" in tool_call["result"]:
                        result_content = tool_call["result"]["text"]
                    elif "result" in tool_call:
                        result_content = str(tool_call["result"])
                    
                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", f"call_{datetime.now().timestamp()}"),
                        "content": result_content
                    })
            else:
                # Regular text message
                openai_messages.append({
                    "role": "assistant" if role == "assistant" else "user",
                    "content": content
                })
        
        return openai_messages
    
    async def generate_response(self, messages: List[Dict], tools: List[Dict] = None, mcp_client: MCPClient = None, mcp_manager: Any = None, step_callback=None) -> Tuple[str, List[Dict]]:
        """Generate response from OpenAI GPT with optional MCP tool support"""
        
        # Reset loop detection for new conversation
        self._reset_loop_detection()
        
        # Convert tools and messages to OpenAI format
        openai_tools = self._convert_mcp_tools_to_openai(tools) if tools else []
        openai_messages = self._convert_messages_to_openai(messages)
        
        # System message
        has_mcp = bool(mcp_client or mcp_manager)
        system_message = SYSTEM_MESSAGE_WITHOUT_MCP if not has_mcp else SYSTEM_MESSAGE_WITH_MCP
        
        # Insert system message at beginning
        openai_messages.insert(0, {"role": "system", "content": system_message})
        
        try:
            # Prepare request payload
            payload = {
                "model": self.model,
                "messages": openai_messages,
                "max_tokens": 4096,
                "temperature": 0.7
            }
            
            if openai_tools:
                payload["tools"] = openai_tools
                payload["tool_choice"] = "auto"
            
            # Make request to OpenAI
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            response_data = response.json()
            
            # Handle response
            choice = response_data["choices"][0]
            message = choice["message"]
            
            tool_calls = []
            response_text = message.get("content", "")
            
            # Process tool calls if present
            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    function = tool_call["function"]
                    tool_name = function["name"]
                    
                    try:
                        arguments = json.loads(function["arguments"])
                    except json.JSONDecodeError:
                        arguments = {}
                    
                    # Notify step callback
                    if step_callback:
                        await step_callback(f"🔧 Calling tool: **{tool_name}**", "tool_start", {
                            "tool_name": tool_name,
                            "arguments": arguments
                        })
                    
                    # Check for infinite loops
                    if self._is_loop_detected(tool_name, arguments):
                        # Return a message about the loop detection
                        return f"Loop detected: The same tool call '{tool_name}' with identical arguments was attempted multiple times. Stopping to prevent infinite loop.", tool_calls
                    
                    # Execute MCP tool (only if client or manager is available)
                    if mcp_client or mcp_manager:
                        tool_result = await self._call_mcp_tool(tool_name, arguments, mcp_client, mcp_manager)
                    else:
                        tool_result = {
                            "type": "text",
                            "text": "Tool execution not available - no MCP server connected"
                        }
                    
                    # Notify step callback
                    if step_callback:
                        await step_callback(f"✅ Tool **{tool_name}** completed", "tool_complete", {
                            "tool_name": tool_name,
                            "result": tool_result
                        })
                    
                    tool_calls.append({
                        "id": tool_call["id"],
                        "name": tool_name,
                        "arguments": arguments,
                        "result": tool_result
                    })
                
                # If there were tool calls, handle multiple rounds
                if tool_calls:
                    # Track all tool calls across rounds
                    all_tool_calls = tool_calls.copy()
                    current_round = 0
                    
                    while current_round < self.max_tool_rounds:
                        # Add assistant message with tool calls
                        openai_messages.append({
                            "role": "assistant",
                            "content": response_text if current_round == 0 else "",
                            "tool_calls": [{
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(tc["arguments"])
                                }
                            } for tc in tool_calls]
                        })
                        
                        # Add tool results
                        for tool_call in tool_calls:
                            openai_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": tool_call["result"]["text"]
                            })
                        
                        # Get next response
                        follow_up_payload = {
                            "model": self.model,
                            "messages": openai_messages,
                            "tools": openai_tools if openai_tools else None,
                            "max_tokens": 4096,
                            "temperature": 0.7
                        }
                        
                        follow_up_response = await self.client.post(
                            f"{self.base_url}/chat/completions",
                            json=follow_up_payload
                        )
                        follow_up_response.raise_for_status()
                        follow_up_data = follow_up_response.json()
                        
                        follow_up_message = follow_up_data["choices"][0]["message"]
                        follow_up_text = follow_up_message.get("content", "")
                        
                        # Check for more tool calls
                        if follow_up_message.get("tool_calls"):
                            tool_calls = []
                            for tool_call in follow_up_message["tool_calls"]:
                                function = tool_call["function"]
                                tool_name = function["name"]
                                
                                # Extract server and actual tool name
                                if "__" in tool_name:
                                    server_name, actual_tool_name = tool_name.split("__", 1)
                                else:
                                    server_name = "default"
                                    actual_tool_name = tool_name
                                
                                try:
                                    arguments = json.loads(function["arguments"])
                                except json.JSONDecodeError:
                                    arguments = {}
                                
                                # Check for infinite loops
                                if self._is_loop_detected(tool_name, arguments):
                                    # Stop processing and return current state
                                    return f"Loop detected: The same tool call '{tool_name}' with identical arguments was attempted multiple times. Stopping to prevent infinite loop.", all_tool_calls
                                
                                # Execute tool with multi-server support
                                tool_result = await mcp_manager.call_tool(
                                    server_name=server_name,
                                    tool_name=actual_tool_name,
                                    arguments=arguments
                                )
                                
                                # Report step
                                if step_callback:
                                    await step_callback({
                                        "type": "tool_result",
                                        "tool_name": tool_name,
                                        "result": tool_result
                                    })
                                
                                tool_calls.append({
                                    "id": tool_call["id"],
                                    "name": tool_name,
                                    "arguments": arguments,
                                    "result": tool_result
                                })
                            
                            all_tool_calls.extend(tool_calls)
                            current_round += 1
                        else:
                            # No more tool calls, return final response
                            return follow_up_text, all_tool_calls
                    
                    # If we hit max rounds, return last response
                    return follow_up_text, all_tool_calls
            
            return response_text, tool_calls
            
        except Exception as e:
            error_msg = f"OpenAI API request failed: {str(e)}"
            return error_msg, []

class OllamaConnector(LLMConnector):
    """Ollama local LLM connector with MCP tool integration"""
    
    def __init__(self, api_key: str = "ollama", model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        super().__init__(api_key)
        self.model = model
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=120.0)  # Longer timeout for local models
    
    def _convert_mcp_tools_to_ollama(self, mcp_tools: List[Dict]) -> List[Dict]:
        """Convert MCP tool definitions to Ollama function format"""
        # Ollama uses OpenAI-compatible function calling format
        ollama_tools = []
        
        for tool in mcp_tools:
            # Use full_name for multi-server support, fall back to name for legacy compatibility
            tool_name = tool.get("full_name", tool.get("name", ""))
            ollama_tool = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                }
            }
            ollama_tools.append(ollama_tool)
        
        return ollama_tools
    
    def _convert_messages_to_ollama(self, messages: List[Dict]) -> List[Dict]:
        """Convert chat messages to Ollama format"""
        ollama_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle assistant messages with tool calls
            if role == "assistant" and "tool_calls" in msg and msg["tool_calls"]:
                # Include tool calls in the message
                ollama_messages.append({
                    "role": "assistant",
                    "content": content if content else "",
                    "tool_calls": msg["tool_calls"]
                })
                
                # Add tool results if present
                for tool_call in msg["tool_calls"]:
                    if "result" in tool_call and "text" in tool_call["result"]:
                        ollama_messages.append({
                            "role": "tool",
                            "content": tool_call["result"]["text"]
                        })
            else:
                # Regular message
                ollama_messages.append({
                    "role": role,
                    "content": content
                })
        
        return ollama_messages
    
    async def generate_response(self, messages: List[Dict], tools: List[Dict] = None, mcp_client: MCPClient = None, mcp_manager: Any = None, step_callback=None) -> Tuple[str, List[Dict]]:
        """Generate response from Ollama with MCP tool support"""
        
        # Reset loop detection for new conversation
        self._reset_loop_detection()
        
        # Convert tools and messages to Ollama format
        ollama_tools = self._convert_mcp_tools_to_ollama(tools) if tools else []
        ollama_messages = self._convert_messages_to_ollama(messages)
        
        # System message
        has_mcp = bool(mcp_client or mcp_manager)
        if not has_mcp:
            system_message = SYSTEM_MESSAGE_WITHOUT_MCP
        else:
            system_message = SYSTEM_MESSAGE_WITH_MCP
        
        # Insert system message at beginning
        ollama_messages.insert(0, {"role": "system", "content": system_message})
        
        try:
            # Check if Ollama server is accessible
            health_response = await self.client.get(f"{self.base_url}/api/tags")
            if health_response.status_code != 200:
                return "Ollama server not accessible. Please ensure Ollama is running on localhost:11434", []
            
            # Initial request with tools
            payload = {
                "model": self.model,
                "messages": ollama_messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 4096
                }
            }
            
            # Add tools if available
            if ollama_tools:
                payload["tools"] = ollama_tools
            
            # Notify step callback about reasoning
            if step_callback:
                await step_callback("🧠 Generating response with Ollama...", "reasoning", {})
            
            # Make initial request
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            response_data = response.json()
            
            message = response_data.get("message", {})
            response_text = message.get("content", "")
            tool_calls = []
            
            # Check for tool calls in the response
            if message.get("tool_calls"):
                # Handle multiple rounds of tool calling
                all_tool_calls = []
                current_round = 0
                
                while current_round < self.max_tool_rounds:
                    round_tool_calls = []
                    
                    for tool_call in message.get("tool_calls", []):
                        function = tool_call.get("function", {})
                        tool_name = function.get("name", "")
                        
                        # Extract server and actual tool name for multi-server support
                        if "__" in tool_name:
                            server_name, actual_tool_name = tool_name.split("__", 1)
                        else:
                            server_name = "default"
                            actual_tool_name = tool_name
                        
                        try:
                            arguments = function.get("arguments", {})
                            if isinstance(arguments, str):
                                arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                        
                        # Report step
                        if step_callback:
                            await step_callback(f"⚡ Calling tool: {tool_name}", "tool_call", {
                                "tool_name": tool_name,
                                "arguments": arguments
                            })
                        
                        # Check for infinite loops
                        if self._is_loop_detected(tool_name, arguments):
                            # Return a message about the loop detection
                            return f"Loop detected: The same tool call '{tool_name}' with identical arguments was attempted multiple times. Stopping to prevent infinite loop.", all_tool_calls
                        
                        # Execute tool with multi-server support
                        if mcp_manager:
                            tool_result = await mcp_manager.call_tool(
                                server_name=server_name,
                                tool_name=actual_tool_name,
                                arguments=arguments
                            )
                        else:
                            # Legacy single-client support
                            tool_result = await self._call_mcp_tool(tool_name, arguments, mcp_client)
                        
                        # Report result
                        if step_callback:
                            await step_callback({
                                "type": "tool_result",
                                "tool_name": tool_name,
                                "result": tool_result
                            })
                        
                        round_tool_calls.append({
                            "id": tool_call.get("id", f"call_{datetime.now().timestamp()}"),
                            "name": tool_name,
                            "arguments": arguments,
                            "result": tool_result
                        })
                    
                    all_tool_calls.extend(round_tool_calls)
                    
                    # Add assistant message with tool calls
                    ollama_messages.append({
                        "role": "assistant",
                        "content": response_text if current_round == 0 else "",
                        "tool_calls": message.get("tool_calls", [])
                    })
                    
                    # Add tool results as separate messages
                    for tc in round_tool_calls:
                        ollama_messages.append({
                            "role": "tool",
                            "content": tc["result"].get("text", "")
                        })
                    
                    # Get next response
                    follow_up_payload = {
                        "model": self.model,
                        "messages": ollama_messages,
                        "tools": ollama_tools if ollama_tools else None,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 4096
                        }
                    }
                    
                    follow_up_response = await self.client.post(
                        f"{self.base_url}/api/chat",
                        json=follow_up_payload
                    )
                    follow_up_response.raise_for_status()
                    follow_up_data = follow_up_response.json()
                    
                    message = follow_up_data.get("message", {})
                    follow_up_text = message.get("content", "")
                    
                    # Check for more tool calls
                    if message.get("tool_calls"):
                        current_round += 1
                    else:
                        # No more tool calls, return final response
                        return follow_up_text, all_tool_calls
                
                # If we hit max rounds, return last response
                return follow_up_text, all_tool_calls
            
            return response_text, tool_calls
            
        except Exception as e:
            error_msg = f"Ollama API request failed: {str(e)}"
            return error_msg, []

class AnthropicConnector(LLMConnector):
    """Anthropic Claude LLM connector with MCP tool integration"""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        super().__init__(api_key)
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def _convert_mcp_tools_to_anthropic(self, mcp_tools: List[Dict]) -> List[Dict]:
        """Convert MCP tool definitions to Anthropic tool format"""
        anthropic_tools = []
        
        for tool in mcp_tools:
            # Convert MCP tool schema to Anthropic format
            # Use full_name for multi-server support, fall back to name for legacy compatibility
            tool_name = tool.get("full_name", tool.get("name", ""))
            anthropic_tool = {
                "name": tool_name,
                "description": tool.get("description", ""),
                "input_schema": tool.get("inputSchema", {
                    "type": "object",
                    "properties": {},
                    "required": []
                })
            }
            anthropic_tools.append(anthropic_tool)
        
        return anthropic_tools
    
    def _convert_messages_to_anthropic(self, messages: List[Dict]) -> List[Dict]:
        """Convert chat messages to Anthropic format"""
        anthropic_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle assistant messages with tool calls
            if role == "assistant" and "tool_calls" in msg and msg["tool_calls"]:
                # Create assistant message with tool_use blocks
                assistant_content = []
                
                # Add text content if present (before tool calls)
                if content and content.strip():
                    assistant_content.append({
                        "type": "text",
                        "text": content
                    })
                
                # Add tool use blocks
                for tool_call in msg["tool_calls"]:
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tool_call.get("id", f"tool_{datetime.now().timestamp()}"),
                        "name": tool_call.get("name", ""),
                        "input": tool_call.get("arguments", {})
                    })
                
                # Add assistant message with tool_use blocks
                anthropic_messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })
                
                # Add corresponding tool_result messages as user messages
                for tool_call in msg["tool_calls"]:
                    tool_result_text = ""
                    if "result" in tool_call and "text" in tool_call["result"]:
                        tool_result_text = tool_call["result"]["text"]
                    elif "result" in tool_call:
                        tool_result_text = str(tool_call["result"])
                    
                    anthropic_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call.get("id", f"tool_{datetime.now().timestamp()}"),
                                "content": tool_result_text
                            }
                        ]
                    })
                    
            else:
                # Regular text message (user or assistant without tool calls)
                anthropic_messages.append({
                    "role": "assistant" if role == "assistant" else "user",
                    "content": content
                })
        
        return anthropic_messages
    
    async def generate_response(self, messages: List[Dict], tools: List[Dict] = None, mcp_client: MCPClient = None, mcp_manager: Any = None, step_callback=None) -> Tuple[str, List[Dict]]:
        """Generate response from Anthropic Claude with optional MCP tool support
        
        Args:
            messages: Chat message history
            tools: Available MCP tools (optional)
            mcp_client: MCP client for tool execution (optional)
            step_callback: Optional callback for step-by-step updates
        """
        
        # Reset loop detection for new conversation
        self._reset_loop_detection()
        
        # Convert tools and messages to Anthropic format
        anthropic_tools = self._convert_mcp_tools_to_anthropic(tools) if tools else []
        anthropic_messages = self._convert_messages_to_anthropic(messages)
        
        # System message based on MCP availability
        has_mcp = bool(mcp_client or mcp_manager)
        system_message = SYSTEM_MESSAGE_WITHOUT_MCP if not has_mcp else SYSTEM_MESSAGE_WITH_MCP
        
        
        try:
            # Make initial request to Claude
            kwargs = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": anthropic_messages,
                "system": system_message
            }
            
            if anthropic_tools:
                kwargs["tools"] = anthropic_tools
            
            response = self.client.messages.create(**kwargs)
            
            # Handle tool calls if present
            tool_calls = []
            assistant_content_blocks = []
            
            if hasattr(response, 'content'):
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        # Notify step callback about tool use
                        if step_callback:
                            await step_callback(f"🔧 Calling tool: **{content_block.name}**", "tool_start", {
                                "tool_name": content_block.name,
                                "arguments": content_block.input
                            })
                        
                        # Check for infinite loops
                        if self._is_loop_detected(content_block.name, content_block.input):
                            # Return a message about the loop detection
                            return f"Loop detected: The same tool call '{content_block.name}' with identical arguments was attempted multiple times. Stopping to prevent infinite loop.", tool_calls
                        
                        # Execute MCP tool (if client or manager is available)
                        if mcp_client or mcp_manager:
                            tool_result = await self._call_mcp_tool(
                                content_block.name,
                                content_block.input,
                                mcp_client,
                                mcp_manager
                            )
                        else:
                            # No MCP client or manager available
                            tool_result = {
                                "type": "text",
                                "text": "Tool execution not available - no MCP server connected"
                            }
                        
                        # Notify step callback about tool completion
                        if step_callback:
                            await step_callback(f"✅ Tool **{content_block.name}** completed", "tool_complete", {
                                "tool_name": content_block.name,
                                "result": tool_result
                            })
                        
                        tool_calls.append({
                            "id": content_block.id,
                            "name": content_block.name,
                            "arguments": content_block.input,
                            "result": tool_result
                        })
                        
                        # Store the assistant's tool_use block
                        assistant_content_blocks.append({
                            "type": "tool_use",
                            "id": content_block.id,
                            "name": content_block.name,
                            "input": content_block.input
                        })
                    elif content_block.type == "text":
                        # Notify step callback about assistant reasoning
                        if step_callback and content_block.text.strip():
                            await step_callback(content_block.text, "reasoning", {})
                        
                        # Store text content from assistant
                        assistant_content_blocks.append({
                            "type": "text",
                            "text": content_block.text
                        })
            
            # If there were tool calls, create proper conversation flow
            if tool_calls:
                # Add the assistant's message with tool_use blocks
                anthropic_messages.append({
                    "role": "assistant",
                    "content": assistant_content_blocks
                })
                
                # Add tool results as user messages
                for tool_call in tool_calls:
                    anthropic_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call["id"],
                                "content": tool_call["result"]["text"]
                            }
                        ]
                    })
                
                # Allow multiple rounds of tool calling
                current_round = 0
                all_tool_calls = tool_calls.copy()
                
                while current_round < self.max_tool_rounds:
                    # Get response from Claude with tool results
                    follow_up_response = self.client.messages.create(
                        model=self.model,
                        max_tokens=4096,
                        messages=anthropic_messages,
                        system=system_message,
                        tools=anthropic_tools if anthropic_tools else None
                    )
                    
                    # Check if Claude made more tool calls
                    follow_up_tool_calls = []
                    follow_up_content_blocks = []
                    
                    if hasattr(follow_up_response, 'content'):
                        for content_block in follow_up_response.content:
                            if content_block.type == "tool_use":
                                # Notify step callback about additional tool use
                                if step_callback:
                                    await step_callback(f"🔧 Calling additional tool: **{content_block.name}**", "tool_start", {
                                        "tool_name": content_block.name,
                                        "arguments": content_block.input
                                    })
                                
                                # Check for infinite loops
                                if self._is_loop_detected(content_block.name, content_block.input):
                                    # Return current state with loop detection message
                                    return f"Loop detected: The same tool call '{content_block.name}' with identical arguments was attempted multiple times. Stopping to prevent infinite loop.", all_tool_calls
                                
                                # Execute additional MCP tool (if client or manager is available)
                                if mcp_client or mcp_manager:
                                    tool_result = await self._call_mcp_tool(
                                        content_block.name,
                                        content_block.input,
                                        mcp_client,
                                        mcp_manager
                                    )
                                else:
                                    # No MCP client or manager available
                                    tool_result = {
                                        "type": "text",
                                        "text": "Tool execution not available - no MCP server connected"
                                    }
                                
                                # Notify step callback about tool completion
                                if step_callback:
                                    await step_callback(f"✅ Additional tool **{content_block.name}** completed", "tool_complete", {
                                        "tool_name": content_block.name,
                                        "result": tool_result
                                    })
                                
                                follow_up_tool_calls.append({
                                    "id": content_block.id,
                                    "name": content_block.name,
                                    "arguments": content_block.input,
                                    "result": tool_result
                                })
                                
                                follow_up_content_blocks.append({
                                    "type": "tool_use",
                                    "id": content_block.id,
                                    "name": content_block.name,
                                    "input": content_block.input
                                })
                            elif content_block.type == "text":
                                # Notify step callback about additional reasoning
                                if step_callback and content_block.text.strip():
                                    await step_callback(content_block.text, "reasoning", {})
                                
                                follow_up_content_blocks.append({
                                    "type": "text",
                                    "text": content_block.text
                                })
                    
                    # If no more tool calls, return the final response
                    if not follow_up_tool_calls:
                        final_text = ""
                        if hasattr(follow_up_response, 'content'):
                            for content_block in follow_up_response.content:
                                if content_block.type == "text":
                                    final_text += content_block.text
                        
                        return final_text, all_tool_calls
                    
                    # Add follow-up tool calls to conversation and continue
                    all_tool_calls.extend(follow_up_tool_calls)
                    
                    # Add assistant message with new tool calls
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": follow_up_content_blocks
                    })
                    
                    # Add tool results as user messages
                    for tool_call in follow_up_tool_calls:
                        anthropic_messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_call["id"],
                                    "content": tool_call["result"]["text"]
                                }
                            ]
                        })
                    
                    current_round += 1
                
                # If we hit the max rounds, get final response
                if current_round >= self.max_tool_rounds:
                    final_response = self.client.messages.create(
                        model=self.model,
                        max_tokens=4096,
                        messages=anthropic_messages,
                        system=system_message
                    )
                    
                    final_text = ""
                    if hasattr(final_response, 'content'):
                        for content_block in final_response.content:
                            if content_block.type == "text":
                                final_text += content_block.text
                    
                    return final_text, all_tool_calls
            else:
                # Extract text from initial response (no tool calls)
                response_text = ""
                if hasattr(response, 'content'):
                    for content_block in response.content:
                        if content_block.type == "text":
                            response_text += content_block.text
                
                return response_text, []
        
        except Exception as e:
            error_msg = f"LLM request failed: {str(e)}"
            return error_msg, []

def create_llm_connector(provider: str, api_key: str, **kwargs) -> LLMConnector:
    """Factory function to create LLM connectors"""
    if provider.lower() == "anthropic":
        model = kwargs.get("model", "claude-3-5-sonnet-20241022")
        return AnthropicConnector(api_key=api_key, model=model)
    elif provider.lower() == "openai":
        model = kwargs.get("model", "gpt-4o")
        base_url = kwargs.get("base_url", "https://api.openai.com/v1")
        return OpenAIConnector(api_key=api_key, model=model, base_url=base_url)
    elif provider.lower() == "ollama":
        model = kwargs.get("model", "llama3.1:8b")
        base_url = kwargs.get("base_url", "http://localhost:11434")
        return OllamaConnector(api_key=api_key, model=model, base_url=base_url)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Supported providers: anthropic, openai, ollama")