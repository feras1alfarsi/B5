# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %% [markdown]
# # Day 3 - Function Calling with Ollama
#
# Welcome to the Generative AI Course!
#
# In this notebook, you will use Ollama's **functiongemma** model with function calling to build a chat interface over a local database. Functiongemma is specifically trained for accurate function/tool calling, making it ideal for tasks that require structured data retrieval.
#
# **Key Features of Functiongemma:**
# - Specialized for function calling with high accuracy
# - Automatically decides when to use tools based on the request
# - Returns structured tool calls, then explains results in natural language
# - Works best with clear, explicit instructions about available tools
#
# **Prerequisites**:
# - Ollama installed and running locally (see https://ollama.ai)
# - The `functiongemma` model pulled: `ollama pull functiongemma`

# %% [markdown]
# ## Setup
#
# ```bash
# pip install -U -q "ollama"
# ollama pull functiongemma
# ```

# %%
import json
from ollama import chat
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax

from tools import list_tables, describe_table, execute_query
from tools.database import setup_database

# %%
# Create a console instance for rich output
console = Console()

# Model to use
model = "qwen3:1.7b"

# Display welcome banner
console.print(Panel(
    "[bold cyan]Day 3 - Function Calling with Ollama[/bold cyan]\n\n"
    f"[white]Using [bold]{model}[/bold] model with database tools[/white]",
    title="🚀 Welcome",
    border_style="cyan"
))

# %% [markdown]
# ## Create a local database
#
# For this example, you'll create a local SQLite database and add some synthetic data about a computer store.

# %%
setup_database()

# Test one function
tables = list_tables()
console.print(Panel(
    f"[bold green]✓ Database initialized[/bold green]\n"
    f"[cyan]Available tables:[/cyan] {', '.join(tables)}",
    title="Database Setup",
    border_style="green"
))

# %% [markdown]
# ## Register the tools
#
# Import all tools from the tools package. Each tool is defined in its own module with clear docstrings and type hints. Functiongemma uses these to construct the schema that is passed to the model.
#
# **Important for Functiongemma:** The model relies on clear docstrings and type hints to understand:
# - What each tool does
# - What parameters it accepts
# - What it returns
#
# Make sure your tool functions have comprehensive docstrings!

# %%
# Register all tools
db_tools = [list_tables, describe_table, execute_query]

# Create a mapping of function names to functions for easy lookup
tool_map = {tool.__name__: tool for tool in db_tools}

# System instruction optimized for functiongemma
# Functiongemma works best with clear, explicit instructions about tool usage
system_instruction = """You are a helpful chatbot that can interact with an SQL database.
You will take the user's questions and turn them into SQL queries using the tools available.
Once you have the information you need, you will answer the user's question using the data returned.

Use list_tables to see what tables are present, describe_table to understand the schema,
and execute_query to issue an SQL SELECT query."""

# %% [markdown]
# ## Helper function for chat with tool calling
#
# Since Ollama requires manual handling of tool calls, we'll create a helper function that handles the conversation loop.

# %%
def chat_with_tools(messages, conversation_history=None, max_iterations=10):
    """Handle a chat conversation with automatic tool calling.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys (new messages to add)
        conversation_history: Optional existing conversation history to continue from
        max_iterations: Maximum number of tool call iterations to prevent infinite loops
        
    Returns:
        Tuple of (final response message, updated conversation_history)
    """
    if conversation_history is None:
        conversation_history = messages.copy()
    else:
        # Append new messages to existing history
        conversation_history = conversation_history.copy()
        conversation_history.extend(messages)
    
    for iteration in range(max_iterations):
        # Add system instruction if this is the first message
        if iteration == 0 and conversation_history[0]["role"] != "system":
            conversation_history.insert(0, {
                "role": "system",
                "content": system_instruction
            })
        
        # Call the model
        response = chat(model, messages=conversation_history, tools=db_tools)
        
        # Check if the model wants to call a tool
        if response.message.tool_calls:
            # Add the assistant's message with tool calls to history
            conversation_history.append(response.message)
            
            # Execute each tool call
            for tool_call in response.message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                
                # Display tool call with rich formatting
                if isinstance(tool_args, dict):
                    args_str = json.dumps(tool_args, indent=2)
                    syntax_lang = "json"
                else:
                    args_str = str(tool_args)
                    # If it looks like SQL, use SQL syntax highlighting
                    if tool_name == "execute_query" or "SELECT" in args_str.upper():
                        syntax_lang = "sql"
                    else:
                        syntax_lang = "text"
                
                console.print(Panel(
                    Syntax(args_str, syntax_lang, theme="monokai", word_wrap=True),
                    title=f"[bold cyan]🔧 Calling: {tool_name}()[/bold cyan]",
                    border_style="cyan"
                ))
                
                # Get the tool function
                if tool_name not in tool_map:
                    result = json.dumps({"error": f"Unknown tool: {tool_name}"})
                    console.print(f"[bold red]❌ Unknown tool: {tool_name}[/bold red]")
                else:
                    try:
                        # Parse arguments - Ollama may return dict or JSON string
                        if isinstance(tool_args, str):
                            try:
                                args_dict = json.loads(tool_args)
                            except json.JSONDecodeError:
                                # If it's not valid JSON, treat as single string arg
                                args_dict = {"query": tool_args} if tool_name == "execute_query" else {}
                        elif isinstance(tool_args, dict):
                            args_dict = tool_args
                        else:
                            # Fallback: try to convert to dict
                            args_dict = dict(tool_args) if hasattr(tool_args, '__dict__') else {}
                        
                        # Call the tool
                        tool_func = tool_map[tool_name]
                        tool_result = tool_func(**args_dict)
                        
                        # Convert result to JSON string if it's not already
                        if isinstance(tool_result, str):
                            result = tool_result
                        else:
                            result = json.dumps(tool_result, default=str, indent=2)
                        
                        # Display result with rich formatting
                        console.print(Panel(
                            Syntax(result, "json", theme="monokai", word_wrap=True),
                            title="[bold green]✅ Tool Result[/bold green]",
                            border_style="green"
                        ))
                    except Exception as e:
                        result = json.dumps({"error": str(e)})
                        console.print(Panel(
                            f"[bold red]{str(e)}[/bold red]",
                            title="[bold red]❌ Tool Error[/bold red]",
                            border_style="red"
                        ))
                
                # Add tool response to conversation
                conversation_history.append({
                    "role": "tool",
                    "content": result
                })
        else:
            # No tool calls, return the final response
            return response, conversation_history
    
    # If we've hit max iterations, return the last response
    return response, conversation_history

# %% [markdown]
# ## Chat with the Data
#
# We'll use our helper function to handle conversations with automatic tool calling.

# %%
# Initialize conversation
# Example 1: A question that requires exploring the database structure first
messages = [{"role": "user", "content": "What is the cheapest product in the store?"}]

console.print(Panel(
    messages[0]["content"],
    title="[bold blue]💬 User Question[/bold blue]",
    border_style="blue"
))

# Get response with automatic tool calling
# Functiongemma should: list_tables -> describe_table(products) -> execute_query to find cheapest
response, conversation_history = chat_with_tools(messages)

console.print(Panel(
    Markdown(response.message.content),
    title="[bold green]🤖 Model Answer[/bold green]",
    border_style="green"
))

# %% [markdown]
# Ask a follow-up question. The model remembers the context (table structures) from the previous turn.

# %%
# Example 2: Multi-step question requiring multiple tool calls
# This demonstrates functiongemma's ability to chain tool calls

console.print("\n")
console.rule("[bold yellow]Example 2: Complex Query[/bold yellow]", style="yellow")

messages = [{"role": "user", "content": "Show me all products and their prices, sorted from cheapest to most expensive."}]

console.print(Panel(
    messages[0]["content"],
    title="[bold blue]💬 User Question[/bold blue]",
    border_style="blue"
))

response, conversation_history = chat_with_tools(messages)

console.print(Panel(
    Markdown(response.message.content),
    title="[bold green]🤖 Model Answer[/bold green]",
    border_style="green"
))

# %%
# Example 3: Follow-up question that leverages conversation context
# Functiongemma should remember the table structure from previous calls

console.print("\n")
console.rule("[bold yellow]Example 3: Follow-up Question[/bold yellow]", style="yellow")

# Add the follow-up question to the existing conversation
follow_up = [{"role": "user", "content": "Which staff member has made the most sales?"}]

console.print(Panel(
    follow_up[0]["content"],
    title="[bold blue]💬 User Question[/bold blue]",
    border_style="blue"
))

# Continue the conversation using the previous history
# Functiongemma should remember table structures and can directly query
response, conversation_history = chat_with_tools(follow_up, conversation_history=conversation_history)

console.print(Panel(
    Markdown(response.message.content),
    title="[bold green]🤖 Model Answer[/bold green]",
    border_style="green"
))

# %% [markdown]
# ## Inspecting the Conversation
#
# You can view the conversation history to see the back-and-forth between the model and the tools.

# %%
# Example: Show how to inspect a conversation
# The conversation_history contains the full conversation including tool calls

# Create a table to display conversation history
table = Table(title="Conversation History", show_header=True, header_style="bold magenta")
table.add_column("#", style="dim", width=4)
table.add_column("Role", style="cyan", width=12)
table.add_column("Content", style="white", overflow="fold")

for i, msg in enumerate(conversation_history, 1):
    role = msg.get("role", "unknown")
    content = msg.get("content", "")
    
    # Handle different message types
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        tool_info = "\n".join([
            f"• {tc.function.name}({json.dumps(tc.function.arguments, indent=2)})"
            for tc in msg.tool_calls
        ])
        table.add_row(str(i), "[yellow]assistant[/yellow]", f"[bold]Tool Calls:[/bold]\n{tool_info}")
    elif content:
        # Truncate very long content
        if len(content) > 500:
            display_content = content[:500] + "\n...[truncated]"
        else:
            display_content = content
        
        # Color code by role
        role_style = {
            "system": "[dim]system[/dim]",
            "user": "[blue]user[/blue]",
            "assistant": "[green]assistant[/green]",
            "tool": "[cyan]tool[/cyan]"
        }.get(role, role)
        
        table.add_row(str(i), role_style, display_content)
    else:
        table.add_row(str(i), role, "[dim][Message object][/dim]")

console.print(table)

# %% [markdown]
# ## Best Practices for Functiongemma
#
# Based on how functiongemma is designed, here are tips for getting the best results:
#
# 1. **Clear Tool Descriptions**: Functiongemma relies heavily on docstrings and type hints. Make them descriptive!
# 2. **Explicit Instructions**: Tell the model when to use tools and in what order (e.g., "list tables first, then describe schema")
# 3. **Natural Language After Tools**: Functiongemma is trained to explain tool results in natural language - let it do that
# 4. **Specific Questions**: Ask specific questions that require tool usage rather than generic ones
# 5. **Conversation Context**: Functiongemma can remember table structures from previous calls, reducing redundant tool usage
#
# ## Try Your Own Questions
#
# Here are some example questions you can try to test functiongemma's capabilities:
#
# - "What's the total value of all products in inventory?"
# - "List all customers who bought a laptop"
# - "Which staff member has the most orders?"
# - "Show me products priced between $50 and $200"
# - "What's the average price of all products?"
#
# Notice how these questions require:
# 1. Understanding the database structure (list_tables, describe_table)
# 2. Writing appropriate SQL queries (execute_query)
# 3. Interpreting and explaining the results (natural language response)

# %% [markdown]
# ## Further reading
#
# To learn more about Ollama and function calling:
# - [Ollama Documentation](https://ollama.ai/docs)
# - [Ollama Python Library](https://github.com/ollama/ollama-python)
# - [Function Calling with Ollama](https://github.com/ollama/ollama/blob/main/docs/function-calling.md)
# - [Functiongemma Model Card](https://ollama.com/library/functiongemma)
#
# And stay tuned for day 4, where you will explore using function calling with grounding tools.
