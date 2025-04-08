# Claude LLM Tools

A library for auto-generating JSON Schema and dispatching tool calls in applications
that use the Anthropic API to interact with Claude.

## Installation
```shell
pip install claude-llm-tools
```

## Usage

```python
import anthropic
import claude_llm_tools


@claude_llm_tools.tool
def add_numbers(req: claude_llm_tools.Request, a: int, b: int) -> int:
    """
    Add two numbers together and return the result.

    Args:
        a: The first number
        b: The second number

    Returns:
        The sum of a and b
    """
    print(req.tool_use.id)
    return a + b


client = anthropic.Anthropic()

# You can implement the Anthropic text editor contract
claude_llm_tools.add_tool(..., tool_type='text_editor_20250124')

message = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=1000,
    temperature=1,
    system="You are a world-class poet. Respond only with short poems about math.",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Why is the ocean salty?"
                }
            ]
        }
    ],
    tools=claude_llm_tools.tools()
)

for block in message.content:
    match block.type:
        case 'text':
            ...
        case 'tool_use':
            result = await claude_llm_tools.dispatch(block)
```
