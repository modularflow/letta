import json
import re
from typing import List, Optional, Union

from letta.llm_api.helpers import make_post_request
from letta.schemas.message import Message
from letta.schemas.openai.chat_completion_request import ChatCompletionRequest, Tool
from letta.schemas.openai.chat_completion_response import (
    ChatCompletionResponse,
    Choice,
    FunctionCall,
)
from letta.schemas.openai.chat_completion_response import (
    Message as ChoiceMessage,  # NOTE: avoid conflict with our own Letta Message datatype
)
from letta.schemas.openai.chat_completion_response import ToolCall, UsageStatistics
from letta.utils import get_utc_time, smart_urljoin
import httpx

BASE_URL = "https://api.anthropic.com/v1"


# https://docs.anthropic.com/claude/docs/models-overview
# Sadly hardcoded
MODEL_LIST = [
    {
        "name": "claude-3-opus-20240229",
        "context_window": 200000,
    },
    {
        "name": "claude-3-sonnet-20240229",
        "context_window": 200000,
    },
    {
        "name": "claude-3-haiku-20240307",
        "context_window": 200000,
    },
]

DUMMY_FIRST_USER_MESSAGE = "User initializing bootup sequence."


async def anthropic_get_model_list_async(url: str, api_key: Union[str, None]) -> List[dict]:
    """Get list of available models from Anthropic API asynchronously"""
    # Currently returns hardcoded list since Anthropic doesn't have a models endpoint
    return MODEL_LIST


def anthropic_get_model_list(url: str, api_key: Union[str, None]) -> List[dict]:
    """Get list of available models from Anthropic API"""
    # Currently returns hardcoded list since Anthropic doesn't have a models endpoint
    return MODEL_LIST


async def anthropic_get_model_context_window_async(url: str, api_key: Union[str, None], model: str) -> int:
    """Get context window size for a model from Anthropic API asynchronously"""
    for model_dict in await anthropic_get_model_list_async(url=url, api_key=api_key):
        if model_dict["name"] == model:
            return model_dict["context_window"]
    raise ValueError(f"Can't find model '{model}' in Anthropic model list")


def anthropic_get_model_context_window(url: str, api_key: Union[str, None], model: str) -> int:
    """Get context window size for a model from Anthropic API"""
    for model_dict in anthropic_get_model_list(url=url, api_key=api_key):
        if model_dict["name"] == model:
            return model_dict["context_window"]
    raise ValueError(f"Can't find model '{model}' in Anthropic model list")


def convert_tools_to_anthropic_format(tools: List[Tool]) -> List[dict]:
    """See: https://docs.anthropic.com/claude/docs/tool-use

    OpenAI style:
      "tools": [{
        "type": "function",
        "function": {
            "name": "find_movies",
            "description": "find ....",
            "parameters": {
              "type": "object",
              "properties": {
                 PARAM: {
                   "type": PARAM_TYPE,  # eg "string"
                   "description": PARAM_DESCRIPTION,
                 },
                 ...
              },
              "required": List[str],
            }
        }
      }
      ]

    Anthropic style:
      "tools": [{
        "name": "find_movies",
        "description": "find ....",
        "input_schema": {
          "type": "object",
          "properties": {
             PARAM: {
               "type": PARAM_TYPE,  # eg "string"
               "description": PARAM_DESCRIPTION,
             },
             ...
          },
          "required": List[str],
        }
      }
      ]

      Two small differences:
        - 1 level less of nesting
        - "parameters" -> "input_schema"
    """
    tools_dict_list = []
    for tool in tools:
        tools_dict_list.append(
            {
                "name": tool.function.name,
                "description": tool.function.description,
                "input_schema": tool.function.parameters,
            }
        )
    return tools_dict_list


def merge_tool_results_into_user_messages(messages: List[dict]):
    """Anthropic API doesn't allow role 'tool'->'user' sequences

    Example HTTP error:
    messages: roles must alternate between "user" and "assistant", but found multiple "user" roles in a row

    From: https://docs.anthropic.com/claude/docs/tool-use
    You may be familiar with other APIs that return tool use as separate from the model's primary output,
    or which use a special-purpose tool or function message role.
    In contrast, Anthropic's models and API are built around alternating user and assistant messages,
    where each message is an array of rich content blocks: text, image, tool_use, and tool_result.
    """

    # TODO walk through the messages list
    # When a dict (dict_A) with 'role' == 'user' is followed by a dict with 'role' == 'user' (dict B), do the following
    # dict_A["content"] = dict_A["content"] + dict_B["content"]

    # The result should be a new merged_messages list that doesn't have any back-to-back dicts with 'role' == 'user'
    merged_messages = []
    if not messages:
        return merged_messages

    # Start with the first message in the list
    current_message = messages[0]

    for next_message in messages[1:]:
        if current_message["role"] == "user" and next_message["role"] == "user":
            # Merge contents of the next user message into current one
            current_content = (
                current_message["content"]
                if isinstance(current_message["content"], list)
                else [{"type": "text", "text": current_message["content"]}]
            )
            next_content = (
                next_message["content"]
                if isinstance(next_message["content"], list)
                else [{"type": "text", "text": next_message["content"]}]
            )
            merged_content = current_content + next_content
            current_message["content"] = merged_content
        else:
            # Append the current message to result as it's complete
            merged_messages.append(current_message)
            # Move on to the next message
            current_message = next_message

    # Append the last processed message to the result
    merged_messages.append(current_message)

    return merged_messages


def remap_finish_reason(stop_reason: str) -> str:
    """Remap Anthropic's 'stop_reason' to OpenAI 'finish_reason'

    OpenAI: 'stop', 'length', 'function_call', 'content_filter', null
    see: https://platform.openai.com/docs/guides/text-generation/chat-completions-api

    From: https://docs.anthropic.com/claude/reference/migrating-from-text-completions-to-messages#stop-reason

    Messages have a stop_reason of one of the following values:
        "end_turn": The conversational turn ended naturally.
        "stop_sequence": One of your specified custom stop sequences was generated.
        "max_tokens": (unchanged)

    """
    if stop_reason == "end_turn":
        return "stop"
    elif stop_reason == "stop_sequence":
        return "stop"
    elif stop_reason == "max_tokens":
        return "length"
    elif stop_reason == "tool_use":
        return "function_call"
    else:
        raise ValueError(f"Unexpected stop_reason: {stop_reason}")


def strip_xml_tags(string: str, tag: Optional[str]) -> str:
    if tag is None:
        return string
    # Construct the regular expression pattern to find the start and end tags
    tag_pattern = f"<{tag}.*?>|</{tag}>"
    # Use the regular expression to replace the tags with an empty string
    return re.sub(tag_pattern, "", string)


def convert_anthropic_response_to_chatcompletion(
    response_json: dict,  # REST response from Google AI API
    inner_thoughts_xml_tag: Optional[str] = None,
) -> ChatCompletionResponse:
    """
    Example response from Claude 3:
    response.json = {
        'id': 'msg_01W1xg9hdRzbeN2CfZM7zD2w',
        'type': 'message',
        'role': 'assistant',
        'content': [
            {
                'type': 'text',
                'text': "<thinking>Analyzing user login event. This is Chad's first
    interaction with me. I will adjust my personality and rapport accordingly.</thinking>"
            },
            {
                'type':
                'tool_use',
                'id': 'toolu_01Ka4AuCmfvxiidnBZuNfP1u',
                'name': 'core_memory_append',
                'input': {
                    'name': 'human',
                    'content': 'Chad is logging in for the first time. I will aim to build a warm
    and welcoming rapport.',
                    'request_heartbeat': True
                }
            }
        ],
        'model': 'claude-3-haiku-20240307',
        'stop_reason': 'tool_use',
        'stop_sequence': None,
        'usage': {
            'input_tokens': 3305,
            'output_tokens': 141
        }
    }
    """
    prompt_tokens = response_json["usage"]["input_tokens"]
    completion_tokens = response_json["usage"]["output_tokens"]

    finish_reason = remap_finish_reason(response_json["stop_reason"])

    if isinstance(response_json["content"], list):
        # inner mono + function call
        # TODO relax asserts
        assert len(response_json["content"]) == 2, response_json
        assert response_json["content"][0]["type"] == "text", response_json
        assert response_json["content"][1]["type"] == "tool_use", response_json
        content = strip_xml_tags(string=response_json["content"][0]["text"], tag=inner_thoughts_xml_tag)
        tool_calls = [
            ToolCall(
                id=response_json["content"][1]["id"],
                type="function",
                function=FunctionCall(
                    name=response_json["content"][1]["name"],
                    arguments=json.dumps(response_json["content"][1]["input"], indent=2),
                ),
            )
        ]
    else:
        # just inner mono
        content = strip_xml_tags(string=response_json["content"], tag=inner_thoughts_xml_tag)
        tool_calls = None

    assert response_json["role"] == "assistant", response_json
    choice = Choice(
        index=0,
        finish_reason=finish_reason,
        message=ChoiceMessage(
            role=response_json["role"],
            content=content,
            tool_calls=tool_calls,
        ),
    )

    return ChatCompletionResponse(
        id=response_json["id"],
        choices=[choice],
        created=get_utc_time(),
        model=response_json["model"],
        usage=UsageStatistics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def anthropic_chat_completions_request(
    url: str,
    api_key: str,
    data: ChatCompletionRequest,
    inner_thoughts_xml_tag: Optional[str] = "thinking",
) -> ChatCompletionResponse:
    """https://docs.anthropic.com/claude/docs/tool-use"""

    url = smart_urljoin(url, "messages")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "tools-2024-04-04",
    }
    
    anthropic_data = convert_to_anthropic_format(data, inner_thoughts_xml_tag)
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=anthropic_data, headers=headers)
        response.raise_for_status()
        response_json = response.json()
        return convert_anthropic_response_to_chatcompletion(
            response_json=response_json, 
            inner_thoughts_xml_tag=inner_thoughts_xml_tag
        )
