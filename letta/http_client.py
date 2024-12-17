from typing import Optional, Union, Dict, Any
import httpx
from letta.errors import LLMError
from letta.utils import printd

async def make_async_request(
    method: str,
    url: str,
    headers: Dict[str, str],
    json_data: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """Make an async HTTP request and return JSON response"""
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            if method.upper() == "POST":
                response = await client.post(url, json=json_data, files=files, headers=headers)
            elif method.upper() == "GET":
                response = await client.get(url, headers=headers)
            elif method.upper() == "DELETE":
                response = await client.delete(url, headers=headers)
            elif method.upper() == "PUT":
                response = await client.put(url, json=json_data, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            printd(f"HTTP request failed: {str(e)}")
            raise LLMError(f"HTTP request failed: {str(e)}")

async def make_async_post_request(
    url: str,
    headers: Dict[str, str],
    json_data: Dict[str, Any],
    files: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """Convenience wrapper for POST requests"""
    return await make_async_request("POST", url, headers, json_data, files, timeout)

async def make_async_delete_request(
    url: str,
    headers: Dict[str, str],
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """Convenience wrapper for DELETE requests"""
    return await make_async_request("DELETE", url, headers, timeout=timeout)