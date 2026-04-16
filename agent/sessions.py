from agent.mcp_client import MCPClient

_clients: dict[str, MCPClient] = {}


async def get_client(thread_id: str) -> MCPClient:
    client = _clients.get(thread_id)
    if client is None or client._session is None:
        client = MCPClient()
        await client.connect()
        _clients[thread_id] = client
    return client


async def close_session(thread_id: str):
    client = _clients.pop(thread_id, None)
    if client:
        await client.close()
