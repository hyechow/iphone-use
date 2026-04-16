from agent.mcp_client import MCPClient

_clients: dict[str, MCPClient] = {}


async def get_client(session_id: str) -> MCPClient:
    client = _clients.get(session_id)
    if client is None or client._session is None:
        client = MCPClient()
        await client.connect()
        _clients[session_id] = client
    return client


async def close_session(session_id: str):
    client = _clients.pop(session_id, None)
    if client:
        await client.close()
