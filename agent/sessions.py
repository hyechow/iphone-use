from agent.sync_mcp_client import SyncMCPClient

_clients: dict[str, SyncMCPClient] = {}


def get_client(thread_id: str) -> SyncMCPClient:
    client = _clients.get(thread_id)
    if client is None or client._proc is None:
        client = SyncMCPClient()
        client.connect()
        _clients[thread_id] = client
    return client


def close_session(thread_id: str):
    client = _clients.pop(thread_id, None)
    if client:
        client.close()
