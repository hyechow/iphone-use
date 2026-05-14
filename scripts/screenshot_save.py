"""Take a screenshot from iPhone and save to Images/."""

import sys
sys.path.insert(0, ".")
from policy_expr.sync_mcp_client import SyncMCPClient
from datetime import datetime
from pathlib import Path

client = SyncMCPClient()
client.connect()

img = client.screenshot()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = Path("Images") / f"screen_{timestamp}.png"
out_path.write_bytes(img)
print(f"Saved: {out_path} ({len(img)} bytes)")

client.close()
