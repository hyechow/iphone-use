import asyncio
import base64
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server = StdioServerParameters(
        command="npx", args=["-y", "mirroir-mcp"]
    )

    async with stdio_client(server) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            print("✅ 连接成功")

            # 截图
            print("📸 截图中...")
            result = await session.call_tool("screenshot", {})

            # 打印原始返回，看看结构
            print("返回内容数量:", len(result.content))
            for i, item in enumerate(result.content):
                print(f"  [{i}] type={item.type}", end="")
                if hasattr(item, "text"):
                    print(f"  text={item.text[:100]}")
                elif hasattr(item, "data"):
                    print(f"  data长度={len(item.data)}  mimeType={getattr(item, 'mimeType', '?')}")
                else:
                    print(f"  attrs={vars(item)}")

            # 找到图片内容并保存
            for item in result.content:
                if item.type == "image" and hasattr(item, "data"):
                    img_bytes = base64.b64decode(item.data)
                    out = Path("screenshot.png")
                    out.write_bytes(img_bytes)
                    print(f"\n✅ 图片已保存：{out.resolve()}  ({len(img_bytes)//1024} KB)")
                    break
            else:
                print("\n⚠️  没找到图片内容，完整返回：")
                print(result)

asyncio.run(main())