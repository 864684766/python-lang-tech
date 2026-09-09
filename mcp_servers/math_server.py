"""
MCP 的「进程 B」（Server）。7-2 的 Client 会把它当子进程拉起。

FastMCP = 官方 SDK 里写 Server 的快捷方式：@mcp.tool 登记函数，
run(stdio) 从 stdin 读 JSON-RPC、往 stdout 写结果。
它不是 Client，也不能拿来连 Figma。Figma 的进程 B 是别人写好的 npx 包。
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("math")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


if __name__ == "__main__":
    mcp.run(transport="stdio")
