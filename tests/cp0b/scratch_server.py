"""A throwaway FastMCP server used only to validate langchain.mcp's contracts
in CP0b. Not part of juena_core's public surface — nothing here ships.
"""

from pydantic import BaseModel

from fastmcp import FastMCP

mcp = FastMCP("cp0b-scratch")


class Sum(BaseModel):
    total: int


@mcp.tool
def echo(text: str) -> str:
    return text


@mcp.tool
def add(a: int, b: int) -> Sum:
    return Sum(total=a + b)


if __name__ == "__main__":
    import os

    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=int(os.environ["CP0B_PORT"]),
        path="/mcp",
        show_banner=False,
    )
