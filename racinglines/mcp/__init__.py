"""
The MCP server (docs/mcp.md): the app's data and simulations for a chat client, over the same functions the
web app and the CLI use. `racinglines mcp` starts it; nothing else in the app imports this package.

    page.py     one envelope for every tabular result: row cap, byte cap, paging, JSON-safe values
    tools.py    the tools as plain functions (conn, args) -> dict, testable without a client
    server.py   the MCP registration (tools, resources, instructions) and the transports
"""
