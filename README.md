# Projects

This is a monorepo with so many AI Projects in Python. Each Project lives separately in its own folder

## Running a Project
- To run any project, `cd project/project-name`
- Run the project with the command `uv run uvicorn src.main:app --reload`
- The command above will start the server and monitor for changes in that directory


## Add Virtual Environment To A Project
- Run `uv venv` to add Virtual environment to a project.

## Activating Virtual Environment After Starting New Terminal
- Since project makes use of `uv`, just run `uv run` and the virtual environment will be activated automatically
- In case .venv doesn't exist yet, we create it with `uv sync` then run `source .venv/bin/activate`            

## Run MCP Inspector
PYTHONPATH=. uv run fastmcp dev inspector src/main.py:stock_mcp
-- Working
(ai-projects) martin@Martin ai-projects % cd mcp_servers
(ai-projects) martin@Martin mcp_servers % PYTHONPATH=. uv run fastmcp dev inspector src/main.py:stock_mcp