"""Agent Runtime

Background-loop entry point. Production tool calling lives in the web chat
path (``src/web/app.py``), which uses Ollama's native ``tool_calls`` protocol
through ``models.gateway.ModelGateway``. This module only provides the
process_task helper for non-streaming, single-turn agent calls (e.g. tests
or scripted automations).
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from models.gateway import ModelGateway
from agent.tools import FileTools, ShellTools, KBTools, build_tool_registry

log = logging.getLogger(__name__)


class AgentRuntime:
    """Background agent runtime loop.

    The web server owns the chat-with-tools path. This class exists so other
    entry points (CLI scripts, scheduled tasks) can issue a single tool-using
    turn without spinning up the SSE pipeline.
    """

    def __init__(
        self,
        model: ModelGateway,
        kb_dir: Path,
        canon_dir: Path,
        memory: Optional["ConversationStore"] = None,
        kb_index: Optional["KBIndex"] = None,
    ):
        self.model = model
        self.kb_dir = kb_dir
        self.canon_dir = canon_dir
        self.memory = memory
        self.kb_index = kb_index
        self.running = False

        self.file_tools = FileTools(kb_dir, canon_dir)
        self.shell_tools = ShellTools()
        self.kb_tools = KBTools(kb_index, kb_dir, canon_dir)
        self._registry = build_tool_registry(self.kb_tools)

    async def run(self):
        """Run the agent loop."""
        self.running = True
        log.info("Agent runtime started")

        while self.running:
            await asyncio.sleep(1)

    def stop(self):
        """Stop the agent."""
        self.running = False
        log.info("Agent runtime stopped")

    def execute_tool(self, name: str, args: dict) -> str:
        """Execute a tool by name. Used by tests and scripted automations.

        Production traffic goes through the web chat path which dispatches
        natively from streamed ``tool_call`` events.
        """
        fn = self._registry.get(name)
        if fn is None:
            # Legacy fallthroughs for the file/shell tools that aren't part
            # of the KB tool registry.
            if name == "read_file":
                path = args.get("path", args.get("0", ""))
                return self.file_tools.read_file(path).output
            if name == "write_file":
                path = args.get("path", args.get("0", ""))
                content = args.get("content", args.get("1", ""))
                return self.file_tools.write_file(path, content).output
            if name == "list_files":
                folder = args.get("folder", args.get("0", "knowledge"))
                return self.file_tools.list_files(folder).output
            if name == "shell":
                cmd = args.get("cmd", args.get("0", ""))
                return self.shell_tools.execute(cmd).output
            return f"Unknown tool: {name}"
        try:
            return fn(**args) if isinstance(args, dict) else fn(*args)
        except TypeError as e:
            return f"Tool error ({name}): bad arguments — {e}"
        except Exception as e:
            return f"Tool error ({name}): {e}"

    async def process_task(self, task: str) -> str:
        """Single-turn agent call. Uses native tool calling end-to-end."""
        messages = [{"role": "user", "content": task}]
        registry_callables = list(self._registry.values())

        for _ in range(3):
            assistant_content = ""
            tool_calls: list[dict] = []
            async for kind, payload in self.model.chat_stream(
                messages, tools=registry_callables, think=False
            ):
                if kind == "content":
                    assistant_content += payload
                elif kind == "tool_call":
                    tool_calls.append(payload)

            assistant_msg: dict = {"role": "assistant", "content": assistant_content}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {"function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in tool_calls
                ]
            messages.append(assistant_msg)

            if not tool_calls:
                return assistant_content

            for tc in tool_calls:
                result = self.execute_tool(tc["name"], tc.get("arguments") or {})
                messages.append({
                    "role": "tool",
                    "tool_name": tc["name"],
                    "content": str(result),
                })

        return messages[-1].get("content", "") if messages else ""