"""
LLM Wiki Agent - Personal AI Assistant

Main entry point. Starts:
1. FastAPI web server (UI + API)
2. File watcher for knowledge base changes
3. Agent runtime loop
"""

import asyncio
import logging
from pathlib import Path

import uvicorn

from agent.watcher import KnowledgeBaseWatcher
from agent.runtime import AgentRuntime
from agent.tools import KBTools
from models.gateway import ModelGateway
from memory.store import ConversationStore
from knowledge.index import KBIndex
import web.app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("llm_wiki_agent")

KB_DIR = Path("/app/knowledge")
CANON_DIR = Path("/app/canon")


async def main():
    log.info("Starting LLM Wiki Agent...")

    # Initialize all components once
    model_gateway = ModelGateway()
    await model_gateway.test_connection()

    memory_store = ConversationStore()
    memory_store.initialize()

    kb_index = KBIndex(model_gateway=model_gateway)
    kb_index.build_index()

    # Create KB tools with shared index
    kb_tools = KBTools(
        kb_index=kb_index,
        kb_dir=KB_DIR,
        canon_dir=CANON_DIR,
        conversation_store=memory_store,
    )

    # Pass shared components to web app BEFORE server starts
    web.app.set_components(model_gateway, kb_index, memory_store, kb_tools)

    # Watcher with reindex callback — single-file reindex preserves LLM summaries
    def _reindex_kb(file_path: Path):
        kb_index.reindex_file(file_path)

    watcher = KnowledgeBaseWatcher(KB_DIR, CANON_DIR, reindex_callback=_reindex_kb)
    runtime = AgentRuntime(model_gateway, KB_DIR, CANON_DIR, memory_store, kb_index)

    # Start watcher (runs in background thread)
    watcher.start()
    log.info(f"Watching knowledge base at {KB_DIR}")

    # Start web server
    server_task = asyncio.create_task(
        asyncio.to_thread(
            lambda: uvicorn.run(
                "web.app:app",
                host="0.0.0.0",
                port=8080,
                log_level="info",
            )
        )
    )
    log.info("Web server started on port 8080")

    # Run agent loop concurrently with web server
    runtime_task = asyncio.create_task(runtime.run())

    # Wait for the server task (runtime is optional)
    try:
        await server_task
    except Exception as e:
        log.error(f"Server task failed: {e}")

    # Cleanup
    runtime.stop()
    watcher.stop()
    runtime_task.cancel()
    try:
        await runtime_task
    except asyncio.CancelledError:
        pass
    log.info("LLM Wiki Agent shut down")


if __name__ == "__main__":
    asyncio.run(main())
