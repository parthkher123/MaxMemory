"""Standalone ingestion worker.

The MCP server runs these loops in-process, which is fine for one instance.
Run this separately when ingestion volume needs its own box, or when you want
extraction to keep draining while the server is restarting.

    python scripts/worker.py
"""

from __future__ import annotations

import asyncio
import logging
import signal

from memory_mcp.memory import MemoryLayer

log = logging.getLogger("worker")


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    layer = MemoryLayer()
    await layer.setup()
    layer.start_workers()
    log.info("worker running: draining episodes and outbox")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows
            signal.signal(sig, lambda *_: stop.set())

    try:
        await stop.wait()
    finally:
        log.info("shutting down")
        await layer.close()


if __name__ == "__main__":
    asyncio.run(main())
