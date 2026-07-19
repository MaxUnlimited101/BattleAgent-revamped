"""Optional live WebSocket bridge for the browser battle visualizer.

The simulation is synchronous, so the broadcaster runs its own asyncio event loop in a
daemon thread. :meth:`LiveBroadcaster.publish` is called from the simulation thread and hands
the message to that loop in a thread-safe way. Every event is also retained in a backlog, so a
browser that connects mid-run is first replayed the full history and then receives new events
live.

The dependency (``websockets``) is imported lazily and guarded: if it is missing the broadcaster
degrades to a no-op and the simulation is unaffected. Enable the feature by constructing a
:class:`LiveBroadcaster` (the :class:`~utils.event_log.EventLogger` does this when the
``BATTLE_LIVE_VIZ`` environment variable is set).
"""
import asyncio
import logging
import threading

logger = logging.getLogger(__name__)


class LiveBroadcaster:
    """Broadcasts JSON event lines to connected WebSocket clients from a background thread."""

    def __init__(self, host="127.0.0.1", port=8765):
        self.host = host
        self.port = port
        self._loop = None
        self._queue = None
        self._clients = set()
        self._backlog = []
        self._ready = threading.Event()
        self._enabled = False

        try:
            import websockets  # noqa: F401
        except ImportError:
            logger.warning("live viz requested but 'websockets' is not installed; "
                           "disabling live streaming (pip install websockets)")
            return

        self._thread = threading.Thread(target=self._run, name="battle-live-viz", daemon=True)
        self._thread.start()
        # Wait briefly for the server to bind so early events are not silently dropped.
        if self._ready.wait(timeout=5.0) and self._enabled:
            logger.info("live viz streaming on ws://%s:%d", self.host, self.port)
        else:
            logger.warning("live viz server did not start within timeout; disabling")

    def _run(self):
        import websockets

        async def main():
            self._queue = asyncio.Queue()
            try:
                async with websockets.serve(self._handler, self.host, self.port):
                    self._enabled = True
                    self._ready.set()
                    await self._broadcast_worker()
            except OSError as exc:
                logger.warning("live viz could not bind %s:%d (%s); disabling",
                               self.host, self.port, exc)
                self._ready.set()

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(main())
        finally:
            self._loop.close()

    async def _handler(self, websocket):
        # Replay history so a client joining mid-run sees the full battle state, then stream live.
        self._clients.add(websocket)
        try:
            for msg in list(self._backlog):
                await websocket.send(msg)
            await websocket.wait_closed()
        finally:
            self._clients.discard(websocket)

    async def _broadcast_worker(self):
        while True:
            msg = await self._queue.get()
            if not self._clients:
                continue
            # Copy the set; a send failure must not abort delivery to the other clients.
            results = await asyncio.gather(
                *(ws.send(msg) for ws in list(self._clients)), return_exceptions=True
            )
            for ws, result in zip(list(self._clients), results):
                if isinstance(result, Exception):
                    self._clients.discard(ws)

    def publish(self, msg):
        """Thread-safe: enqueue a JSON string for broadcast. No-op if the server never started."""
        if not self._enabled or self._loop is None:
            return
        self._backlog.append(msg)

        def _enqueue():
            if self._queue is not None:
                self._queue.put_nowait(msg)

        self._loop.call_soon_threadsafe(_enqueue)

    def close(self):
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
