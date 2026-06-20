# server_asgi.py
# ASGI server using FastAPI and uvicorn
# Run with: uvicorn server_asgi:app --host 0.0.0.0 --port 8000

import os
import time
import uuid
import asyncio
import json
from typing import List, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Hedge Fund API ASGI Stub")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to import user integration adapter
INTEGRATED = False
try:
    # hf_integration must expose:
    # get_portfolio() -> dict
    # get_market_quotes(symbols: List[str]) -> List[dict]
    # list_recent_orders(limit:int) -> List[dict]
    # place_order(payload: dict) -> dict
    import hf_integration as hf
    INTEGRATED = True
except Exception:
    INTEGRATED = False
    # Fallback mocks
    def get_portfolio():
        return {
            "accountId": "demo-acc",
            "cash": 125000.50,
            "totalEquity": 312345.12,
            "unrealizedPnl": 2345.67,
            "positions": [
                {"symbol": "AAPL", "qty": 100, "avgPrice": 170.5, "marketPrice": 172.6, "unrealizedPnl": 210.0},
                {"symbol": "TSLA", "qty": 10, "avgPrice": 650.0, "marketPrice": 660.0, "unrealizedPnl": 100.0}
            ],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

    def get_market_quotes(symbols: List[str]):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return [{"symbol": s, "bid": 100.0, "ask": 100.5, "last": 100.25, "timestamp": now} for s in symbols]

    _ORDERS: List[Dict[str, Any]] = []
    def list_recent_orders(limit: int = 50):
        return list(reversed(_ORDERS[-limit:]))

    def place_order(payload: Dict[str, Any]):
        order = {
            "orderId": str(uuid.uuid4()),
            "symbol": payload.get("symbol"),
            "qty": payload.get("qty"),
            "side": payload.get("side"),
            "type": payload.get("type", "Market"),
            "timeInForce": payload.get("timeInForce", "GTC"),
            "status": "New",
            "filledQty": 0,
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        _ORDERS.append(order)
        return order

# Adapter functions that call either hf_integration or mocks
def _get_portfolio():
    return hf.get_portfolio() if INTEGRATED else get_portfolio()

def _get_market_quotes(symbols: List[str]):
    return hf.get_market_quotes(symbols) if INTEGRATED else get_market_quotes(symbols)

def _list_recent_orders(limit: int = 50):
    return hf.list_recent_orders(limit) if INTEGRATED else list_recent_orders(limit)

def _place_order(payload: Dict[str, Any]):
    return hf.place_order(payload) if INTEGRATED else place_order(payload)

# REST endpoints
@app.get("/v1/portfolio")
async def api_portfolio():
    try:
        return _get_portfolio()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/marketdata")
async def api_marketdata(symbols: str = ""):
    if not symbols:
        return {"data": []}
    symbols_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    try:
        quotes = _get_market_quotes(symbols_list)
        return {"data": quotes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/orders")
async def api_orders(limit: int = 50):
    try:
        orders = _list_recent_orders(limit)
        return {"orders": orders}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/orders", status_code=201)
async def api_place_order(request: Request):
    body = await request.json()
    if not body or not body.get("symbol") or not body.get("qty") or not body.get("side"):
        raise HTTPException(status_code=400, detail="symbol, qty and side are required")
    try:
        order = _place_order(body)
        # broadcast to websocket clients via broadcaster
        await broadcaster.broadcast_json({"type": "order_update", "payload": order})
        return order
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "integrated": INTEGRATED}

# WebSocket manager and broadcaster
class Broadcaster:
    def __init__(self):
        self.connections: List[WebSocket] = []
        self.lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self.lock:
            self.connections.append(ws)

    async def disconnect(self, ws: WebSocket):
        async with self.lock:
            if ws in self.connections:
                self.connections.remove(ws)

    async def broadcast_json(self, message: Dict[str, Any]):
        payload = json.dumps(message)
        async with self.lock:
            conns = list(self.connections)
        for ws in conns:
            try:
                await ws.send_text(payload)
            except Exception:
                # ignore send errors; cleanup will happen on disconnect
                pass

broadcaster = Broadcaster()

@app.websocket("/v1/stream")
async def websocket_endpoint(ws: WebSocket):
    # Accept connection and optionally validate token query param
    token = ws.query_params.get("token")
    await broadcaster.connect(ws)
    try:
        # send initial portfolio snapshot
        try:
            await ws.send_text(json.dumps({"type": "portfolio_update", "payload": _get_portfolio()}))
        except Exception:
            pass
        while True:
            # keep connection alive; client may send pings
            data = await ws.receive_text()
            # echo or ignore; no action required
            # if client sends "subscribe:SYMBOL" you could implement per-client subscriptions here
            await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        await broadcaster.disconnect(ws)
    except Exception:
        await broadcaster.disconnect(ws)

# Background task to emit market ticks periodically
async def market_tick_task(interval: float = 2.0):
    symbols = ["AAPL", "TSLA", "MSFT"]
    while True:
        try:
            ticks = _get_market_quotes(symbols)
            for t in ticks:
                await broadcaster.broadcast_json({"type": "market_tick", "payload": t})
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(interval)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    app.state.tick_task = asyncio.create_task(market_tick_task(2.0))

@app.on_event("shutdown")
async def shutdown_event():
    task = app.state.tick_task
    if task:
        task.cancel()
        try:
            await task
        except Exception:
            pass