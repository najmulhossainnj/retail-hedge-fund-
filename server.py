# server.py
# Minimal Flask + Flask-SocketIO server stub for the Hedge Fund Dashboard
# Place this at the repo root and run with: python server.py

import os
import json
import time
import uuid
from threading import Thread, Event

from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from flask_socketio import SocketIO, emit, disconnect

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.environ.get('HF_SECRET', 'dev-secret')
socketio = SocketIO(app, cors_allowed_origins="*")

# Integration points: try to import your repo functions
# Replace these module names with the actual files in your repo if different
try:
    # expected functions:
    # get_portfolio() -> dict
    # get_market_quotes(symbols: list) -> list[dict]
    # list_recent_orders(limit:int) -> list[dict]
    # place_order(payload: dict) -> dict
    from hf_integration import get_portfolio, get_market_quotes, list_recent_orders, place_order
    INTEGRATED = True
except Exception:
    INTEGRATED = False

    # Fallback mock implementations
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

    def get_market_quotes(symbols):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        out = []
        for s in symbols:
            out.append({"symbol": s, "bid": 100.0, "ask": 100.5, "last": 100.25, "timestamp": now})
        return out

    _ORDERS = []
    def list_recent_orders(limit=50):
        return list(reversed(_ORDERS[-limit:]))

    def place_order(payload):
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

# REST endpoints

@app.route("/v1/portfolio", methods=["GET"])
def api_portfolio():
    try:
        p = get_portfolio()
        return jsonify(p)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/v1/marketdata", methods=["GET"])
def api_marketdata():
    symbols = request.args.get("symbols", "")
    symbols_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not symbols_list:
        return jsonify({"data": []})
    try:
        quotes = get_market_quotes(symbols_list)
        return jsonify({"data": quotes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/v1/orders", methods=["GET"])
def api_orders():
    limit = int(request.args.get("limit", 50))
    try:
        orders = list_recent_orders(limit)
        return jsonify({"orders": orders})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/v1/orders", methods=["POST"])
def api_place_order():
    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({"error": "invalid json body"}), 400
    # Basic validation
    if not body.get("symbol") or not body.get("qty") or not body.get("side"):
        return jsonify({"error": "symbol, qty and side are required"}), 400
    try:
        order = place_order(body)
        # emit order_update to connected websocket clients
        socketio.emit("order_update", {"type": "order_update", "payload": order})
        return jsonify(order), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# WebSocket stream
# Clients connect to /v1/stream via SocketIO
@socketio.on("connect")
def ws_connect():
    token = request.args.get("token") or request.headers.get("Authorization")
    # Basic token check placeholder
    if not token:
        # allow anonymous for local testing, but in production require token
        pass
    emit("connected", {"message": "connected"})
    # send initial portfolio snapshot
    try:
        emit("portfolio_update", {"type": "portfolio_update", "payload": get_portfolio()})
    except Exception:
        pass

# Background thread to broadcast market ticks periodically for demo
stop_event = Event()
def market_tick_broadcaster(interval=2.0):
    symbols = ["AAPL", "TSLA", "MSFT"]
    while not stop_event.wait(interval):
        ticks = get_market_quotes(symbols)
        for t in ticks:
            socketio.emit("market_tick", {"type": "market_tick", "payload": t})

# Start background thread when server starts
@socketio.on("connect")
def start_broadcaster():
    # ensure only one thread
    if not hasattr(start_broadcaster, "thread_started"):
        start_broadcaster.thread_started = True
        t = Thread(target=market_tick_broadcaster, daemon=True)
        t.start()

# Graceful shutdown
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "integrated": INTEGRATED})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    try:
        socketio.run(app, host="0.0.0.0", port=port)
    finally:
        stop_event.set()