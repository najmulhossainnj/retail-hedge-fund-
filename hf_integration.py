# hf_integration.py
# Adapter that maps your repo functions to the server contract

# Example imports from your repo
# from my_repo.portfolio import snapshot as portfolio_snapshot
# from my_repo.market import quotes_for_symbols
# from my_repo.orders import recent_orders, submit_order

def get_portfolio():
    # return dict matching API contract
    # return portfolio_snapshot()
    return {
        "accountId": "real-acc",
        "cash": 100000.0,
        "totalEquity": 120000.0,
        "unrealizedPnl": 2000.0,
        "positions": [],
        "timestamp": "2026-06-20T12:00:00Z"
    }

def get_market_quotes(symbols):
    # return list of dicts: symbol, bid, ask, last, timestamp
    now = "2026-06-20T12:00:00Z"
    return [{"symbol": s, "bid": 10.0, "ask": 10.5, "last": 10.25, "timestamp": now} for s in symbols]

def list_recent_orders(limit=50):
    return []

def place_order(payload):
    # implement order placement and return created order dict
    return {
        "orderId": "o-demo",
        "symbol": payload.get("symbol"),
        "qty": payload.get("qty"),
        "side": payload.get("side"),
        "status": "New",
        "filledQty": 0,
        "createdAt": "2026-06-20T12:01:00Z"
    }