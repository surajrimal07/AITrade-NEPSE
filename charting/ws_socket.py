import json
import threading
import websocket
from global_var import JWT_TOKEN

def on_message(ws, message):
    data = json.loads(message)
    if 'portfolio' in data:
        tradingdashboard_portfolio = [portfolio for portfolio in data['portfolio'] if portfolio['name'] == 'tradingdashboard']
        if tradingdashboard_portfolio:
            return tradingdashboard_portfolio[0]
        else:
            print("Tradingdashboard portfolio not found in the response")

def on_close(ws):
    print("Connection closed")

def on_open(ws):
    print("Connected to socket successfully")

def fetch_live_portfolio():
    global JWT_TOKEN
    def start_websocket():
#        token = "string:0d4bdd8c8dd5469a1a7eb165637d3cfd9a02f61e67e071e1e4a7172ac412673bbb6c9fe2d2bcea536c186db25b4361f5280718f4c5b11c3ad7d25923eec3222dc549b359f8c2b1c07d37385b2bc98dba1c8a6f4a9bdce09f1847933c9e5c78a455ab023c68e6db650a9f1d560eda503d281a63f28e2700df271ba3553d93438cb277c7d49dc901e06df90de45493abd567e39cd7f60b954bc88c832ba2a71ff10565494de0c37ad983"
        ws = websocket.WebSocketApp(
            "wss://localhost:8081/?room=portfolio&jwt=" + JWT_TOKEN ,
            on_open=on_open,
            on_message=on_message,
            on_close=on_close
        )
        ws.run_forever()

    thread = threading.Thread(target=start_websocket)
    thread.daemon = True
    thread.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Terminating WebSocket connection")

if __name__ == "__main__":
    fetch_live_portfolio()
