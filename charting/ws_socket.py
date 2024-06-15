import json
import threading
import websocket
from global_var import *

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
    socketUrl = "wss://api.zorsha.com.np/?room=portfolio&jwt="+JWT_TOKEN
    def start_websocket():
        ws = websocket.WebSocketApp(
            socketUrl ,
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
