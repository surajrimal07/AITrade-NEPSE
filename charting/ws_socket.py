import asyncio
import websockets
import json

async def connect_websocket():
    uri = "wss://localhost:8081?email=suraj@rimal.com&password=111111"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            if data.get('type') == 'index':
                index_data = data.get('data')
                if index_data:
                    time = index_data.get('time')
                    open_price = index_data.get('open')
                    high_price = index_data.get('high')
                    low_price = index_data.get('low')
                    close_price = index_data.get('close')
                    print(f"time: {time}, open: {open_price}, high: {high_price}, low: {low_price}, close: {close_price}")

asyncio.run(connect_websocket())
