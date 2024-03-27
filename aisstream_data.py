import asyncio
import websockets
import json
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv('.env')
ais_key = os.getenv('AIS_KEY')

async def connect_ais_stream():

    async with websockets.connect("wss://stream.aisstream.io/v0/stream") as websocket:
        #TROCAR CHAVE API
        subscribe_message = {"APIKey": ais_key , "BoundingBoxes": [[[-23.997, -46.316], [-23.991, -46.302]]]} #[[[-23.996, -46.31], [-23.990, -43.3]]]}

        subscribe_message_json = json.dumps(subscribe_message)
        await websocket.send(subscribe_message_json)

        async for message_json in websocket:
            message = json.loads(message_json)
            message_type = message["MessageType"]

            if message_type == "PositionReport":
                # the message parameter contains a key of the message type which contains the message itself
                ais_message = message['Message']['PositionReport']
                print(f"PR: [{datetime.now()}] ShipId: {ais_message['UserID']} Latitude: {ais_message['Latitude']} Longitude: {ais_message['Longitude']}")

            if message_type == "ShipStaticData":
                # the message parameter contains a key of the message type which contains the message itself
                ais_message = message['Message']['ShipStaticData']
                print(f"SSD: [{datetime.now()}] ShipId: {ais_message['UserID']} Name:{ais_message['Name']} Destination:{ais_message['Destination']} ETA:{ais_message['Eta']} Valid:{ais_message['Valid']}")

            if message_type == "ExtendedClassBPositionReport":
                # the message parameter contains a key of the message type which contains the message itself
                ais_message = message['Message']['ExtendedClassBPositionReport']
                print(f"ECBPR: [{datetime.now()}] ShipId: {ais_message['UserID']} Name:{ais_message['Name']} Type:{ais_message['Type']} Latitude: {ais_message['Latitude']} Longitude: {ais_message['Longitude']} Valid:{ais_message['Valid']}")

if __name__ == "__main__":
    asyncio.run(connect_ais_stream())