from utils import annotate_text, save_detection
from fastapi import FastAPI, Response, Query
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
import json
from vidgear.gears import CamGear
import cv2
from ultralytics import YOLO
import asyncio

app = FastAPI()

model = YOLO('models/vedit-std_v1.2.pt')

video_src = "https://www.youtube.com/watch?v=tWUIUDd4DgE" #Porto de Santos
#stream_url = "https://www.youtube.com/watch?v=6IKZS6guYO0" #Harbor Marina
stream = CamGear(source=video_src, stream_mode=True, logging=True).start()


@app.get("/")
async def main_page():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Video Stream</title>
    </head>
    <body>
        <h1>Links</h1>
        <a href=http://localhost:8000/detect/>Detect</a>
        <a href=http://localhost:8000/stream/>Stream</a>
        <h2>Video Stream</h2>
        <img src="http://localhost:8000/stream/" alt="Video Stream" id="video">
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/set_source/")
async def set_source(src: str, is_stream: bool):
    global stream
    #global stream_src
    
    if stream:
        stream.stop()
    stream = CamGear(source=src, stream_mode=is_stream, logging=True).start()
    return {"message": "Source changed"}



# @app.get("/frame/")
# async def get_frame():
#     global stream
#     frame = stream.read()
#     if frame is None:
#         return Response(content="Novo frame indisponivel", status_code=204)
    
#     _, image = cv2.imencode('.jpg', frame)
    
#     return Response(content=image.tobytes(), media_type="image/jpeg")

async def generate_frame():
    global stream
    while True:
        try:
            frame = stream.read()
            if frame is None:
                break
            
            _, image = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + image.tobytes() + b'\r\n')
            await asyncio.sleep(0.03)
        except Exception as exc:
            print(f'Error: {exc}')
            break

@app.get("/stream/")
async def get_stream():
    return StreamingResponse(generate_frame(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/detect/")
async def detect():
    global stream

    frame = stream.read()
    if frame is None:
        return Response(content="Stream ended or no frame available", status_code=204)

    #results = model.predict(frame, stream=True, conf=0.1, imgsz=256, visualize=True)
    results = model.predict(frame, conf=0.1, augment=True, half=True, agnostic_nms=True)

    results = list(results)

    annotated_frame = results[0].plot()

    #Anote o número de navios detectados no frame
    annotated_frame = annotate_text(('Navios: ' + str(len(results[0].boxes.cls))),
                annotated_frame,
                20, 
                annotated_frame.shape[0] - 40,
                1.2
                )

    _, encoded_image = cv2.imencode('.jpg', annotated_frame)

    return Response(content=encoded_image.tobytes(), media_type="image/jpeg")


@app.post("/ship_detect/")
async def ship_detect():
    global stream

    frame = stream.read()
    if frame is None:
        return Response(content="Stream ended or no frame available", status_code=204)
    
    results = model.predict(frame, conf=0.1, augment=True, agnostic_nms=True)

    results = list(results)

    json_results = results[0].tojson()

    data = json.loads(json_results)

    #print(f'RESULTS JSON: {json_results}')
    save_detection(frame, data)

    for index, obj in enumerate(data):
        print(f"Detection {index+1}")
        print(f"Name: {obj['name']}")
        print(f"Class: {obj['class']}")
        print(f"Confidence: {obj['confidence']}")
    
    return JSONResponse(content=json_results)

    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)