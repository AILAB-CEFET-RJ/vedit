from vidgear.gears import CamGear
import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime

def annotate_text(text, frame, pos_x, pos_y, font_size):

    shape = np.zeros_like(frame, np.uint8)

    width= int(len(text) * font_size * 17.5)
    height = int(font_size * 15)

    cv2.rectangle(shape, (max(pos_x-10, 0), max(int(pos_y-(font_size*30)), 0)), (pos_x+width, pos_y+height), (1,1,1), -1)
    mask = shape.astype(bool)
    frame[mask] = cv2.addWeighted(frame, 0.4, shape, 0.6, 1)[mask]

    cv2.putText(
    frame,
    text,
    (pos_x, pos_y),
    cv2.FONT_HERSHEY_SIMPLEX,
    font_size,
    (0, 255, 0),
    2,
    )

    return frame

def annotate_bounding_box(ship_class, prob, dim, frame):

    if ship_class == 0:
        border_color = (255, 0, 0)
        font_color = (255,255,255)
    elif ship_class == 1:
        border_color = (0, 255, 255)
        font_color = (255,255,255)
    elif ship_class == 2:
        border_color = (255, 255, 0)
        font_color = (255,255,255)
    elif ship_class == 3:
        border_color = (0, 255, 0)
        font_color = (255,255,255)
    elif ship_class == 4:
        border_color = (0, 0, 255)
        font_color = (255,255,255)

    ship_dict = {
        0: "Cargo",
        1: "Carrier",
        2: "Cruise",
        3: "Military",
        4: "Tanker"
    }

    font_size = 0.8
    class_identifier = f"{ship_dict[int(ship_class)]}: {(prob*100):.2f}%"

    ann_frame1 = cv2.rectangle(frame, (int(dim[0]),int(dim[1])), (int(dim[2]),int(dim[3])), border_color, thickness=3)
    ann_frame2 = cv2.rectangle(ann_frame1, (int(dim[0]),int(dim[1])), (int(dim[0] + (len(class_identifier) * (font_size * 20))),int(dim[1] + (font_size * 40))), border_color, thickness=-1)

    ann_frame3 = cv2.putText(
    ann_frame2,
    class_identifier,
    (int(dim[0] + (font_size * 10)), int(dim[1] + (font_size * 30))),
    cv2.FONT_HERSHEY_SIMPLEX,
    font_size,
    font_color,
    2,
    )

    return ann_frame3

    

def main():

    model = YOLO('models/vedit-std_v1.2.pt')

    stream_url = input("Insira url do video/stream a ser analisado: ")
    stream = CamGear(source=stream_url, stream_mode=True, logging=True).start()

    while True:
        rec_input = input("Deseja registrar a detecção como video? (sim/nao)")

        if rec_input.lower() in ["sim","s","yes","y"]:
            recording = True
            print("Gravacao habilitada, esta sera armazenada na pasta \'rec\'.")
            break
        elif rec_input.lower() in ["nao","não","n","no"]:
            recording = False
            print('Gravacao desabilitada.')
            break
        else:
            print('Input invalido')



    frame_cur = 0

    frame = stream.read()

    if recording:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid_size = (frame.shape[1],frame.shape[0])
        #Inicia um a gravação de um video
        out_vid = cv2.VideoWriter(f'rec/detection_output-{(datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")}.mp4',fourcc, 20.0, vid_size)

    pred_frame = 15

    while True:

        if frame is None:
            print('END OF STREAM')
            break

        frame_cur += 1

        #CLASSES: Standard YOLOV8:{8: 'ship'} 
        #Ship Model:{0: 'Cargo', 1: 'Carrier', 2: 'Cruise', 3: 'Military', 4: 'Tanker'}
        if pred_frame >= 15:
            results = model.predict(frame, stream=True, conf=0.3)
            pred_frame = 0
        pred_frame += 1

        results = list(results)

        #Desenha bounding boxes da deteccao
        annotated_frame = frame
        for index, it in enumerate(results[0].boxes.cls):
            annotated_frame = annotate_bounding_box(it, results[0].boxes.conf[index], results[0].boxes.xyxy[index], annotated_frame)

        #Metodo antigo para gerar bounding box na tela
        #annotated_frame = results[0].plot()
        
        #Anote o número de navios detectados no frame
        annotated_frame = annotate_text(('Navios: ' + str(len(results[0].boxes.cls))),
                    annotated_frame,
                    20, 
                    annotated_frame.shape[0] - 40,
                    1.2
                    )
        
        #Anote o número do frame atual
        annotated_frame = annotate_text(("Quadro atual: "+ str(frame_cur)),
                    annotated_frame,
                    20, 
                    40,
                    1.2
                    )
        

        #Exiba o frame com anotações
        cv2.imshow('Detection Results', annotated_frame)

        #Registre o frame com anotações no vídeo
        if recording:
            out_vid.write(cv2.resize(annotated_frame, vid_size))


        #Interrompa a execução pressionando a tecla 'Q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print('EARLY STOP')
            break

        frame = stream.read()

    if recording:
        out_vid.release()

    cv2.destroyAllWindows()
    stream.stop()


if __name__ == "__main__":
    main()