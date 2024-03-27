from vidgear.gears import CamGear
import cv2
from ultralytics import YOLO
import numpy as np

#Diferentes modelos do YOLOV8 (ordem crescente de peso/eficacia)
#model = YOLO('models/yolov8n.pt') # Modelo menos pesado/eficaz
#model = YOLO('models/yolov8s.pt') # 
#model = YOLO('models/yolov8m.pt') # Modelo atual
#model = YOLO('models/yolov8l.pt') # 
#model = YOLO('models/yolov8x.pt') # Modelo mais pesado/eficaz

#Diferentes vídeos para teste
# 'https://www.youtube.com/watch?v=CubAd2gt4rU' #Navio Royalty Free
# 'https://www.youtube.com/watch?v=8WD3lAVvbHo' #Navio Rodando
# 'https://www.youtube.com/watch?v=uDOTRV-chaE' #Cruseiro




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


def main():

    model = YOLO('models/yolov8m.pt')

    recording = False
    stream_url = input("Insira url do video/stream a ser analisado: ")
    stream = CamGear(source=stream_url, stream_mode=True, logging=True).start()

    frame_cur = 0

    frame = stream.read()

    if recording:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid_size = (frame.shape[1],frame.shape[0])
        #Inicia um a gravação de um video
        out_vid = cv2.VideoWriter('rec/porto_de_santos.mp4',fourcc, 20.0, vid_size)

    while frame_cur < 54000:

        if frame is None:
            print('END OF STREAM')
            break

        frame_cur += 1

        #CLASSES: Standard YOLOV8:{8: 'ship'} 
        #Ship Model:{0: 'container', 1: 'cruise', 2: 'fish-b', 3: 'sail boat', 4: 'warship'}
        results = model.predict(frame, stream=True, conf=0.3, classes=[8])

        results = list(results)

        annotated_frame = results[0].plot()
        
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