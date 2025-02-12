from vidgear.gears import CamGear
import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import json
import os

ship_dict = {
    0: "Cargo",
    1: "Cruise",
    2: "Military",
    3: "Offshore",
    4: "Passenger",
    5: "Tanker",
    6: "Tug"
}

font_size = 0.5
recording = False

def annotate_text(text, frame, pos_x, pos_y, font_size):
    shape = np.zeros_like(frame, np.uint8)

    width = int(len(text) * font_size * 17.5)
    height = int(font_size * 15)

    cv2.rectangle(
        shape,
        (max(pos_x - 10, 0), max(int(pos_y - (font_size * 30)), 0)),
        (pos_x + width, pos_y + height),
        (1, 1, 1),
        -1,
    )
    mask = shape.astype(bool)
    frame[mask] = cv2.addWeighted(frame, 0.4, shape, 0.6, 1)[mask]

    cv2.putText(
        frame,
        text,
        (pos_x, pos_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (255, 255, 255),
        2,
    )

    return frame

def annotate_bounding_box(ship_class, prob, dim, frame, ship_info={}):
    ship_data_string = ""
    font_color = (0, 0, 0)  # Cor do texto
    border_color = (0, 0, 0)  # Cor da borda do texto

    # Cor da borda da caixa baseado na classe
    match ship_class:
        case 0:
            border_color = (255, 0, 0)
        case 1:
            border_color = (0, 255, 255)
        case 2:
            border_color = (255, 255, 0)
        case 3:
            border_color = (0, 255, 0)
        case 4:
            border_color = (0, 0, 255)
        case 5:
            border_color = (0, 50, 50)
        case 6:
            border_color = (100, 100, 100)

    # Texto da classe e probabilidade
    class_identifier = f"{ship_dict[int(ship_class)]}: {(prob * 100):.2f}%"

    # Desenhar a caixa de detecção
    ann_frame = cv2.rectangle(
        frame,
        (int(dim[0]), int(dim[1])),
        (int(dim[2]), int(dim[3])),
        border_color,
        thickness=2,
    )

    # Dimensões do frame
    frame_height, frame_width = frame.shape[:2]

    # Determinar a largura e altura do retângulo para o texto
    text_width = max(len(class_identifier), 40) * int(font_size * 10)
    text_height = int(font_size * 40 * (1 + len(ship_info)))  # Altura ajustada para múltiplas linhas

    # Calcular a posição padrão (acima da caixa)
    text_x = int(dim[0])  # Começo alinhado com a borda esquerda da caixa
    text_y = int(dim[1]) - text_height - 10  # Texto acima da caixa

    # Ajustar para lateral se o texto ultrapassar o topo do frame
    if text_y < 0:
        # Posição à direita se houver espaço, caso contrário, à esquerda
        if dim[2] + text_width < frame_width:  # Verificar se cabe à direita
            text_x = int(dim[2]) + 10
            text_y = int(dim[1])
        else:  # Caso contrário, colocar à esquerda
            text_x = int(dim[0]) - text_width - 10
            text_y = int(dim[1])

    # Garantir que o texto não saia dos limites do frame lateralmente
    text_x = max(0, min(text_x, frame_width - text_width))

    # Criar uma máscara para o retângulo transparente
    shape = np.zeros_like(frame, dtype=np.uint8)
    cv2.rectangle(
        shape,
        (text_x, max(0, text_y)),  # Garantir que o texto não saia do frame
        (text_x + text_width, max(0, text_y + text_height)),
        border_color,
        thickness=-1,  # Preenchido
    )

    # Aplicar transparência
    frame_with_transparency = cv2.addWeighted(frame, 0.9, shape, 0.5, 0)

    # Adicionar o texto principal (classe e probabilidade)
    cv2.putText(
        frame_with_transparency,
        class_identifier,
        (text_x + 10, max(20, text_y + int(font_size * 30))),  # Posição segura dentro do retângulo
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        font_color,
        thickness=1,
    )

    # Adicionar informações extras do navio (uma linha por vez)
    for index, (key, value) in enumerate(ship_info.items()):
        ship_data_string = f"{key}: {value}"
        cv2.putText(
            frame_with_transparency,
            ship_data_string,
            (text_x + 10, max(20, text_y + int(font_size * 30 * (index + 2)))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            font_color,
            thickness=1,
        )

    return frame_with_transparency




def main():
    global recording
    ship_data = load_json_data('./videos/cinematicas')

    model = YOLO('models/vedit-std_v2.4.pt')
    video_path = r"C:\Users\rodri\OneDrive\Área de Trabalho\VEDIT\vedit-main\vedit-main\videos\vid3.mp4"
    if not os.path.exists(video_path):
        print(f"Erro: arquivo de vídeo '{video_path}' não encontrado.")
        return

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    frame_cur = 0
    pred_frame = 0
    results = []

    recording = False
    if recording:
        print("Gravação habilitada.")
    else:
        print("Gravação desabilitada.")

    out_vid = None
    if recording:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        ret, frame = capture.read()
        if not ret:
            print("Erro ao capturar o vídeo.")
            return
        vid_size = (frame.shape[1], frame.shape[0])
        out_vid = cv2.VideoWriter(
            f'rec/detection_output-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mp4',
            fourcc,
            20.0,
            vid_size,
        )

    # Ajustar o tamanho da janela
    cv2.namedWindow('Detection Results', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection Results', 1280, 720)  

    while True:
        ret, frame = capture.read()
        if not ret:
            print('Fim do vídeo.')
            break

        frame_cur += 1
        # Redimensionar o frame para alguma resolução
        frame = cv2.resize(frame, (1280, 720))

        if pred_frame >= 24:
            results = list(model.track(frame, tracker="bytetrack.yaml", conf=0.7, iou=0.5, stream=True))
            pred_frame = 0
        pred_frame += 1

        if results:
            for index, it in enumerate(results[0].boxes.cls):
                frame = annotate_bounding_box(
                    it,
                    results[0].boxes.conf[index],
                    results[0].boxes.xyxy[index],
                    frame,
                    get_ship_data(ship_data)
                )

        frame = annotate_text(f'Navios: {len(results[0].boxes.cls) if results else 0}', frame, 20, frame.shape[0] - 40, 1.2)
        frame = annotate_text(f"Quadro atual: {frame_cur}", frame, 20, 40, 1.2)

        cv2.imshow('Detection Results', frame)

        if recording:
            out_vid.write(frame)

        if cv2.waitKey(1) & 0xFF in [ord("q"), ord("Q")]:
            print("Execução interrompida.")
            break

    if recording:
        out_vid.release()

    capture.release()
    cv2.destroyAllWindows()


def load_json_data(path):
    data = {}
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), 'r') as file:
                json_data = json.load(file)
                timestamp = filename.split(".")[0]
                data[timestamp] = json_data
    return data

def get_ship_data(json_arr, time_obj=None):
    nome_json = "2024-04-09T15_50_54"

    
    if nome_json in json_arr:
        ship_data = json_arr[nome_json]

        nome = ship_data.get("identificacao", {}).get("nome", "Desconhecido")
        imo = ship_data.get("identificacao", {}).get("imo", "Desconhecido")
        mmsi = ship_data.get("identificacao", {}).get("mmsi", "Desconhecido")
        rumo = ship_data.get("cinematica", {}).get("rumo", {}).get("fundo", "Desconhecida")
        velocidade = ship_data.get("cinematica", {}).get("velocidade", {}).get("fundo", "Desconhecida")
        
        return {"Nome": nome, "IMO": imo, "MMSI": mmsi, "Rumo": rumo , "Velocidade": velocidade}
    else:
        print(f"Alerta: Dados não encontrados para o timestamp {nome_json}.")
        return {}



if __name__ == "__main__":
    main()
