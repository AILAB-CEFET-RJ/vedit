import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv, Concat, Focus
from ultralytics.nn.modules.block import (
    DFL, C2f, Bottleneck, SPPF, C3, C3TR, C3x, C2, C3Ghost, BottleneckCSP
)
from torch.nn import ModuleList, Sequential
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import SiLU, ReLU, LeakyReLU
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.upsampling import Upsample
from ultralytics.nn.modules.head import Classify, Detect

# Permitir uso seguro dessas classes customizadas
add_safe_globals([
    Conv, C2f, Bottleneck, SPPF, C3, C3TR, C3x, C2, C3Ghost, BottleneckCSP,
    DetectionModel, Detect, ModuleList, Sequential, Conv2d, BatchNorm2d,
    SiLU, ReLU, LeakyReLU, MaxPool2d, Upsample, Concat, Focus, Classify, DFL
])

# Carregar o checkpoint diretamente com segurança
ckpt = torch.load('models/vedit-std_v3.0.pt', map_location='cpu', weights_only=False)

# Criar o modelo de detecção e carregar os pesos
model = DetectionModel(cfg=ckpt['model'].yaml)  # Usa a configuração salva no checkpoint
model.load_state_dict(ckpt['model'].state_dict())

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
    cv2.rectangle(shape, (max(pos_x - 10, 0), max(int(pos_y - (font_size * 30)), 0)),
                  (pos_x + width, pos_y + height), (1, 1, 1), -1)
    mask = shape.astype(bool)
    frame[mask] = cv2.addWeighted(frame, 0.4, shape, 0.6, 1)[mask]
    cv2.putText(frame, text, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_size, (255, 255, 255), 2)
    return frame

def annotate_bounding_box(ship_class, prob, dim, frame, ship_info={}):
    ship_data_string = ""
    font_color = (0, 0, 0)
    border_color = [
        (255, 0, 0), (0, 255, 255), (255, 255, 0),
        (0, 255, 0), (0, 0, 255), (0, 50, 50), (100, 100, 100)
    ][int(ship_class)]
    class_identifier = f"{ship_dict[int(ship_class)]}: {(prob * 100):.2f}%"
    frame = cv2.rectangle(frame, (int(dim[0]), int(dim[1])),
                          (int(dim[2]), int(dim[3])), border_color, 2)
    frame_height, frame_width = frame.shape[:2]
    text_width = max(len(class_identifier), 40) * int(font_size * 10)
    text_height = int(font_size * 40 * (1 + len(ship_info)))
    text_x = int(dim[0])
    text_y = int(dim[1]) - text_height - 10
    if text_y < 0:
        text_x = int(dim[2]) + 10 if dim[2] + text_width < frame_width else int(dim[0]) - text_width - 10
        text_y = int(dim[1])
    text_x = max(0, min(text_x, frame_width - text_width))
    shape = np.zeros_like(frame, dtype=np.uint8)
    cv2.rectangle(shape, (text_x, max(0, text_y)),
                  (text_x + text_width, max(0, text_y + text_height)),
                  border_color, -1)
    frame = cv2.addWeighted(frame, 0.9, shape, 0.5, 0)
    cv2.putText(frame, class_identifier, (text_x + 10, max(20, text_y + int(font_size * 30))),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, 1)
    for i, (k, v) in enumerate(ship_info.items()):
        cv2.putText(frame, f"{k}: {v}", (text_x + 10, max(20, text_y + int(font_size * 30 * (i + 2)))),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, 1)
    return frame

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
        s = json_arr[nome_json]
        idf = s.get("identificacao", {})
        cine = s.get("cinematica", {})
        return {
            "Nome": idf.get("nome", "Desconhecido"),
            "IMO": idf.get("imo", "Desconhecido"),
            "MMSI": idf.get("mmsi", "Desconhecido"),
            "Rumo": cine.get("rumo", {}).get("fundo", "Desconhecida"),
            "Velocidade": cine.get("velocidade", {}).get("fundo", "Desconhecida")
        }
    else:
        print(f"Alerta: Dados não encontrados para {nome_json}.")
        return {}

def main():
    global recording
    ship_data = load_json_data('./videos')
    model = YOLO(r'models/vedit-std_v3.0.pt')
    video_path = r"videos/K120-AIS-710102038/K120-AIS-710102038.mp4"
    if not os.path.exists(video_path):
        print(f"Erro: vídeo '{video_path}' não encontrado.")
        return
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    frame_cur = pred_frame = 0
    results = []

    if recording:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        ret, frame = capture.read()
        if not ret:
            print("Erro ao capturar o vídeo.")
            return
        vid_size = (frame.shape[1], frame.shape[0])
        out_vid = cv2.VideoWriter(f'rec/detection_output-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mp4',
                                  fourcc, 20.0, vid_size)

    cv2.namedWindow('Detection Results', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection Results', 1280, 720)

    while True:
        ret, frame = capture.read()
        if not ret:
            print('Fim do vídeo.')
            break

        frame_cur += 1
        frame = cv2.resize(frame, (1280, 720))

        if pred_frame >= 24:
            results = list(model.track(frame, tracker="bytetrack.yaml", conf=0.7, iou=0.5, stream=True))
            pred_frame = 0
        pred_frame += 1

        if results:
            for index, it in enumerate(results[0].boxes.cls):
                frame = annotate_bounding_box(it,
                                              results[0].boxes.conf[index],
                                              results[0].boxes.xyxy[index],
                                              frame,
                                              get_ship_data(ship_data))

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

if __name__ == "__main__":
    main()
