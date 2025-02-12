# VEDIT
Vessel detection, identification and tracking system

## Installation
É necessário que se tenha Python instalado e utilize um programa para rodar com êxito. No caso, foi utilizado o VSCode pela familiaridade com o projeto.

É possível realizar a instalação das dependências dos scripts do projeto através do arquivo **requirements.txt**.

Navegue até o diretório onde se encontra o projeto, e no terminal/cmd do seu sistema operacional, use o comando abaixo (requer pip).
```
pip install -r requirements.txt
```
### Setting Current Model

Para que o script funcione, é necessário que se tenha um modelo compativel com o mesmo. Atualmente, o VEDIT está configurado para trabalhar com o modelo `vedit-std_v1.2.pt`, treinado com base neste [dataset](https://universe.roboflow.com/projetos-9tpwk/vedit). Você pode baixar o modelo já treinado [aqui](https://drive.google.com/file/d/1V8qnRz-PJO23ZLZMNV0gl32FrVRWPXlF/view).

Após baixado, apenas transfira o modelo para a pasta `/models`.

## Ship Detection

Através do script **ship_detect.py** é possível executar um processo de detecção de barcos/navios sobre a URL de um vídeo ou stream.

Ao executar o script, você tera que inserir um link válido. Então, o programa funcionara sobre o video/stream até o seu final, ou até que a tecla **'Q'** seja pressionada.
