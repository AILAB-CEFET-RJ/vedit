# VEDIT
Vessel detection, identification and tracking system

## Installation
É necessário que se tenha Python instalado e utilize um programa para rodar com êxito. No caso, foi utilizado o VSCode pela familiaridade com o projeto.

É possível realizar a instalação das dependências dos scripts do projeto através do arquivo **requirements.txt**.

Navegue até o diretório onde se encontra o projeto, e no terminal/cmd do seu sistema operacional, use o comando abaixo (requer pip).
```
pip install -r requirements.txt
```



## Ship Detection

Através do script **ship_detect.py** é possível executar um processo de detecção de barcos/navios sobre a URL de um vídeo ou stream.

Ao executar o script, você tera que inserir um link válido. Então, o programa funcionara sobre o video/stream até o seu final, ou até que a tecla **'Q'** seja pressionada.

## AIS Data

Ao rodar **aisstream_data.py** será possível receber informações AIS de três diferentes tipos (*PositionReport*, *ShipStaticData* e *ExtendedClassBPositionReport*) dentro de um território delimitado por um retângulo criado através de duas diferentes coordenadas geográficas.

Atualmente, a área selecionada pelo programa é consideravelmente pequena, então é possível ficar uma quantidade significativa de tempo sem receber qualquer mensagem.

**É necessário uma chave para a API do AIS Stream para a execução deste script. Ele lerá um arquivo .env, que deve possuir a chave no formato 'AIS_KEY=*chave*'.**
