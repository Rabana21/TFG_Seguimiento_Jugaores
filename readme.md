# Proyecto de detección y seguimiento de jugadores de balonamano en un a secuancia de vídeo.
---
Los scripts más importantes dentro del proyecto son:
* [hander](./model-server/ressources/torchserve_handler.py)
* [servidor](./detection/detection/views.py)

1. torchserve_handler
Dentro de este script veremos diferentes metodos a tener en cuneta para la ejecución del proecto:
- init -> inicializarán los parametros que serán utilizados en el handler
- inference -> el encargado de llamar al YOLO para hacer las detecciones
- preprocess -> dividir el vídeo en frames para poder procesarlo con el inference
- postprocess -> manejar la detección proporcionada por la inference, llamar a deepsort para que haga el seguimiento y devolver el resultado en el formato deseado
- non_max_suppression -> hará un filtro de las detecciones hechas por el YOLO
- deepSort -> con los datos devueltos por non_max_suppresion, hace el seguimiento de los objetos detectados en el mismo
- remove_out_range -> filtro para eliminar detecciones con un area demasiado grande y eliminar a objetos que probablemente sean aficionados
- xywh2xyxy -> convertirlo al bormato deseado para el bounding_box

2. Views servidor
- index -> encargada de llamar al handler para hacer la inferencia del vídeo. Recoge también los valores max_age y n_init para poder hacer la inicialización del deepsort.
- download_file -> devolverá al usuario el vídeo de la detección más el json

## Ejecución del proyecto
---
Introducrnos en la carpeta `cd yolov5/` habrá que instalar sus requerimientos y a ejecutar el `../model-server/ressources/download_weights.sh` copiamos el archivo .pt dentro de ressources `cp ./yolov5x.pt ../model-server/ressources` y por ultimo generar el archivo torchsript necesario para el handler:
```
python ./export.py --weights ./yolov5x.pt --img 640 --batch 1
```
Tras haber hecho esto, generar el modelo para hacer la inferencia para las detecciones:
```
torch-model-archiver --model-name my_model_name \
--version 0.1 --serialized-file ./yolov5x.torchscript \
--handler ../model-server/ressources/torchserve_handler.py \
--extra-files ../model-server/ressources/index_to_name.json,../model-server/ressources/torchserve_handler.py
```
Y por últmo mover el archivo generado en model_store 
`mv my_model_name.mar ../model-store`

Habiendo hecho esto, podremos ejecutar satisfactoriamente el handler, ubicandonos en la carpeta /model-server
```python
import os 
os.system('torchserve --start \
           --ncs \
           --ts-config ../model-store/config.properties \
           --model-store ../model-store \
           --models my_model_name.mar')
```
Para parar la ejecución del handler `os.system('torchserve --stop')` y , después se ejecuta el servidor que acepta las peticiones ubicandonos en [detection](/detection) y hacer `python manage.py runserver`.
---
Para la ejecución del proyecto se puede utulizar el [video de muestra](./detection/media/video.mp4)

## Resultados
Los resultados obtenidos podemos verlo dentro de la carpeta [media](./detecion/media) donde lo encontraremos con el sufijo *_detection* y el resultado serí así:
![Resultado](./detecion/media/video_detecion.webm)

## Requerimientos 
Es recomendable utilizar [Anaconda](https://www.anaconda.com/) para poder manejar mejor las versiones de los paquetes. Teniendolo instalado y operativo descargar los siguientes paquetes:
```
 sudo apt install default-jre 
 conda install -c pytorch torchserve 
 conda install -c pytorch torch-model-archiver
 conda install -c pytorch torch-workflow-archiver
 sudo apt-get install -y libgl1-mesa-glx
 sudo apt-get install -y libglib2.0-0
 sudo apt-get install -y python3-distutils
 conda install -c pytorch-lts torchvision 
 conda install -c fastai opencv-python-headless
 conda install -c conda-forge onnx
 conda install -c conda-forge coremltools 
 conda install -c anaconda django
 conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```
