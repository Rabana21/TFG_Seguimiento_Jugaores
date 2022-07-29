Proyecto de detección y seguimiento de jugadores de balonamano en un a secuancia de vídeo.
---
Los scripts más importantes dentro del proyecto son:
* torchserve_handler.py
* views.py

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

2. views
- index -> encargada de llamar al handler para hacer la inferencia del vídeo. Recoge también los valores max_age y n_init para poder hacer la inicialización del deepsort.
- download_file -> devolverá al usuario el vídeo de la detección más el json

Ejecución del proyecto
---
ANTES DE EJECTUAR:
```
cd yolov5/
```
```
python ./export.py --weights ./yolov5x.pt --img 640 --batch 1
```
```
torch-model-archiver --model-name my_model_name \
--version 0.1 --serialized-file ./yolov5x.torchscript \
--handler ../model-server/ressources/torchserve_handler.py \
--extra-files ../model-server/ressources/index_to_name.json,../model-server/ressources/torchserve_handler.py
mv my_model_name.mar ../model-store
```

Para poder ejecutar el proyecto, seguir las instrucciones detalladas en el manual de instalación de la memoria.
