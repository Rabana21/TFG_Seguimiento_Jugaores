# Transform the input image into a bytes object
import cv2
import json
from PIL import Image
from io import BytesIO

# file_name = "/home/ruben/Documentos/torchserve/model-server/_113942313_mediaitem113942310.jpg"
file_name = "/home/ruben/Documentos/torchserve/model-server/_partido_champion_recortado.mp4"

# image = Image.fromarray(cv2.imread(file_name))
# image2bytes = BytesIO()
# image.save(image2bytes, format="PNG")
# image2bytes.seek(0)
# image_as_bytes = image2bytes.read()


cap = cv2.VideoCapture(file_name)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# dict = {
#     "path":file_name,
#     "video": False,
#     "data": image_as_bytes,
# }

dict = {
    "path":file_name,
    "cap":cap,
    "video": True,
    "frames": frames
}

# Send the HTTP POST request to TorchServe
import requests

req = requests.post("http://localhost:9080/predictions/my_model_name", data=dict)
if req.status_code == 200: 
    res = req.json()
else:
    res = req.json()

with open("result.json","w") as file:
    json.dump(res,file)