from django.shortcuts import render
from django.http import Http404, JsonResponse
from django.http import HttpResponse
import base64
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.conf import settings 
from django.template import  Context
# from tensorflow.python.keras.backend import set_session
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
from detection.settings import MEDIA_ROOT, MEDIA_URL
import numpy as np
# from keras.applications import vgg16
import datetime
import traceback
import requests
import cv2
import os
import json
import tempfile
import mimetypes


def index(request):
    if  request.method == "POST":
        f=request.FILES['sentFile'] # here you get the files needed
        max_age = request.POST['max_age']
        n_init = request.POST['n_init']
        response = {}
        file_name = "video.mp4"
        file_name_2 = default_storage.save(file_name, f)
        file_url = default_storage.url(file_name_2)
        # cap = cv2.VideoCapture(f)
        # frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        file_video = file_name_2.split('.')[0]
        file_video_2 = MEDIA_ROOT+'/'+file_video+"_detecion.webm"
        dict = {
            "path": os.path.join(MEDIA_ROOT, file_name_2),
            "path_video": file_video_2,
            "max_age": max_age,
            "n_init": n_init,
        }

        req = requests.post("http://localhost:9080/predictions/my_model_name", data=dict)
        if req.status_code == 200: 
            res = req.json()
        else:
            res = req.json()
        print(f"FILE VIDEO PATH::::{file_video_2}")
        
        with tempfile.TemporaryFile("w+t") as file:
            json.dump(res,file)
            response['name'] = res
            file_name = "result.json"
            file_name_2 = default_storage.save(file_name, file)
            file_url = default_storage.url(file_name_2)
            response['url'] = file_url
        
            return render(request,'download_file.html',{'filename': file_name_2,'videofile': file_video+"_detecion.webm"})
    else:
        return render(request,'homepage.html')


def download_file(request, filename='', videofile=''):
    if filename != '':
        url_video = "/"+MEDIA_URL+videofile
        # Define Django project base directory
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Define the full file path
        filepath = BASE_DIR + '/media/' + filename
        # Open the file for reading content
        path = open(filepath, 'rb')
        mime_type, _ = mimetypes.guess_type(filepath)
        # Set the return value of the HttpResponse
        response = HttpResponse(path, content_type=mime_type)
        if videofile!='':
            videopath = BASE_DIR + '/media/' + videofile
            path2 = open(videopath, 'rb')
            # Set the mime type
            mime_type2, _ = mimetypes.guess_type(videopath)
            # Set the return value of the HttpResponse
            response = HttpResponse(path2, content_type=mime_type2)
        # Set the HTTP header for sending to browser
        response['Content-Disposition'] = "attachment; filename=%s ; videofile=%s" % (filename, videofile)
        # Return the response value
        return response
    else:
        # Load the template
        print("FILE NAME:::::::: NO TIENE")

        return render(request, 'file.html')


