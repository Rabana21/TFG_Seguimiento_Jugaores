"""Custom TorchServe model handler for YOLOv5 models.
"""
from ts.torch_handler.base_handler import BaseHandler
import numpy as np
import base64
import torch
import torchvision.transforms as tf
import torchvision
import io
from PIL import Image
# import tensorflow as tf


#Importado por mi
import cv2
import sys
sys.path.append("/home/ruben/Documentos/torchserve/model-server/ressources")
sys.path.append("/home/ruben/Documentos/torchserve/model-server/ressources/Yolov5_StrongSORT_OSNet")
sys.path.append("/home/ruben/Documentos/torchserve/model-server/ressources/Yolov5_StrongSORT_OSNet/deep_sort/deep/reid")
sys.path.append("/home/ruben/Documentos/torchserve/model-server/ressources/Yolov5_StrongSORT_OSNet/yolov5")
from LoadImages import LoadImages
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from pathlib import Path

# from yolov5.utils.datasets import LoadImages, LoadStreams, VID_FORMATS


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    img_size = 640
    imsize= [640,640]
    PATH_SAVE_IMGS = "/home/ruben/Documentos/torchserve/model-server/resultados/"
    """Image size (px). Images will be resized to this resolution before inference.
    """
    deepsort_list = []
    im0 = None

    def __init__(self):
        # call superclass initializer
        self.context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        # deepsort_list = self.init_deepsort()
        # super().__init__()    


    def init_deepsort(self):
        # deep_sort_model = 'osnet_x0_25' #Este es el que estaba en el original
        deep_sort_model = 'osnet_x0_25_msmt17.pt'
        nr_sources = 1
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        RUTA_YAML_DS = '/home/ruben/Documentos/torchserve/model-server/ressources/Yolov5_StrongSORT_OSNet/deep_sort/configs/deep_sort.yaml'
  
        cfg = get_config()
        cfg.merge_from_file(RUTA_YAML_DS)
        
        for i in range(nr_sources):
            self.deepsort_list.append(
                DeepSort(
                    deep_sort_model,
                    device,
                    max_dist=0.25,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=self.max_age, n_init=self.n_init, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                )
            )

        return self.deepsort_list


    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.context = context
        self.initialized = True
        super().initialize(context)


    def inference(self, model_input):
        return super().inference(model_input)


    def handle(self, data, context):
        """
            Invoke by TorchServe for prediction request.
            Do pre-processing of data, prediction using model and postprocessing of prediciton output
            :param data: Input data for prediction
            :param context: Initial context contains model server system properties.
            :return: prediction output
        """        
        dataset, device = self.preprocess(data)
        result = [[] for _ in range(dataset.frames)]
        for frame_idx, (path, im, im0s, self.vid_cap) in enumerate(dataset):
            self.im0 = im0s.copy()
            self.final_size = im0s.shape
            im = torch.from_numpy(im).to(device)

            #CONVIERTO LA IMAGEN AL FORMATO ADECUADO
            im = im.float()  
            im /= 255.0  
            if len(im.shape) == 3:
                im = im[None]  
            self.im = im

            model_output = self.inference(im) 

            detection = self.postprocess(model_output, frame_idx)
            result[frame_idx] = detection
        self.vid_writer[0].release()
        return [result]


    def preprocess(self, data):
        for row in data:
            source = row.get("path").decode()
            self.video_path = row.get("path_video").decode()
            self.n_init = int(row.get("n_init").decode())
            self.max_age = int(row.get("max_age").decode())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        yolo_model = '/home/ruben/Documentos/torchserve/model-server/ressources/yolov5x.pt'
        model = DetectMultiBackend(yolo_model, device=device, dnn=False)
        self.names = model.module.names if hasattr(model, 'module') else model.names
        stride, _ , pt = model.stride, model.names, model.pt
        self.imsize = check_img_size(self.imsize, s=stride) 
        dataset = LoadImages(path=source, img_size=self.imsize, stride=stride, auto=pt)
        nr_sources = 1
        self.vid_path, self.vid_writer, self.txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources
        self.deepsort_list = self.init_deepsort()
        return dataset, device


    def postprocess(self, inference_output, frame):
        # perform NMS (nonmax suppression) on model outputs
        pred = non_max_suppression(inference_output[0],conf_thres=0.35, iou_thres=0.45, classes=[0])

        # pred[0][:, :4] = scale_coords(self.im.shape[2:], pred[0][:, :4], self.im0.shape).round()
        self.im0 = cv2.resize(self.im0, (640, 640))
        # print(f'Predicción de YOLOv5: {pred}')
        pred = deepSort(self, pred)
        print(f'Predicción de DeepSORT: {pred}')
        detections = [[] for _ in range(len(pred))]
        for i,image_detections in enumerate(pred):
            print(f'Predicción de DeepSORT{image_detections}')
            annotator = Annotator(self.im0, line_width=2, pil=not ascii)
            for det in image_detections:
                # print(f'Jugador detectado {det}')
                xyxy = det[0] # / self.img_size
                id_player = det[1]
                label = det[2]
                conf = det[3]
                detections[i].append({
                    "x1": xyxy[0].item(),
                    "y1": xyxy[1].item(),
                    "x2": xyxy[2].item(),
                    "y2": xyxy[3].item(),
                    "id_player" :id_player,
                    "confidence": conf,
                    "class": label
                })
                c = int(label)  
                label = f'{id_player:0.0f} {self.names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
            filename = self.PATH_SAVE_IMGS+'FRAME-'+str(frame)+'.jpg'
            self.im0 = annotator.result()
            self.im0 = cv2.resize(self.im0, (self.final_size[1],self.final_size[0])) #Estaban  las posiciones cambiadas
            cv2.imwrite(filename, self.im0)
            if self.vid_path[i] != self.video_path:  # new video
                self.vid_path[i] = self.video_path
                if isinstance(self.vid_writer[i], cv2.VideoWriter):
                    self.vid_writer[i].release()  # release previous video writer
                if self.vid_cap:  # video
                    fps = self.vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, self.final_size[1],self.final_size[0]
                self.video_path = str(Path(self.video_path).with_suffix('.webm'))  # force *.mp4 suffix on results videos
                self.vid_writer[i] = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (w, h))
            self.vid_writer[i].write(self.im0)

        return detections


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    output = [torch.zeros((0, 6), device=prediction.device)
              ] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[
                conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = torchvision.box_iou(
                boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float(
            ) / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]      

    return output
    

def deepSort(self, pred):
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    # device = select_device('cpu') 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """ yolo_model = '/home/ruben/Documentos/torchserve/model-server/ressources/Yolov5_StrongSORT_OSNet/yolov5/weights/crowdhuman_yolov5m.pt'
    model = DetectMultiBackend(yolo_model, device=device, dnn=False)
    stride, names, pt = model.stride, model.names, model.pt
    dataset = LoadImages(self.path, img_size=self.img_size, stride=stride, auto=pt)  """
    # self.im0 = cv2.imread()
    # self.im0 = cv2.imread(path)
    # names = model.module.names if hasattr(model, 'module') else model.names

    nr_sources = 1
    outputs = [None] * nr_sources
    frame_detection = []
    result = []
    for i, det in enumerate(pred):
        seen += 1
        'Si la detección no es nula, se procesa'
        if det is not None and len(det):      
           
          'Se re-escala el bbox'
          T = 0.25 #0.3
          det = remove_out_range(self,det, T)

          'A continuación, se establece como se van a almacenar los resultados en el fichero'
          for c in det[:, -1].unique(): 
            n = (det[:, -1] == c).sum()  
            # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  
            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]
            'Acto seguido, se le pasan al deepsort para establecer el tracking'
            t4 = time_sync()
            outputs[i] = self.deepsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), self.im0)
            # print(f'PREDICCIÓN DE DeepSORT{outputs[i]}')

            t5 = time_sync()
            dt[3] += t5 - t4

            'Dibujo las bbox de acuerdo con los elementos trackeados'
            
            if len(outputs[i]) > 0:
              for j, (output) in enumerate(outputs[i]):
                detection_player = []
                'Si te fijas aquí definas las coordenadas de la bbox, el id del deepsort, la clase y su score o puntuación'
                detection_player.append(output[0:4])
                detection_player.append(output[4])
                detection_player.append(output[5])
                detection_player.append(output[6])
                frame_detection.append(detection_player)
        result.append(frame_detection)
    return result


def remove_out_range(self, det, T):
  det_aux = det
  range = self.img_size * T
  i = 0
  medium_area = []
  for det_person in det:
    # print(det_person)
    height = det_person[3] - det_person[1]
    wide = det_person[0] + det_person[2]
    medium_height = ( det_person[1] + det_person[3]) / 2
    area = height * wide
    if medium_height < range: 
      det_aux = torch.cat((det_aux[:i], det_aux[i+1:]))
    elif area >= 500000: #500000
      medium_area.append(area)
      det_aux = torch.cat((det_aux[:i], det_aux[i+1:]))
    else:
      medium_area.append(area)
      i += 1
  # print(det_aux)
  medium_area.sort()
  return det_aux 


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y




