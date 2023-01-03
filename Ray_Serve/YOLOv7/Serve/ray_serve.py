from pathlib import Path
from threading import Thread
from fastapi import FastAPI, Body
from urllib import request
from PIL import Image
from io import BytesIO
from starlette.requests import Request

import numpy as np
import torch
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel


from ray import serve
import pandas as pd

def tensor2json(inp_ten):
    out_json = []
    for batch in inp_ten:
        out_batch=[]
        for res in batch:
            df = pd.DataFrame(res.numpy())
            df = df.to_json(orient='values')
            out_batch.append(df)
        out_json.append(out_batch)
    return out_json

app = FastAPI()

@serve.deployment(route_prefix="/")
@serve.ingress(app)
class YOLOModelRay:
    def __init__(self):        
        # Initialize/load model and set device
        print('inside test')
        self.weights = '/tmp/yolov7.pt'
        self.batch_size = 64
        self.imgsz = 640
        self.conf_thres = 0.001
        self.iou_thres = 0.65
        self.project = 'runs/test'
        self.name = 'kyle_test'
        self.cpugpu = 'cpu'
        self.single_cls = False
        self.plots = False
        
        weight_url = 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt'
        request.urlretrieve(weight_url, self.weights)

        set_logging()
        self.device = select_device(self.cpugpu, batch_size=self.batch_size)

        # Directories
        self.save_dir = Path(increment_path(Path(self.project) / self.name, exist_ok=False))  # increment run
        (self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        print(self.save_dir)

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.gs = max(int(self.model.stride.max()), 32)  # grid size (max stride)
        self.imgsz = check_img_size(self.imgsz, s=self.gs)  # check img_size

        # Configure
        self.model.eval()
    
    @app.post('/image_predict')      
    async def inference_call(self, starlette_request: Request):
        image_payload_bytes = await starlette_request.body()
        pil_image = Image.open(BytesIO(image_payload_bytes))
        pil_image = pil_image.resize((self.imgsz,self.imgsz))
        np_image = np.asarray(pil_image).transpose(2,0,1)
        print(np_image.shape)
        img = torch.Tensor([np_image])
        print(img.shape)
        t0, t1 = 0, 0   
        res = []
        img = img.to(self.device, non_blocking=True)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            out, train_out = self.model(img, augment=False)  # inference and training outputs
            t0 += time_synchronized() - t

            lb = []  # for autolabelling
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=self.conf_thres, iou_thres=self.iou_thres, labels=lb, multi_label=True)
            t1 += time_synchronized() - t
            res.append(out)

        print(f"Results saved to {self.save_dir}")
        return tensor2json(res)

    @app.get('/healthcheck')
    def healthcheck(self):
        return "Healthy"


yolo_model_ray = YOLOModelRay.bind()
serve.run(yolo_model_ray)