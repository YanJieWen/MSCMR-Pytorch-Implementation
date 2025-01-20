'''
@File: save_json.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 12月 11, 2024
@HomePage: https://github.com/YanJieWen
'''


import numpy as np
import os
from tqdm import tqdm
import sys
import json

from ultralytics import YOLO


class Res_json():
    def __init__(self, img_root, ckpt_path, save_dir, eval_batch=10):
        self.ckpt_path = ckpt_path
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_list = os.listdir(img_root)
        file_list = np.array([os.path.join(img_root, x) for x in file_list if x.endswith('.jpg')])
        batch_id = np.arange(len(file_list)) // eval_batch
        nb = max(batch_id) + 1
        self.file_list = [list(file_list[batch_id == i]) for i in range(nb)]
        self.file_list = img_root

    def __call__(self):
        # pbar = tqdm(self.file_list, desc='Detection...', file=sys.stdout, position=0, leave=True)
        _results = []
        model = YOLO(self.ckpt_path)
        results = model.predict(source=self.file_list, device='cuda:0', iou=0.6, conf=0.05, max_det=300)
        for res in results:
            img_name = os.path.basename(res.path)
            boxes = res.boxes.xyxy.detach().to('cpu').numpy()
            clss = res.boxes.cls.detach().to('cpu').numpy()
            conf = res.boxes.conf.detach().to('cpu').numpy()
            img_id = [int(img_name.split('.')[0])] * boxes.shape[0]
            boxes[:, 2], boxes[:, 3] = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
            for b, c, s, i in zip(boxes, clss, conf, img_id):
                b = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                _results.append({"image_id": int(i), 'category_id': int(c),
                                 'bbox': b, 'score': float(s)})
        # for img_list in pbar:
        #     if isinstance(img_list, str):
        #         img_list = [img_list]
        #     else:
        #         pass
        # results = model(img_list, verbose=False)

        # for res, img_path in zip(results, img_list):
        #     img_name = os.path.basename(img_path)
        #     # 同时检测会造成内存溢出
        #     desc = f'{"*" * 5}{img_name}{"*" * 5}'
        #     boxes = res.boxes.xyxy.detach().to('cpu').numpy()
        #     clss = res.boxes.cls.detach().to('cpu').numpy() + 1  # coco是从1开始计数类别的
        #     conf = res.boxes.conf.detach().to('cpu').numpy()
        #     img_id = [int(img_name.split('.')[0])] * boxes.shape[0]
        #     boxes[:, 2], boxes[:, 3] = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        #     for b, c, s, i in zip(boxes, clss, conf, img_id):
        #         b = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
        #         _results.append({"image_id": int(i), 'category_id': int(c),
        #                             'bbox': b, 'score': float(s)})
        x = json.dumps(_results)
        with open(os.path.join(self.save_dir, 'result.json'), 'w') as w:
            w.write(x)
        w.close()