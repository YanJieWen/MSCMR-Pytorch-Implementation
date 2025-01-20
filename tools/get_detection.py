'''
@File: get_detection.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 1月 19, 2025
@HomePage: https://github.com/YanJieWen
'''


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
import json
from glob import glob
from functools import partial
from pycocotools.coco import COCO
import PIL.ImageDraw as Imagedraw
import sys


colors = {0:(69, 123, 157),1:(255, 159, 28),2:(173, 181, 189),3:(252, 243, 0),4:(255, 77, 109)}

class VisualDeter(object):
    def __init__(self,img_root,gt_annpath,pred_path,base_path,vis_root,iou_threshold,):
        self.img_root = img_root
        self.gt_annfile = gt_annpath
        self.pred_file = pred_path
        self.baseline = base_path
        self.vis_root = vis_root
        self.iou_threshold = iou_threshold
        self.coco_gt = COCO(self.gt_annfile)
        self.coco_dt = self.coco_gt.loadRes(self.pred_file)
        self.base_dt = self.coco_gt.loadRes(self.baseline)
        if not os.path.exists(vis_root):
            os.makedirs(vis_root, exist_ok=True)
        imgs = np.random.permutation(glob(os.path.join(img_root, '*.jpg')))[:100]
        sp_img_files = [os.path.basename(x) for x in imgs]
        img_ids = {v['file_name']: k for k, v in self.coco_gt.imgs.items()}
        self.img_ids = sorted([img_ids.get(x) for x in sp_img_files])

    def xywh2xyxy(self,box):
        box[:, 2] = box[:, 0] + box[:, 2]
        box[:, 3] = box[:, 1] + box[:, 3]
        return box

    def iou(self,box1, box2):
        x11, y11, x12, y12 = np.split(box1, 4, axis=1)
        x21, y21, x22, y22 = np.split(box2, 4, axis=1)

        xa = np.maximum(x11, np.transpose(x21))
        xb = np.minimum(x12, np.transpose(x22))
        ya = np.maximum(y11, np.transpose(y21))
        yb = np.minimum(y12, np.transpose(y22))

        area_inter = np.maximum(0, (xb - xa + 1)) * np.maximum(0, (yb - ya + 1))

        area_1 = (x12 - x11 + 1) * (y12 - y11 + 1)
        area_2 = (x22 - x21 + 1) * (y22 - y21 + 1)
        area_union = area_1 + np.transpose(area_2) - area_inter

        iou = area_inter / area_union
        return iou

    def draw_tp_box(self,img, pred):
        draw = Imagedraw.Draw(img)
        # color = ImageColor.getrgb(colors[int(pred[0])])
        color = colors.get(int(pred[0]))
        left, top, right, bottom = pred[1:5]
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=6, fill=color)
        return img

    def draw_fail_box(self,img, bbox, color=None):
        draw = Imagedraw.Draw(img)
        left, top, right, bottom = bbox
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=3, fill=color)
        return img

    def get_single_model(self,coco_gt,coco_dt,postprefix='mamba'):
        for img_id in tqdm(self.img_ids, desc='caching quality...', file=sys.stdout):
            img_filename = coco_gt.loadImgs(img_id)[0]['file_name']
            img_filename = os.path.join(self.img_root, img_filename)
            _posname = os.path.basename(img_filename).split('.')[0]
            img = Image.open(img_filename).convert('RGB')
            ann_ids = coco_gt.getAnnIds(img_id)
            pred_ids = coco_dt.getAnnIds(img_id)
            if len(pred_ids) != 0:
                anns = coco_gt.loadAnns(ann_ids)
                preds = coco_dt.loadAnns(pred_ids)
                anns = np.array([[ann['category_id'] - 1, *ann['bbox']] for ann in anns])
                preds = np.array([[pred['category_id'], *pred['bbox'], pred['score']] for pred in preds])
                preds[:, 1:5] = self.xywh2xyxy(preds[:, 1:5])
                anns[:, 1:5] = self.xywh2xyxy(anns[:, 1:])
                # 选择置信度高于0.25的
                preds = preds[preds[:, -1] > 0.25]
                preds = preds[np.argsort(preds[:, -1])][::-1]
                preds = list(preds)
            else:
                preds = []
            # 计算TP,FP,FN
            num_tp, num_fn, num_fp = 0, 0, 0
            for i in range(anns.shape[0]):  # 遍历每一个正样本
                if len(preds) == 0: break
                ious = self.iou(anns[i:i + 1, 1:], np.array(preds)[:, 1:5])[0]
                iou_argsort = ious.argsort()[::-1]
                missing = True
                for j in iou_argsort:
                    if ious[j] < self.iou_threshold: break  # 最匹配的正例
                    if anns[i, 0] == preds[j][0]:  # 如果属于同一个类
                        img = self.draw_tp_box(img, preds[j])
                        preds.pop(j)
                        missing = False
                        num_tp += 1
                        break
                if missing:  # 漏检
                    img = self.draw_fail_box(img, anns[i, 1:], color=(0, 255, 0))
                    num_fn += 1
            if len(preds):  # 误检
                for j in range(len(preds)):
                    img = self.draw_fail_box(img, preds[j][1:5], color=(255, 0, 0))
                    num_fp += 1
            out_prefix = f'{_posname}-tp@{int(num_tp)}-fn@{int(num_fn)}-fp@{int(num_fp)}.jpg'
            new_out_root = os.path.join(self.vis_root,postprefix)
            if not os.path.exists(new_out_root):
                os.makedirs(new_out_root,exist_ok=True)
            cv2.imwrite(os.path.join(new_out_root, out_prefix), np.array(img)[..., ::-1])

    def __call__(self):
        _func = partial(self.get_single_model,coco_gt=self.coco_gt)
        _func(coco_dt=self.coco_dt,postprefix='mamba')
        _func(coco_dt=self.base_dt,postprefix='yolov8')
