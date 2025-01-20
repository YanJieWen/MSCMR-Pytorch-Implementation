'''
@File: analysis.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 12月 15, 2024
@HomePage: https://github.com/YanJieWen
'''

import os
import os.path as osp
from glob import glob
import sys

from tqdm import tqdm


from tools import Drawer,save_fig,Yolov8_erf,Yolov8_heatmap,Res_json,PR_curve,VisualDeter
from ultralytics import YOLO

from PIL import Image
import matplotlib.pyplot as plt


_GETJSON = {
    'img_root':'./datasets/COCO/test2017/test2017/',
    'ckpt_path': './pretrained/yolov8n.pt',
    'save_dir': './results',
    'eval_batch': 10,
}
_GETHEATMAP = {
    'device': 'cuda:0',
    'method': 'GradCAM',
    'layer': [15,18,21],
    'backward_type': 'all',
    'conf_threshold': 0.2,
    'ratio': 0.02,
    'show_box': False,
    'renormalize': False,
}

_GETDRAWER = {
    'type':'voc',
    'if_inference':True,
    'if_shuffle': True,
    'cls_dict': None,
}

_GETERF = {
    'device':'cuda:0',
    'layer':'9', #2,4.6,9
    'dataset':'./demo/',
    'num_images':50,
    'save_path':'results.png'
}

_GETPRCURVE = {
    'gt_path':'./crash_test.json',
    'thr': -1,
    'pred_path': None
}

_GETVISCOMPARION = {'img_root':'./datasets/Crash2024/test/',
                    'gt_annpath': './datasets/Crash2024/annotations/test.json',
                    'pred_path': './results/crash-comamba-new.json',
                    'base_path': './results/crash-yolov8-new.json',
                    'vis_root': './demo/crash_res/',
                    'iou_threshold': 0.5
}

class Fashionana(object):
    def __init__(self,if_json=False,if_heat=True,if_draw=True,if_erf=True,if_ap=True,if_visdet=True,
                 pretrain_weights='./pretrained/yolov8l.pt',out_root='results'):
        self.heat = if_heat
        self.draw_box = if_draw
        self.erf = if_erf
        self.ap = if_ap
        self.json = if_json
        self.vis = if_visdet
        if not osp.exists(out_root):
            os.makedirs(out_root)
        self.out_root = out_root
        #check ckpt
        assert osp.isfile(pretrain_weights), f'{pretrain_weights} is not found,check it'
        if self.json and len(glob(osp.join(self.out_root,'*.json')))==0:
            _GETJSON['ckpt_path'] = pretrain_weights
            self.resloader = Res_json(**_GETJSON)
        else:
            self.resloader = None
        if self.heat:
            _GETHEATMAP['weight'] = pretrain_weights
            self.heatmapper = Yolov8_heatmap(**_GETHEATMAP)
        if self.draw_box:
            self.model = YOLO(pretrain_weights)
            self.drawer = Drawer(**_GETDRAWER)
        if self.erf:
            _GETERF['weight'] = pretrain_weights
            self.rfer = Yolov8_erf(**_GETERF)
        if self.ap:
            self.cal_aper = PR_curve(**_GETPRCURVE)
        if self.vis:
            self.visuer = VisualDeter(**_GETVISCOMPARION)

    def __call__(self, img_roots):
        if self.resloader is not None:
            self.resloader()
        else:
            pass
        if self.vis:
            self.visuer()
        if self.erf:
            self.rfer.process()
        if self.ap:
            self.cal_aper()
        imgs = [osp.join(img_roots,x) for x in os.listdir(img_roots) if x.endswith('.jpg')]
        pbar = tqdm(imgs,desc='Anaysising...',file=sys.stdout)
        for img in pbar: #only for demo
            _img = Image.open(img).convert("RGB")
            basic_name = osp.basename(img).split('.')[0]
            if self.heat:
                self.heatmapper(img,self.out_root)
            if self.draw_box:
                #默认配置
                results = self.model([img],verbose=False,conf=0.25,iou=0.7)[0]
                boxes = results.boxes.xyxy.detach().to('cpu').numpy()
                clss = results.boxes.cls.detach().to('cpu').numpy() + 1
                scores = results.boxes.conf.detach().to('cpu').numpy()
                img = self.drawer(_img,boxes,clss,scores)
                plt.imshow(img)
                save_fig(f'{basic_name}_bbox',self.out_root)
                plt.show()
            else:
                pass


        print(f'{"*"*10}Visualization has been finished{"*"*10}')


if __name__ == '__main__':
    imgroot = './demo'
    visualizer = Fashionana(if_json=False,if_heat=False,if_draw=False,if_erf=False,
                            if_ap=False,if_visdet=True)
    visualizer(imgroot)







