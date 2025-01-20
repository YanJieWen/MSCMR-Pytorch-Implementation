'''
@File: get_heat.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 12月 11, 2024
@HomePage: https://github.com/YanJieWen
'''


import os
import shutil

import cv2

import torch
import torch.nn as nn

import numpy as np
from tqdm import trange

from ultralytics.utils.ops import xywh2xyxy, non_max_suppression
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.nn.tasks import attempt_load_weights

from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM

from PIL import Image

from .get_bboxes import Drawer

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True,
              scaleFill=False, scaleup=True, stride=32):
    '''
    缩放图像至指定的尺寸
    Args:
        im: cv2读取array
        new_shape: 缩放指定的图像
        color: 填充颜色
        auto: bool
        scaleFill:bool
        scaleup: 是否放大
        stride: 指定步长

    Returns:

    '''
    shape = im.shape[:2]#h,w,c
    if isinstance(new_shape,int):
        new_shape = (new_shape,new_shape)#h,w
    r = min(new_shape[0]/shape[0],new_shape[1]/shape[1])#h_r,w_r
    if not scaleup: #只缩小不扩大
        r = min(r,1.0)
    ratio = r,r# w, h ratios-->等比例缩放
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))#w,h
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]#w,h
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

class ActivationsAndGradients:
    def __init__(self,model,target_layers,reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform#for VIT
        self.handles = []
        for tgt_l in target_layers:
            self.handles.append(tgt_l.register_forward_hook(hook=self.save_activation))
            #不适用backward_hook:https://github.com/pytorch/pytorch/issues/61519
            self.handles.append(tgt_l.register_forward_hook(hook=self.save_gradient))

    def post_process(self, result):
        '''

        Args:
            result: b,a,4+nc

        Returns:对于单张图像(a,c),(a,4)xywh,(a,4)xyxy-->排序好的

        '''
        logits_ = result[:, 4:]#b,c,a
        boxes_ = result[:, :4]#b,4,a
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)#返回每个预测最大的值
        return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
            indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()

    def save_activation(self,m,input,output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        post_result, pre_post_boxes, post_boxes = self.post_process(model_output[0])
        return [[post_result, pre_post_boxes]]

    def release(self):
        for handle in self.handles:
            handle.remove()


class yolov8_target(nn.Module):
    def __init__(self, ouput_type, conf, ratio) -> None:
        #选择一部分anchors-->定位或分类求和
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio

    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result)


class Yolov8_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_box, renormalize):
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        model = attempt_load_weights(weight, device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()

        target = yolov8_target(backward_type, conf_threshold, ratio)
        target_layers = [model.model[l] for l in layer]#存储以sequential保存
        method = eval(method)(model, target_layers, use_cuda=device.type == 'cuda')
        method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)

        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype("int")
        self.__dict__.update(locals())#可以将局部变量转为类别的属性

    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.65)[0]
        return result

    # def draw_detections(self, box, color, name, img):
    #     xmin, ymin, xmax, ymax = list(map(int, list(box)))
    #     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
    #     cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
    #                 lineType=cv2.LINE_AA)
    #     return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1]
        inside every bounding boxes, and zero outside of the bounding boxes. """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized

    def process(self, img_path, save_path):
        # img process
        img = cv2.imread(img_path)
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        try:
            grayscale_cam = self.method(tensor, [self.target])
        except AttributeError as e:
            return

        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

        pred = self.model(tensor)[0]
        pred = self.post_process(pred)
        if self.renormalize:
            cam_image = self.renormalize_cam_in_bounding_boxes(pred[:, :4].cpu().detach().numpy().astype(np.int32), img,
                                                               grayscale_cam)
        if self.show_box:
            draw = Drawer(if_inference=True, cls_dict=self.model_names)
            pred = pred.cpu().detach().numpy()
            bboxes,clss,scores= pred[:,:4],pred[:,-1],pred[:,-2]
            cam_image = np.array(draw(Image.fromarray(cam_image),bboxes,clss,scores))
            # for data in pred:
            #     data = data.cpu().detach().numpy()
            #     cam_image = self.draw_detections(data[:4], self.colors[int(data[4:].argmax())],
            #                                      f'{self.model_names[int(data[4:].argmax())]} {float(data[4:].max()):.2f}',
            #                                      cam_image)

        cam_image = Image.fromarray(cam_image)
        print(save_path)
        cam_image.save(save_path)

    def __call__(self, img_path, save_path):
        # remove dir if exist
        base_name = os.path.basename(img_path)
        # if os.path.exists(save_path):
        #     shutil.rmtree(save_path)
        # # make dir if not exist
        # os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(img_path):
            for img_path_ in os.listdir(img_path):
                self.process(f'{img_path}/{img_path_}', f'{save_path}/{img_path_}')
        else:
            self.process(img_path, f'{save_path}/{base_name}')