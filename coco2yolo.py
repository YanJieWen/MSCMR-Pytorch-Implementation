'''
@File: coco2yolo.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 6月 03, 2024
@HomePage: https://github.com/YanJieWen
'''
import copy
import os
import json

import numpy as np
from tqdm import tqdm
import argparse
import sys


# def convert_(boxes,w,h):
#     x_c = boxes[:,1]+boxes[:,3]/2.
#     y_c = boxes[:,2]+boxes[:,4]/2.
#     w_i = boxes[:,3]
#     h_i = boxes[:,4]
#     boxes[:,1] = x_c/w
#     boxes[:,2] = y_c/h
#     boxes[:,3] = w_i/w
#     boxes[:,4] = h_i/h

# def coco2yolo(args):
#     json_file = args.json_file#'./datasets/Crash2024/annotations/train.json'
#     ann_save_root = args.ann_save_root#'./datasets/CrashYolo/labels/train2024'

#     with open(json_file,'r') as r:
#         data = json.load(r)
#     if not os.path.exists(ann_save_root):
#         os.makedirs(ann_save_root)
#     #将id置为0
#     id_map = {}
#     for i,cat in enumerate(data['categories']):
#         id_map[cat['id']] = i #{1:0}
#     #先将annotations包裹起来
#     img_metas = {}
#     for i,anns in enumerate(data['annotations']):
#         if anns['image_id'] not in img_metas.keys():
#             info = [[int(id_map.get(anns['category_id']))]+anns['bbox']]
#             img_metas[anns['image_id']] = info
#         else:
#             info = [int(id_map.get(anns['category_id']))]+anns['bbox']
#             img_metas[anns['image_id']].append(info)
#     pbar = tqdm(data['images'],file=sys.stdout,desc='Transform....')
#     for img in pbar:
#         file_name = img["file_name"]
#         img_width = img['width']
#         img_height = img['height']
#         img_id = img['id']
#         pbar.desc = f'*{file_name}*'
#         head,tail = os.path.splitext(file_name)
#         ann_txt_name = head+'.txt'
#         origin_info = img_metas.get(img_id)
#         origin_boxes = np.array(copy.deepcopy(origin_info)) #[N,5]
#         convert_(origin_boxes,img_width,img_height)
#         np.savetxt(os.path.join(ann_save_root,ann_txt_name),origin_boxes)





# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--json_file',default='./dataset/annotations/instances_train2017.json',type=str)
#     parser.add_argument('--ann_save_root',default='./COCO/labels/train2017',type=str)

#     args = parser.parse_args()
#     print(args)
#     coco2yolo(args)


import os 
import json
from tqdm import tqdm
import argparse
 
parser = argparse.ArgumentParser()
parser.add_argument('--json_path', default='dataset/annotations/instances_train2017.json',type=str, help="input: coco format(json)")
parser.add_argument('--save_path', default='datasets/coco/labels/train2017', type=str, help="specify where to save the output dir of labels")
arg = parser.parse_args()
 
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
 
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)
 
if __name__ == '__main__':
    json_file =   arg.json_path # COCO Object Instance 类型的标注
    ana_txt_save_path = arg.save_path  # 保存的路径
 
    data = json.load(open(json_file, 'r'))
    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)
    
    id_map = {} # coco数据集的id不连续！重新映射一下再输出！
    for i, category in enumerate(data['categories']): 
        id_map[category['id']] = i
 
    # 通过事先建表来降低时间复杂度
    max_id = 0
    for img in data['images']:
        max_id = max(max_id, img['id'])
    # 注意这里不能写作 [[]]*(max_id+1)，否则列表内的空列表共享地址
    img_ann_dict = [[] for i in range(max_id+1)] 
    for i, ann in enumerate(data['annotations']):
        img_ann_dict[ann['image_id']].append(i)
 
    for img in tqdm(data['images']):
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, tail = os.path.splitext(filename)
        ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
        f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
        '''for ann in data['annotations']:
            if ann['image_id'] == img_id:
                box = convert((img_width, img_height), ann["bbox"])
                f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))'''
        # 这里可以直接查表而无需重复遍历
        for ann_id in img_ann_dict[img_id]:
            ann = data['annotations'][ann_id]
            box = convert((img_width, img_height), ann["bbox"])
            f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
        f_txt.close()