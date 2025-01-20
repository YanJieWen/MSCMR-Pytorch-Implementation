'''
@File: cal_map.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 1月 13, 2025
@HomePage: https://github.com/YanJieWen
'''


import os
import numpy as np
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pandas as pd
from glob import glob


class PR_curve():
    def __init__(self,gt_path='./crash_test.json',thr=-1,pred_path=None):
        '''
        需要提前将预测结果保存为json文件
        precisions[T, R, K, A, M]
        T: iou thresholds [0.5 : 0.05 : 0.95], idx from 0 to 9
        R: recall thresholds [0 : 0.01 : 1], idx from 0 to 100
        K: category, idx from 0 to ...
        A: area range, (all, small, medium, large), idx from 0 to 3
        M: max dets, (1, 10, 100), idx from 0 to 2
        :param data_root: './datasets/POCOCO'
        :param data_type:'test'
        :param thr:[0.5:0.05:0.95],-1 当为-1的时候为为0.5:0.95的评估
        :param save_root:'./weights/cspdark_dyhead_our/'
        :return:
        '''
        coco = COCO(gt_path)
        coco_dt = coco.loadRes(pred_path)
        self.coco_eval = COCOeval(coco,coco_dt,'bbox')
        self.thr = thr
        self.save_name = os.path.basename(pred_path).split('.')[0]
        self.coco_eval.evaluate()
        self.coco_eval.accumulate()
        self.coco_eval.summarize()
    def __call__(self, *args, **kwargs):
        p_value = self.coco_eval.eval['precision']
        recall = np.mat(np.arange(0.0,1.01,0.01)).T
        max_dets = -1
        if self.thr==-1:
            map_all_pr = np.mean(p_value[:,:,:,0,max_dets],axis=0)
        else:
            T = int((self.thr-0.5)/0.05)
            map_all_pr = p_value[T,:,:,0,max_dets]
        data = np.hstack((np.hstack((recall, map_all_pr)),
                          np.mat(np.mean(map_all_pr, axis=1)).T))
        df = pd.DataFrame(data)
        save_name = f'PR_{self.save_name}@ap{int(self.thr)}.xlsx'
        save_path = os.path.join('./results/',save_name)
        df.to_excel(save_path,index=False)
        print(f'{"*"*10}Table has been saved to results dir{"*"*10}')