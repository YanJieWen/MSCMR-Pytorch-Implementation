'''
@File: get_bboxes.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 12月 11, 2024
@HomePage: https://github.com/YanJieWen
'''

import PIL.ImageDraw as Imagedraw
from PIL import ImageColor,ImageFont
import matplotlib.pyplot as plt
import numpy as np

import os


class Drawer(object):
    def __init__(self,type='voc',if_inference=False,if_shuffle=True,cls_dict=None) -> None:
        colors =  [
            'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
            'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
            'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
            'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
            'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
            'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
            'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
            'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
            'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
            'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
            'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
            'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
            'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
            'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
            'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
            'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
            'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
            'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
            'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
            'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
            'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
            'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
            'WhiteSmoke', 'Yellow', 'YellowGreen']
        if if_shuffle:
            self.colors = np.random.permutation(colors)
        else:
            self.colors = colors
        self.infer = if_inference
        self.type = type

        self.cls_dict = cls_dict#dict{int:str}
    def convert_bbox(self,img,bbox):
        w,h = img.size
        if self.type=='voc':
            return bbox
        elif self.type=='yolo':
            norm_cx,norm_cy,norm_w,norm_h = bbox
            left, top = (norm_cx - norm_w / 2) * w, (norm_cy - norm_h / 2) * h
            right, bottom = (norm_cx + norm_w / 2) * w, (norm_cy + norm_h / 2) * h
            return [left,top,right,bottom]
        elif self.type=='coco':
            left,top,_w,_h = bbox
            right,bottom = left+_w,top+_h
            return [left,top,right,bottom]
        else:
            raise ValueError(f'{self.type} is not found')
    @staticmethod
    def draw_text(draw,box,cls,score,cls_dict,color,font='arial.ttf',font_size=18):
        try:
            font = ImageFont.truetype(font, font_size)
        except IOError:
            font = ImageFont.load_default()
        left, top, right, bottom = box
        display_str = f"{cls_dict[int(cls)]}: {int(100 * score)}%" if cls_dict is not None else \
            f'{int(cls)}: {int(100 * score)}%'
        display_str_heights = [font.getsize(ds)[1] for ds in display_str]
        display_str_height = (1 + 2 * 0.05) * max(display_str_heights)
        if top > display_str_height:
            text_top = top - display_str_height
            text_bottom = top
        else:
            text_top = bottom
            text_bottom = bottom + display_str_height

        for ds in display_str:
            text_width, text_height = font.getsize(ds)
            margin = np.ceil(0.05 * text_width)
            draw.rectangle([(left, text_top),
                            (left + text_width + 2 * margin, text_bottom)], fill=color)
            draw.text((left + margin, text_top),
                    ds,
                    fill='black',
                    font=font)
            left += text_width

    def __call__(self,img,bboxes,clss,scores=None):
        colors = [ImageColor.getrgb(self.colors[cls % len(self.colors)]) for cls in
                      np.asarray(clss, dtype=int)]
        draw = Imagedraw.Draw(img)
        if not self.infer:
            scores = np.ones_like(clss)
        for box,cls,score,color in zip(bboxes,clss,scores,colors):
            bbox= self.convert_bbox(img,box)
            left,top,right,bottom = bbox
            draw.line([(left, top), (left, bottom), (right, bottom),
                       (right, top), (left, top)], width=3, fill=color)
            self.draw_text(draw,bbox, int(cls), float(score), self.cls_dict, color)
        return img
def save_fig(fig_id,out_root, tight_layout=True, fig_extension="png", resolution=300):
    # IMAGES_PATH = './out/'
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    path = os.path.join(out_root, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)