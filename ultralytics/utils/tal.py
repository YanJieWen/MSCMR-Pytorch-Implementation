# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn

from .checks import check_version
from .metrics import bbox_iou,probiou,rf_overlaps
from .ops import xywhr2xyxyxyxy

TORCH_1_10 = check_version(torch.__version__, "1.10.0")


class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9,assign_type='point',quality=False):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.assign_type = assign_type
        self.quality = quality
        #新增的rfla参数
        if assign_type=='rf':
            #修改PLA的参数
            self.assiner = HieAssigner(topk=[3,1],assign_mertic='kl',ratio=0.9)
            print('HieAssigner is adopted')
        else:
            self.assiner = None
            print('point-based assigner is adopted')

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, rfields, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment. Reference code is available at
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py.

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            rfields (Tensor): shape(num_total_anchors, 4)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        # print(f'anchors:{anc_points.shape},rfields:{rfields.shape},pd_bbooxes:{pd_bboxes.shape}')

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx).to(device),
                torch.zeros_like(pd_bboxes).to(device),
                torch.zeros_like(pd_scores).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
            )
        #修改1： 将anc_points替换为感受野rfields进行overlaps计算,获取正样本蒙版
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points,
            rfields if self.assiner is not None else None, mask_gt
        )
        #step2: 过滤掉1个anchor对应多个gt的情况，仅保留最高的gt索引
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        #step3：获取正负样本标签
        # Assigned target，上述步骤是为了获取正负样本的蒙版以保证与anchor的数目对齐，该步骤是根据索引进行标签分配
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes,
                                                                       target_gt_idx, fg_mask,overlaps)

        # step4:Normalize，动态分配策略-->根据aligen和overlaps获取值
        target_scores = self.get_norm_tgt_scores(align_metric,overlaps,target_scores,mask_pos)
        # align_metric *= mask_pos#对齐矩阵过滤
        # pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        # #重叠度矩阵过滤
        # pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        # #(GT与所有anchor的对齐值 x 每个GT与所有anchor最大IOU / 每个GT与所有anchor最大对齐值)，求每个anchor与所有gt的最大值->获得惩罚因子
        # norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        # target_scores = target_scores * norm_align_metric#为类别添加惩罚项，预测结果越差则align越小，loss越大

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx
    def get_norm_tgt_scores(self,align_metric,overlaps,tgt_scores,mask_pos):
        align_metric *= mask_pos#1个gt可以被多个anchor负责预测而1个anchor最多负责预测1个gt预测
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        # (GT与所有anchor的对齐值 x 每个GT与所有anchor最大IOU / 每个GT与所有anchor最大对齐值)，求每个anchor与所有gt的最大值->获得惩罚因子
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        if self.quality:
            # return tgt_scores*norm_align_metric
            return tgt_scores
        else:
            # print(tgt_scores.shape,norm_align_metric.shape)
            return tgt_scores*norm_align_metric
            # return tgt_scores

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, rfields, mask_gt):
        """Get in_gts mask, (b, max_num_obj, h*w)."""
        # bs,a,_ = pd_bboxes.shape
        # bs, m, _ = gt_bboxes.shape
        # mask_in_gts = gt_bboxes.new_full((bs,m,a),1)
        #step1: 初步筛选出中心点在GT内部的样本-->基于中心点的匹配以及基于感受野的匹配
        if rfields is None:
            mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        else:
            mask_in_gts = self.select_rfla_over_gts(rfields,gt_bboxes)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        #对于实例i网格j预测为类别c的置信度
        #获取每个样本中每个网格对应gt类别的预测得分，存储每个网格对应正确分类的得分，表示对于每个正样本，网格给出的置信度
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for horizontal bounding boxes."""
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            topk_mask (Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.

        Returns:
            (Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """

        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)#没有实例的地方为0，有实例的地方变为m的索引
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            #以topk_ids为index，ones为src,往count_tensor中填充值，topk(index)必须和ones的形状保持一致，将GT与anchor进行关联
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8, device=topk_idxs.device))
        # Filter invalid bboxes-->排序存在相同的索引？
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx,fg_mask,overlaps=None):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                          positive anchor points.
                - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                          for positive anchor points.
                - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                          for positive anchor points, where num_classes is the number
                                          of object classes.
        """
        #是否启用质量损失
        if self.quality and overlaps is not None:
            num_anchors,num_batch = target_gt_idx.shape[1],target_gt_idx.shape[0]
            b_idx = torch.arange(end=num_batch,dtype=torch.long,device=overlaps.device).view(-1,1).expand(-1,num_anchors)
            m_idx = torch.arange(end=num_anchors,dtype=torch.long,device=overlaps.device).view(1,-1).expand(num_batch,-1)
            _overlaps = overlaps[b_idx,target_gt_idx,m_idx].unsqueeze(-1)#[b,m,1]
        else:
            _overlaps = torch.ones_like(target_gt_idx.unsqueeze(-1),dtype=torch.int64,device=target_gt_idx.device)


        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)通过展平获得连续的tgt索引
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)获得按照索引排序的tgt_labels

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot(),可以改进的地方使用质量表征损失，赋值不是而是overlaps的值
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64 if not self.quality else torch.float,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), _overlaps)#此处将定值1修改为overlaps
        #因为tgt_id在没有任何实例的情况下也会被添加overlap,因此需要根据还需要再过滤一遍，即过滤掉负样本
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        Select the positive anchor center in gt.基于点的分配

        Args:
            xy_centers (Tensor): shape(h*w, 2)
            gt_bboxes (Tensor): shape(b, n_boxes, 4)[xyxy]

        Returns:
            (Tensor): shape(b, n_boxes, h*w)
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
        #amin为返回最小值
        return bbox_deltas.amin(3).gt_(eps)

    def select_rfla_over_gts(self,rfields,gt_bboxes):
        '''
        计算感受野与gt的overlaps并进行初步筛选,为每个gt匹配k个正样本
        Args:
            gt_bboxes: shape(b, n_boxes, 4)[xyxy]
            rfields: shape[A,4]

        Returns:

        '''
        mask_in_gts = self.assiner.assign(rfields,gt_bboxes)
        #分层匹配方案
        return mask_in_gts.bool()




    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        If an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.

        Args:
            mask_pos (Tensor): shape(b, n_max_boxes, h*w)
            overlaps (Tensor): shape(b, n_max_boxes, h*w)

        Returns:
            target_gt_idx (Tensor): shape(b, h*w)
            fg_mask (Tensor): shape(b, h*w)
            mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        """
        # (b, n_max_boxes, h*w) -> (b, h*w)
        #1个gt可被用于多个anchors的表示
        #排除1个anchor对应多个GT的情况
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)返回最大IOU中GT的索引

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            #构造的空矩阵将最大CIOU的位置置为1，其余的元素为0-->根据最大的IoU索引保证每个anchor仅由1个gt负责预测
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            #过滤掉重复匹配的情况 where(condition,input,other),将大于1的位置元素赋值为0
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos


class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for rotated bounding boxes."""
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes):
        """
        Select the positive anchor center in gt for rotated bounding boxes.

        Args:
            xy_centers (Tensor): shape(h*w, 2)
            gt_bboxes (Tensor): shape(b, n_boxes, 5)

        Returns:
            (Tensor): shape(b, n_boxes, h*w)
        """
        # (b, n_boxes, 5) --> (b, n_boxes, 4, 2)
        corners = xywhr2xyxyxyxy(gt_bboxes)
        # (b, n_boxes, 1, 2)
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a
        ad = d - a

        # (b, n_boxes, h*w, 2)
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(dim=-1)
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap * ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)  # is_in_box


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    rfields = []
    trfs = gen_trf('s')
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
        center_point = torch.stack((sx, sy), -1).view(-1, 2)#A,2
        rfields.append(gen_erf(center_point,stride=stride,trfs=trfs,dist_ratio=1/3,num_index=i))
        anchor_points.append(center_point)
        #生成一个张量并填充为stride
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

    return torch.cat(anchor_points), torch.cat(stride_tensor),torch.cat(rfields)

def gen_erf(center_point,stride,trfs,dist_ratio=1/2,num_index=0):
    '''
    获取有效感受野
    Args:
        center_point: 每个特征图的中心点坐标,原文是左上角坐标
        trfs: 所有特征层的感受野
        dist_ratio: 估计的有效感受野相对理论感受野的缩放比例
        num_index: 层索引

    Returns:tensor[A,4]
    '''
    # center_point = (center_point)*stride+stride//2
    center_point = (center_point - 0.5) * stride+stride//2
    rfnum = num_index+1
    if rfnum==0:
        rf = trfs[rfnum]*dist_ratio
    elif rfnum==1:
        rf = trfs[rfnum]*dist_ratio
    elif rfnum==2:
        rf = trfs[rfnum]*dist_ratio
    elif rfnum==3:
        rf = trfs[rfnum]*dist_ratio
    elif rfnum==4:
        rf = trfs[rfnum]*dist_ratio
    else:
        raise ValueError('out of index')
    px1 = center_point[...,0]- rf/2
    py1 = center_point[..., 1] - rf / 2
    px2 = center_point[..., 0] + rf / 2
    py2 = center_point[..., 1] + rf / 2
    rfield = torch.cat((px1[..., None], py1[..., None]), dim=1)
    rfield = torch.cat((rfield, px2[..., None]), dim=1)
    rfield = torch.cat((rfield, py2[..., None]), dim=1)
    return rfield



def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """
    Decode predicted object bounding box coordinates from anchor points and distribution.

    Args:
        pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).
        anchor_points (torch.Tensor): Anchor points, (h*w, 2).
    Returns:
        (torch.Tensor): Predicted rotated bounding boxes, (bs, h*w, 4).
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)


def gen_trf(size_type='n'):
    '''
    获得理论感受野--基于YOLOV8的骨干结构
    Args:ref: https://distill.pub/2019/computing-receptive-fields/
    #https://github.com/Chasel-Tsui/mmdet-rfla/blob/main/mmdet/models/dense_heads/rfla_fcos_head.py
        size_type: ['n', 's', 'm', 'l', 'x']

    Returns: 各特征图的感受野

    '''
    base = [1,1,3,1,6,1,6,1,3]
    depth_ratios = [0.33, 0.33, 0.67, 1, 1]
    names = ['n', 's', 'm', 'l', 'x']
    all_sizes = {}
    for name, depth in zip(names, depth_ratios):
        if name not in all_sizes.keys():
            all_sizes[name] = list(map(lambda x: max(round(x * depth), 1) if x > 1 else x, base))
        else:
            pass
    assert size_type in names, f'{size_type} is not in the list'
    num_blocks = all_sizes[size_type]
    jin = [1]  # 特征间隔
    for i in range(5):
        j = jin[i] * 2
        jin.append(j)
    r0 = 1
    r1 = r0 + (3 - 1) * jin[0] * num_blocks[0]

    r2 = r1 + (3 - 1) * jin[1] * num_blocks[1]
    trf_p2 = r2 + (3 - 1) * jin[2] * num_blocks[2]

    r3 = trf_p2 + (3 - 1) * jin[2] * num_blocks[3]
    trf_p3 = r3 + (3 - 1) * jin[3] * num_blocks[4]

    r4 = trf_p3 + (3 - 1) * jin[3] * num_blocks[5]
    trf_p4 = r4 + (3 - 1) * jin[4] * num_blocks[6]

    r5 = trf_p4 + (3 - 1) * jin[4] * num_blocks[7]
    trf_p5 = r5 + (3 - 1) * jin[5] * num_blocks[8]

    trfs = [trf_p2, trf_p3, trf_p4, trf_p5]
    return trfs


class HieAssigner():
    def __init__(self,topk=[200,100],ratio=0.9,assign_mertic='kl'):
        self.topk = topk
        self.ratio = ratio
        self.assign_metric = assign_mertic

    def assign(self,bboxes,gt_bboxes):
        '''
        层次化标签分配
        Args:
            bboxes:(m,4)
            gt_bboxes: (b,n,4)

        Returns:

        '''
        #1.计算感受野与gt的kl距离
        num_gts = gt_bboxes.size(0)
        overlaps = rf_overlaps(gt_bboxes,bboxes[None].repeat(num_gts,1,1),mode=self.assign_metric)#[b,n,m]
        bboxes2 = self.anchor_rescale(bboxes,self.ratio)
        overlaps2 = rf_overlaps(gt_bboxes,bboxes2[None].repeat(num_gts,1,1),mode=self.assign_metric)
        k1,k2 = self.topk[0],self.topk[1]
        assigned_gt_inds = self.assign_wrt_ranking(overlaps, k1)
        mask_in_gts = self.reassign_wrt_ranking(assigned_gt_inds, overlaps2, k2)

        return mask_in_gts

    @staticmethod
    def assign_wrt_ranking(overlaps,k1=200):
        '''
        初次筛选：待完成半径衰减，二次筛选
        Args:
            overlaps:(b,n,m)
            k1:int

        Returns:

        '''
        # TODO: ensure each gt is assigened and each anchor only match single GT-->COLUM ONLY 1
        num_gts,num_bboxes = overlaps.size(1),overlaps.size(2)
        # assigned_gt_inds = overlaps.new_full((num_bboxes,),-1,dtype=torch.long)#令正样本的数值为-1
        # mask_in_gts = torch.zeros_like(overlaps,dtype=torch.long,device=overlaps.device)
        mask_in_gts = torch.zeros(overlaps.shape, dtype=torch.int8,device=overlaps.device)
        if num_gts==0 or num_bboxes==0:
            return mask_in_gts
        _, gt_argmax_overlaps = overlaps.topk(k1, dim=-1, largest=True, sorted=True)  # 与gt最匹配的先验索引(n,k)
        ones = torch.ones_like(gt_argmax_overlaps[...,:1],dtype=torch.int8,device=overlaps.device)
        for k in range(k1):
            mask_in_gts.scatter_add_(-1,gt_argmax_overlaps[...,k:k+1],ones)
        mask_in_gts.masked_fill_(mask_in_gts > 1, 0)#过滤无效的bboxes
        # src = torch.ones_like(gt_argmax_overlaps,dtype=torch.long,device=overlaps.device)
        # mask_in_gts.scatter_(-1,gt_argmax_overlaps,src)#核心语句将src按照索引赋值给mask_in_gts
        # max_overlaps,argmax_overlaps = overlaps.max(dim=0)#与网格最匹配的gt索引(m)
        # gt_max_overlaps,gt_argmax_overlaps =overlaps.topk(k,dim=1,largest=True,sorted=True)#与gt最匹配的网格索引(n,k)
        # assigned_gt_inds[(max_overlaps>=0)&(max_overlaps<0.8)]=0#令满足条件的网格置为0，剩下的保持为-1，由于kl值很小，基本仍然为-1
        #
        # for i in range(num_gts):
        #     for j in range(k):
        #         max_overlaps_inds = overlaps[i,:]==gt_max_overlaps[i,j]
        #         assigned_gt_inds[max_overlaps_inds] = i+1
        return mask_in_gts

    @staticmethod
    def reassign_wrt_ranking(once_assign_results,overlaps,k2=100):
        num_gts,num_bboxes = overlaps.size(1),overlaps.size(2)
        mask1 = once_assign_results<=0
        mask2 = once_assign_results>0
        # mask_in_gts = torch.zeros_like(overlaps,dtype=torch.long,device=overlaps.device)
        mask_in_gts = torch.zeros(overlaps.shape,dtype=torch.int8,device=overlaps.device)
        if num_gts==0 or num_bboxes==0:
            return mask_in_gts
        _, gt_argmax_overlaps = overlaps.topk(k2, dim=-1, largest=True, sorted=True)
        ones = torch.ones_like(gt_argmax_overlaps[...,:1],dtype=torch.int8,device=overlaps.device)
        for k in range(k2):
            mask_in_gts.scatter_add_(-1,gt_argmax_overlaps[...,k:k+1],ones)
        # src = torch.ones_like(gt_argmax_overlaps, dtype=torch.long, device=overlaps.device)
        # mask_in_gts.scatter_(-1, gt_argmax_overlaps, src)
        mask_in_gts = mask_in_gts*mask1+once_assign_results*mask2
        return mask_in_gts


    @staticmethod
    def anchor_rescale(bboxes,ratio):
        center_x2 =(bboxes[...,2]+bboxes[...,0])/2
        center_y2 = (bboxes[...,3]+bboxes[...,1])/2
        w2 = bboxes[...,2]-bboxes[...,0]
        h2 = bboxes[...,3]-bboxes[...,1]
        bboxes[..., 0] = center_x2 - w2 * ratio / 2
        bboxes[..., 1] = center_y2 - h2 * ratio / 2
        bboxes[..., 2] = center_x2 + w2 * ratio / 2
        bboxes[..., 3] = center_y2 + h2 * ratio / 2
        return bboxes

