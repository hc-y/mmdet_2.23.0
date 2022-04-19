# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from ..transforms import bbox_xyxy_to_cxcywh
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class UniformAssigner(BaseAssigner):
    """Uniform Matching between the anchors and gt boxes, which can achieve
    balance in positive anchors, and gt_bboxes_ignore was not considered for
    now.

    Args:
        pos_ignore_thr (float): the threshold to ignore positive anchors
        neg_ignore_thr (float): the threshold to ignore negative anchors
        match_times(int): Number of positive anchors for each gt box.
           Default 4.
        iou_calculator (dict): iou_calculator config
    """

    def __init__(self,
                 pos_ignore_thr,
                 neg_ignore_thr,
                 match_times=4,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.match_times = match_times
        self.pos_ignore_thr = pos_ignore_thr
        self.neg_ignore_thr = neg_ignore_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self,
               bbox_pred,
               anchor,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              0,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            assign_result = AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
            assign_result.set_extra_property(
                'pos_idx', bbox_pred.new_empty(0, dtype=torch.bool))
            assign_result.set_extra_property('pos_predicted_boxes',
                                             bbox_pred.new_empty((0, 4)))
            assign_result.set_extra_property('target_boxes',
                                             bbox_pred.new_empty((0, 4)))
            return assign_result

        # 2. Compute the L1 cost between boxes
        # Note that we use anchors and predict boxes both
        cost_bbox = torch.cdist(
            bbox_xyxy_to_cxcywh(bbox_pred),
            bbox_xyxy_to_cxcywh(gt_bboxes),
            p=1)
        cost_bbox_anchors = torch.cdist(
            bbox_xyxy_to_cxcywh(anchor), bbox_xyxy_to_cxcywh(gt_bboxes), p=1)

        # We found that topk function has different results in cpu and
        # cuda mode. In order to ensure consistency with the source code,
        # we also use cpu mode.
        # TODO: Check whether the performance of cpu and cuda are the same.
        C = cost_bbox.cpu()
        C1 = cost_bbox_anchors.cpu()

        # self.match_times x n  # hc-y_note1214:n表示num_gts
        index = torch.topk(
            C,  # c=b,n,x c[i]=n,x
            k=self.match_times,
            dim=0,
            largest=False)[1]  # hc-y_note1214:for each gt, which pred best matchs with it

        # self.match_times x n
        index1 = torch.topk(C1, k=self.match_times, dim=0, largest=False)[1]  # hc-y_note1214:for each gt, which anchor best matchs with it
        # (self.match_times*2) x n
        indexes = torch.cat((index, index1),
                            dim=1).reshape(-1).to(bbox_pred.device)  # hc-y_note1216:indexes中存在相同的值, indexes.unique().numel()

        pred_overlaps = self.iou_calculator(bbox_pred, gt_bboxes)  # hc-y_note1214:torch.Size([num_preds, num_gts])
        anchor_overlaps = self.iou_calculator(anchor, gt_bboxes)  # hc-y_note1214:torch.Size([num_anchors, num_gts])
        pred_max_overlaps, _ = pred_overlaps.max(dim=1)  # hc-y_note1214:for each pred, which gt best overlaps with it
        anchor_max_overlaps, _ = anchor_overlaps.max(dim=0)  # hc-y_note1214:for each gt, which anchor best overlaps with it

        # 3. Compute the ignore indexes use gt_bboxes and predict boxes  # hc-y_note1214:使得负样本不包括 IoU(pred,gt)>0.7重叠很好的anchor位置
        ignore_idx = pred_max_overlaps > self.neg_ignore_thr
        assigned_gt_inds[ignore_idx] = -1

        # 4. Compute the ignore indexes of positive sample use anchors and gt_boxes  # hc-y_note1214:使得正样本不包括 IoU(anchor,gt)<0.15重叠很差的anchor位置
        pos_gt_index = torch.arange(
            0, C1.size(1),
            device=bbox_pred.device).repeat(self.match_times * 2)  # hc-y_note1214:torch.Size([num_gts*self.match_times*2])
        pos_ious = anchor_overlaps[indexes, pos_gt_index]  # hc-y_note1214:torch.Size([self.match_times*num_gts*2])
        pos_ignore_idx = pos_ious < self.pos_ignore_thr

        flag_switch = 'Default'
        flag_switch = 'Modified'  # hc-y_modify1217:
        if flag_switch == 'Default':
            pos_gt_index_with_ignore = pos_gt_index + 1
            pos_gt_index_with_ignore[pos_ignore_idx] = -1
            # hc-y_note1216:由于indexes中存在相同的值, 故这里会发生后写入的值覆盖先写入的值, 从而导致torch.nonzero(pos_gt_index_with_ignore > 0, as_tuple=False).squeeze(-1).numel() 
            # 会多于 torch.nonzero(assigned_gt_inds[indexes] > 0, as_tuple=False).squeeze(-1).numel();
            assigned_gt_inds[indexes] = pos_gt_index_with_ignore
        else:
            pos_idx_candidate_sorted, pos_idx_candidate_sorted_ssubindices = indexes[~pos_ignore_idx].sort()
            pos_indices_mask = bbox_pred.new_ones((pos_ignore_idx.size(0),), dtype=torch.bool)
            pos_idx_candidate = torch.arange(0, pos_ignore_idx.size(0), device=bbox_pred.device)
            pos_idx_oncemore = pos_idx_candidate_sorted[torch.nonzero((pos_idx_candidate_sorted[1:] - pos_idx_candidate_sorted[:-1]) == 0).view(-1)+1]  # hc-y_note1217:找出~pos_ignore_idx中属于"某个index可匹配多个gt_bbox"情况的index
            # 对于"某个index可匹配多个gt_bbox"的情况, 若该index位于由index1得到的索引index_anchor中则保留anchor_overlaps最大的那1个;
            # 若该index不位于由index1得到的索引index_anchor中则在由index得到的索引index_pred中保留pred_overlaps最大的那1个; 
            # 通过这样做, 可以确保给 torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1) 这些正样本标记的类别更合理而不是覆盖该写入;
            for _pos_idx_oncemore in pos_idx_oncemore:
                indexes_1, _pos_idx_candidate_1, pos_ious_1 = indexes[index.numel():], ~pos_ignore_idx[index.numel():], pos_ious[index.numel():]
                _ssubindices_1 = torch.nonzero(indexes_1[_pos_idx_candidate_1] == _pos_idx_oncemore).view(-1)
                if _ssubindices_1.numel() > 0:
                    pos_idx_1_exclude = pos_idx_candidate[:index.numel()][_pos_idx_candidate_1][
                        _ssubindices_1[pos_ious_1[_pos_idx_candidate_1][_ssubindices_1].sort(descending=True)[1][1:]]]
                    pos_indices_mask[pos_idx_1_exclude] = False  # pos_gt_index[pos_idx_1_exclude]

                    indexes_2, _pos_idx_candidate_2, pos_ious_2 = indexes[:index.numel()], ~pos_ignore_idx[:index.numel()], pos_ious[:index.numel()]
                    _ssubindices_2 = torch.nonzero(indexes_2[_pos_idx_candidate_2] == _pos_idx_oncemore).view(-1)
                    if _ssubindices_2.numel() > 0:
                        pos_idx_2_exclude = pos_idx_candidate[:index.numel()][_pos_idx_candidate_2][_ssubindices_2]
                        pos_indices_mask[pos_idx_2_exclude] = False  # pos_gt_index[pos_idx_2_exclude]
                else:
                    indexes_2, _pos_idx_candidate_2, pos_ious_2 = indexes[:index.numel()], ~pos_ignore_idx[:index.numel()], pos_ious[:index.numel()]
                    _ssubindices_2 = torch.nonzero(indexes_2[_pos_idx_candidate_2] == _pos_idx_oncemore).view(-1)
                    if _ssubindices_2.numel() > 0:
                        pos_idx_2_exclude = pos_idx_candidate[:index.numel()][_pos_idx_candidate_2][
                            _ssubindices_2[pos_ious_2[_pos_idx_candidate_2][_ssubindices_2].sort(descending=True)[1][1:]]]
                        pos_indices_mask[pos_idx_2_exclude] = False  # pos_gt_index[pos_idx_2_exclude]
                # _ssubindices = torch.nonzero(indexes[~pos_ignore_idx] == _pos_idx_oncemore).view(-1)
                # pos_ious[~pos_ignore_idx][_ssubindices].sort(descending=True)

            pos_indices_mask[pos_ignore_idx] = False
            assigned_gt_inds[indexes[pos_indices_mask]] = pos_gt_index[pos_indices_mask] + 1
            # 可以观测到: (~pos_ignore_idx).sum() 多于 pos_indices_mask.sum(); indexes[~pos_ignore_idx].unique().numel() 等于 indexes[pos_indices_mask].unique().numel();

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        assign_result = AssignResult(
            num_gts,
            assigned_gt_inds,
            anchor_max_overlaps,
            labels=assigned_labels)
        # hc-y_note1216:**indexes中存在相同的值**, 作者在计算reg loss时使用的是pos_idx而不是torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1)
        # 作为sampling_pos_inds, 可以使得each gt_bbox有足够的bbox_pred与之匹配, 但会导致"某个index可匹配多个gt_bbox"的情况, 即,
        # 例如, "(anchor_index1,gt_1),(pred_index1,gt_1),(pred_index1,gt_2)"均符合pos_idx的要求, 但这之中的index1既可以匹配gt_1又可以匹配gt_2;
        # hc-y_note1217:作者在计算cls loss时对于正样本使用的是torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1)而不是pos_idx
        # 作为sampling_pos_inds;
        assign_result.set_extra_property('pos_idx', ~pos_ignore_idx)
        assign_result.set_extra_property('pos_predicted_boxes',
                                         bbox_pred[indexes])
        assign_result.set_extra_property('target_boxes',
                                         gt_bboxes[pos_gt_index])
        return assign_result
