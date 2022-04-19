# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class ArgoverseDataset(CocoDataset):

    CLASSES = ('person',  'bicycle',  'car',  'motorcycle',  'bus',  'truck',  
                'traffic_light',  'stop_sign')

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        sequences = self.coco.dataset['sequences']  # hc-y_add1031:
        seq_dirs = self.coco.dataset['seq_dirs']  # hc-y_add1031:
        data_infos = []
        total_ann_ids = []
        # hc-y_note1231:只采样出少量的图片用于训练和评估;
        if 'train.json' in ann_file:
            self.img_ids = self.img_ids[::4]  # './../datasets/Argoverse-1.1/annotations/train.json' 39384 images
        elif 'val.json' in ann_file:
            self.img_ids = self.img_ids[::5]  # './../datasets/Argoverse-1.1/annotations/val.json' 15062 images
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = seq_dirs[info['sid']] + '/' + info['name']  # hc-y_modify1031:
            # from pathlib import Path  # hc-y_add1031:
            # Path('/mnt/data1/yuhangcheng/yhc_workspace/datasets/Argoverse-1.1/images/' + seq_dirs[info['sid']]+'/'+info['name']).exists()  # hc-y_add1031:
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def evaluate_v2(self,
                 results,
                 metric='bbox',
                 logger=None,
                 runner=None,  # hc-y_add0108:correspond to /envs/mmlab/lib/python3.9/site-packages/mmcv/runner/hooks/evaluation.py
                 val_dir=None,  # hc-y_add0110:path to the folder where checkpoint_file locates;
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        flag_train_last_epoch = False if runner is None else runner.epoch == runner._max_epochs - 1
        if jsonfile_prefix is None:  # hc-y_add0118:
            _jsonfile_prefix = (osp.dirname(val_dir) if val_dir is not None else runner.work_dir) + '/results'
            if not osp.exists(_jsonfile_prefix + '.bbox.json'):
                raise KeyError(f'json file {_jsonfile_prefix}.bbox.json does not exist!')
        # for the 'bbox predictions' type of results
        result_files, tmp_dir = dict(bbox = f'{_jsonfile_prefix}.bbox.json', proposal = f'{_jsonfile_prefix}.bbox.json'), None

        import json
        with open(self.ann_file, 'r') as f:
            json_val = json.load(f)
        

        eval_results = OrderedDict()
        cocoGt = self.coco  # COCO(ann_file)
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                cocoDt = cocoGt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11,
                'mAP_60': 12,  # hc-y_add1231:
                'mAP_70': 13,  # hc-y_add1231:
                'mAP_80': 14,  # hc-y_add1231:
                'mAP_90': 15,  # hc-y_add1231:
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)
                if val_dir is not None or flag_train_last_epoch:  # hc-y_add0108:输出 Class mAP mAP50 mAP60 mAP70 mAP75 mAP80 mAP90 AP_s AP_m AP_l AR100 AR300 AR1000 AR_s AR_m AR_l
                    stasts_csv_header = ['Class', 'mAP', 'mAP50', 'mAP60', 'mAP70', 'mAP75', 'mAP80', 'mAP90', 
                        'AP_s', 'AP_m', 'AP_l', 'AR100', 'AR300', 'AR1000', 'AR_s', 'AR_m', 'AR_l']
                    stasts_csv_data = np.concatenate((cocoEval.stats[:2], cocoEval.stats[12:14], cocoEval.stats[2:3], cocoEval.stats[14:16], cocoEval.stats[3:12]),0)
                    from pathlib import Path
                    stats_ap_csv = Path(val_dir).parent / f'stats_ap_02.csv' if val_dir is not None else Path(runner.work_dir) / f'stats_ap.csv'
                    stats_ap_num = len(stasts_csv_header) - 1
                    str_stasts_csv_header = '%20s' % stasts_csv_header[0] + ('%11s,' * stats_ap_num % tuple(stasts_csv_header[1:])).rstrip(',') + '\n'
                    str_stasts_csv_data = '%20s' % 'all' + ('%11.4g,' * stats_ap_num % tuple(stasts_csv_data)).rstrip(',') + '\n'
                    with open(stats_ap_csv, 'a') as f:
                        f.write(str_stasts_csv_header + str_stasts_csv_data + '\n')

                if classwise or flag_train_last_epoch:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    recalls = cocoEval.eval['recall']  # hc-y_add0108:recall: (iou, cls, area range, max dets)
                    iouThrs_inds = [0, 2, 4, 5, 6, 8]
                    assert len(self.cat_ids) == precisions.shape[2]

                    str_stasts_csv_data_per_cls = ''
                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        # hc-y_add0108:输出 per_Class mAP mAP50 mAP60 mAP70 mAP75 mAP80 mAP90 AP_s AP_m AP_l AR100 / / AR_s AR_m AR_l
                        recall = recalls[:, idx, 0, -1]
                        recall = recall[recall > -1]
                        if recall.size:
                            ar = np.mean(recall)
                        else:
                            ar = float('nan')
                        p_selected = precisions[iouThrs_inds, :, idx, 0, -1]
                        p_selected = (p_selected.sum(axis=1) / (p_selected > -1).sum(axis=1)).clip(min=-1)
                        p_allsml = precisions[:, :, idx, :, -1].reshape(-1, 4)
                        p_allsml = (p_allsml.sum(axis=0) / (p_allsml > -1).sum(axis=0)).clip(min=-1)
                        r100_allsml = recalls[:, idx, :, -1]
                        r100_allsml = (r100_allsml.sum(axis=0) / (r100_allsml > -1).sum(axis=0)).clip(min=-1)
                        _val_per_cls = np.concatenate((p_allsml[0:1], p_selected, p_allsml[1:], r100_allsml[0:1], np.array([-1, -1]), r100_allsml[1:]))
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.4f}', f'{float(ar):0.4f}'))
                        str_stasts_csv_data_per_cls += '%20s' % nm["name"] + ('%11.4g,' * stats_ap_num % tuple(_val_per_cls)).rstrip(',') + '\n'
                    with open(stats_ap_csv, 'a') as f:
                        f.write(str_stasts_csv_data_per_cls)

                    num_columns = min(9, len(results_per_category) * 3)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP', 'AR'] * (num_columns // 3)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l',
                        'mAP_60', 'mAP_70', 'mAP_80', 'mAP_90',  # hc-y_add1231:
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                ap_extra = cocoEval.stats[-4:]  # hc-y_add1231:
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f} '
                    f'{ap_extra[-4]:.3f} {ap_extra[-3]:.3f} {ap_extra[-2]:.3f} {ap_extra[-1]:.3f}')  # hc-y_add1231:
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
