#  modify from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
# %matplotlib inline
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tools.split_argoverse_dataset import split_bboxes_by_scale
import numpy as np
import pylab
import json
import itertools
from terminaltables import AsciiTable
from pathlib import Path
pylab.rcParams['figure.figsize'] = (10.0, 8.0)


def split_gts_scale_type(json_dir):  # hc-y_add0119:
    scale_type = ['all', 's_inner', 'ml_inner', 's_outer', 'ml_outer']
    # scale_type = ['all', 's', 'ml']
    # scale_type = ['all', 'inner', 'outer']
    num_scale_type = len(scale_type) - 1
    with open(json_dir, 'r') as f:
        a = json.load(f)
    a_sub = [[] for _ in range(num_scale_type)]
    img_id_list = sorted(set([_inst['image_id'] for _inst in a]))
    for _img_id in img_id_list:
        preds_per_img = [_inst for _inst in a if _inst['image_id'] == _img_id]
        if len(preds_per_img) == 0:
            continue
        # The predicted box format is [ctr_x, ctr_y, width, height]
        _pred_bboxes = np.array([_val['bbox'] for _val in preds_per_img], dtype=np.float64)
        _img_hw = (preds_per_img[0]['height'], preds_per_img[0]['width'])
        _pred_bboxes[:, [0, 2]] /= _img_hw[1]  # normalize x
        _pred_bboxes[:, [1, 3]] /= _img_hw[0]  # normalize y
        _pred_idx_split = split_bboxes_by_scale(_pred_bboxes, split_type=scale_type)
        for idx_type in range(num_scale_type):
            for _pred_idx in _pred_idx_split[idx_type+1]:
                a_sub[idx_type].append(preds_per_img[_pred_idx])
    
    path_to_new_file = Path(json_dir).parents[0] / 'preds_scale_type_800_split_by_iof'
    if not path_to_new_file.exists():
        path_to_new_file.mkdir(parents=True, exist_ok=True)
    for idx_type in range(num_scale_type):
        json.dump(a_sub[idx_type], open(path_to_new_file  / f'results.bbox_{scale_type[idx_type+1]}.json', 'w'), indent=4)
    print('\nfinish split_gts_scale_type()!')


def main():
    classwise = False
    CLASSES = ('person',  'bicycle',  'car',  'motorcycle',  'bus',  'truck',  
                'traffic_light',  'stop_sign')
    proposal_nums=(100, 300, 1000)
    annType = ['segm','bbox','keypoints']
    iou_type = annType[1]  # specify type here
    iou_thrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
    print('Evaluating *%s*...'%(iou_type))

    # gt_json_dir='/mnt/data1/yuhangcheng/yhc_workspace/datasets/Argoverse-HD-mini'
    # gt_json_folder='annotations_scale_type_800_sub_split_by_iof'
    gt_json_dir='/mnt/data1/yuhangcheng/yhc_workspace/datasets/Argoverse-1.1'
    gt_json_folder='annotations_scale_type_800_split_by_iof'
    pred_json_dir='/mnt/data1/yuhangcheng/yhc_workspace/mmdet_1213/work_dirs_poc_archive'
    # pred_json_folder='yolof_r50_c5_8x8_1x_argoverse_hd_poc_01071635'
    # pred_json_folder='yolof_r50_c5_8x8_1x_argoverse_hd_poc_01082146'
    # pred_json_folder='yolofv0v1_r50_c5_8x8_1x_argoverse_hd_poc_01092053'
    # pred_json_folder='yolofmbranchv1v2_r50_c5_8x8_1x_argoverse_hd_poc_grad_01090456'
    # pred_json_folder='yolofmbranchv2v1_r50_c5_8x8_1x_argoverse_hd_poc_grad_01091215'
    pred_json_folder='yolofmbranchv2v2_r50_c5_8x8_1x_argoverse_hd_poc_grad_01091334'
    pred_json_folder2='preds_scale_type_800_split_by_iof'
    pred_all_json='%s/%s/results.bbox.json'%(pred_json_dir, pred_json_folder)
    print(f'pred_json_folder: {pred_json_folder}')

    # split_gts_scale_type(pred_all_json)

    scale_type = ['all', 's_inner', 'ml_inner', 's_outer', 'ml_outer']
    num_scale_type = len(scale_type)
    for idx_type in range(num_scale_type):
        str_separator_line = '\n' + '---' * 20 + 'scale_type: ' + scale_type[idx_type] + '---' * 20
        print(str_separator_line, '\n')

        #initialize COCO ground truth api
        if idx_type == 0:
            annFile = '%s/annotations/val.json'%(gt_json_dir)
        else:
            annFile = '%s/%s/val_%s.json'%(gt_json_dir, gt_json_folder, scale_type[idx_type])
        cocoGt=COCO(annFile)

        #initialize COCO detections api
        if idx_type == 0:
            resFile = '%s/%s/results.bbox.json'%(pred_json_dir, pred_json_folder)
        else:
            resFile = '%s/%s/%s/results.bbox_%s.json'%(pred_json_dir, pred_json_folder, pred_json_folder2, scale_type[idx_type])
        cocoDt=cocoGt.loadRes(resFile)

        imgIds=sorted(cocoGt.getImgIds())
        imgIds=imgIds[::5]  # hc-y_note1231:只采样出少量的图片用于训练和评估; './../datasets/Argoverse-1.1/annotations/val.json' 15062 images
        # imgId = imgIds[np.random.randint(len(imgIds))]

        # running evaluation
        cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
        cat_ids = cocoGt.getCatIds(CLASSES)
        cocoEval.params.catIds = cat_ids
        cocoEval.params.imgIds  = imgIds
        cocoEval.params.maxDets = list(proposal_nums)
        cocoEval.params.iouThrs = iou_thrs
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        if idx_type == 0:  # hc-y_add0108:输出 Class mAP mAP50 mAP60 mAP70 mAP75 mAP80 mAP90 AP_s AP_m AP_l AR100 AR300 AR1000 AR_s AR_m AR_l
            stasts_csv_header = ['Class', 'mAP', 'mAP50', 'mAP60', 'mAP70', 'mAP75', 'mAP80', 'mAP90', 
                'AP_s', 'AP_m', 'AP_l', 'AR100', 'AR300', 'AR1000', 'AR_s', 'AR_m', 'AR_l']
            stasts_csv_data = np.concatenate((cocoEval.stats[:2], cocoEval.stats[12:14], cocoEval.stats[2:3], cocoEval.stats[14:16], cocoEval.stats[3:12]),0)
            # stasts_csv_data = list(_stasts_csv_data)
        else:
            stasts_csv_header.extend([f'{_str}_{scale_type[idx_type]}' for _str in stasts_csv_header[1:17]])
            _stasts_csv_data = np.concatenate((cocoEval.stats[:2], cocoEval.stats[12:14], cocoEval.stats[2:3], cocoEval.stats[14:16], cocoEval.stats[3:12]),0)
            stasts_csv_data = np.concatenate((stasts_csv_data, _stasts_csv_data), 0)
            # stasts_csv_data.extend(list(_stasts_csv_data))

        if classwise:  # Compute per-category AP
            _stasts_csv_data_per_cls = []
            stasts_csv_data_cls_name = []
            # Compute per-category AP
            # from https://github.com/facebookresearch/detectron2/
            precisions = cocoEval.eval['precision']
            # precision: (iou, recall, cls, area range, max dets)
            recalls = cocoEval.eval['recall']  # hc-y_add0108:recall: (iou, cls, area range, max dets)
            iouThrs_inds = [0, 2, 4, 5, 6, 8]
            assert len(cat_ids) == precisions.shape[2]

            results_per_category = []
            for idx, catId in enumerate(cat_ids):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                nm = cocoGt.loadCats(catId)[0]
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
                _stasts_csv_data_per_cls.append(_val_per_cls)
                if idx_type == 0:
                    stasts_csv_data_cls_name.append(nm["name"])
            _stasts_csv_data_per_cls = np.stack(_stasts_csv_data_per_cls, 0)
            # if idx_type == 0:
            #     stasts_csv_data_cls_name.append(nm["name"])
            # else:
            #     stasts_csv_data_per_cls = np.concatenate((stasts_csv_data_per_cls, _val_per_cls), 0)
            # if idx_type == 4:
            #     pass

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
            print('\n' + table.table)

    stats_ap_csv = Path(f'{pred_json_dir}/{pred_json_folder}/stats_ap_03.csv')
    stats_ap_num = len(stasts_csv_header) - 1
    str_stasts_csv_header = '%20s' % stasts_csv_header[0] + ('%16s,' * stats_ap_num % tuple(stasts_csv_header[1:])).rstrip(',') + '\n'
    str_stasts_csv_data = '%20s' % 'all' + ('%16.4g,' * stats_ap_num % tuple(stasts_csv_data)).rstrip(',') + '\n'
    str_stasts_csv_data_per_cls = ''
    # stasts_csv_data_per_cls += '%20s' % nm["name"] + ('%11.4g,' * stats_ap_num % tuple(_val_per_cls)).rstrip(',') + '\n'
    with open(stats_ap_csv, 'a') as f:
        f.write(str_stasts_csv_header + str_stasts_csv_data + '\n')
        if classwise:
            f.write(str_stasts_csv_data_per_cls)

    print('\nfinish!')


if __name__ == "__main__":
    # opt = parse_opt()
    # main(opt)
    main()
