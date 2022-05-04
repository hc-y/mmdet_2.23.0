# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    flag_show_result = 'Default'
    flag_show_result = 'labels_pred'
    data_loader.dataset.det_results = []  # hc-y_add0501:供 mmdet/datasets/custom.py def prepare_test_img(self, idx) 使用
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if isinstance(result, tuple):  # hc-y_add0501:
            result_tensor = result[1]
            result = result[0]
            img_metas = data['img_metas'][0].data[0]
            data['img'][0].data[0] = torch.stack([_chips[0] for _chips in data['img'][0].data[0]], dim=0)
            for j, img_meta in enumerate(img_metas):
                det_bboxes, det_labels = result_tensor[j]
                if isinstance(det_bboxes, torch.Tensor):
                    det_bboxes = det_bboxes.detach().cpu().numpy()
                    det_labels = det_labels.detach().cpu().numpy()
                det_result = dict()
                det_result['image_id'] = data_loader.dataset.img_ids[i+j]  # 参考 mmdet/datasets/coco.py def _det2json(self, results)
                det_result['sid'] = data_loader.dataset.data_infos[i+j]['sid']
                det_result['fid'] = data_loader.dataset.data_infos[i+j]['fid']
                det_result['det_bbox'] = det_bboxes#.tolist()  # (x1,y1,x2,y2)
                det_result['det_label'] = det_labels#.tolist()  # det_result['det_label']是 mmdetection 加载转化后的 label, 而不是 .json 中的 'category_id'
                det_result['img_shape'] = img_meta['ori_shape']
                data_loader.dataset.det_results.append(det_result)

        if flag_show_result == 'labels_pred':
            iter_idx = i  # hc-y_add0121:
        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if flag_show_result == 'labels_pred':
                    if out_dir:
                        # out_file_labels = osp.join(out_dir, f'val_iter{iter_idx}_bs{i}_labels.jpg')  # hc-y_add0121:
                        out_file_pred = osp.join(out_dir, f'val_iter{iter_idx}_bs{i}_pred.jpg')  # hc-y_add0121:
                    else:
                        out_file = None
                    # model.module.show_result(
                    #     img_show,
                    #     gt_bboxes,  # TODO:
                    #     show=show,
                    #     bbox_color=PALETTE,
                    #     text_color=PALETTE,
                    #     mask_color=PALETTE,
                    #     out_file=out_file_labels,
                    #     score_thr=show_score_thr)
                    model.module.show_result(
                        img_show,
                        result[i],
                        bbox_color=PALETTE,
                        text_color=PALETTE,
                        mask_color=PALETTE,
                        show=show,
                        out_file=out_file_pred,
                        score_thr=show_score_thr)
                else:
                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        bbox_color=PALETTE,
                        text_color=PALETTE,
                        mask_color=PALETTE,
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        # This logic is only used in panoptic segmentation test.
        elif isinstance(result[0], dict) and 'ins_results' in result[0]:
            for j in range(len(result)):
                bbox_results, mask_results = result[j]['ins_results']
                result[j]['ins_results'] = (bbox_results,
                                            encode_mask_results(mask_results))

        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
            # This logic is only used in panoptic segmentation test.
            elif isinstance(result[0], dict) and 'ins_results' in result[0]:
                for j in range(len(result)):
                    bbox_results, mask_results = result[j]['ins_results']
                    result[j]['ins_results'] = (
                        bbox_results, encode_mask_results(mask_results))

        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
