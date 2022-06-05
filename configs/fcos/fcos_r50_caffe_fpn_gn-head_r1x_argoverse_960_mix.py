# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/fcos/fcos_r50_caffe_fpn_gn-head_r1x_argoverse_640.py 4 
_base_ = [
    '../_base_/datasets/argoverse_hd_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='FCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=8,  # hc-y_modify1031:
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(960, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(960, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
dataset_type = 'ArgoverseDataset'
classes = ('person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic_light', 'stop_sign')
data_root = './../datasets/Argoverse-1.1/'
# data_root = './../datasets/Argoverse-HD-mini/'
# ---------- Concatenate dataset: 方式一 ---------- #
dataset_train = dict(
    type=dataset_type,
    # explicitly add your class names to the field `classes`
    classes=classes,
    ann_file=[data_root + _val for _val in ['annotations/train.json', 'annotations_chip/train_5lft_euc_1.2_3.2_chip.json']],
    img_prefix=[data_root +  _val for _val in ['images/', 'images_chip/']],
    pipeline=train_pipeline)
data = dict(
    samples_per_gpu=12,  # hc-y_modify0515:原始为2;
    workers_per_gpu=12,  # hc-y_modify0515:原始为2;
    train=dataset_train,
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# ---------- Concatenate dataset: 方式二 ---------- #
# dataset_A_train = dict(
#     type=dataset_type,
#     # explicitly add your class names to the field `classes`
#     classes=classes,
#     ann_file=data_root + 'annotations/train.json',
#     img_prefix=data_root + 'images/',
#     pipeline=train_pipeline)
# dataset_B_train = dict(
#     type=dataset_type,
#     # explicitly add your class names to the field `classes`
#     classes=classes,
#     ann_file=data_root + 'annotations_chip/train_5lft_euc_1.2_3.2_chip.json',
#     img_prefix=data_root + 'images_chip/',
#     pipeline=train_pipeline)
# data = dict(
#     samples_per_gpu=24,  # hc-y_modify0515:原始为2;
#     workers_per_gpu=24,  # hc-y_modify0515:原始为2;
#     train=[dataset_A_train, dataset_B_train],
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    lr=0.01*12*4/(2*8), paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy  # configs/_base_/schedules/schedule_1x.py
lr_config = dict(
    policy='step',
    warmup='linear',  # hc-y_modify1031:
    warmup_iters=1500,  # hc-y_modify1031:
    warmup_ratio=1.0 / 3,
    step=[8, 11])
evaluation = dict(interval=1, metric='bbox')  # configs/_base_/datasets/argoverse_hd_detection.py
runner = dict(type='EpochBasedRunner', max_epochs=12)  # configs/_base_/schedules/schedule_1x.py

checkpoint_config = dict(interval=1)  # configs/_base_/default_runtime.py
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'work_dirs/fcos_coco/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth'
resume_from = None
workflow = [('train', 1)]
