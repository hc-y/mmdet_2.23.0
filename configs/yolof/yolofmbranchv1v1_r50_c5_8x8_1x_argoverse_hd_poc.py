# hc-y_modify1103:
# CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_train2.sh configs/yolof/yolofmbranchv1v1_r50_c5_8x8_1x_argoverse_hd_poc.py 4 
_base_ = [
    '../_base_/datasets/argoverse_hd_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='YOLOFMBranchV1v1',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='DilatedEncoder',
        in_channels=2048,
        out_channels=512,
        block_mid_channels=128,
        num_residual_blocks=4),
    local_branch=dict(
        with_no_grad=True),
    bbox_head=dict(
        type='YOLOFHeadV1v1',
        num_classes=8,  # hc-y_modify1103:
        in_channels=512,
        reg_decoded_bbox=True,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[1, 2, 4, 8, 16],
            strides=[32]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1., 1., 1., 1.],
            add_ctr_clamp=True,
            ctr_clamp=32),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='UniformAssigner', pos_ignore_thr=0.15, neg_ignore_thr=0.7),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.12*2*8/(8*8),  # hc-y_modify1103:原始为lr=0.12 with num_GPU=8, samles_per_gpu=8; num_GPU=4, samples_per_gpu=6时, 则lr乘以4*6/(8*8);
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        norm_decay_mult=0., custom_keys={'backbone': dict(lr_mult=1. / 3)}))
# lr_config = dict(warmup_iters=1500, warmup_ratio=0.00066667)
# learning policy  # configs/_base_/schedules/schedule_1x.py
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.002 / 3,  # hc-y_modify1103:0.002/3 = 0.0006666666666666666
    step=[8, 11])

# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='ResizeChipsV1v1', img_scale=(1280, 800), keep_ratio=True),  # (1280, 800), (1024, 640)
    dict(type='RandomFlip', flip_ratio=0.),
    # dict(type='RandomShift', shift_ratio=0.5, max_shift_px=32),
    dict(type='NormalizeChipsV1v1', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundleChipsV1v1'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'chip4'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 800),  # (1280, 800), (1024, 640)
        flip=False,
        transforms=[
            dict(type='ResizeChipsV1v1', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='NormalizeChipsV1v1', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensorChipsV1v1', keys=['img']),
            dict(type='Collect', keys=['img', 'chip4']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,  # hc-y_modify1103:8 8, 2 4; on local 1050, 1, 2;
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

evaluation = dict(interval=2, metric='bbox')  # configs/_base_/datasets/argoverse_hd_detection.py
runner = dict(type='EpochBasedRunner', max_epochs=12)  # configs/_base_/schedules/schedule_1x.py

checkpoint_config = dict(interval=12)  # configs/_base_/default_runtime.py
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
load_from = None
resume_from = None
workflow = [('train', 1)]
