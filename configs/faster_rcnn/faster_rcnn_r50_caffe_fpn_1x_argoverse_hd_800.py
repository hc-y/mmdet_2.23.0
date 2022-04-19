# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_argoverse_hd_800.py 4 
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/argoverse_hd_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    roi_head=dict(
        bbox_head=dict(
            num_classes=8)))
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1280, 800), keep_ratio=True),  # (1280, 800), (1024, 640)
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
        img_scale=(1280, 800),  # (1280, 800), (1024, 640)
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
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=6,  # hc-y_modify0109:10 10, 8 8, 2 4
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.02*6*4/(2*8), momentum=0.9, weight_decay=0.0001)  # hc-y_modify0109:
optimizer_config = dict(grad_clip=None)
# learning policy  # configs/_base_/schedules/schedule_1x.py
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1500,  # hc-y_modify0109:
    warmup_ratio=1.0 / 3,  # hc-y_modify0109:
    step=[8, 11])
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
