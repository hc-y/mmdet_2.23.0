_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/argoverse_hd_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(bbox_head=dict(num_classes=8))
# optimizer
optimizer = dict(type='SGD', lr=0.01*3/2, momentum=0.9, weight_decay=0.0001)  # hc-y_modify1103:原始为lr=0.01 with num_GPU=8, samles_per_gpu=2; num_GPU=4, samples_per_gpu=6时, 则lr乘以4*6/(8*2);
# learning policy  # configs/_base_/schedules/schedule_1x.py
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])

evaluation = dict(interval=2, metric='bbox')  # configs/_base_/datasets/argoverse_hd_detection.py
runner = dict(type='EpochBasedRunner', max_epochs=12)  # configs/_base_/schedules/schedule_1x.py

checkpoint_config = dict(interval=2)  # configs/_base_/default_runtime.py
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
