_base_ = [
    "./henancrop_1024x1024.py",
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset={{_base_.train_henancrop}}, #设置训练数据集
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset={{_base_.val_henancrop}}, #设置验证数据集
)
test_dataloader = val_dataloader
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type="DefaultSampler", shuffle=False),
#     dataset={{_base_.test_henancrop}}, #设置测试数据集
# )
val_evaluator = dict(
    type="DGIoUMetric", iou_metrics=["mIoU","mDice","mFscore"], dataset_keys=["henan"]
)
test_evaluator=val_evaluator
