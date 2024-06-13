henancrop_type = "LoveDADataset"
henancrop_root = "data/loveda/"
henancrop_crop_size = (1024, 1024)
henancrop_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(1024, 1024)),
    dict(type="RandomCrop", crop_size=henancrop_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
henancrop_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type="LoadAnnotations"), #无需加载测试标签
    dict(type="PackSegInputs"),
]
train_henancrop = dict(
    type=henancrop_type,
    data_root=henancrop_root,
    data_prefix=dict(
        img_path="img_dir/train",
        seg_map_path="ann_dir/train",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=henancrop_train_pipeline,
)
val_henancrop = dict(
    type=henancrop_type,
    data_root=henancrop_root,
    data_prefix=dict(
        img_path="img_dir/val",
        seg_map_path="ann_dir/val",
    ),
    pipeline=henancrop_test_pipeline,
)
test_henancrop = dict(
    type=henancrop_type,
    data_root=henancrop_root,
    data_prefix=dict(
        img_path="img_dir/test",
        seg_map_path="ann_dir/test",
    ),
    pipeline=henancrop_test_pipeline,
)
