henancrop_type = "BigWheatDataset"
henancrop_root = "data/bigwheat/"
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
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_henancrop = dict(
    type=henancrop_type,
    data_root=henancrop_root,
    data_prefix=dict(
        img_path="train/image",
        seg_map_path="train/label",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=henancrop_train_pipeline,
)
val_henancrop = dict(
    type=henancrop_type,
    data_root=henancrop_root,
    data_prefix=dict(
        img_path="val/image",
        seg_map_path="val/label",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=henancrop_test_pipeline,
)
test_henancrop = dict(
    type=henancrop_type,
    data_root=henancrop_root,
    data_prefix=dict(
        img_path="test/image",
        seg_map_path="test/image",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=henancrop_test_pipeline,
)
