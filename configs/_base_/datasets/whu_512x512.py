whu_type = "WHUDataset"
whu_root = "E:/xienan/project/Rein-train/data/whu/"
whu_crop_size = (512, 512)
whu_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(512, 512)),
    dict(type="RandomCrop", crop_size=whu_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
whu_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_whu = dict(
    type=whu_type,
    data_root=whu_root,
    data_prefix=dict(
        img_path="train/image",
        seg_map_path="train/label",
    ),
    img_suffix=".tif",
    seg_map_suffix=".tif",
    pipeline=whu_train_pipeline,
)
val_whu = dict(
    type=whu_type,
    data_root=whu_root,
    data_prefix=dict(
        img_path="val/image",
        seg_map_path="val/label",
    ),
    img_suffix=".tif",
    seg_map_suffix=".tif",
    pipeline=whu_test_pipeline,
)
test_whu = dict(
    type=whu_type,
    data_root=whu_root,
    data_prefix=dict(
        img_path="test/image",
        seg_map_path="test/label",
    ),
    img_suffix=".tif",
    seg_map_suffix=".tif",
    pipeline=whu_test_pipeline,
)