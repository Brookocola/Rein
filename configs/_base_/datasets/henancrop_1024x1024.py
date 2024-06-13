henancrop_type = "HenanCropDataset"
henancrop_root = "data/henancrop/"
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
        img_path="train/images",
        seg_map_path="train/labels",
    ),
    img_suffix=".tif",
    seg_map_suffix=".tif",
    pipeline=henancrop_train_pipeline,
)
val_henancrop = dict(
    type=henancrop_type,
    data_root=henancrop_root,
    data_prefix=dict(
        img_path="val/images",
        seg_map_path="val/labels",
    ),
    img_suffix=".tif",
    seg_map_suffix=".tif",
    pipeline=henancrop_test_pipeline,
)
test_henancrop=val_henancrop
# test_henancrop = dict(
#     type=henancrop_type,
#     data_root=henancrop_root,
#     data_prefix=dict(
#         img_path="test/images",
#         seg_map_path="test/labels",
#     ),
#     img_suffix=".tif",
#     seg_map_suffix=".tif",
#     pipeline=henancrop_test_pipeline,
# )
