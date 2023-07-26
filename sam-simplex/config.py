from box import Box
config = {
    "n_runs":1000,
    "n_ways":5,
    "n_shot":1,
    "n_queries":15,
    "n_classes":100,
    "elements_per_class":100*[600],
    "miniImageNet_path":"/nasbrain/datasets/miniimagenetimages",
    "runs_path":"/nasbrain/f21lin/runs",
    "crops_path":"/nasbrain/f21lin/crops",
    "sam_path":"/nasbrain/f21lin/sam_vit_h_4b8939.pth",
    "backbone_path":"/homes/f21lin/stageFred2A/mini1.pt1",
    "features_path":"/nasbrain/f21lin/features",
    "packages_path":"~/venv/fredVenv/lib/python3.10/site-packages/",
}
config = Box(config)