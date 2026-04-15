from pathlib import Path


BRIDGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = BRIDGE_ROOT.parent
EASYNCO_ROOT = WORKSPACE_ROOT / "EasyNCO"
DATASETS_ROOT = EASYNCO_ROOT / "data" / "datasets"
RESULTS_ROOT = EASYNCO_ROOT / "results"

V3_DATA_ROOT = DATASETS_ROOT / "offline_init_v3"
V3_NSS_COPY_ROOT = V3_DATA_ROOT / "nss_copy"
V3_ATSP_ROOT = V3_DATA_ROOT / "atsp"
V3_MVRP_ROOT = V3_DATA_ROOT / "mvrp"
V3_PCTSP_ROOT = V3_DATA_ROOT / "pctsp"

V4_DATA_ROOT = DATASETS_ROOT / "offline_init_v4"
V4_MVRP_DIVERSE_ROOT = V4_DATA_ROOT / "mvrp_diverse_v1"
V4_NSS_STYLE_SPLITS_ROOT = V4_DATA_ROOT / "nss_style_splits_v1"
V4_ATSP_SPLIT_ROOT = V4_NSS_STYLE_SPLITS_ROOT / "atsp"
V4_MVRP_SPLIT_ROOT = V4_NSS_STYLE_SPLITS_ROOT / "mvrp_diverse"

NSS_SOURCE_ROOT = DATASETS_ROOT / "nss_varying"
NSS_SOURCE_MANIFEST = NSS_SOURCE_ROOT / "nss_manifest.json"
NSS_COPY_MANIFEST = V3_NSS_COPY_ROOT / "manifest.json"

MANIFEST_ROOT = BRIDGE_ROOT / "manifests"
COMMAND_ROOT = BRIDGE_ROOT / "commands"
LOG_ROOT = BRIDGE_ROOT / "logs"

CONDA_ENV = "easynco_zhoucl"
PYTHONPATH_ROOT = WORKSPACE_ROOT
DEFAULT_GPU = "0"

SEED = 20260414
MVRP_MODE = 1


def build_dense_scale_specs(min_scale: int, max_scale: int, total_instances: int):
    scales = list(range(min_scale, max_scale + 1))
    if not scales:
        raise ValueError("Scale range must be non-empty.")
    if total_instances < len(scales):
        raise ValueError(
            f"Total instances ({total_instances}) must be >= number of scales ({len(scales)})."
        )
    base = total_instances // len(scales)
    remainder = total_instances % len(scales)
    return {
        scale: base + (1 if idx < remainder else 0)
        for idx, scale in enumerate(scales)
    }


def default_mvrp_demand_scaler(problem_size: int) -> int:
    anchors = (
        (20, 30),
        (50, 40),
        (100, 50),
        (1000, 200),
        (2000, 300),
        (5000, 300),
        (7000, 300),
    )
    if problem_size <= anchors[0][0]:
        return anchors[0][1]
    for (left_scale, left_value), (right_scale, right_value) in zip(anchors, anchors[1:]):
        if left_scale <= problem_size <= right_scale:
            ratio = (problem_size - left_scale) / float(right_scale - left_scale)
            return int(round(left_value + ratio * (right_value - left_value)))
    return anchors[-1][1]

NSS_DATASETS = ("TSPtrain", "CVRPtrain")
NSS_COPY_DATASET_NAMES = {
    "TSPtrain": "TSPtrain_v3copy",
    "CVRPtrain": "CVRPtrain_v3copy",
}

ATSP_SCALE_RANGE = (20, 100)
ATSP_TOTAL_INSTANCES = 10000
ATSP_SPECS = build_dense_scale_specs(
    min_scale=ATSP_SCALE_RANGE[0],
    max_scale=ATSP_SCALE_RANGE[1],
    total_instances=ATSP_TOTAL_INSTANCES,
)

PCTSP_SPECS = {
    20: 2000,
    100: 2000,
    500: 6000,
}

# Base CVRP is handled separately by copying NSS data, so the MVRP shard
# generator only covers the non-base variants here.
MVRP_VARIANTS = (
    "OVRP",
    "VRPB",
    "VRPL",
    "VRPTW",
    "OVRPTW",
    "OVRPB",
    "OVRPL",
    "VRPBL",
    "VRPBTW",
    "VRPLTW",
    "OVRPBL",
    "OVRPBTW",
    "OVRPLTW",
    "VRPBLTW",
    "OVRPBLTW",
)

MVRP_SCALE_RANGE = (50, 100)
MVRP_TOTAL_INSTANCES = 10000
MVRP_SPECS = build_dense_scale_specs(
    min_scale=MVRP_SCALE_RANGE[0],
    max_scale=MVRP_SCALE_RANGE[1],
    total_instances=MVRP_TOTAL_INSTANCES,
)

ACTIVE_EXPERIMENT_GROUPS = ("atsp", "mvrp")

NSS_TSP_METHODS = (
    "pointerformer",
    "invit",
    "elg",
    "lehd",
    "icam",
    "lih",
    "dact",
    "udc",
    "omni",
    "difusco",
    "t2t",
    "glop",
)

NSS_CVRP_METHODS = (
    "invit",
    "elg",
    "lehd",
    "icam",
    "lih",
    "dact",
    "udc",
    "omni",
)

NSS_PROGRESS_EXPECTED = {
    "TSPtrain_v3copy": len(NSS_TSP_METHODS),
    "CVRPtrain_v3copy": len(NSS_CVRP_METHODS),
}

ATSP_METHODS = ("matnet", "matpoenet", "glop")
MVRP_METHODS = ("mtpomo", "mvmoe")
MVRP_VARIANT_GROUPS = (
    ("part1", MVRP_VARIANTS[:8]),
    ("part2", MVRP_VARIANTS[8:]),
)

ATSP_MODEL_OVERRIDES = {
    "matnet": {
        "settings": "matnet_settings",
        "model_dirpath": "pretrained/matnet_pretrain",
        "model_filename": "matnet_atsp100.ckpt",
        "batch_size": 4,
    },
    "matpoenet": {
        "settings": "matpoenet_settings",
        "model_dirpath": "pretrained/matpoenet",
        "model_filename": "MatNet-POE_mix.pt",
        "batch_size": 4,
    },
    "glop": {
        "settings": "glop_settings",
        "model_dirpath": "pretrained/glop",
        "model_filename": "glop_policy_atsp.pt",
        "batch_size": 1,
    },
}

MVRP_MODEL_OVERRIDES = {
    "mtpomo": {
        "settings": "mtpomo_settings",
        "model_dirpath": "pretrained/mtpomo/mtpomo",
        "model_filename": "mtpomo_mvrp_100.ckpt",
        "batch_size": 64,
    },
    "mvmoe": {
        "settings": "mvmoe_settings",
        "model_dirpath": "pretrained/mvmoe/mvmoe",
        "model_filename": "mvmoe_mvrp100.ckpt",
        "batch_size": 64,
    },
}


def ensure_layout() -> None:
    for path in (
        MANIFEST_ROOT,
        COMMAND_ROOT,
        LOG_ROOT,
        V3_DATA_ROOT,
        V3_NSS_COPY_ROOT,
        V3_ATSP_ROOT,
        V3_MVRP_ROOT,
        V3_PCTSP_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def relative_to_datasets(path: Path) -> str:
    return str(path.relative_to(DATASETS_ROOT))


def variant_dir_name(variant: str) -> str:
    return variant.lower()
