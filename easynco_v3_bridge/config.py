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

NSS_DATASETS = ("TSPtrain", "CVRPtrain")
NSS_COPY_DATASET_NAMES = {
    "TSPtrain": "TSPtrain_v3copy",
    "CVRPtrain": "CVRPtrain_v3copy",
}

ATSP_SPECS = {
    20: 2500,
    50: 3500,
    100: 4000,
}

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

MVRP_SPECS = {
    50: 5000,
    100: 5000,
}

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

ATSP_METHODS = ("matnet", "matpoenet", "glop")
MVRP_METHODS = ("mtpomo", "mvmoe")

ATSP_MODEL_OVERRIDES = {
    "matnet": {
        20: {
            "settings": "matnet_settings",
            "model_dirpath": "pretrained/matnet_pretrain",
            "model_filename": "matnet_atsp20.ckpt",
            "batch_size": 4,
        },
        50: {
            "settings": "matnet_settings",
            "model_dirpath": "pretrained/matnet_pretrain",
            "model_filename": "matnet_atsp50.ckpt",
            "batch_size": 4,
        },
        100: {
            "settings": "matnet_settings",
            "model_dirpath": "pretrained/matnet_pretrain",
            "model_filename": "matnet_atsp100.ckpt",
            "batch_size": 4,
        },
    },
    "matpoenet": {
        20: {
            "settings": "matpoenet_settings",
            "model_dirpath": "pretrained/matpoenet",
            "model_filename": "MatNet-POE_mix.pt",
            "batch_size": 4,
        },
        50: {
            "settings": "matpoenet_settings",
            "model_dirpath": "pretrained/matpoenet",
            "model_filename": "MatNet-POE_mix.pt",
            "batch_size": 4,
        },
        100: {
            "settings": "matpoenet_settings",
            "model_dirpath": "pretrained/matpoenet",
            "model_filename": "MatNet-POE_mix.pt",
            "batch_size": 4,
        },
    },
    "glop": {
        20: {
            "settings": "glop_settings",
            "model_dirpath": "pretrained/glop",
            "model_filename": "glop_policy_atsp.pt",
            "batch_size": 1,
        },
        50: {
            "settings": "glop_settings",
            "model_dirpath": "pretrained/glop",
            "model_filename": "glop_policy_atsp.pt",
            "batch_size": 1,
        },
        100: {
            "settings": "glop_settings",
            "model_dirpath": "pretrained/glop",
            "model_filename": "glop_policy_atsp.pt",
            "batch_size": 1,
        },
    },
}

MVRP_MODEL_OVERRIDES = {
    "mtpomo": {
        50: {
            "settings": "mtpomo_settings",
            "model_dirpath": "pretrained/mtpomo/mtpomo",
            "model_filename": "mtpomo_mvrp_50.ckpt",
            "batch_size": 64,
        },
        100: {
            "settings": "mtpomo_settings",
            "model_dirpath": "pretrained/mtpomo/mtpomo",
            "model_filename": "mtpomo_mvrp_100.ckpt",
            "batch_size": 64,
        },
    },
    "mvmoe": {
        50: {
            "settings": "mvmoe_settings",
            "model_dirpath": "pretrained/mvmoe/mvmoe",
            "model_filename": "mvmoe_mvrp50.ckpt",
            "batch_size": 64,
        },
        100: {
            "settings": "mvmoe_settings",
            "model_dirpath": "pretrained/mvmoe/mvmoe",
            "model_filename": "mvmoe_mvrp100.ckpt",
            "batch_size": 64,
        },
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
