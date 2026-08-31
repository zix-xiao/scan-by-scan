import logging
import yaml
from yacs.config import CfgNode as ConfigurationNode

Logger = logging.getLogger(__name__)

# YACS overwrite these settings using YAML, all YAML variables MUST BE defined here first
# as this is the master list of ALL attributes.


def get_cfg_defaults(singleton: ConfigurationNode):
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    return singleton.clone()


def _filter_cfg_dict(
    cfg_node: ConfigurationNode, user_dict: dict, prefix: str = ""
) -> tuple[dict, list[str]]:
    """Recursively keep only keys that exist in cfg_node, collecting unknown key paths."""
    filtered: dict = {}
    unknown: list[str] = []
    for k, v in user_dict.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if k not in cfg_node:
            unknown.append(full_key)
        elif isinstance(v, dict) and isinstance(cfg_node[k], ConfigurationNode):
            sub_filtered, sub_unknown = _filter_cfg_dict(
                cfg_node[k], v, prefix=full_key
            )
            filtered[k] = sub_filtered
            unknown.extend(sub_unknown)
        else:
            filtered[k] = v
    return filtered, unknown


def merge_cfg_from_file(cfg: ConfigurationNode, cfg_filename: str) -> None:
    """Like cfg.merge_from_file() but logs a warning for unknown keys instead of raising."""
    with open(cfg_filename) as f:
        user_dict = yaml.safe_load(f)
    if not user_dict:
        return
    if "DATA_PATH" in user_dict and isinstance(user_dict["DATA_PATH"], str):
        user_dict["DATA_PATH"] = [user_dict["DATA_PATH"]]
    if "FDR" in user_dict and isinstance(user_dict["FDR"], dict):
        fdr = user_dict["FDR"]
        if isinstance(fdr.get("METHOD"), str):
            # Pre-dates FDR.METHOD becoming a list (one or more rescoring methods
            # per run) -- effective_config.yaml files written before that change
            # still record it as a bare scalar (e.g. "percolator").
            fdr["METHOD"] = [fdr["METHOD"]]
        if isinstance(fdr.get("METHOD"), list) and any(
            isinstance(m, str) for m in fdr["METHOD"]
        ):
            # Pre-dates FDR.METHOD becoming a list of {TRAINING_DATA, METHOD} dicts --
            # effective_config.yaml files written before that change still record
            # entries as bare strings ("percolator" / "mokapot_trusted"), with the
            # engine choice for "mokapot_trusted" nested under FDR.MOKAPOT_TRUSTED.
            trusted = fdr.pop("MOKAPOT_TRUSTED", {}) or {}
            for k in ("DECOY_TARGET_RATIO", "DECOY_MSMS_ONLY", "SEED"):
                if k in trusted:
                    fdr[k] = trusted[k]
            model_type = trusted.get("MODEL_TYPE", "percolator")
            new_methods = []
            for m in fdr["METHOD"]:
                if m == "percolator":
                    new_methods.append({"TRAINING_DATA": "All", "METHOD": "semi-supervised"})
                elif m == "mokapot_trusted":
                    new_methods.append(
                        {
                            "TRAINING_DATA": "MS/MS",
                            "METHOD": "supervised" if model_type == "supervised" else "semi-supervised",
                        }
                    )
                else:
                    new_methods.append(m)  # unrecognized, let downstream validation catch it
            fdr["METHOD"] = new_methods
        if "PERCOLATOR_POST_PROCESSING" in fdr:
            # Pre-dates FDR.POST_PROCESSING becoming a list.
            old_pp = fdr.pop("PERCOLATOR_POST_PROCESSING")
            fdr.setdefault("POST_PROCESSING", [old_pp] if isinstance(old_pp, str) else old_pp)
    filtered_dict, unknown_keys = _filter_cfg_dict(cfg, user_dict)
    if unknown_keys:
        Logger.warning(
            "Config file '%s' contains %d unrecognised key(s) that will be ignored: %s",
            cfg_filename,
            len(unknown_keys),
            unknown_keys,
        )
    cfg.merge_from_other_cfg(ConfigurationNode.load_cfg(yaml.dump(filtered_dict)))
