import argparse
import configparser
import copy
import os

import numpy as np
import torch

from lib.predifineGraph import cal_lape, get_adjacency_matrix

PEMS_DATASETS = {"PEMS03", "PEMS04", "PEMS07", "PEMS08", "PEMS07M"}
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def _load_physical_adjacency(dataset, args_base):
    filepath = os.path.join(PROJECT_ROOT, "data", dataset)
    num_nodes = args_base.num_nodes_dict[dataset]
    csv_path = os.path.join(filepath, f"{dataset}.csv")
    npy_path = os.path.join(filepath, f"{dataset}_rn_adj.npy")

    if os.path.exists(csv_path):
        adjacency, _ = get_adjacency_matrix(csv_path, num_nodes)
    elif os.path.exists(npy_path):
        adjacency = np.load(npy_path)
    else:
        raise FileNotFoundError(
            f"SA-MGSTFN requires a PeMS physical graph at {csv_path} or {npy_path}."
        )
    adjacency = adjacency.astype(np.float32)
    adjacency = adjacency + np.eye(adjacency.shape[0], dtype=np.float32)
    return adjacency


def _normalize_adjacency(adjacency):
    degree = np.sum(adjacency, axis=1)
    degree[degree <= 1e-6] = 1.0
    inv_sqrt_degree = np.diag(1.0 / np.sqrt(degree))
    normalized = np.eye(adjacency.shape[0], dtype=np.float32) + inv_sqrt_degree @ adjacency @ inv_sqrt_degree
    return normalized.astype(np.float32)


def _load_shortest_hop(adjacency, dataset, type_short_path):
    cache_dir = os.path.join(PROJECT_ROOT, "data", "SA-MGSTFN", dataset)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{dataset}_sh_mx.npy")
    shortest = adjacency.copy().astype(np.float32)
    if type_short_path == "hop":
        if not os.path.exists(cache_path):
            shortest[shortest > 0] = 1
            shortest[shortest == 0] = 511
            for idx in range(shortest.shape[0]):
                shortest[idx, idx] = 0
            np.save(cache_path, shortest)
        shortest = np.load(cache_path)
    return shortest.astype(np.float32)


def parse_args(parser: argparse.ArgumentParser, args_base):
    config_file = os.path.join(PROJECT_ROOT, "conf", "SA-MGSTFN", "SA-MGSTFN.conf")
    config = configparser.ConfigParser()
    config.read(config_file, encoding="utf-8-sig")

    parser.add_argument("--input_window", type=int, default=config.getint("data", "input_window"))
    parser.add_argument("--output_window", type=int, default=config.getint("data", "output_window"))

    parser.add_argument("--embed_dim", type=int, default=config.getint("model", "embed_dim"))
    parser.add_argument("--semantic_dim", type=int, default=config.getint("model", "semantic_dim"))
    parser.add_argument("--semantic_raw_dim", type=int, default=config.getint("model", "semantic_raw_dim"))
    parser.add_argument("--num_heads", type=int, default=config.getint("model", "num_heads"))
    parser.add_argument("--num_layers", type=int, default=config.getint("model", "num_layers"))
    parser.add_argument("--mlp_ratio", type=int, default=config.getint("model", "mlp_ratio"))
    parser.add_argument("--drop", type=float, default=config.getfloat("model", "drop"))
    parser.add_argument("--attn_drop", type=float, default=config.getfloat("model", "attn_drop"))
    parser.add_argument("--semantic_threshold", type=float, default=config.getfloat("model", "semantic_threshold"))
    parser.add_argument("--flow_topk", type=int, default=config.getint("model", "flow_topk"))
    parser.add_argument("--far_mask_delta", type=int, default=config.getint("model", "far_mask_delta"))
    parser.add_argument("--type_short_path", type=str, default=config.get("model", "type_short_path"))
    parser.add_argument("--lape_dim", type=int, default=config.getint("model", "lape_dim"))
    parser.add_argument("--osm_radius", type=int, default=config.getint("model", "osm_radius"))
    parser.add_argument("--osm_timeout", type=int, default=config.getint("model", "osm_timeout"))
    parser.add_argument("--osm_endpoint", type=str, default=config.get("model", "osm_endpoint"))
    parser.add_argument("--semantic_model_name", type=str, default=config.get("model", "semantic_model_name"))
    parser.add_argument("--semantic_batch_size", type=int, default=config.getint("model", "semantic_batch_size"))
    parser.add_argument("--semantic_max_length", type=int, default=config.getint("model", "semantic_max_length"))
    parser.add_argument("--semantic_force_refresh", type=_as_bool, default=config.getboolean("model", "semantic_force_refresh"))
    parser.add_argument("--semantic_disable_osm", type=_as_bool, default=config.getboolean("model", "semantic_disable_osm"))

    parser.add_argument("--seed", type=int, default=config.getint("train", "seed"))
    parser.add_argument("--seed_mode", type=_as_bool, default=config.getboolean("train", "seed_mode"))
    parser.add_argument("--xavier", type=_as_bool, default=config.getboolean("train", "xavier"))
    parser.add_argument("--loss_func", type=str, default=config.get("train", "loss_func"))
    parser.set_defaults(real_value=config.getboolean("train", "real_value"))
    parser.add_argument("--lambda_orth", type=float, default=config.getfloat("train", "lambda_orth"))
    parser.add_argument("--lambda_recon", type=float, default=config.getfloat("train", "lambda_recon"))

    args_predictor, _ = parser.parse_known_args()
    args_predictor.his = args_predictor.input_window
    args_predictor.pred = args_predictor.output_window
    args_base.his = args_predictor.input_window
    args_base.pred = args_predictor.output_window

    unsupported = [dataset for dataset in args_base.dataset_use if dataset not in PEMS_DATASETS]
    if unsupported:
        raise ValueError(
            "SA-MGSTFN v1 supports only PeMS03/PEMS04/PEMS07/PEMS08/PEMS07M. "
            f"Unsupported dataset(s): {unsupported}."
        )

    adj_mx_dict = {}
    raw_adj_mx_dict = {}
    sh_mx_dict = {}
    lap_mx_dict = {}
    semantic_meta_path_dict = {}
    semantic_cache_dir_dict = {}

    for dataset in args_base.dataset_use:
        adjacency = _load_physical_adjacency(dataset, args_base)
        meta_path = os.path.join(PROJECT_ROOT, "data", dataset, f"{dataset}_meta.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"SA-MGSTFN requires PeMS node coordinates at {meta_path}. "
                "The CSV must contain latitude/longitude columns so OSM semantic texts can be generated."
            )

        raw_adj_mx_dict[dataset] = torch.FloatTensor(adjacency)
        adj_mx_dict[dataset] = torch.FloatTensor(_normalize_adjacency(adjacency))
        sh_mx_dict[dataset] = torch.FloatTensor(_load_shortest_hop(adjacency, dataset, args_predictor.type_short_path))
        lap_mx_dict[dataset] = torch.FloatTensor(cal_lape(copy.deepcopy(adjacency)))
        semantic_meta_path_dict[dataset] = meta_path
        semantic_cache_dir_dict[dataset] = os.path.join(PROJECT_ROOT, "data", "SA-MGSTFN", dataset)

    args_predictor.adj_mx_dict = adj_mx_dict
    args_predictor.raw_adj_mx_dict = raw_adj_mx_dict
    args_predictor.sh_mx_dict = sh_mx_dict
    args_predictor.lap_mx_dict = lap_mx_dict
    args_predictor.semantic_meta_path_dict = semantic_meta_path_dict
    args_predictor.semantic_cache_dir_dict = semantic_cache_dir_dict
    return args_predictor
