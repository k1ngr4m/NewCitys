import json
import os
import re
import time
from collections import Counter
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch


LATITUDE_CANDIDATES = ("lat", "latitude", "Lat", "Latitude", "LAT")
LONGITUDE_CANDIDATES = ("lng", "lon", "longitude", "Lng", "Lon", "Longitude", "LNG", "LON")
NODE_ID_CANDIDATES = ("node_id", "sensor_id", "id", "ID", "index", "idx", "ID2")


def _first_existing_column(columns: Iterable[str], candidates: Iterable[str], role: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Missing {role} column. Expected one of: {list(candidates)}")


def _clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"[^0-9A-Za-z,.;:()_\\-/ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _safe_tags(element: Dict) -> Dict:
    tags = element.get("tags", {})
    return tags if isinstance(tags, dict) else {}


def _summarize_osm_elements(elements: List[Dict]) -> Dict[str, str]:
    highway_counter = Counter()
    poi_counter = Counter()
    landuse_counter = Counter()

    for element in elements:
        tags = _safe_tags(element)
        if "highway" in tags:
            highway_counter[tags["highway"]] += 1
        if "amenity" in tags:
            poi_counter[f"amenity:{tags['amenity']}"] += 1
        if "shop" in tags:
            poi_counter[f"shop:{tags['shop']}"] += 1
        if "tourism" in tags:
            poi_counter[f"tourism:{tags['tourism']}"] += 1
        if "landuse" in tags:
            landuse_counter[tags["landuse"]] += 1

    def summarize(counter: Counter, fallback: str) -> str:
        if not counter:
            return fallback
        return ", ".join(f"{name}({count})" for name, count in counter.most_common(5))

    return {
        "road_classes": summarize(highway_counter, "unknown road class"),
        "poi_types": summarize(poi_counter, "no nearby POI returned"),
        "landuse": summarize(landuse_counter, "unknown land use"),
    }


def _query_overpass(lat: float, lon: float, radius: int, endpoint: str, timeout: int) -> List[Dict]:
    try:
        import requests
    except ImportError as exc:
        raise ImportError("SA-MGSTFN OSM preprocessing requires requests. Install requirements.txt first.") from exc

    query = f"""
    [out:json][timeout:{timeout}];
    (
      way(around:{radius},{lat},{lon})["highway"];
      node(around:{radius},{lat},{lon})["amenity"];
      node(around:{radius},{lat},{lon})["shop"];
      node(around:{radius},{lat},{lon})["tourism"];
      way(around:{radius},{lat},{lon})["landuse"];
      relation(around:{radius},{lat},{lon})["landuse"];
    );
    out tags center 50;
    """
    response = requests.post(endpoint, data={"data": query}, timeout=timeout + 10)
    response.raise_for_status()
    payload = response.json()
    return payload.get("elements", [])


def _build_fallback_text(row: pd.Series, node_id: str, lat_col: str, lon_col: str) -> str:
    known_fields = []
    for column in row.index:
        if column in {lat_col, lon_col}:
            continue
        value = row[column]
        if pd.isna(value):
            continue
        known_fields.append(f"{column}={value}")
    field_text = "; ".join(known_fields[:8]) if known_fields else "no structured metadata"
    return (
        f"Traffic sensor {node_id} is located at latitude {row[lat_col]}, longitude {row[lon_col]}. "
        f"Metadata: {field_text}. Road context is inferred from topology when OSM data is unavailable."
    )


def build_or_load_semantic_texts(
    dataset: str,
    meta_path: str,
    cache_dir: str,
    radius: int,
    endpoint: str,
    timeout: int,
    force_refresh: bool = False,
    disable_osm: bool = False,
) -> List[str]:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "semantic_texts.json")
    if os.path.exists(cache_path) and not force_refresh:
        with open(cache_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        return [item["text"] for item in sorted(payload["nodes"], key=lambda item: item["order"])]

    meta = pd.read_csv(meta_path)
    lat_col = _first_existing_column(meta.columns, LATITUDE_CANDIDATES, "latitude")
    lon_col = _first_existing_column(meta.columns, LONGITUDE_CANDIDATES, "longitude")
    id_col = None
    for candidate in NODE_ID_CANDIDATES:
        if candidate in meta.columns:
            id_col = candidate
            break

    nodes = []
    for order, row in meta.reset_index(drop=True).iterrows():
        node_id = str(row[id_col]) if id_col is not None else str(order)
        fallback_text = _build_fallback_text(row, node_id, lat_col, lon_col)

        if disable_osm:
            text = fallback_text
        else:
            elements = _query_overpass(float(row[lat_col]), float(row[lon_col]), radius, endpoint, timeout)
            summary = _summarize_osm_elements(elements)
            text = (
                f"Traffic sensor {node_id} is located at latitude {row[lat_col]}, longitude {row[lon_col]}. "
                f"Nearby road classes: {summary['road_classes']}. "
                f"Nearby POI types: {summary['poi_types']}. "
                f"Nearby land use: {summary['landuse']}. "
                f"{fallback_text}"
            )
            time.sleep(0.2)

        nodes.append({"order": int(order), "node_id": node_id, "text": _clean_text(text)})

    with open(cache_path, "w", encoding="utf-8") as file:
        json.dump({"dataset": dataset, "nodes": nodes}, file, ensure_ascii=False, indent=2)
    return [item["text"] for item in nodes]


def build_or_load_distilbert_embeddings(
    texts: List[str],
    cache_dir: str,
    model_name: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
    force_refresh: bool = False,
) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "semantic_embeddings.npy")
    if os.path.exists(cache_path) and not force_refresh:
        return np.load(cache_path).astype(np.float32)

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError("SA-MGSTFN semantic encoding requires transformers. Install requirements.txt first.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    encoded_batches = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            tokens = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            output = model(**tokens)
            attention_mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (output.last_hidden_state * attention_mask).sum(dim=1)
            pooled = pooled / attention_mask.sum(dim=1).clamp_min(1.0)
            encoded_batches.append(pooled.cpu().numpy())

    embeddings = np.concatenate(encoded_batches, axis=0).astype(np.float32)
    np.save(cache_path, embeddings)
    return embeddings
