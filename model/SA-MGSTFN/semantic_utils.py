import hashlib
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch


LATITUDE_CANDIDATES = ("lat", "latitude", "Lat", "Latitude", "LAT")
LONGITUDE_CANDIDATES = ("lng", "lon", "longitude", "Lng", "Lon", "Longitude", "LNG", "LON")
NODE_ID_CANDIDATES = ("node_id", "sensor_id", "id", "ID", "index", "idx", "ID2")
OVERPASS_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass
class OverpassQueryResult:
    elements: List[Dict]
    status: str
    error: str = ""
    status_code: Optional[int] = None
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.status in {"ok", "empty"}


def _first_existing_column(columns: Iterable[str], candidates: Iterable[str], role: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Missing {role} column. Expected one of: {list(candidates)}")


def _clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"[^0-9A-Za-z,.;:()_/\\ -]+", " ", text)
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


def _response_snippet(text: str, limit: int = 240) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _log_osm_api_call(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[SA-MGSTFN][OSM] {message}", flush=True)


def _query_overpass_with_diagnostics(
    lat: float,
    lon: float,
    radius: int,
    endpoint: str,
    timeout: int,
    output_limit: int = 200,
    max_retries: int = 2,
    retry_backoff: float = 1.5,
    log_api_calls: bool = False,
    node_label: str = "",
) -> OverpassQueryResult:
    try:
        import requests
    except ImportError as exc:
        raise ImportError("SA-MGSTFN OSM preprocessing requires requests. Install requirements.txt first.") from exc

    output_clause = f"out tags center {output_limit};" if output_limit > 0 else "out tags center;"
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
    {output_clause}
    """
    headers = {
        "User-Agent": "SA-MGSTFN-NewCitys/1.0 (semantic preprocessing; contact: local)",
        "Accept": "application/json",
    }
    attempts = max(max_retries, 0) + 1
    last_result = OverpassQueryResult([], status="request_error", attempts=0)
    for attempt in range(1, attempts + 1):
        started_at = time.perf_counter()
        _log_osm_api_call(
            log_api_calls,
            (
                f"request node={node_label or 'unknown'} attempt={attempt}/{attempts} lat={lat:.6f} lon={lon:.6f} "
                f"radius={radius} output_limit={output_limit} timeout={timeout}s endpoint={endpoint}"
            ),
        )
        try:
            response = requests.post(endpoint, data={"data": query}, headers=headers, timeout=timeout + 10)
        except requests.Timeout as exc:
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            last_result = OverpassQueryResult([], "timeout", str(exc), attempts=attempt)
            _log_osm_api_call(
                log_api_calls,
                f"response node={node_label or 'unknown'} attempt={attempt}/{attempts} status=timeout duration_ms={elapsed_ms} error={_response_snippet(str(exc))}",
            )
        except requests.RequestException as exc:
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            last_result = OverpassQueryResult([], "request_error", str(exc), attempts=attempt)
            _log_osm_api_call(
                log_api_calls,
                f"response node={node_label or 'unknown'} attempt={attempt}/{attempts} status=request_error duration_ms={elapsed_ms} error={_response_snippet(str(exc))}",
            )
        else:
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            if response.status_code != 200:
                error = f"HTTP {response.status_code} {response.reason}: {_response_snippet(response.text)}"
                last_result = OverpassQueryResult(
                    [],
                    "http_error",
                    error,
                    status_code=response.status_code,
                    attempts=attempt,
                )
                _log_osm_api_call(
                    log_api_calls,
                    (
                        f"response node={node_label or 'unknown'} attempt={attempt}/{attempts} status=http_error "
                        f"http_status={response.status_code} reason={response.reason} "
                        f"duration_ms={elapsed_ms} error={_response_snippet(response.text)}"
                    ),
                )
                if response.status_code not in OVERPASS_RETRY_STATUS_CODES:
                    return last_result
            else:
                try:
                    payload = response.json()
                except ValueError as exc:
                    error = f"{exc}: {_response_snippet(response.text)}"
                    last_result = OverpassQueryResult(
                        [],
                        "json_error",
                        error,
                        status_code=response.status_code,
                        attempts=attempt,
                    )
                    _log_osm_api_call(
                        log_api_calls,
                        (
                            f"response node={node_label or 'unknown'} attempt={attempt}/{attempts} status=json_error "
                            f"http_status={response.status_code} duration_ms={elapsed_ms} error={_response_snippet(error)}"
                        ),
                    )
                else:
                    elements = payload.get("elements", [])
                    if not isinstance(elements, list):
                        error = "Overpass JSON response did not contain a list-valued elements field."
                        _log_osm_api_call(
                            log_api_calls,
                            (
                                f"response node={node_label or 'unknown'} attempt={attempt}/{attempts} status=api_error "
                                f"http_status={response.status_code} duration_ms={elapsed_ms} error={error}"
                            ),
                        )
                        return OverpassQueryResult(
                            [],
                            "api_error",
                            error,
                            status_code=response.status_code,
                            attempts=attempt,
                        )
                    if not elements:
                        remark = payload.get("remark", "")
                        error = _response_snippet(str(remark)) if remark else ""
                        _log_osm_api_call(
                            log_api_calls,
                            (
                                f"response node={node_label or 'unknown'} attempt={attempt}/{attempts} status=empty "
                                f"http_status={response.status_code} elements=0 duration_ms={elapsed_ms} error={error}"
                            ),
                        )
                        return OverpassQueryResult(
                            [],
                            "empty",
                            error,
                            status_code=response.status_code,
                            attempts=attempt,
                        )
                    _log_osm_api_call(
                        log_api_calls,
                        (
                            f"response node={node_label or 'unknown'} attempt={attempt}/{attempts} status=ok "
                            f"http_status={response.status_code} elements={len(elements)} duration_ms={elapsed_ms}"
                        ),
                    )
                    return OverpassQueryResult(
                        elements,
                        "ok",
                        status_code=response.status_code,
                        attempts=attempt,
                    )

        if attempt < attempts:
            sleep_seconds = max(retry_backoff, 0.0) * attempt
            _log_osm_api_call(
                log_api_calls,
                f"retry_sleep node={node_label or 'unknown'} seconds={sleep_seconds:.2f} next_attempt={attempt + 1}/{attempts}",
            )
            time.sleep(sleep_seconds)
    return last_result


def _query_overpass(lat: float, lon: float, radius: int, endpoint: str, timeout: int, output_limit: int = 200) -> List[Dict]:
    return _query_overpass_with_diagnostics(lat, lon, radius, endpoint, timeout, output_limit=output_limit).elements


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


def _build_osm_text(row: pd.Series, node_id: str, lat_col: str, lon_col: str, summary: Dict[str, str]) -> str:
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
        f"Metadata: {field_text}. "
        f"OpenStreetMap nearby road classes: {summary['road_classes']}. "
        f"OpenStreetMap nearby POI types: {summary['poi_types']}. "
        f"OpenStreetMap nearby land use: {summary['landuse']}."
    )


def _cache_matches_request(payload: Dict, disable_osm: bool, radius: int, endpoint: str, output_limit: int) -> bool:
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        return False
    source = payload.get("source")
    if not isinstance(source, dict):
        return disable_osm
    requested_osm_enabled = not disable_osm
    if bool(source.get("osm_enabled")) != requested_osm_enabled:
        return False
    if requested_osm_enabled:
        return (
            source.get("osm_endpoint") == endpoint
            and int(source.get("osm_radius", -1)) == int(radius)
            and int(source.get("osm_output_limit", -1)) == int(output_limit)
        )
    return True


def _write_osm_report(cache_dir: str, dataset: str, source: Dict, nodes: List[Dict]) -> None:
    status_counts = Counter(node.get("osm_status", "unknown") for node in nodes)
    error_nodes = [
        {
            "order": node.get("order"),
            "node_id": node.get("node_id"),
            "status": node.get("osm_status"),
            "status_code": node.get("osm_status_code"),
            "attempts": node.get("osm_attempts"),
            "error": node.get("osm_error"),
        }
        for node in nodes
        if node.get("osm_error")
    ]
    report = {
        "dataset": dataset,
        "source": source,
        "status_counts": dict(status_counts),
        "error_count": len(error_nodes),
        "sample_errors": error_nodes[:50],
    }
    report_path = os.path.join(cache_dir, "semantic_osm_report.json")
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


def _texts_fingerprint(texts: List[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def build_or_load_semantic_texts(
    dataset: str,
    meta_path: str,
    cache_dir: str,
    radius: int,
    endpoint: str,
    timeout: int,
    output_limit: int = 200,
    max_retries: int = 2,
    retry_backoff: float = 1.5,
    rate_limit_interval: float = 0.2,
    log_api_calls: bool = False,
    force_refresh: bool = False,
    disable_osm: bool = False,
) -> List[str]:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "semantic_texts.json")
    if os.path.exists(cache_path) and not force_refresh:
        with open(cache_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if _cache_matches_request(payload, disable_osm, radius, endpoint, output_limit):
            _log_osm_api_call(log_api_calls, f"cache_hit dataset={dataset} path={cache_path} api_calls=0")
            return [item["text"] for item in sorted(payload["nodes"], key=lambda item: item["order"])]
        _log_osm_api_call(log_api_calls, f"cache_stale dataset={dataset} path={cache_path} reason=request_changed")
    else:
        _log_osm_api_call(log_api_calls, f"cache_miss dataset={dataset} path={cache_path}")

    meta = pd.read_csv(meta_path)
    lat_col = _first_existing_column(meta.columns, LATITUDE_CANDIDATES, "latitude")
    lon_col = _first_existing_column(meta.columns, LONGITUDE_CANDIDATES, "longitude")
    id_col = None
    for candidate in NODE_ID_CANDIDATES:
        if candidate in meta.columns:
            id_col = candidate
            break

    nodes = []
    _log_osm_api_call(
        log_api_calls,
        (
            f"generate_semantic_texts dataset={dataset} nodes={len(meta)} "
            f"osm_enabled={not disable_osm} radius={radius} output_limit={output_limit}"
        ),
    )
    for order, row in meta.reset_index(drop=True).iterrows():
        node_id = str(row[id_col]) if id_col is not None else str(order)
        fallback_text = _build_fallback_text(row, node_id, lat_col, lon_col)
        osm_status = "disabled"
        osm_error = ""
        osm_status_code = None
        osm_attempts = 0
        osm_element_count = 0
        osm_summary = None

        if disable_osm:
            text = fallback_text
        else:
            result = _query_overpass_with_diagnostics(
                float(row[lat_col]),
                float(row[lon_col]),
                radius,
                endpoint,
                timeout,
                output_limit=output_limit,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
                log_api_calls=log_api_calls,
                node_label=f"{dataset}:{node_id}",
            )
            osm_status = result.status
            osm_error = result.error
            osm_status_code = result.status_code
            osm_attempts = result.attempts
            osm_element_count = len(result.elements)
            if result.elements:
                osm_summary = _summarize_osm_elements(result.elements)
                text = _build_osm_text(row, node_id, lat_col, lon_col, osm_summary)
            else:
                text = fallback_text
            time.sleep(max(rate_limit_interval, 0.0))

        node_payload = {
            "order": int(order),
            "node_id": node_id,
            "text": _clean_text(text),
            "osm_status": osm_status,
            "osm_error": osm_error,
            "osm_status_code": osm_status_code,
            "osm_attempts": osm_attempts,
            "osm_element_count": osm_element_count,
        }
        if osm_summary is not None:
            node_payload["osm_summary"] = osm_summary
        nodes.append(node_payload)

    source = {
        "cache_version": 2,
        "osm_enabled": not disable_osm,
        "osm_endpoint": endpoint,
        "osm_radius": radius,
        "osm_output_limit": output_limit,
        "osm_timeout": timeout,
        "osm_max_retries": max_retries,
        "osm_log_api_calls": log_api_calls,
    }
    with open(cache_path, "w", encoding="utf-8") as file:
        json.dump({"dataset": dataset, "source": source, "nodes": nodes}, file, ensure_ascii=False, indent=2)
    _write_osm_report(cache_dir, dataset, source, nodes)
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
    meta_path = os.path.join(cache_dir, "semantic_embeddings_meta.json")
    expected_meta = {
        "cache_version": 2,
        "model_name": model_name,
        "max_length": max_length,
        "text_count": len(texts),
        "texts_sha256": _texts_fingerprint(texts),
    }
    if os.path.exists(cache_path) and os.path.exists(meta_path) and not force_refresh:
        with open(meta_path, "r", encoding="utf-8") as file:
            cached_meta = json.load(file)
        if all(cached_meta.get(key) == value for key, value in expected_meta.items()):
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
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(expected_meta, file, ensure_ascii=False, indent=2)
    return embeddings
