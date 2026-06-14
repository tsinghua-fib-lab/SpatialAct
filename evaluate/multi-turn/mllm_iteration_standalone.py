#!/usr/bin/env python3
"""
 MLLM 


-  src/iteration_controller.py  blender_scripts/
-  region_multi_step_error_construct  error_scene.blend
-  MLLM  top/isometric final_scene.blend
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import threading
from urllib import error as urllib_error
from urllib import request as urllib_request

import httpx
from openai import AzureOpenAI, OpenAI


PROJECT_ROOT = "SpatialAct"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
BLENDER = os.path.join(PROJECT_ROOT, "blender-3.2.2-linux-x64/blender")
BLENDER_SCRIPT = os.path.join(THIS_DIR, "blender_iter_renderer.py")
METRICS_SCRIPT = os.path.join(THIS_DIR, "metrics.py")
REGION_COMPLEX_SPLIT_BLEND_ROOT = os.environ.get(
    "REGION_COMPLEX_SPLIT_BLEND_ROOT",
    os.path.join(PROJECT_ROOT, "benchmark", "data", "region_complex_blend"),
)
REGION_BASIC_SPLIT_BLEND_ROOT = os.environ.get(
    "REGION_BASIC_SPLIT_BLEND_ROOT",
    os.path.join(PROJECT_ROOT, "benchmark", "data", "region_basic_blend"),
)
DEFAULT_AZURE_ENDPOINT = ""
DEFAULT_AZURE_API_VERSION = ""
_BLENDER_RENDER_SEMAPHORE = None


@dataclass
class RegionSample:
    region_id: int
    anomalies: List[Dict]
    initial_images: List[str]
    error_images: List[str]
    labels_map: Dict[str, str]
    scene_name: str = ""
    glb_name: str = ""
    source_scene_path: str = ""
    labeled_objects_hint: str = ""
    source_step: int = 0
    source_sample_id: str = ""
    source_sample_index: int = -1


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _append_jsonl(path: str, payload: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _path_candidates(path_value: str) -> List[str]:
    raw = str(path_value or "").strip()
    if not raw:
        return []

    candidates: List[str] = [raw]
    if raw.startswith("SpatialAct/"):
        rel = raw.split("/", 1)[1]
        candidates.append(os.path.join(REPO_ROOT, rel))
        candidates.append(os.path.join(os.path.dirname(REPO_ROOT), raw))
        candidates.append(os.path.join(PROJECT_ROOT, rel))
    elif not os.path.isabs(raw):
        candidates.append(os.path.join(REPO_ROOT, raw))
        candidates.append(os.path.join(os.path.dirname(REPO_ROOT), raw))

    out: List[str] = []
    seen = set()
    for p in candidates:
        if not p:
            continue
        norm = os.path.normpath(p)
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _resolve_existing_path(path_value: str) -> str:
    for candidate in _path_candidates(path_value):
        if os.path.exists(candidate):
            return candidate
    return str(path_value or "").strip()


def _normalize_path_list(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths or []:
        value = str(p or "").strip()
        if value:
            out.append(_resolve_existing_path(value))
    return out


def _to_int(value) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except Exception:
        return 0


def _normalize_token_usage(raw_usage) -> Dict[str, int]:
    usage_dict: Dict = {}
    if raw_usage is None:
        usage_dict = {}
    elif isinstance(raw_usage, dict):
        usage_dict = raw_usage
    elif hasattr(raw_usage, "model_dump"):
        usage_dict = raw_usage.model_dump() or {}
    elif hasattr(raw_usage, "dict"):
        usage_dict = raw_usage.dict() or {}

    prompt_tokens = _to_int(usage_dict.get("prompt_tokens"))
    completion_tokens = _to_int(usage_dict.get("completion_tokens"))
    # Responses API uses input_tokens/output_tokens.
    if prompt_tokens <= 0:
        prompt_tokens = _to_int(usage_dict.get("input_tokens"))
    if completion_tokens <= 0:
        completion_tokens = _to_int(usage_dict.get("output_tokens"))
    total_tokens = _to_int(usage_dict.get("total_tokens"))
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _safe_model_dump(value):
    if value is None:
        return None
    if isinstance(value, (dict, list, str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return value.dict()
        except Exception:
            pass
    return None


def _safe_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_safe_jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}

    dumped = _safe_model_dump(value)
    if dumped is not None and dumped is not value:
        return _safe_jsonable(dumped)
    return str(value)


def _chat_messages_to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        raw_content = msg.get("content", "")
        content_items: List[Dict[str, Any]] = []

        if isinstance(raw_content, str):
            content_items.append({"type": "input_text", "text": raw_content})
        elif isinstance(raw_content, list):
            for part in raw_content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "text":
                    txt = part.get("text")
                    if isinstance(txt, str):
                        content_items.append({"type": "input_text", "text": txt})
                elif part_type == "image_url":
                    img = part.get("image_url", {})
                    if not isinstance(img, dict):
                        continue
                    url = img.get("url")
                    detail = img.get("detail", "auto")
                    if isinstance(url, str) and url.strip():
                        item: Dict[str, Any] = {"type": "input_image", "image_url": url}
                        if isinstance(detail, str) and detail.strip():
                            item["detail"] = detail
                        content_items.append(item)

        if content_items:
            converted.append({"role": role, "content": content_items})
    return converted


def _extract_responses_text(resp: Any) -> str:
    out_text = getattr(resp, "output_text", None)
    if isinstance(out_text, str) and out_text.strip():
        return out_text.strip()

    output = getattr(resp, "output", None)
    if not isinstance(output, list):
        return ""

    parts: List[str] = []
    for item in output:
        if getattr(item, "type", None) != "message":
            continue
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            continue
        for c in content:
            c_type = getattr(c, "type", None)
            if c_type in {"output_text", "text"}:
                txt = getattr(c, "text", None)
                if isinstance(txt, str):
                    parts.append(txt)
    return "\n".join(parts).strip()


def _extract_responses_reasoning(resp: Any) -> Any:
    output = getattr(resp, "output", None)
    if not isinstance(output, list):
        return None

    reasoning_items: List[Any] = []
    for item in output:
        if getattr(item, "type", None) != "reasoning":
            continue
        summary = getattr(item, "summary", None)
        if summary is None:
            reasoning_items.append(_safe_jsonable(item))
            continue
        reasoning_items.append(_safe_jsonable(summary))

    if not reasoning_items:
        return None
    if len(reasoning_items) == 1:
        return reasoning_items[0]
    return reasoning_items


def _flatten_reasoning_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)

    dumped = _safe_model_dump(value)
    if dumped is not None and dumped is not value:
        return _flatten_reasoning_text(dumped)

    if isinstance(value, list):
        parts = []
        for item in value:
            text = _flatten_reasoning_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()

    if isinstance(value, dict):
        # Common reasoning payload keys across providers.
        for key in ("reasoning", "reasoning_content", "text", "content", "summary"):
            text = _flatten_reasoning_text(value.get(key))
            if text:
                return text
        return ""

    return ""


def _progress_bar(current: int, total: int, width: int = 20) -> str:
    total_safe = max(1, int(total))
    current_safe = max(0, min(int(current), total_safe))
    filled = int(round(width * (current_safe / total_safe)))
    return f"[{'#' * filled}{'.' * (width - filled)}] {current_safe}/{total_safe}"


def _fmt_elapsed(seconds: float) -> str:
    sec = max(0.0, float(seconds))
    if sec < 60:
        return f"{sec:.1f}s"
    minutes = int(sec // 60)
    remain = sec - minutes * 60
    return f"{minutes}m{remain:04.1f}s"


def _infer_llm_provider(provider: str, model: str) -> str:
    p = str(provider or "").strip().lower()
    if p in {"gpt", "openrouter"}:
        return p
    return "gpt" if "gpt" in str(model or "").lower() else "openrouter"


def _build_llm_runtime(args) -> Dict[str, Any]:
    provider = _infer_llm_provider(getattr(args, "llm_provider", "auto"), args.llm_model)
    runtime: Dict[str, Any] = {
        "provider": provider,
        "model": str(args.llm_model),
        "timeout": int(args.llm_timeout),
        "max_retries": int(args.llm_max_retries),
    }

    if provider == "gpt":
        api_key = os.environ.get("AZURE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(" Azure API Key : AZURE_API_KEY")
        runtime.update(
            {
                "api_key": api_key,
                "azure_endpoint": str(args.azure_endpoint or DEFAULT_AZURE_ENDPOINT),
                "azure_api_version": str(args.azure_api_version or DEFAULT_AZURE_API_VERSION),
            }
        )
        runtime["gpt_use_responses_api"] = bool(getattr(args, "gpt_use_responses_api", True))
        if bool(getattr(args, "gpt_no_reasoning", False)):
            runtime["gpt_reasoning"] = None
        else:
            effort = str(getattr(args, "gpt_reasoning_effort", "medium") or "").strip().lower()
            summary = str(getattr(args, "gpt_reasoning_summary", "auto") or "").strip().lower()
            gpt_reasoning: Dict[str, str] = {}
            if effort:
                gpt_reasoning["effort"] = effort
            if summary and summary != "none":
                gpt_reasoning["summary"] = summary
            runtime["gpt_reasoning"] = gpt_reasoning if gpt_reasoning else None
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            #  --api-key-env 
            api_key = os.environ.get(str(args.api_key_env or ""), "").strip()
        if not api_key:
            raise RuntimeError(
                " OpenRouter API Key : OPENROUTER_API_KEY "
                f"( {args.api_key_env})"
            )
        runtime.update(
            {
                "api_key": api_key,
                "base_url": str(args.openrouter_base_url),
            }
        )
    return runtime


def _create_llm_client(llm_runtime: Dict[str, Any], timeout: int):
    provider = llm_runtime.get("provider", "openrouter")
    if provider == "gpt":
        return AzureOpenAI(
            api_version=llm_runtime["azure_api_version"],
            azure_endpoint=llm_runtime["azure_endpoint"],
            api_key=llm_runtime["api_key"],
        )

    proxy_url = llm_runtime.get("proxy", "")
    if proxy_url and not proxy_url.startswith(("http://", "https://", "socks5://", "socks4://")):
        proxy_url = f"http://{proxy_url}"
    timeout_config = httpx.Timeout(connect=10.0, read=float(timeout), write=10.0, pool=5.0)
    client_http = httpx.Client(proxy=proxy_url, timeout=timeout_config) if proxy_url else httpx.Client(timeout=timeout_config)
    return OpenAI(
        base_url=llm_runtime["base_url"],
        api_key=llm_runtime["api_key"],
        http_client=client_http,
    )


def _copy(src: str, dst: str) -> bool:
    src = _resolve_existing_path(src)
    if not src or not os.path.exists(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    import shutil
    shutil.copy2(src, dst)
    return True


def _safe_fs_name(text: str) -> str:
    s = str(text or "").strip()
    s = s.replace(os.sep, "_")
    if os.altsep:
        s = s.replace(os.altsep, "_")
    s = re.sub(r"[^\w.\-]+", "_", s)
    s = s.strip("._")
    return s


def _parse_regions(spec: str | None, candidates: List[int]) -> List[int]:
    if not spec:
        return sorted(candidates)
    out: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            s, e = int(a), int(b)
            if s > e:
                s, e = e, s
            out.extend(range(s, e + 1))
        else:
            out.append(int(token))
    out = sorted(set(out))
    return [item for item in out if item in candidates]


def _load_samples(qa_json: str) -> Dict[int, RegionSample]:
    data = _load_json(qa_json)
    if not isinstance(data, list):
        raise ValueError("qa json  list")
    out: Dict[int, RegionSample] = {}
    for row in data:
        rid = int(row.get("region_id", -1))
        if rid < 0:
            continue
        qa = row.get("qa", {}).get("multi_step", {})
        out[rid] = RegionSample(
            region_id=rid,
            anomalies=row.get("anomalies", []) or [],
            initial_images=_normalize_path_list(qa.get("initial_images", []) or []),
            error_images=_normalize_path_list(qa.get("images", []) or []),
            labels_map={str(k): str(v) for k, v in (row.get("labels_map", {}) or {}).items()},
            scene_name=str(row.get("scene_name", "") or ""),
            glb_name=str(row.get("glb_name", "") or ""),
            source_scene_path=_resolve_existing_path(str(row.get("source_scene_path", "") or "")),
            labeled_objects_hint=str(row.get("labeled_objects_hint", "") or ""),
            source_step=_safe_int(row.get("source_step"), default=0),
            source_sample_id=str(row.get("source_sample_id", "") or ""),
            source_sample_index=_safe_int(row.get("source_sample_index"), default=-1),
        )
    return out


def _extract_labeled_objects_hint(question_text: str) -> str:
    text = str(question_text or "")
    if not text.strip():
        return ""
    m = re.search(r"(?im)^\s*labeled\s+objects\s*:\s*(.+?)\s*$", text)
    if not m:
        return ""
    detail = re.sub(r"\s+", " ", m.group(1).strip())
    if not detail:
        return ""
    return f"Labeled objects: {detail}"


def _extract_scene_path_from_row(row: Dict) -> str:
    if not isinstance(row, dict):
        return ""
    for key in ("error_scene", "error_scene_glb", "error_blend", "source_scene_path", "scene"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return _resolve_existing_path(value)
    images = row.get("images")
    if isinstance(images, dict):
        for key in ("error_scene", "error_scene_glb", "error_blend", "source_scene_path", "scene"):
            value = str(images.get(key, "") or "").strip()
            if value:
                return _resolve_existing_path(value)
    return ""


def _labels_map_from_task_row(row: Dict) -> Dict[str, str]:
    labels_map = row.get("labels_map") if isinstance(row, dict) else None
    if isinstance(labels_map, dict) and labels_map:
        return {str(k): str(v) for k, v in labels_map.items()}

    label_rows = row.get("label_mapping", []) if isinstance(row, dict) else []
    out: Dict[str, str] = {}
    if isinstance(label_rows, list):
        for item in label_rows:
            if not isinstance(item, dict):
                continue
            lb = item.get("label_id")
            if lb is None or not str(lb).isdigit():
                continue
            target = item.get("instance_index")
            if target is None or str(target).strip() == "":
                target = item.get("scene_object_id")
            if target is not None and str(target).strip():
                out[str(int(lb))] = str(target).strip()
    if out:
        return out

    return {}


def _build_direct_task_row(row: Dict, region_id: int) -> Dict:
    images = row.get("images") or []
    if isinstance(images, dict):
        task_images = images.get("images") or images.get("multi_step") or []
        if not isinstance(task_images, list):
            task_images = []
    elif isinstance(images, list):
        task_images = images
    else:
        task_images = []

    initial_images = row.get("initial_images") or []
    if not isinstance(initial_images, list):
        initial_images = []

    source_scene_path = _extract_scene_path_from_row(row)
    labels_map = _labels_map_from_task_row(row)
    scene_name = str(row.get("scene_name", "") or "").strip()
    if not scene_name and source_scene_path:
        scene_name = os.path.basename(os.path.dirname(source_scene_path)) or os.path.splitext(os.path.basename(source_scene_path))[0]

    return {
        "region_id": int(region_id),
        "scene_name": scene_name or f"scene_{region_id}",
        "glb_name": os.path.basename(source_scene_path) if source_scene_path else "",
        "labels_map": labels_map,
        "labeled_objects_hint": _extract_labeled_objects_hint(str(row.get("question", "") or "")),
        "source_scene_path": source_scene_path,
        "source_step": 0,
        "source_sample_id": str(row.get("sample_id", "") or ""),
        "source_sample_index": int(region_id),
        "anomalies": row.get("anomalies", []) or [],
        "qa": {
            "multi_step": {
                "images": _normalize_path_list([str(x) for x in task_images if str(x or "").strip()]),
                "initial_images": _normalize_path_list([str(x) for x in initial_images if str(x or "").strip()]),
            }
        },
    }


def _write_iteration_inputs_from_rows(rows: List[Dict], sampled_metadata_json: str, qa_out: str, region_data_out: str) -> tuple[str, str]:
    if not rows:
        raise RuntimeError(f"sampled/task metadata : {sampled_metadata_json}")
    _write_json(qa_out, rows)
    region_rows = []
    for row in rows:
        labels_map = row.get("labels_map") or {}
        ordered = sorted([k for k in labels_map.keys() if str(k).isdigit()], key=lambda x: int(x))
        region_rows.append({"region_id": int(row["region_id"]), "building_ids": [labels_map[k] for k in ordered]})
    _write_json(
        region_data_out,
        {
            "regions": region_rows,
            "meta": {"generated_from_sampled": os.path.abspath(sampled_metadata_json)},
        },
    )
    return qa_out, region_data_out


def _sample_prefers_isometric(sample: RegionSample) -> bool:
    if len(sample.error_images) < 2:
        return False
    return bool(str(sample.error_images[1] or "").strip())


def _resolve_region_data_path_for_name(region_name: str) -> str:
    token = str(region_name or "").strip()
    if not token:
        return ""
    candidates: List[str] = []
    if token.endswith("_complex"):
        candidates.extend(
            [
                os.path.join(PROJECT_ROOT, f"benchmark/data_construct/model_process/results/{token}/region_data_clean.json"),
                os.path.join(PROJECT_ROOT, f"benchmark/data_construct/model_process/results/{token}_kmeans/region_data_clean.json"),
            ]
        )
    else:
        candidates.extend(
            [
                os.path.join(PROJECT_ROOT, f"benchmark/data_construct/model_process/results/{token}_kmeans/region_data_clean.json"),
                os.path.join(PROJECT_ROOT, f"benchmark/data_construct/model_process/results/{token}/region_data_clean.json"),
            ]
        )
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0] if candidates else ""


def _pick_first_existing(paths: List[str]) -> str:
    for p in paths:
        resolved = _resolve_existing_path(p)
        if resolved and os.path.exists(resolved):
            return resolved
    return ""


def _safe_int(x, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _resolve_split_region_blend_path(region_name: str, src_region_id: int, source_step: int = 0) -> str:
    rid = _safe_int(src_region_id, default=-1)
    if rid < 0:
        return ""

    rn = str(region_name or "").strip().lower()
    family = ""
    sub_dir = ""
    prefix = ""


    root = REGION_COMPLEX_SPLIT_BLEND_ROOT if family == "complex" else REGION_BASIC_SPLIT_BLEND_ROOT
    candidates_region_dirs = []
    step_i = _safe_int(source_step, default=0)
    if step_i > 0:
        candidates_region_dirs.append(os.path.join(root, sub_dir, f"step{step_i}", f"region_{rid}"))
    candidates_region_dirs.append(os.path.join(root, sub_dir, f"region_{rid}"))

    for region_dir in candidates_region_dirs:
        if not os.path.isdir(region_dir):
            continue
        candidates = [
            os.path.join(region_dir, f"{prefix}_region_{rid}_clean.blend"),
            os.path.join(region_dir, f"{sub_dir}_region_{rid}_clean.blend"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p

        wildcard = sorted(
            p for p in glob.glob(os.path.join(region_dir, f"*region_{rid}*_clean.blend"))
            if os.path.isfile(p)
        )
        if wildcard:
            return wildcard[0]

        fallback = sorted(
            p for p in glob.glob(os.path.join(region_dir, "*.blend"))
            if os.path.isfile(p)
        )
        if fallback:
            return fallback[0]
    return ""


def _normalize_sampled_source(row: Dict) -> Dict:
    src = row.get("_source") or {}
    if not isinstance(src, dict):
        return {}
    return src


def _extract_multi_step_case_tag(paths: List[str]) -> str:
    for p in paths:
        s = str(p or "").strip()
        if not s:
            continue
        m = re.search(r"(multi_step_case_\d+)", s)
        if m:
            return m.group(1)
    return ""


def _infer_indoor_mode_from_src(src_info: Dict) -> str:
    mode = str(src_info.get("indoor_mode", "") or "").strip().lower()
    if mode in {"simple", "complex"}:
        return mode
    for key in ("region", "source_file", "scene_name"):
        token = str(src_info.get(key, "") or "").lower()
        if "simple" in token:
            return "simple"
        if "complex" in token:
            return "complex"
    return "unknown"


def _resolve_indoor_source_file(source_file: str, step_value) -> str:
    sf = str(source_file or "").strip()
    if sf and os.path.exists(sf):
        return sf
    base_dir = os.path.dirname(sf) if sf else ""
    step_i = _safe_int(step_value, default=0)
    candidates = []
    if base_dir:
        if step_i > 0:
            candidates.append(os.path.join(base_dir, f"metadata_indoor_steps{step_i}.json"))
        candidates.append(os.path.join(base_dir, "metadata_indoor.json"))
        candidates.extend(sorted(glob.glob(os.path.join(base_dir, "metadata_indoor_steps*.json"))))
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return sf


def _build_sampled_region_row(
    sampled_row: Dict,
    src_info: Dict,
    source_row: Dict,
    region_name: str,
    region_id: int,
    region_data_cache: Dict[str, Dict],
) -> Dict:
    source_file = str(src_info.get("source_file", "") or "")
    source_dir = os.path.dirname(source_file) if source_file else ""
    source_step = _safe_int(src_info.get("step"), default=0)
    src_region_id = _safe_int(src_info.get("region_id"), default=-1)

    qa_ms = (source_row.get("qa", {}) or {}).get("multi_step", {}) or {}
    sampled_imgs = sampled_row.get("images") or []
    sampled_init = sampled_row.get("initial_images") or []

    top_err = _pick_first_existing(
        [
            str(sampled_imgs[0]) if len(sampled_imgs) > 0 else "",
            str((qa_ms.get("images") or [None])[0]) if isinstance(qa_ms.get("images"), list) else "",
            os.path.join(source_dir, f"region_{src_region_id}", "top_multi_error.png"),
        ]
    )
    iso_err = _pick_first_existing(
        [
            str(sampled_imgs[1]) if len(sampled_imgs) > 1 else "",
            str((qa_ms.get("images") or [None, None])[1]) if isinstance(qa_ms.get("images"), list) and len(qa_ms.get("images", [])) > 1 else "",
            os.path.join(source_dir, f"region_{src_region_id}", "iso_multi_error.png"),
            os.path.join(source_dir, f"region_{src_region_id}", "isometric_multi_error.png"),
        ]
    )
    top_init = _pick_first_existing(
        [
            str(sampled_init[0]) if len(sampled_init) > 0 else "",
            str((qa_ms.get("initial_images") or [None])[0]) if isinstance(qa_ms.get("initial_images"), list) else "",
            os.path.join(source_dir, f"region_{src_region_id}", "top.png"),
        ]
    )
    iso_init = _pick_first_existing(
        [
            str(sampled_init[1]) if len(sampled_init) > 1 else "",
            str((qa_ms.get("initial_images") or [None, None])[1]) if isinstance(qa_ms.get("initial_images"), list) and len(qa_ms.get("initial_images", [])) > 1 else "",
            os.path.join(source_dir, f"region_{src_region_id}", "isometric.png"),
            os.path.join(source_dir, f"region_{src_region_id}", "iso.png"),
        ]
    )

    task_scene_path = _extract_scene_path_from_row(sampled_row)
    split_region_blend = _resolve_split_region_blend_path(region_name, src_region_id, source_step=source_step)
    source_scene_path = _pick_first_existing(
        [
            task_scene_path,
            split_region_blend,
            str(sampled_row.get("error_scene", "") or ""),
            str(sampled_row.get("error_scene_glb", "") or ""),
            str(sampled_row.get("error_blend", "") or ""),
            str(qa_ms.get("error_blend", "") or ""),
            os.path.join(source_dir, f"{region_name}_steps{source_step}_global_error_scene.blend"),
            os.path.join(source_dir, f"{region_name}_steps{source_step}_global_error_scene.glb"),
            os.path.join(source_dir, f"{region_name}_global_error_scene.blend"),
            os.path.join(source_dir, f"{region_name}_global_error_scene.glb"),
        ]
    )

    labels_map = source_row.get("labels_map") or {}
    labels_map = {str(k): str(v) for k, v in labels_map.items()}
    if not labels_map:
        #  region_multi_step_error_construct.py 
        #  region  labels.json
        #  region_data 
        region_labels_path = os.path.join(source_dir, f"region_{src_region_id}", "labels.json")
        if os.path.exists(region_labels_path):
            try:
                region_labels = _load_json(region_labels_path)
                if isinstance(region_labels, dict):
                    labels_map = {
                        str(k): str(v)
                        for k, v in region_labels.items()
                        if str(k).isdigit() and str(v).strip()
                    }
            except Exception:
                labels_map = {}

    if not labels_map:
        if region_name not in region_data_cache:
            region_data_path = _resolve_region_data_path_for_name(region_name)
            if region_data_path and os.path.exists(region_data_path):
                region_data_cache[region_name] = _load_json(region_data_path)
            else:
                region_data_cache[region_name] = {"regions": []}
        labels_map = _labels_from_region_data(region_data_cache[region_name], src_region_id)

    return {
        "region_id": int(region_id),
        "scene_name": f"{region_name}/region_{src_region_id}",
        "glb_name": os.path.basename(source_scene_path) if source_scene_path else "",
        "labels_map": labels_map,
        "source_scene_path": source_scene_path,
        "source_step": int(source_step),
        "source_sample_id": str(src_info.get("sample_id", "") or ""),
        "source_sample_index": _safe_int(src_info.get("sample_index"), default=-1),
        "anomalies": source_row.get("anomalies", []) or [],
        "qa": {
            "multi_step": {
                "images": [x for x in [top_err, iso_err] if x],
                "initial_images": [x for x in [top_init, iso_init] if x],
            }
        },
    }


def _build_sampled_indoor_row(
    sampled_row: Dict,
    src_info: Dict,
    metadata_row: Dict,
    region_id: int,
) -> Dict:
    sampled_imgs = sampled_row.get("images") or []
    sampled_init = sampled_row.get("initial_images") or []
    meta_imgs = metadata_row.get("images") or {}

    top_err = _pick_first_existing([str(sampled_imgs[0]) if len(sampled_imgs) > 0 else "", str(meta_imgs.get("top", "") or "")])
    iso_err = _pick_first_existing([str(sampled_imgs[1]) if len(sampled_imgs) > 1 else "", str(meta_imgs.get("isometric", "") or "")])
    top_init = _pick_first_existing([str(sampled_init[0]) if len(sampled_init) > 0 else "", str(meta_imgs.get("top_init", "") or "")])
    iso_init = _pick_first_existing([str(sampled_init[1]) if len(sampled_init) > 1 else "", str(meta_imgs.get("isometric_init", "") or "")])
    source_scene_path = _pick_first_existing(
        [
            _extract_scene_path_from_row(sampled_row),
            _extract_scene_path_from_row(metadata_row),
            str(meta_imgs.get("error_scene_glb", "") or ""),
            str(meta_imgs.get("error_scene", "") or ""),
        ]
    )

    labels_map: Dict[str, str] = {}
    for row in metadata_row.get("label_mapping", []) or []:
        lb = row.get("label_id")
        if lb is None or (not str(lb).isdigit()):
            continue
        target = row.get("instance_index")
        if target is None or str(target).strip() == "":
            target = row.get("scene_object_id")
        if target is None:
            continue
        labels_map[str(int(lb))] = str(target).strip()
    labeled_objects_hint = _extract_labeled_objects_hint(
        ((metadata_row.get("qa") or {}).get("multi_step") or {}).get("question", "")
    ) or _extract_labeled_objects_hint(sampled_row.get("question", ""))
    source_step = _safe_int(src_info.get("step"), default=0)

    return {
        "region_id": int(region_id),
        "scene_name": str(src_info.get("scene_name", "") or metadata_row.get("scene_name", "") or f"scene_{region_id}"),
        "glb_name": str(metadata_row.get("glb_name", "") or ""),
        "labels_map": labels_map,
        "labeled_objects_hint": labeled_objects_hint,
        "source_scene_path": source_scene_path,
        "source_step": int(source_step),
        "source_sample_id": str(src_info.get("sample_id", "") or ""),
        "source_sample_index": _safe_int(src_info.get("sample_index"), default=-1),
        "anomalies": metadata_row.get("anomalies", []) or [],
        "qa": {
            "multi_step": {
                "images": [x for x in [top_err, iso_err] if x],
                "initial_images": [x for x in [top_init, iso_init] if x],
            }
        },
    }


def _build_iteration_inputs_from_sampled_metadata(
    sampled_metadata_json: str,
    scene_type: str,
    workspace_root: str,
    indoor_mode_filter: str = "all",
) -> tuple[str, str]:
    records = _load_json(sampled_metadata_json)
    if not isinstance(records, list):
        raise ValueError(f"sampled metadata  list: {sampled_metadata_json}")

    sampled_dir = os.path.join(workspace_root, "_sampled_inputs")
    os.makedirs(sampled_dir, exist_ok=True)
    qa_out = os.path.join(sampled_dir, "qa_from_sampled.json")
    region_data_out = os.path.join(sampled_dir, "region_data_from_sampled.json")

    rows: List[Dict] = []
    seen = set()
    source_cache: Dict[str, List[Dict]] = {}
    metadata_cache: Dict[str, List[Dict]] = {}
    region_data_cache: Dict[str, Dict] = {}
    next_region_id = 0

    has_source_rows = any(_normalize_sampled_source(rec) for rec in records if isinstance(rec, dict))
    if not has_source_rows:
        rows = [
            _build_direct_task_row(rec, region_id=i)
            for i, rec in enumerate(records)
            if isinstance(rec, dict)
        ]
        return _write_iteration_inputs_from_rows(rows, sampled_metadata_json, qa_out, region_data_out)

    for rec in records:
        src_info = _normalize_sampled_source(rec)
        if not src_info:
            continue
        source_file = str(src_info.get("source_file", "") or "")
        if not source_file:
            continue

        if scene_type == "indoor":
            inferred_mode = _infer_indoor_mode_from_src(src_info)
            if indoor_mode_filter in {"simple", "complex"} and inferred_mode != indoor_mode_filter:
                continue
            scene_name = str(src_info.get("scene_name", "") or "").strip()
            sampled_imgs = rec.get("images") or []
            sampled_imgs_key = tuple(str(x) for x in sampled_imgs[:2])
            key = ("indoor", source_file, scene_name, sampled_imgs_key)
            if key in seen:
                continue
            resolved_source = _resolve_indoor_source_file(source_file, src_info.get("step"))
            if resolved_source not in metadata_cache:
                if not os.path.exists(resolved_source):
                    continue
                metadata_cache[resolved_source] = _load_json(resolved_source)
            md_rows = metadata_cache[resolved_source]
            if not isinstance(md_rows, list):
                continue
            scene_hits: List[tuple[int, Dict]] = []
            for i, row in enumerate(md_rows):
                if str(row.get("scene_name", "") or "").strip() == scene_name:
                    scene_hits.append((i, row))
            hit = None
            src_sample_index = _safe_int(src_info.get("sample_index"), default=-1)
            # Prefer exact row by source sample index when it points to this scene.
            if 0 <= src_sample_index < len(md_rows):
                row = md_rows[src_sample_index]
                if str(row.get("scene_name", "") or "").strip() == scene_name:
                    hit = row
            # Fallback: infer case from sampled image path (multi_step_case_x).
            if hit is None and scene_hits:
                sampled_imgs = rec.get("images") or []
                sampled_init = rec.get("initial_images") or []
                case_tag = _extract_multi_step_case_tag(
                    [str(x) for x in (list(sampled_imgs) + list(sampled_init))]
                )
                if case_tag:
                    for _, row in scene_hits:
                        if str(row.get("case_tag", "") or "").strip() == case_tag:
                            hit = row
                            break
            if hit is None and scene_hits:
                hit = scene_hits[0][1]
            if hit is None:
                continue
            rows.append(_build_sampled_indoor_row(rec, src_info, hit, region_id=next_region_id))
            seen.add(key)
            next_region_id += 1
        else:
            src_region_id = _safe_int(src_info.get("region_id"), default=-1)
            region_name = str(src_info.get("region", "") or "").strip()
            if src_region_id < 0 or not region_name:
                continue
            key = ("region", source_file, region_name, src_region_id)
            if key in seen:
                continue
            if source_file not in source_cache:
                if not os.path.exists(source_file):
                    continue
                source_cache[source_file] = _load_json(source_file)
            qa_rows = source_cache[source_file]
            if not isinstance(qa_rows, list):
                continue
            hit = None
            for row in qa_rows:
                if _safe_int(row.get("region_id"), default=-1) == src_region_id:
                    hit = row
                    break
            if hit is None:
                continue
            rows.append(
                _build_sampled_region_row(
                    sampled_row=rec,
                    src_info=src_info,
                    source_row=hit,
                    region_name=region_name,
                    region_id=next_region_id,
                    region_data_cache=region_data_cache,
                )
            )
            seen.add(key)
            next_region_id += 1

    return _write_iteration_inputs_from_rows(rows, sampled_metadata_json, qa_out, region_data_out)


def _labels_from_region_data(region_data: Dict, region_id: int) -> Dict[str, str]:
    for region in region_data.get("regions", []):
        if int(region.get("region_id", -1)) == int(region_id):
            building_ids = list(region.get("building_ids", []) or [])
            return {str(i + 1): bid for i, bid in enumerate(building_ids)}
    return {}


def _move_dir_code_to_word(code: str) -> str:
    c = str(code or "").strip().upper()
    mapping = {
        "N": "North",
        "S": "South",
        "E": "East",
        "W": "West",
        "NE": "Northeast",
        "NW": "Northwest",
        "SE": "Southeast",
        "SW": "Southwest",
    }
    return mapping.get(c, "North")


def _indoor_scale_factor(up: bool, percent: float) -> float:
    p = max(0.0, float(percent))
    if bool(up):
        return 1.0 + p / 100.0
    return max(1e-4, 1.0 - p / 100.0)


def _convert_indoor_metadata_to_iteration_inputs(
    metadata_json: str,
    qa_json_out: str,
    region_data_json_out: str,
) -> tuple[str, str]:
    records = _load_json(metadata_json)
    if not isinstance(records, list):
        raise ValueError(f"metadata  list: {metadata_json}")

    qa_rows: List[Dict] = []
    region_rows: List[Dict] = []

    for idx, rec in enumerate(records):
        region_id = int(idx)

        label_rows = rec.get("label_mapping", []) or []
        labels_map: Dict[str, str] = {}
        for row in label_rows:
            lb = row.get("label_id")
            if lb is None or (not str(lb).isdigit()):
                continue
            # indoor export  instance_index
            target = row.get("instance_index")
            if target is None or str(target).strip() == "":
                target = row.get("scene_object_id")
            if target is None:
                continue
            labels_map[str(int(lb))] = str(target).strip()

        anomalies_out: List[Dict] = []
        for st in rec.get("anomalies", []) or []:
            inject_action = (st.get("inject_action") or {}).copy()
            op = str(inject_action.get("op", "")).lower()
            bid = inject_action.get("id")
            if bid is None:
                continue
            action_payload: Dict = {"op": op}
            if op == "move":
                action_payload["dir_word"] = _move_dir_code_to_word(inject_action.get("dir"))
                action_payload["dist_m"] = float(inject_action.get("dist", 0.0) or 0.0)
            elif op == "rotate":
                action_payload["deg"] = float(inject_action.get("deg", 0.0) or 0.0)
                action_payload["clockwise"] = bool(inject_action.get("clockwise", False))
            elif op == "scale":
                up = bool(inject_action.get("up", True))
                pct = float(inject_action.get("percent", 0.0) or 0.0)
                action_payload["scale_factor"] = _indoor_scale_factor(up, pct)
            else:
                continue

            anomalies_out.append(
                {
                    "step_id": int(st.get("step_id", 0) or 0),
                    "case_tag": str(st.get("case_tag", "")),
                    "bid_label": int(bid),
                    "action": action_payload,
                }
            )

        images = rec.get("images", {}) or {}
        error_top = (
            images.get("top_multi_error")
            or images.get("top")
            or (images.get("images", [None])[0] if isinstance(images.get("images"), list) else None)
            or ""
        )
        error_iso = (
            images.get("isometric_multi_error")
            or images.get("isometric")
            or (images.get("images", [None, None])[1] if isinstance(images.get("images"), list) and len(images.get("images", [])) > 1 else None)
            or ""
        )
        init_top = images.get("top_init") or ""
        init_iso = images.get("isometric_init") or ""
        source_scene_path = _pick_first_existing(
            [
                _extract_scene_path_from_row(rec),
                str(images.get("error_scene_glb", "") or ""),
                str(images.get("error_scene", "") or ""),
            ]
        )
        labeled_objects_hint = _extract_labeled_objects_hint(
            ((rec.get("qa") or {}).get("multi_step") or {}).get("question", "")
        )

        qa_rows.append(
            {
                "region_id": region_id,
                "scene_name": rec.get("scene_name", ""),
                "glb_name": rec.get("glb_name", ""),
                "labels_map": labels_map,
                "labeled_objects_hint": labeled_objects_hint,
                "source_scene_path": source_scene_path,
                "anomalies": anomalies_out,
                "qa": {
                    "multi_step": {
                        "images": [x for x in [error_top, error_iso] if x],
                        "initial_images": [x for x in [init_top, init_iso] if x],
                    }
                },
            }
        )

        ordered_label_keys = sorted(
            [k for k in labels_map.keys() if str(k).isdigit()],
            key=lambda x: int(x),
        )
        building_ids = [labels_map[k] for k in ordered_label_keys]
        region_rows.append({"region_id": region_id, "building_ids": building_ids})

    _write_json(
        region_data_json_out,
        {
            "regions": region_rows,
            "meta": {
                "scene_type": "indoor",
                "generated_from": os.path.abspath(metadata_json),
                "region_count": len(region_rows),
            },
        },
    )
    _write_json(qa_json_out, qa_rows)
    return qa_json_out, region_data_json_out


def _direction_to_move_key(direction: str) -> Optional[str]:
    key = (direction or "").strip().lower()
    mapping = {
        "n": "Up", "north": "Up", "up": "Up",
        "s": "Down", "south": "Down", "down": "Down",
        "e": "Right", "east": "Right", "right": "Right",
        "w": "Left", "west": "Left", "left": "Left",
    }
    return mapping.get(key)


def _direction_to_move_components(direction: str) -> Dict[str, float]:
    key = (direction or "").strip().lower().replace("-", "").replace("_", "")
    diag = 1.0 / math.sqrt(2.0)
    mapping: Dict[str, Dict[str, float]] = {
        "n": {"Up": 1.0},
        "north": {"Up": 1.0},
        "up": {"Up": 1.0},
        "s": {"Down": 1.0},
        "south": {"Down": 1.0},
        "down": {"Down": 1.0},
        "e": {"Right": 1.0},
        "east": {"Right": 1.0},
        "right": {"Right": 1.0},
        "w": {"Left": 1.0},
        "west": {"Left": 1.0},
        "left": {"Left": 1.0},
        "ne": {"Up": diag, "Right": diag},
        "northeast": {"Up": diag, "Right": diag},
        "nw": {"Up": diag, "Left": diag},
        "northwest": {"Up": diag, "Left": diag},
        "se": {"Down": diag, "Right": diag},
        "southeast": {"Down": diag, "Right": diag},
        "sw": {"Down": diag, "Left": diag},
        "southwest": {"Down": diag, "Left": diag},
    }
    return mapping.get(key, {})


def _seed_from_anomalies(anomalies: List[Dict], unit_scale: float) -> Dict[str, Dict]:
    cumulative: Dict[str, Dict] = {}

    def _ensure(label: str):
        if label not in cumulative:
            cumulative[label] = {
                "Move": {"Up": 0.0, "Down": 0.0, "Left": 0.0, "Right": 0.0},
                "Rotate": 0.0,
                "Scale": None,
            }

    for item in anomalies:
        label = str(item.get("bid_label", "")).strip()
        if not label:
            continue
        _ensure(label)
        action = item.get("action", {}) or {}
        op = str(action.get("op", "")).lower()
        if op == "move":
            direction = action.get("dir_word") or action.get("dir")
            dist_m = float(action.get("dist_m", 0.0) or 0.0)
            comps = _direction_to_move_components(str(direction))
            if comps and dist_m > 0:
                for k, coeff in comps.items():
                    cumulative[label]["Move"][k] += dist_m * unit_scale * float(coeff)
        elif op == "rotate":
            deg = float(action.get("deg", 0.0) or 0.0)
            clockwise = bool(action.get("clockwise", False))
            cumulative[label]["Rotate"] += (deg if clockwise else -deg)
        elif op == "scale":
            sf = action.get("scale_factor")
            if sf is not None:
                cumulative[label]["Scale"] = _accumulate_scale(cumulative[label]["Scale"], float(sf))
    return cumulative


def _consolidate(cumulative: Dict[str, Dict]) -> List[Dict]:
    def _fmt_num(value: float, precision: int = 3) -> str:
        text = f"{float(value):.{precision}f}"
        text = text.rstrip("0").rstrip(".")
        return text if text else "0"

    out: List[Dict] = []
    for label in sorted(cumulative.keys(), key=lambda x: int(x)):
        record = cumulative[label]
        moves = record["Move"]
        net_ns = moves.get("Up", 0.0) - moves.get("Down", 0.0)
        net_ew = moves.get("Right", 0.0) - moves.get("Left", 0.0)
        actions: List[str] = []
        if net_ns > 0:
            actions.append(f"Move(North, {_fmt_num(net_ns)})")
        elif net_ns < 0:
            actions.append(f"Move(South, {_fmt_num(-net_ns)})")
        if net_ew > 0:
            actions.append(f"Move(East, {_fmt_num(net_ew)})")
        elif net_ew < 0:
            actions.append(f"Move(West, {_fmt_num(-net_ew)})")

        rot = float(record.get("Rotate", 0.0) or 0.0)
        if abs(rot) > 1e-6:
            actions.append(f"Rotate({_fmt_num(rot)})")

        scale = record.get("Scale")
        if scale is not None:
            actions.append(f"Scale({_fmt_num(float(scale), precision=4)})")

        if actions:
            out.append({"id": int(label), "action": actions})
    return out


def _parse_action_line(action_str: str):
    text = action_str.strip()
    m = re.match(r"Move\s*\(\s*(\w+)\s*,\s*([-\d.]+)\s*\)", text, re.IGNORECASE)
    if m:
        return ("Move", m.group(1).capitalize(), float(m.group(2)))
    r = re.match(r"Rotate\s*\(\s*([-\d.]+)\s*\)", text, re.IGNORECASE)
    if r:
        return ("Rotate", float(r.group(1)))
    s = re.match(r"Scale\s*\(\s*([\d.]+)\s*\)", text, re.IGNORECASE)
    if s:
        return ("Scale", float(s.group(1)))
    return None


def _normalize_action_list(raw_action) -> List[str]:
    """Normalize model output action field to a stable list[str]."""
    if raw_action is None:
        return []

    if isinstance(raw_action, str):
        text = raw_action.strip()
        return [text] if text else []

    if isinstance(raw_action, (list, tuple)):
        out: List[str] = []
        for item in raw_action:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return out

    text = str(raw_action).strip()
    return [text] if text else []


def _accumulate_scale(prev_scale, new_factor: float):
    factor = float(new_factor)
    if prev_scale is None:
        return factor
    return float(prev_scale) * factor


def _normalize_issue_status(raw_status) -> str:
    token = str(raw_status or "").strip().upper()
    if not token:
        return ""
    token = re.sub(r"[\s\-]+", "_", token)
    mapping = {
        "FIXED": "FIXED",
        "STILL": "STILL_HAS_ISSUE",
        "STILL_HAS_ISSUE": "STILL_HAS_ISSUE",
        "STILL_ISSUE": "STILL_HAS_ISSUE",
        "HAS_ISSUE": "STILL_HAS_ISSUE",
        "WORSE": "WORSE",
        "NEW": "NEW",
        "NEW_ISSUE": "NEW",
    }
    return mapping.get(token, "")


def _is_actionable_issue_item(item: Dict) -> bool:
    status = _normalize_issue_status(item.get("issue_status"))
    actions = _normalize_action_list(item.get("action", []))
    if status == "FIXED" and not actions:
        return False
    if status in {"STILL_HAS_ISSUE", "WORSE", "NEW"}:
        return True
    if actions:
        return True
    return False


def _merge_analysis_into_cumulative(
    cumulative: Dict[str, Dict],
    analysis_items: List[Dict],
    unit_scale: float,
):
    for item in analysis_items:
        label = str(item.get("id", "")).strip()
        if not label:
            continue
        if label not in cumulative:
            cumulative[label] = {
                "Move": {"Up": 0.0, "Down": 0.0, "Left": 0.0, "Right": 0.0},
                "Rotate": 0.0,
                "Scale": None,
            }
        actions = _normalize_action_list(item.get("action", []))
        for text in actions:
            parsed = _parse_action_line(text)
            if not parsed:
                continue
            if parsed[0] == "Move":
                _, direction, distance = parsed
                comps = _direction_to_move_components(direction)
                if comps:
                    for k, coeff in comps.items():
                        cumulative[label]["Move"][k] += float(distance) * unit_scale * float(coeff)
            elif parsed[0] == "Rotate":
                _, angle = parsed
                cumulative[label]["Rotate"] += float(angle)
            elif parsed[0] == "Scale":
                _, factor = parsed
                cumulative[label]["Scale"] = _accumulate_scale(cumulative[label]["Scale"], float(factor))

def _extract_json_array(text: str) -> List[Dict]:
    if not text:
        return []
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:].strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    match = re.search(r"\[[\s\S]*\]", cleaned)
    if not match:
        match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        arr = json.loads(match.group(0))
        return arr if isinstance(arr, list) else []
    except Exception:
        return []


def _convert_image_to_public_url(
    path: str,
    convert_endpoint: str,
    timeout_seconds: int,
    expires_in: int,
    public_base_url: str = "",
) -> str:
    image_path = os.path.abspath(str(path))
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f" URL: {image_path}")

    payload = {"image_source": image_path}
    if int(expires_in) > 0:
        payload["expires_in"] = int(expires_in)
    if str(public_base_url or "").strip():
        payload["public_base_url"] = str(public_base_url).strip()

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib_request.Request(
        str(convert_endpoint).strip(),
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=float(timeout_seconds)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib_error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        raise RuntimeError(f" URL  HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f" URL : {exc}") from exc

    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f" URL  JSON : {raw[:300]}") from exc

    image_url = str(parsed.get("image_url", "")).strip()
    if not image_url:
        raise RuntimeError(f" URL  image_url : {parsed}")
    return image_url


def _encode_image(path: str) -> str:
    path = _resolve_existing_path(path)
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def _build_cumulative_summary(cumulative_actions: List[Dict], unit_scale: float, max_items: int = 12) -> str:
    def _fmt_num(value: float, precision: int = 3) -> str:
        text = f"{float(value):.{precision}f}"
        text = text.rstrip("0").rstrip(".")
        return text if text else "0"

    def _to_meter_actions(actions: List[str]) -> List[str]:
        converted: List[str] = []
        scale = float(unit_scale) if float(unit_scale) > 1e-9 else 1.0
        for text in actions:
            m = re.match(r"^\s*Move\s*\(\s*(\w+)\s*,\s*([-\d.]+)\s*\)\s*$", str(text), re.IGNORECASE)
            if m:
                direction = m.group(1)
                distance_scene = float(m.group(2))
                distance_meter = distance_scene / scale
                converted.append(f"Move({direction}, {_fmt_num(distance_meter)})")
            else:
                converted.append(str(text))
        return converted

    if not cumulative_actions:
        return "No cumulative actions applied yet."
    lines: List[str] = []
    for item in cumulative_actions[:max_items]:
        bid = item.get("id")
        actions = item.get("action", []) or []
        if actions:
            meter_actions = _to_meter_actions(actions)
            lines.append(f"- id={bid}: {', '.join(meter_actions)}")
    if len(cumulative_actions) > max_items:
        lines.append(f"- ... {len(cumulative_actions) - max_items} more buildings omitted")
    return "\n".join(lines) if lines else "No cumulative actions applied yet."


def _build_previous_actions_summary(previous_analysis: List[Dict], max_items: int = 16) -> str:
    if not previous_analysis:
        return "No actions were applied in the previous iteration."
    lines: List[str] = []
    for item in previous_analysis:
        actions = _normalize_action_list(item.get("action", []))
        if not actions:
            continue
        bid = item.get("id", "?")
        error_type = item.get("error_type") or "Unknown"
        lines.append(f"  - Building {bid} ({error_type}): {', '.join(actions)}")
        if len(lines) >= max_items:
            break
    if not lines:
        return "No actions were applied in the previous iteration."
    if len(previous_analysis) > max_items:
        lines.append(f"  - ... more actions omitted")
    return "\n".join(lines)


def _build_first_round_prompt_region(valid_ids: List[int]) -> str:
    return f"""
# Task Description
Analyze images to identify building issues and fix them. BRIGHT YELLOW areas are roads.

## Cautious Decision Rules
- Only flag issues that are visibly obvious from the images.
- Prefer leaving buildings unchanged over risky modifications.
- Please make only small adjustments to resolve issues, and do not move any object outside the current view.

## Coordinate Reference
- Rotation: clockwise = positive (+), counter-clockwise = negative (-), in degrees.
- Distance: use the visible white 1unit scale bar as reference.
- Direction: use north/south/east/west according to the red N arrow.

## Error Types (Check Independently For Every Building)
1. Overlap: physical intersection or z-fighting between building volumes.
2. RoadViolation: building footprint intersects BRIGHT YELLOW roads.
   - Check all sides against all nearby roads.
   - Roads should run alongside buildings, not through interiors/courtyards.
   - At most one short dead-end access road may connect to a building.
3. AlignmentError: building axes are not parallel/perpendicular to the road grid (not near 0°/90° relative to road axis), OR clearly inconsistent with nearby parallel buildings that should share orientation.

## One Building Can Have Multiple Issues
- The same building may have Overlap, RoadViolation, and AlignmentError at the same time.
- Output one JSON entry per building per error type.

## Output Requirements
- Output only a JSON array.
- No extra prose outside JSON.

Each item:
- id: integer building id
- error_type: "Overlap" | "RoadViolation" | "AlignmentError"
- reason: short issue description
- action: list of commands using `Move(direction, distance)`, `Rotate(angle)`, `Scale(factor)`
Direction must use: North/South/East/West, and diagonals (Northeast/Northwest/Southeast/Southwest).

## Examples
```json
[
  {{
    "id": 9,
    "error_type": "RoadViolation",
    "reason": "Building 9 intersects a yellow arterial road",
    "action": ["Move(North, 4)"]
  }}
]
```

```json
[
  {{
    "id": 6,
    "error_type": "AlignmentError",
    "reason": "Building 6 is rotated about 45° from the road grid",
    "action": ["Rotate(-45)"]
  }},
  {{
    "id": 6,
    "error_type": "RoadViolation",
    "reason": "Building 6 still intersects the bottom yellow road",
    "action": ["Move(North, 0.5)"]
  }}
]
```

When a road passes through a building interior/courtyard, move perpendicular to road direction:
- road north-south -> move east or west
- road east-west -> move north or south

Only use ids from this allowed set: {valid_ids}
If no issue, return EXACTLY [] (no extra text).
"""


def _build_followup_prompt_region(
    valid_ids: List[int],
    previous_actions_summary: str,
    cumulative_summary: str,
    has_isometric_ref: bool,
) -> str:
    if has_isometric_ref:
        image_reference = (
            "## Image Reference\n"
            "- Image 1: ORIGINAL state (top-down view before any adjustments)\n"
            "- Image 2: ORIGINAL state (isometric view before any adjustments)\n"
            "- Image 3: LAST ITERATION result (top-down view after previous adjustments)\n"
            "- Image 4: LAST ITERATION result (isometric view after previous adjustments)"
        )
    else:
        image_reference = (
            "## Image Reference\n"
            "- Image 1: ORIGINAL state (top-down view before any adjustments)\n"
            "- Image 2: LAST ITERATION result (top-down view after previous adjustments)"
        )

    return f"""# Task Description
You are now in a follow-up round. Detect building geometry errors and fix them.
Previous operations have already been applied, so compare original-vs-current image pairs and find remaining clear issues. BRIGHT YELLOW areas are roads.

{image_reference}

## Coordinate Reference
- Rotation: clockwise = positive (+), counter-clockwise = negative (-), in degrees.
- Distance: use the visible white 1unit scale bar as reference.
- Direction: use north/south/east/west according to the red N arrow.

## Error Types
1. Overlap: physical intersection or z-fighting between building volumes.
2. RoadViolation: building footprint intersects BRIGHT YELLOW roads.
   - Roads should run alongside buildings, not through interiors/courtyards.
   - At most one short dead-end access road may connect to a building.
3. AlignmentError: building axes are not parallel/perpendicular to the road grid (not near 0°/90° relative to road axis), OR clearly inconsistent with nearby parallel buildings that should share orientation.

## Your Analysis Steps
1. Compare original and current images by view type.
2. For every numbered building, check all three error types independently. Note that one building can have multiple issues.
3. For each previously detected and attempted error type:
   - FIXED: this specific error type is now clearly resolved.
   - STILL_HAS_ISSUE: this specific error type still exists or needs further adjustment.
   - WORSE: this specific error type became worse after previous action.
4. Also report newly visible issue types that were missed before. For newly discovered error types, set issue_status to NEW.
5. Please make only small adjustments to resolve issues, and do not move any object outside the current view.

## Output Requirements
- Output only a JSON array.
- One item = one building + one error type.

Each item:
- id: integer building id
- issue_status: "FIXED" | "STILL_HAS_ISSUE" | "WORSE" | "NEW"
- error_type: "Overlap" | "RoadViolation" | "AlignmentError"
- reason: short status explanation for this error type
- action: incremental action list from current state; each command must be `Move(direction, distance)` (North/South/East/West, and diagonals (Northeast/Northwest/Southeast/Southwest)), `Rotate(angle)`, or `Scale(factor)`

## Example
```json
[
  {{
    "id": 6,
    "issue_status": "FIXED",
    "error_type": "AlignmentError",
    "reason": "Building 6 is now aligned with nearby roads and parallel neighbors",
    "action": []
  }},
  {{
    "id": 6,
    "issue_status": "STILL_HAS_ISSUE",
    "error_type": "RoadViolation",
    "reason": "Building 6 still intersects the bottom yellow road",
    "action": ["Move(North, 0.5)"]
  }},
  {{
    "id": 12,
    "issue_status": "NEW",
    "error_type": "Overlap",
    "reason": "After prior edits, building 12 now appears oversized and overlaps building 14",
    "action": ["Scale(0.9)"]
  }}
]
```

## Previous Iteration Actions (Just Applied)
{previous_actions_summary}

## Cumulative Adjustments Summary
{cumulative_summary}

Only use ids from this allowed set: {valid_ids}
If all issues are fixed or no issue exists, return EXACTLY [] (no extra text).
"""


def _build_first_round_prompt_indoor(valid_ids: List[int], has_isometric_ref: bool, labeled_objects_hint: str = "") -> str:
    if has_isometric_ref:
        wall_hint = (
            "- In top-view image: the white outer boundary represents walls.\n"
            "- In isometric image: the translucent/transparent outer shell represents walls."
        )
        
    else:
        wall_hint = "- Top-view image: the white outer boundary represents walls."

    return f"""
# Task Description
Analyze indoor images to detect clear geometry issues and propose conservative fixes.

## Cautious Decision Rules
- Only report visually obvious issues.
- Prefer no change over risky modification.
- Please make only small adjustments to resolve issues, and do not move any object outside the current view.

## Wall Rendering Hint
{wall_hint}

## Coordinate Reference
- Distance: use unit, with the visible white 1unit scale bar as reference.
- Direction: use North / South / East / West according to the red N arrow in each image.

## Error Types 
Check independently for each labeled object:
1. Overlap: physical intersection between object volumes/footprints.
2. WallConflict: object footprint collides with room walls (intersects/penetrates wall boundary), or extends outside wall bounds.
3. Orientation: for rectangular/non-round furniture, major axis is clearly misaligned with room axis (not near 0°/90°).

## Important Rules
- Detect these three error_type values: "Overlap", "WallConflict", "Orientation".
- One object may have multiple error types; output one entry per object per error type.
- You may propose actions ONLY for labeled IDs in the allowed set.
- Error validity is global: judge geometry on the WHOLE scene (including unlabeled objects).
- Do NOT fix labeled objects by creating new overlap/wall/orientation issues on unlabeled objects.

## Output Requirements
- Output ONLY a JSON array.
- No extra prose outside JSON.

Each item:
- id: integer object label
- error_type: "Overlap" | "WallConflict" | "Orientation"
- reason: short issue explanation
- action: list of incremental commands from Move/Rotate/Scale

## Action Command Semantics
- Move(direction, distance): move in the current image compass frame. Direction supports North/South/East/West and diagonals (Northeast/Northwest/Southeast/Southwest). Use positive distance.
- Rotate(angle): rotate around object center on Z axis. Positive = clockwise, negative = counter-clockwise.
- Scale(factor): uniform XYZ scaling around object center. Use positive factor (e.g., 0.8 shrink, 1.1 enlarge).

## Example
```json
[
  {{
    "id": 6,
    "error_type": "Orientation",
    "reason": "Object 6 is clearly tilted relative to room axis",
    "action": ["Rotate(-20)"]
  }},
  {{
    "id": 2,
    "error_type": "Overlap",
    "reason": "Object 2 intersects object 5",
    "action": ["Move(West, 0.4)"]
  }},
  {{
    "id": 3,
    "error_type": "WallConflict",
    "reason": "Object 3 (sofa) looks oversized and extends beyond the wall boundary",
    "action": ["Scale(0.7)"]
  }}
]
```

Only use ids from this allowed set: {valid_ids}
{labeled_objects_hint if labeled_objects_hint else ""}
If no issue, return EXACTLY [] (no extra text).
"""


def _build_followup_prompt_indoor(
    valid_ids: List[int],
    previous_actions_summary: str,
    cumulative_summary: str,
    has_isometric_ref: bool,
    labeled_objects_hint: str = "",
) -> str:
    if has_isometric_ref:
        image_reference = (
            "## Image Reference\n"
            "- Top-view pair: Image 1 (ORIGINAL top-down) vs Image 3 (CURRENT top-down after previous actions)\n"
            "- Isometric pair: Image 2 (ORIGINAL isometric) vs Image 4 (CURRENT isometric after previous actions)"
        )
        wall_hint = (
            "- In top-view image: the white outer boundary represents walls.\n"
            "- In isometric image: the translucent/transparent outer shell represents walls."
        )
      
    else:
        image_reference = (
            "## Image Reference\n"
            "- Top-view pair: Image 1 (ORIGINAL top-down) vs Image 2 (CURRENT top-down after previous actions)"
        )
        wall_hint = "- Top-view image: the white outer boundary represents walls."

    return f"""# Task Description
You are now in a follow-up round for indoor geometry correction.
Previous operations have already been applied, so compare original-vs-current image pairs and identify remaining clear issues.

{image_reference}

## Wall Rendering Hint
{wall_hint}

## Coordinate Reference
- Distance: use unit, with the visible white 1unit scale bar in the image as reference.
- Direction: use North / South / East / West according to the red N arrow in each image.

## Error Types (Indoor Only)
Check independently for each labeled object:
1. Overlap: physical intersection between object volumes/footprints.
2. WallConflict: object collides with room walls (intersects/penetrates wall boundary), or extends outside wall bounds.
3. Orientation: for rectangular/non-round furniture, major axis is clearly misaligned with room axis (not near 0°/90°).

## Your Analysis Steps
1. Compare each original/current image pair by view type.
2. For every labeled object, check all three error types independently.
3. For each previously detected and attempted error type:
   - FIXED: this specific error type is now clearly resolved.
   - STILL_HAS_ISSUE: this specific error type still exists or needs further adjustment.
   - WORSE: this specific error type became worse after previous action.
4. Also report newly visible issue types that were missed before. For newly discovered error types, set issue_status to NEW.

## Important Rules
- Detect these three error_type values: "Overlap", "WallConflict", "Orientation".
- One object may have multiple error types; output one entry per object per error type.
- You may propose actions ONLY for labeled IDs in the allowed set.
- Error validity is global: judge geometry on the WHOLE scene (including unlabeled objects).
- Never resolve a labeled-object issue by creating new overlap/wall/orientation errors on unlabeled objects.

## Output Requirements
- Return ONLY a JSON array.
- One object = one object id + one error type.

Each item:
- id: integer object label
- issue_status: "FIXED" | "STILL_HAS_ISSUE" | "WORSE" | "NEW"
- error_type: "Overlap" | "WallConflict" | "Orientation"
- reason: concise status explanation for this error type
- action: incremental fix actions from current state (Move/Rotate/Scale)

## Action Command Semantics
- Move(direction, distance): move in the current image compass frame. Direction supports North/South/East/West and diagonals (Northeast/Northwest/Southeast/Southwest). Use positive distance.
- Rotate(angle): rotate around object center on Z axis. Positive = clockwise, negative = counter-clockwise.
- Scale(factor): uniform XYZ scaling around object center. Use positive factor (e.g., 0.8 shrink, 1.1 enlarge).

## Example
```json
[
  {{
    "id": 4,
    "issue_status": "FIXED",
    "error_type": "WallConflict",
    "reason": "Object 4 is now fully inside the room boundary",
    "action": []
  }},
  {{
    "id": 9,
    "issue_status": "STILL_HAS_ISSUE",
    "error_type": "Overlap",
    "reason": "Object 9 still intersects object 11",
    "action": ["Move(South, 0.3)"]
  }},
  {{
    "id": 12,
    "issue_status": "NEW",
    "error_type": "Orientation",
    "reason": "Object 12 now appears clearly tilted relative to room axis",
    "action": ["Rotate(15)"]
  }}
]
```

## Previous Iteration Actions (JUST APPLIED)
{previous_actions_summary}

## Cumulative Adjustments Summary (Active Objects Only)
{cumulative_summary}

Only use ids from this allowed set: {valid_ids}
{labeled_objects_hint if labeled_objects_hint else ""}
If all issues are fixed or no issue exists, return EXACTLY [] (no extra text).
"""


def _call_mllm(
    image_paths: List[str],
    llm_runtime: Dict[str, Any],
    image_url_convert_endpoint: str,
    image_url_convert_timeout: int,
    image_url_expires_in: int,
    image_url_public_base_url: str,
    scene_type: str,
    valid_ids: List[int],
    iteration: int,
    previous_analysis: List[Dict],
    cumulative_actions: List[Dict],
    unit_scale: float,
    has_isometric_ref: bool,
    labeled_objects_hint: str = "",
    timeout: int = 120,
    max_retries: int = 2,
    max_completion_tokens: int = 8096,
) -> Dict[str, str]:
    provider = llm_runtime.get("provider", "openrouter")
    model = llm_runtime.get("model", "")
    client = _create_llm_client(llm_runtime=llm_runtime, timeout=timeout)

    previous_actions_summary = _build_previous_actions_summary(previous_analysis)
    cumulative_summary = _build_cumulative_summary(cumulative_actions, unit_scale=unit_scale)
    if scene_type == "indoor":
        if iteration <= 1:
            system_prompt = _build_first_round_prompt_indoor(
                valid_ids,
                has_isometric_ref,
                labeled_objects_hint=labeled_objects_hint,
            )
        else:
            system_prompt = _build_followup_prompt_indoor(
                valid_ids,
                previous_actions_summary,
                cumulative_summary,
                has_isometric_ref,
                labeled_objects_hint=labeled_objects_hint,
            )
    else:
        if iteration <= 1:
            system_prompt = _build_first_round_prompt_region(valid_ids)
        else:
            system_prompt = _build_followup_prompt_region(
                valid_ids,
                previous_actions_summary,
                cumulative_summary,
                has_isometric_ref,
            )

    image_role_texts: List[str] = []
    if len(image_paths) == 1:
        image_role_texts = [
            "Image 1 = current baseline top-down view.",
        ]
    elif len(image_paths) == 2:
        if iteration == 1 and has_isometric_ref:
            image_role_texts = [
                "Image 1 = current baseline top-down view.",
                "Image 2 = current baseline isometric view.",
            ]
        else:
            image_role_texts = [
                "Image 1 = ORIGINAL baseline top-down view (before this iteration action).",
                "Image 2 = LAST ITERATION result top-down view.",
            ]
    elif len(image_paths) == 4:
        image_role_texts = [
            "Image 1 = ORIGINAL baseline top-down view (before this iteration action).",
            "Image 2 = ORIGINAL baseline isometric view (before this iteration action).",
            "Image 3 = LAST ITERATION result top-down view.",
            "Image 4 = LAST ITERATION result isometric view.",
        ]
    else:
        image_role_texts = [f"Image {i+1}" for i in range(len(image_paths))]

    contents = []
    input_image_paths: List[str] = []
    for idx, path in enumerate(image_paths, start=1):
        normalized_path = os.path.abspath(path)
        image_base64 = _encode_image(normalized_path)
        data_url = f"data:image/png;base64,{image_base64}"
        input_image_paths.append(normalized_path)
        role_text = image_role_texts[idx - 1] if idx - 1 < len(image_role_texts) else f"Image {idx}"
        contents.append({"type": "text", "text": role_text})
        contents.append({"type": "image_url", "image_url": {"url": data_url, "detail": "high"}})

    user_prompt_text = "Start Analyze"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": contents + [{"type": "text", "text": user_prompt_text}]},
    ]
    gpt_reasoning = llm_runtime.get("gpt_reasoning")
    gpt_use_responses_api = bool(llm_runtime.get("gpt_use_responses_api", True))

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            if provider == "gpt" and gpt_use_responses_api:
                req: Dict[str, Any] = {
                    "model": model,
                    "input": _chat_messages_to_responses_input(messages),
                    "max_output_tokens": max(1, int(max_completion_tokens)),
                }
                if isinstance(gpt_reasoning, dict) and gpt_reasoning:
                    req["reasoning"] = gpt_reasoning
                response = client.responses.create(**req)
                response_reasoning = _extract_responses_reasoning(response)
                raw_text = _extract_responses_text(response)
                return {
                    "raw_response": raw_text,
                    "system_prompt": system_prompt,
                    "user_prompt_text": user_prompt_text,
                    "token_usage": _normalize_token_usage(getattr(response, "usage", None)),
                    "reasoning": _flatten_reasoning_text(response_reasoning),
                    #  base64
                    "input_image_urls": input_image_paths,
                    "input_image_paths": input_image_paths,
                }

            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": 0.1,
                "max_completion_tokens": max(1, int(max_completion_tokens)),
            }
            if provider == "gpt":
                effort = None
                if isinstance(gpt_reasoning, dict):
                    effort = str(gpt_reasoning.get("effort", "") or "").strip().lower()
                if effort:
                    kwargs["reasoning_effort"] = effort
            else:
                kwargs["extra_body"] = {
                    "reasoning": {"enabled": True},
                    "provider": {
                        "only": [
                            "deepinfra/fp4",
                            "io-net/int4",
                            "parasail/int4",
                            "inceptron/int4",
                            "z-ai/fp8",
                            "deepinfra/fp8",
                            "atlas-cloud/fp8",
                            "google-ai-studio",
                            "google-vertex",
                        ]
                    },
                }
            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0] if getattr(response, "choices", None) else None
            message = getattr(choice, "message", None) if choice is not None else None
            reasoning = (
                getattr(message, "reasoning", None)
                or getattr(message, "reasoning_content", None)
                or getattr(message, "reasoning_details", None)
                or getattr(choice, "reasoning", None)
            )
            return {
                "raw_response": (getattr(message, "content", None) or ""),
                "system_prompt": system_prompt,
                "user_prompt_text": user_prompt_text,
                "token_usage": _normalize_token_usage(getattr(response, "usage", None)),
                "reasoning": _flatten_reasoning_text(reasoning),
                #  base64
                "input_image_urls": input_image_paths,
                "input_image_paths": input_image_paths,
            }
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            wait_seconds = min(8, 2 ** attempt)
            print(f"[WARN] MLLM  ({attempt + 1}/{max_retries}): {exc}")
            time.sleep(wait_seconds)

    raise RuntimeError(f"MLLM  {max_retries} : {last_error}")


def _run_blender_render(
    region_dir: str,
    output_dir: str,
    cumulative_path: str,
    input_blend: str,
    output_top: str,
    output_iso: str,
    output_blend: Optional[str],
    gltf_path: str,
    unit_scale: float,
    scene_type: str,
    region_mode: str,
    action_log_path: Optional[str] = None,
):
    env = os.environ.copy()
    env["BLENDER_REGION_DIR"] = region_dir
    env["BLENDER_OUTPUT_DIR"] = output_dir
    env["BLENDER_CUMULATIVE_PATH"] = cumulative_path
    env["BLENDER_INPUT_BLEND"] = input_blend or ""
    env["BLENDER_OUTPUT_TOP"] = output_top
    env["BLENDER_OUTPUT_ISO"] = output_iso
    env["GLTF_PATH"] = gltf_path
    env["BLENDER_UNIT_SCALE"] = str(unit_scale)
    env["BLENDER_SCENE_TYPE"] = str(scene_type or "region")
    env["BLENDER_REGION_MODE"] = str(region_mode or "")
    if output_blend:
        env["BLENDER_OUTPUT_BLEND"] = output_blend
    if action_log_path:
        env["BLENDER_ACTION_LOG_PATH"] = action_log_path

    cmd = [BLENDER, "--background", "--python", BLENDER_SCRIPT]
    expected_top = os.path.join(output_dir, output_top)
    expected_iso = os.path.join(output_dir, output_iso)
    max_attempts = 2
    last_res = None
    for attempt in range(1, max_attempts + 1):
        res = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT, capture_output=True, text=True)
        last_res = res
        if res.returncode != 0:
            if attempt < max_attempts:
                print(f"[WARN] Blender  ({attempt}/{max_attempts})", flush=True)
                time.sleep(0.8)
                continue
            raise RuntimeError(f"Blender : {res.stderr[-1200:]}")

        missing_outputs: List[str] = []
        if not os.path.isfile(expected_top):
            missing_outputs.append(expected_top)
        if not os.path.isfile(expected_iso):
            missing_outputs.append(expected_iso)
        if missing_outputs:
            if attempt < max_attempts:
                print(
                    f"[WARN] Blender  ({attempt}/{max_attempts}): {missing_outputs}",
                    flush=True,
                )
                time.sleep(0.8)
                continue
            out_files = []
            try:
                out_files = sorted(os.listdir(output_dir))
            except Exception:
                out_files = []
            raise RuntimeError(
                "Blender "
                f" missing={missing_outputs}, output_dir={output_dir}, "
                f"files={out_files[:40]}, "
                f"stdout_tail={(res.stdout or '')[-1200:]}, stderr_tail={(res.stderr or '')[-1200:]}"
            )
        return

    if last_res is not None and last_res.returncode != 0:
        raise RuntimeError(f"Blender : {(last_res.stderr or '')[-1200:]}")


def _run_blender_render_limited(**kwargs):
    sem = _BLENDER_RENDER_SEMAPHORE
    if sem is None:
        return _run_blender_render(**kwargs)
    with sem:
        return _run_blender_render(**kwargs)


def _run_metrics_for_region(
    scene_type: str,
    region_dir: str,
    steps: str,
    input_root: str,
    after_blend_path: str = "",
) -> Dict[str, object]:
    if not os.path.isfile(METRICS_SCRIPT):
        return {
            "status": "skipped",
            "reason": f"metrics script missing: {METRICS_SCRIPT}",
            "output_path": None,
        }

    step_suffix = str(steps or "").strip()
    output_name = f"metrics_region_blend_steps{step_suffix}.json" if step_suffix else "metrics_region_blend.json"
    output_path = os.path.join(region_dir, output_name)

    if scene_type == "region":
        cmd = [
            sys.executable,
            METRICS_SCRIPT,
            "--mode", "region",
            "--region-dir", region_dir,
            "--input-root", input_root,
            "--output", output_path,
        ]
        if step_suffix:
            cmd += ["--steps", step_suffix]
        if after_blend_path:
            cmd += ["--after-blend-path", str(after_blend_path)]
    else:
        final_scene_path = _pick_first_existing(
            [
                os.path.join(region_dir, f"final_scene_steps{step_suffix}.blend") if step_suffix else "",
                os.path.join(region_dir, f"final_scene_steps{step_suffix}.glb") if step_suffix else "",
                os.path.join(region_dir, "final_scene.blend"),
                os.path.join(region_dir, "final_scene.glb"),
            ]
        )
        if not final_scene_path:
            return {
                "status": "skipped",
                "reason": f"final scene missing under {region_dir}",
                "output_path": output_path,
            }
        cmd = [
            sys.executable,
            METRICS_SCRIPT,
            "--mode", "indoor",
            "--blend-path", final_scene_path,
            "--scene-dir", region_dir,
            "--output", output_path,
        ]
        if step_suffix:
            cmd += ["--steps", step_suffix]

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return {
            "status": "failed",
            "output_path": output_path,
            "elapsed_sec": elapsed,
            "cmd": cmd,
            "stdout_tail": proc.stdout[-1200:],
            "stderr_tail": proc.stderr[-1200:],
        }
    return {
        "status": "ok",
        "output_path": output_path,
        "elapsed_sec": elapsed,
        "cmd": cmd,
    }


def _stable_key_for_path(path_value: str) -> str:
    p = os.path.abspath(str(path_value or "").strip())
    return os.path.normcase(p)


def _consolidate_generic(cumulative: Dict[str, Dict]) -> List[Dict]:
    def _fmt_num(value: float, precision: int = 3) -> str:
        text = f"{float(value):.{precision}f}"
        text = text.rstrip("0").rstrip(".")
        return text if text else "0"

    def _key(k: str) -> Tuple[int, str]:
        return (0, f"{int(k):09d}") if str(k).isdigit() else (1, str(k))

    out: List[Dict] = []
    for label in sorted(cumulative.keys(), key=_key):
        record = cumulative[label]
        moves = record["Move"]
        net_ns = moves.get("Up", 0.0) - moves.get("Down", 0.0)
        net_ew = moves.get("Right", 0.0) - moves.get("Left", 0.0)
        actions: List[str] = []
        if net_ns > 0:
            actions.append(f"Move(North, {_fmt_num(net_ns)})")
        elif net_ns < 0:
            actions.append(f"Move(South, {_fmt_num(-net_ns)})")
        if net_ew > 0:
            actions.append(f"Move(East, {_fmt_num(net_ew)})")
        elif net_ew < 0:
            actions.append(f"Move(West, {_fmt_num(-net_ew)})")

        rot = float(record.get("Rotate", 0.0) or 0.0)
        if abs(rot) > 1e-6:
            actions.append(f"Rotate({_fmt_num(rot)})")

        scale = record.get("Scale")
        if scale is not None:
            actions.append(f"Scale({_fmt_num(float(scale), precision=4)})")

        if actions:
            out.append({"id": int(label) if str(label).isdigit() else label, "action": actions})
    return out


def _merge_actions_into_cumulative(cumulative: Dict[str, Dict], key: str, actions: List[str]) -> None:
    if key not in cumulative:
        cumulative[key] = {
            "Move": {"Up": 0.0, "Down": 0.0, "Left": 0.0, "Right": 0.0},
            "Rotate": 0.0,
            "Scale": None,
        }
    for text in _normalize_action_list(actions):
        parsed = _parse_action_line(text)
        if not parsed:
            continue
        if parsed[0] == "Move":
            _, direction, distance = parsed
            comps = _direction_to_move_components(direction)
            if comps:
                for move_key, coeff in comps.items():
                    cumulative[key]["Move"][move_key] += float(distance) * float(coeff)
        elif parsed[0] == "Rotate":
            _, angle = parsed
            cumulative[key]["Rotate"] += float(angle)
        elif parsed[0] == "Scale":
            _, factor = parsed
            cumulative[key]["Scale"] = _accumulate_scale(cumulative[key]["Scale"], float(factor))


def _build_shared_scene_actions(
    region_ids: List[int],
    prepared: Dict[int, str],
) -> Tuple[Dict[str, str], Dict[str, Dict], List[str]]:
    object_cumulative: Dict[str, Dict] = {}
    warnings: List[str] = []

    for rid in region_ids:
        region_dir = prepared[rid]
        labels_path = os.path.join(region_dir, "labels.json")
        actions_path = os.path.join(region_dir, "final_actions.json")
        if not os.path.isfile(labels_path) or not os.path.isfile(actions_path):
            warnings.append(f"region_{rid}: missing labels/actions for shared merge")
            continue
        labels_map = _load_json(labels_path) or {}
        final_actions = _load_json(actions_path) or {}
        consolidated = final_actions.get("consolidated_actions", []) or []
        for item in consolidated:
            local_id = str(item.get("id", "")).strip()
            if not local_id:
                continue
            target_obj_id = str(labels_map.get(local_id, "")).strip()
            if not target_obj_id:
                warnings.append(f"region_{rid}: label {local_id} missing object mapping")
                continue
            _merge_actions_into_cumulative(
                object_cumulative,
                key=target_obj_id,
                actions=_normalize_action_list(item.get("action", [])),
            )

    if not object_cumulative:
        return {}, {}, warnings

    object_ids = sorted(object_cumulative.keys())
    synthetic_labels: Dict[str, str] = {}
    synthetic_cumulative: Dict[str, Dict] = {}
    for idx, obj_id in enumerate(object_ids, start=1):
        k = str(idx)
        synthetic_labels[k] = obj_id
        synthetic_cumulative[k] = object_cumulative[obj_id]
    return synthetic_labels, synthetic_cumulative, warnings


def _safe_group_slug(scene_path: str, index: int) -> str:
    stem = os.path.splitext(os.path.basename(str(scene_path or "")))[0]
    slug = _safe_fs_name(stem) or f"group_{index}"
    return f"shared_scene_{slug}_{index}"


def _scan_resume_state(region_dir: str, max_iterations: int, has_isometric_ref: bool) -> Dict[str, object]:
    """
    
    - should_run=False:  region  max_iterations 
    - should_run=True:  start_iteration /
    """
    start_iteration = 1
    stop_reason = "fresh_start"
    converged = False

    for iteration in range(1, max_iterations + 1):
        iter_dir = os.path.join(region_dir, f"iter_{iteration}")
        analysis_path = os.path.join(iter_dir, "mllm_analysis.json")
        if not os.path.isfile(analysis_path):
            start_iteration = iteration
            stop_reason = "missing_analysis"
            break

        try:
            analysis_payload = _load_json(analysis_path) or {}
        except Exception:
            start_iteration = iteration
            stop_reason = "invalid_analysis_json"
            break

        llm_call_failed = bool(analysis_payload.get("llm_call_failed", False))
        actionable = analysis_payload.get("actionable_analysis", []) or []
        if not isinstance(actionable, list):
            actionable = []

        if llm_call_failed:
            start_iteration = iteration
            stop_reason = "llm_call_failed"
            break

        if len(actionable) == 0:
            converged = True
            start_iteration = iteration + 1
            stop_reason = "no_actionable_issues"
            break

        top_path = os.path.join(iter_dir, "top.png")
        iso_path = os.path.join(iter_dir, "isometric.png")
        render_state_path = os.path.join(iter_dir, "cumulative_actions_render.json")
        if not os.path.isfile(top_path):
            start_iteration = iteration
            stop_reason = "missing_iter_top_render"
            break
        if has_isometric_ref and (not os.path.isfile(iso_path)):
            start_iteration = iteration
            stop_reason = "missing_iter_isometric_render"
            break
        if not os.path.isfile(render_state_path):
            start_iteration = iteration
            stop_reason = "missing_iter_cumulative_state"
            break
    else:
        start_iteration = max_iterations + 1
        stop_reason = "reached_max_iterations"

    should_run = (start_iteration <= max_iterations) and (not converged)
    return {
        "should_run": should_run,
        "start_iteration": int(start_iteration),
        "stop_reason": stop_reason,
        "converged": bool(converged),
    }


def _load_resume_prefix_state(
    region_dir: str,
    start_iteration: int,
) -> Tuple[Dict, List[Dict], List[Dict], Dict[str, int]]:
    """
    
    - cumulative: start_iteration 
    - previous_analysis:  actionable_analysis
    - history_prefix: iteration < start_iteration
    - token_usage_prefix:  token
    """
    cumulative: Dict = {}
    cumulative_path = os.path.join(region_dir, "cumulative_actions.json")
    if os.path.isfile(cumulative_path):
        try:
            cumulative = (_load_json(cumulative_path) or {}).get("cumulative_actions", {}) or {}
        except Exception:
            cumulative = {}

    previous_analysis: List[Dict] = []
    if start_iteration > 1:
        prev_iter = start_iteration - 1
        prev_analysis_path = os.path.join(region_dir, f"iter_{prev_iter}", "mllm_analysis.json")
        if os.path.isfile(prev_analysis_path):
            try:
                prev_payload = _load_json(prev_analysis_path) or {}
                prev_analysis = prev_payload.get("actionable_analysis", []) or []
                if not isinstance(prev_analysis, list):
                    prev_analysis = []
            except Exception:
                previous_analysis = []

        prev_cumulative_path = os.path.join(region_dir, f"iter_{prev_iter}", "cumulative_actions_render.json")
        if os.path.isfile(prev_cumulative_path):
            try:
                prev_cumulative_payload = _load_json(prev_cumulative_path) or {}
                prev_cumulative = prev_cumulative_payload.get("cumulative_actions", {}) or {}
                if isinstance(prev_cumulative, dict):
                    cumulative = prev_cumulative
            except Exception:
                pass

    history_prefix: List[Dict] = []
    history_path = os.path.join(region_dir, "iteration_history.json")
    if os.path.isfile(history_path):
        try:
            history_payload = _load_json(history_path) or []
            if isinstance(history_payload, list):
                for row in history_payload:
                    if not isinstance(row, dict):
                        continue
                    if _to_int(row.get("iteration")) < start_iteration:
                        history_prefix.append(row)
        except Exception:
            history_prefix = []

    token_usage_prefix = {
        "request_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    if history_prefix:
        last = history_prefix[-1]
        cumulative_tokens = last.get("token_usage_cumulative", {}) if isinstance(last, dict) else {}
        if isinstance(cumulative_tokens, dict):
            token_usage_prefix["request_count"] = _to_int(cumulative_tokens.get("request_count"))
            token_usage_prefix["prompt_tokens"] = _to_int(cumulative_tokens.get("prompt_tokens"))
            token_usage_prefix["completion_tokens"] = _to_int(cumulative_tokens.get("completion_tokens"))
            token_usage_prefix["total_tokens"] = _to_int(cumulative_tokens.get("total_tokens"))

    return cumulative, previous_analysis, history_prefix, token_usage_prefix


def _truncate_interaction_log(path: str, keep_before_iteration: int) -> None:
    if not os.path.isfile(path):
        return
    kept_lines: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if _to_int(payload.get("iteration")) < keep_before_iteration:
                kept_lines.append(json.dumps(payload, ensure_ascii=False))
    with open(path, "w", encoding="utf-8") as handle:
        for line in kept_lines:
            handle.write(line + "\n")


def _cleanup_iteration_artifacts_for_retry(iter_dir: str) -> None:
    for filename in (
        "mllm_analysis.json",
        "cumulative_actions_render.json",
        "applied_actions_log.json",
        "top.png",
        "isometric.png",
    ):
        path = os.path.join(iter_dir, filename)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except Exception:
                pass


def _prepare_region_workspace(
    sample: RegionSample,
    region_data: Dict,
    workspace_root: str,
    scene_type: str,
    region_mode: str,
    use_error_scene: bool,
    unit_scale: float,
    shared_error_scene_path: str,
    gltf_path: str,
    error_steps_tag: str,
    resume_incomplete: bool = False,
) -> str:
    if scene_type == "indoor":
        stem = os.path.splitext(os.path.basename(str(sample.glb_name or "")))[0].strip()
        preferred_name = sample.scene_name.strip() or stem or f"scene_{sample.region_id}"
        source_name = _safe_fs_name(preferred_name)
        if source_name:
            region_dir_name = f"{source_name}__rid_{sample.region_id}"
        else:
            region_dir_name = f"scene_{sample.region_id}"
    else:
        # sampled region 
        source_name = _safe_fs_name(sample.scene_name.strip()) if sample.scene_name else ""
        if source_name:
            region_dir_name = f"{source_name}__rid_{sample.region_id}"
        else:
            region_dir_name = f"region_{sample.region_id}"
    region_dir = os.path.join(workspace_root, region_dir_name)
    os.makedirs(region_dir, exist_ok=True)

    labels = sample.labels_map or _labels_from_region_data(region_data, sample.region_id)
    if not labels:
        raise RuntimeError(f"region_{sample.region_id}  region_data  building_ids")

    top_err = sample.error_images[0] if len(sample.error_images) > 0 else ""
    iso_err = sample.error_images[1] if len(sample.error_images) > 1 else ""
    top_clean = sample.initial_images[0] if len(sample.initial_images) > 0 else ""
    iso_clean = sample.initial_images[1] if len(sample.initial_images) > 1 else ""
    source_scene_path = str(sample.source_scene_path or "").strip()

    if use_error_scene:
        top_input_path = os.path.join(region_dir, "top_input_qa.png")
        iso_input_path = os.path.join(region_dir, "isometric_input_qa.png")
        if not _copy(top_err, top_input_path):
            raise RuntimeError(f"region_{sample.region_id}  top ")
        if not _copy(iso_err, iso_input_path):
            raise RuntimeError(f"region_{sample.region_id}  isometric ")
    else:
        top_copied = _copy(top_err, os.path.join(region_dir, "top.png"))
        iso_copied = _copy(iso_err, os.path.join(region_dir, "isometric.png"))
        if (not top_copied) and (not source_scene_path):
            raise RuntimeError(f"region_{sample.region_id}  top ")
        if _sample_prefers_isometric(sample) and (not iso_copied) and (not source_scene_path):
            raise RuntimeError(f"region_{sample.region_id} images  isometric")

    _copy(top_clean, os.path.join(region_dir, "top_clean.png"))
    _copy(iso_clean, os.path.join(region_dir, "isometric_clean.png"))

    _write_json(os.path.join(region_dir, "labels.json"), labels)
    effective_error_scene = shared_error_scene_path if use_error_scene else ""

    _write_json(
        os.path.join(region_dir, "region_info.json"),
        {
            "region_id": sample.region_id,
            "region_dir_name": region_dir_name,
            "scene_name": sample.scene_name,
            "glb_name": sample.glb_name,
            "building_ids": list(labels.values()),
            "source": {
                "source_step": int(sample.source_step or 0),
                "source_sample_id": str(sample.source_sample_id or ""),
                "source_sample_index": int(sample.source_sample_index),
                "initial_images": sample.initial_images,
                "error_images": sample.error_images,
                "error_blend": effective_error_scene or None,
                "error_scene": effective_error_scene or None,
                "source_scene_path": source_scene_path or None,
            },
        },
    )

    if use_error_scene and not os.path.exists(effective_error_scene):
        raise RuntimeError(
            f"region_{sample.region_id}  error scene "
            f": {effective_error_scene or '<empty>'}"
        )
    if source_scene_path and not os.path.exists(source_scene_path):
        raise RuntimeError(f"region_{sample.region_id} source_scene_path : {source_scene_path}")
    if (scene_type == "region") and (not use_error_scene) and (not source_scene_path):
        raise RuntimeError(
            f"region_{sample.region_id}  source_scene_path"
            " multi_step_error  *_global_error_scene.glb/.blend"
            " step  sampled "
        )

    # indoor  error_scene.glb seed anomalies
    seed = {} if (use_error_scene or source_scene_path) else _seed_from_anomalies(sample.anomalies, unit_scale)
    cumulative_actions_path = os.path.join(region_dir, "cumulative_actions.json")
    if not (resume_incomplete and os.path.isfile(cumulative_actions_path)):
        _write_json(cumulative_actions_path, {"cumulative_actions": seed})
    _write_json(os.path.join(region_dir, "seed_anomalies.json"), sample.anomalies)

    if use_error_scene:
        input_blend = effective_error_scene if effective_error_scene.lower().endswith(".blend") else ""
        input_gltf = effective_error_scene if (effective_error_scene and (not effective_error_scene.lower().endswith(".blend"))) else gltf_path
        base_render_payload = os.path.join(region_dir, "base_render_actions.json")
        _write_json(base_render_payload, {"consolidated_actions": []})
        _run_blender_render_limited(
            region_dir=region_dir,
            output_dir=region_dir,
            cumulative_path=base_render_payload,
            input_blend=input_blend,
            output_top="top.png",
            output_iso="isometric.png",
            output_blend=None,
            gltf_path=input_gltf,
            unit_scale=unit_scale,
            scene_type=scene_type,
            region_mode=region_mode,
            action_log_path=os.path.join(region_dir, "base_applied_actions_log.json"),
        )
    elif source_scene_path:
        #  source_scene_path 
        top_png = os.path.join(region_dir, "top.png")
        iso_png = os.path.join(region_dir, "isometric.png")
        if (not os.path.exists(top_png)) or (not os.path.exists(iso_png)):
            base_render_payload = os.path.join(region_dir, "base_render_actions.json")
            _write_json(base_render_payload, {"consolidated_actions": []})
            _run_blender_render_limited(
                region_dir=region_dir,
                output_dir=region_dir,
                cumulative_path=base_render_payload,
                input_blend=source_scene_path if source_scene_path.lower().endswith(".blend") else "",
                output_top="top.png",
                output_iso="isometric.png",
                output_blend=None,
                gltf_path=(source_scene_path if (not source_scene_path.lower().endswith(".blend")) else gltf_path),
                unit_scale=unit_scale,
                scene_type=scene_type,
                region_mode=region_mode,
                action_log_path=os.path.join(region_dir, "base_applied_actions_log.json"),
            )

    return region_dir


def _iterate_region(
    region_id: int,
    region_tag: str,
    region_dir: str,
    scene_type: str,
    region_mode: str,
    llm_runtime: Dict[str, Any],
    max_iterations: int,
    gltf_path: str,
    use_error_scene: bool,
    unit_scale: float,
    llm_timeout: int,
    llm_max_retries: int,
    llm_max_completion_tokens: int,
    image_url_convert_endpoint: str,
    image_url_convert_timeout: int,
    image_url_expires_in: int,
    image_url_public_base_url: str,
    has_isometric_ref: bool,
    labeled_objects_hint: str,
    error_steps_tag: str,
    shared_error_scene_path: str,
    source_scene_path: str = "",
    run_metrics_per_region: bool = False,
    metrics_input_root: str = "",
    metrics_steps_value: str = "",
    cleanup_final_blend_after_metrics: bool = False,
    save_region_final_blend: bool = True,
    resume_incomplete: bool = False,
):
    cumulative_path = os.path.join(region_dir, "cumulative_actions.json")
    interaction_log_path = os.path.join(region_dir, "llm_interactions.jsonl")
    token_usage_summary_path = os.path.join(region_dir, "token_usage_summary.json")
    display_name = str(region_tag or f"region_{region_id}")
    labels_map = _load_json(os.path.join(region_dir, "labels.json"))
    valid_ids = sorted(int(k) for k in labels_map.keys() if str(k).isdigit())
    valid_id_set = set(valid_ids)

    final_blend_filename = f"final_scene_{error_steps_tag}.blend" if error_steps_tag else "final_scene.blend"
    source_scene_path = str(source_scene_path or "").strip()
    if source_scene_path:
        if source_scene_path.lower().endswith(".blend"):
            base_blend = source_scene_path
            base_gltf_path = gltf_path
        else:
            base_blend = ""
            base_gltf_path = source_scene_path
    else:
        if use_error_scene and shared_error_scene_path:
            if shared_error_scene_path.lower().endswith(".blend"):
                base_blend = shared_error_scene_path
                base_gltf_path = gltf_path
            else:
                base_blend = ""
                base_gltf_path = shared_error_scene_path
        else:
            base_blend = ""
            base_gltf_path = gltf_path

    start_iteration = 1
    cumulative: Dict = {}
    history: List[Dict] = []
    token_usage_cumulative = {
        "request_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    previous_analysis: List[Dict] = []

    if resume_incomplete:
        resume_state = _scan_resume_state(
            region_dir=region_dir,
            max_iterations=max_iterations,
            has_isometric_ref=has_isometric_ref,
        )
        start_iteration = int(resume_state["start_iteration"])
        if not bool(resume_state["should_run"]):
            existing_token_usage = {"request_count": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            if os.path.isfile(token_usage_summary_path):
                try:
                    existing_token_usage = (
                        (_load_json(token_usage_summary_path) or {}).get("token_usage", {}) or existing_token_usage
                    )
                except Exception:
                    pass
            print(
                f"[RESUME-SKIP] {display_name}: already completed "
                f"(reason={resume_state['stop_reason']}, start_iteration={start_iteration})",
                flush=True,
            )
            return {
                "region_id": region_id,
                "region_tag": display_name,
                "iterations": max(0, start_iteration - 1),
                "status": "skipped_already_complete",
                "region_dir": region_dir,
                "token_usage": {
                    "request_count": _to_int(existing_token_usage.get("request_count")),
                    "prompt_tokens": _to_int(existing_token_usage.get("prompt_tokens")),
                    "completion_tokens": _to_int(existing_token_usage.get("completion_tokens")),
                    "total_tokens": _to_int(existing_token_usage.get("total_tokens")),
                },
                "token_usage_summary_path": token_usage_summary_path,
                "metrics": None,
                "final_blend_path": (os.path.join(region_dir, final_blend_filename) if save_region_final_blend else ""),
                "final_blend_removed": False,
                "resume": resume_state,
            }

        cumulative, previous_analysis, history, token_usage_cumulative = _load_resume_prefix_state(
            region_dir=region_dir,
            start_iteration=start_iteration,
        )
        _truncate_interaction_log(interaction_log_path, keep_before_iteration=start_iteration)
        print(
            f"[RESUME] {display_name}: continue from iter {start_iteration}/{max_iterations}",
            flush=True,
        )
    else:
        cumulative = (_load_json(cumulative_path) or {}).get("cumulative_actions", {}) or {}

    for iteration in range(start_iteration, max_iterations + 1):
        iter_dir = os.path.join(region_dir, f"iter_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)
        if resume_incomplete:
            _cleanup_iteration_artifacts_for_retry(iter_dir)

        if iteration == 1:
            if has_isometric_ref:
                input_images = [
                    os.path.join(region_dir, "top.png"),
                    os.path.join(region_dir, "isometric.png"),
                ]
            else:
                input_images = [os.path.join(region_dir, "top.png")]
        else:
            if has_isometric_ref:
                input_images = [
                    os.path.join(region_dir, "top.png"),
                    os.path.join(region_dir, "isometric.png"),
                    os.path.join(region_dir, f"iter_{iteration-1}", "top.png"),
                    os.path.join(region_dir, f"iter_{iteration-1}", "isometric.png"),
                ]
            else:
                input_images = [
                    os.path.join(region_dir, "top.png"),
                    os.path.join(region_dir, f"iter_{iteration-1}", "top.png"),
                ]

        cumulative_for_prompt = _consolidate(cumulative)

        iter_bar = _progress_bar(iteration, max_iterations)
        print(
            f"[API-START] {display_name} iter {iteration}/{max_iterations} {iter_bar} | sending request...",
            flush=True,
        )
        api_start_ts = time.time()
        llm_call_failed = False
        llm_error = ""
        try:
            llm_result = _call_mllm(
                input_images,
                llm_runtime=llm_runtime,
                image_url_convert_endpoint=image_url_convert_endpoint,
                image_url_convert_timeout=image_url_convert_timeout,
                image_url_expires_in=image_url_expires_in,
                image_url_public_base_url=image_url_public_base_url,
                scene_type=scene_type,
                valid_ids=valid_ids,
                iteration=iteration,
                previous_analysis=previous_analysis,
                cumulative_actions=cumulative_for_prompt,
                unit_scale=unit_scale,
                has_isometric_ref=has_isometric_ref,
                labeled_objects_hint=labeled_objects_hint,
                timeout=llm_timeout,
                max_retries=llm_max_retries,
                max_completion_tokens=llm_max_completion_tokens,
            )
        except Exception as exc:
            llm_call_failed = True
            llm_error = f"{type(exc).__name__}: {exc}"
            print(
                f"[WARN] {display_name} iter {iteration}: MLLM "
                " region"
                f" reason={llm_error}",
                flush=True,
            )
            llm_result = {
                "raw_response": "[]",
                "system_prompt": "",
                "user_prompt_text": "",
                "token_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "reasoning": "",
                "input_image_urls": [os.path.abspath(p) for p in input_images],
                "input_image_paths": [os.path.abspath(p) for p in input_images],
            }
        api_elapsed = time.time() - api_start_ts
        raw = llm_result.get("raw_response", "")
        reasoning = llm_result.get("reasoning", "") or ""
        system_prompt = llm_result.get("system_prompt", "")
        user_prompt_text = llm_result.get("user_prompt_text", "")
        input_image_urls = llm_result.get("input_image_urls", []) or []
        token_usage = _normalize_token_usage(llm_result.get("token_usage"))
        print(
            "[API-DONE] "
            f"{display_name} iter {iteration}/{max_iterations} {iter_bar} | "
            f"time={_fmt_elapsed(api_elapsed)}, "
            f"tokens(prompt={token_usage['prompt_tokens']}, completion={token_usage['completion_tokens']}, total={token_usage['total_tokens']})",
            flush=True,
        )
        token_usage_cumulative["request_count"] += 1
        token_usage_cumulative["prompt_tokens"] += token_usage["prompt_tokens"]
        token_usage_cumulative["completion_tokens"] += token_usage["completion_tokens"]
        token_usage_cumulative["total_tokens"] += token_usage["total_tokens"]
        analysis_raw = _extract_json_array(raw)
        analysis = []
        dropped = []
        for item in analysis_raw:
            try:
                label_id = int(item.get("id"))
            except Exception:
                dropped.append(item)
                continue
            error_type = str(item.get("error_type", "")).strip()
            if error_type.lower() == "strangeappearance":
                dropped.append({"dropped_reason": "disallowed_error_type", "item": item})
                continue
            if scene_type == "indoor":
                token = re.sub(r"[\s_-]+", "", error_type.strip().lower())
                if token == "overlap":
                    normalized_error_type = "Overlap"
                elif token in {"wallconflict", "walldoorconflict", "doorconflict", "pathconflict", "wall"}:
                    normalized_error_type = "WallConflict"
                elif token in {"orientation", "alignmenterror", "alignment"}:
                    normalized_error_type = "Orientation"
                else:
                    dropped.append({"dropped_reason": "out_of_schema_error_type", "item": item})
                    continue
                item = dict(item)
                item["error_type"] = normalized_error_type

            actions = _normalize_action_list(item.get("action", []))
            filtered_actions = []
            has_delete_action = False
            for action_text in actions:
                if re.match(r"^\s*Delete\s*\(\s*\)\s*$", str(action_text), re.IGNORECASE):
                    has_delete_action = True
                else:
                    filtered_actions.append(action_text)
            item = dict(item)
            item["action"] = filtered_actions
            normalized_issue_status = _normalize_issue_status(item.get("issue_status"))
            if normalized_issue_status:
                item["issue_status"] = normalized_issue_status
            if has_delete_action and (not filtered_actions):
                dropped.append({"dropped_reason": "disallowed_delete_action", "item": item})
                continue

            if label_id in valid_id_set:
                analysis.append(item)
            else:
                dropped.append(item)

        actionable_analysis = [it for it in analysis if _is_actionable_issue_item(it)]

        _write_json(
            os.path.join(iter_dir, "mllm_analysis.json"),
            {
                "region_id": region_id,
                "region_tag": display_name,
                "iteration": iteration,
                "model": llm_runtime.get("model"),
                "provider": llm_runtime.get("provider"),
                "base_url": llm_runtime.get("base_url", llm_runtime.get("azure_endpoint", "")),
                "input_images": input_images,
                "input_image_urls": input_image_urls,
                "system_prompt": system_prompt,
                "user_prompt_text": user_prompt_text,
                "reasoning": reasoning,
                "analysis": analysis,
                "raw": raw,
                "raw_response": raw,
                "parsed_analysis": analysis,
                "actionable_analysis": actionable_analysis,
                "valid_ids": valid_ids,
                "dropped_invalid_id_items": dropped,
                "token_usage": token_usage,
                "token_usage_cumulative": dict(token_usage_cumulative),
                "llm_call_failed": llm_call_failed,
                "llm_error": llm_error,
            },
        )
        iter_status = "analysis_returned"
        if llm_call_failed:
            iter_status = "llm_call_failed_no_action"
        elif not actionable_analysis:
            iter_status = "no_actionable_issues"
        _append_jsonl(
            interaction_log_path,
            {
                "region_id": region_id,
                "region_tag": display_name,
                "iteration": iteration,
                "model": llm_runtime.get("model"),
                "provider": llm_runtime.get("provider"),
                "base_url": llm_runtime.get("base_url", llm_runtime.get("azure_endpoint", "")),
                "valid_ids": valid_ids,
                "input_images": input_images,
                "input_image_urls": input_image_urls,
                "system_prompt": system_prompt,
                "user_prompt_text": user_prompt_text,
                "raw_response": raw,
                "parsed_analysis": analysis,
                "actionable_analysis": actionable_analysis,
                "dropped_invalid_id_items": dropped,
                "analysis_count": len(analysis),
                "actionable_count": len(actionable_analysis),
                "status": iter_status,
                "token_usage": token_usage,
                "token_usage_cumulative": dict(token_usage_cumulative),
                "llm_call_failed": llm_call_failed,
                "llm_error": llm_error,
            },
        )

        if not actionable_analysis:
            history.append(
                {
                    "iteration": iteration,
                    "analysis_count": len(analysis),
                    "actionable_count": 0,
                    "status": iter_status,
                    "token_usage": token_usage,
                    "token_usage_cumulative": dict(token_usage_cumulative),
                    "llm_call_failed": llm_call_failed,
                    "llm_error": llm_error,
                }
            )
            break

        previous_analysis = actionable_analysis

        _merge_analysis_into_cumulative(cumulative, actionable_analysis, unit_scale=unit_scale)
        consolidated = _consolidate(cumulative)

        cumulative_render_path = os.path.join(iter_dir, "cumulative_actions_render.json")
        _write_json(cumulative_render_path, {"consolidated_actions": consolidated, "cumulative_actions": cumulative})

        _run_blender_render_limited(
            region_dir=region_dir,
            output_dir=iter_dir,
            cumulative_path=cumulative_render_path,
            input_blend=base_blend,
            output_top="top.png",
            output_iso="isometric.png",
            output_blend=None,
            gltf_path=base_gltf_path,
            unit_scale=unit_scale,
            scene_type=scene_type,
            region_mode=region_mode,
            action_log_path=os.path.join(iter_dir, "applied_actions_log.json"),
        )
        top_out = os.path.join(iter_dir, "top.png")
        iso_out = os.path.join(iter_dir, "isometric.png")
        missing_outputs: List[str] = []
        if not os.path.isfile(top_out):
            missing_outputs.append(top_out)
        if has_isometric_ref and (not os.path.isfile(iso_out)):
            missing_outputs.append(iso_out)
        if missing_outputs:
            raise RuntimeError(
                "Blender "
                f" region={display_name}, iter={iteration}, missing={missing_outputs}. "
                " workspace "
            )

        history.append(
            {
                "iteration": iteration,
                "analysis_count": len(analysis),
                "actionable_count": len(actionable_analysis),
                "status": "success",
                "token_usage": token_usage,
                "token_usage_cumulative": dict(token_usage_cumulative),
            }
        )

    _write_json(os.path.join(region_dir, "cumulative_actions.json"), {"cumulative_actions": cumulative})
    final_consolidated = _consolidate(cumulative)
    final_actions = {"region_id": region_id, "consolidated_actions": final_consolidated, "cumulative_actions": cumulative}
    _write_json(os.path.join(region_dir, "final_actions.json"), final_actions)

    #  region  blendregion 
    final_render_payload = os.path.join(region_dir, "final_render_actions.json")
    _write_json(final_render_payload, {"consolidated_actions": final_consolidated})

    final_blend_path = os.path.join(region_dir, final_blend_filename) if save_region_final_blend else ""
    _run_blender_render_limited(
        region_dir=region_dir,
        output_dir=region_dir,
        cumulative_path=final_render_payload,
        input_blend=base_blend,
        output_top="top_final.png",
        output_iso="isometric_final.png",
        output_blend=(final_blend_path or None),
        gltf_path=base_gltf_path,
        unit_scale=unit_scale,
        scene_type=scene_type,
        region_mode=region_mode,
        action_log_path=os.path.join(region_dir, "final_applied_actions_log.json"),
    )

    _write_json(os.path.join(region_dir, "iteration_history.json"), history)
    _write_json(
        token_usage_summary_path,
        {
            "region_id": region_id,
            "region_tag": display_name,
            "model": llm_runtime.get("model"),
            "provider": llm_runtime.get("provider"),
            "base_url": llm_runtime.get("base_url", llm_runtime.get("azure_endpoint", "")),
            "token_usage": token_usage_cumulative,
            "interaction_log_path": interaction_log_path,
        },
    )
    print(
        "[TOKENS] "
        f"{display_name}: requests={token_usage_cumulative['request_count']}, "
        f"prompt={token_usage_cumulative['prompt_tokens']}, "
        f"completion={token_usage_cumulative['completion_tokens']}, "
        f"total={token_usage_cumulative['total_tokens']}"
    )
    metrics_result = None
    if run_metrics_per_region:
        print(f"[METRICS-START] {display_name}: running per-region metrics...", flush=True)
        metrics_result = _run_metrics_for_region(
            scene_type=scene_type,
            region_dir=region_dir,
            steps=metrics_steps_value,
            input_root=metrics_input_root,
        )
        m_status = str((metrics_result or {}).get("status", "unknown"))
        if m_status == "ok":
            print(
                f"[METRICS-DONE] {display_name}: "
                f"{metrics_result.get('output_path')} "
                f"(elapsed={_fmt_elapsed(float(metrics_result.get('elapsed_sec', 0.0) or 0.0))})",
                flush=True,
            )
        else:
            print(f"[METRICS-{m_status.upper()}] {display_name}: {metrics_result}", flush=True)

    final_blend_removed = False
    if cleanup_final_blend_after_metrics and run_metrics_per_region and scene_type == "region":
        try:
            if final_blend_path and os.path.isfile(final_blend_path):
                os.remove(final_blend_path)
                final_blend_removed = True
                print(f"[CLEANUP] {display_name}: removed {final_blend_path}", flush=True)
        except Exception as exc:
            print(f"[WARN] {display_name}: failed to remove final blend {final_blend_path}: {exc}", flush=True)

    return {
        "region_id": region_id,
        "region_tag": display_name,
        "iterations": len(history),
        "status": history[-1]["status"] if history else "unknown",
        "region_dir": region_dir,
        "token_usage": token_usage_cumulative,
        "token_usage_summary_path": token_usage_summary_path,
        "metrics": metrics_result,
        "final_blend_path": final_blend_path,
        "final_blend_removed": final_blend_removed,
        "resume": {
            "enabled": bool(resume_incomplete),
            "start_iteration": int(start_iteration),
        },
    }


def _render_shared_final_scene_groups(
    targets: List[int],
    samples: Dict[int, RegionSample],
    prepared: Dict[int, str],
    workspace_root: str,
    scene_type: str,
    region_mode: str,
    gltf_path: str,
    unit_scale: float,
    error_steps_tag: str,
) -> List[Dict[str, object]]:
    if scene_type != "region":
        return []

    grouped: Dict[str, Dict[str, object]] = {}
    for rid in targets:
        source_scene_path = str(samples[rid].source_scene_path or "").strip()
        if not source_scene_path:
            continue
        if not os.path.isfile(source_scene_path):
            print(f"[WARN] region_{rid}: source_scene_path missing, skip shared group: {source_scene_path}", flush=True)
            continue
        key = _stable_key_for_path(source_scene_path)
        item = grouped.setdefault(key, {"source_scene_path": source_scene_path, "region_ids": []})
        item["region_ids"].append(rid)

    if not grouped:
        return []

    out_root = os.path.join(workspace_root, "_shared_final_scenes")
    os.makedirs(out_root, exist_ok=True)
    results: List[Dict[str, object]] = []

    for idx, item in enumerate(grouped.values(), start=1):
        source_scene_path = str(item["source_scene_path"])
        region_ids = sorted(int(x) for x in (item.get("region_ids") or []))
        synthetic_labels, synthetic_cumulative, warnings = _build_shared_scene_actions(region_ids=region_ids, prepared=prepared)
        for w in warnings:
            print(f"[WARN] shared-group {idx}: {w}", flush=True)
        if not synthetic_labels:
            print(f"[WARN] shared-group {idx}: no merged actions, skip rendering shared final scene", flush=True)
            continue

        group_dir_name = _safe_group_slug(source_scene_path, idx)
        group_dir = os.path.join(out_root, group_dir_name)
        os.makedirs(group_dir, exist_ok=True)

        _write_json(os.path.join(group_dir, "labels.json"), synthetic_labels)
        consolidated = _consolidate_generic(synthetic_cumulative)
        _write_json(
            os.path.join(group_dir, "final_render_actions.json"),
            {"consolidated_actions": consolidated},
        )

        final_blend_filename = f"final_scene_{error_steps_tag}.blend" if error_steps_tag else "final_scene.blend"
        shared_blend_path = os.path.join(group_dir, final_blend_filename)
        input_blend = source_scene_path if source_scene_path.lower().endswith(".blend") else ""
        input_gltf = source_scene_path if (not source_scene_path.lower().endswith(".blend")) else gltf_path
        print(
            f"[SHARED-BLEND] group {idx}: source={source_scene_path}, regions={region_ids}, output={shared_blend_path}",
            flush=True,
        )
        _run_blender_render_limited(
            region_dir=group_dir,
            output_dir=group_dir,
            cumulative_path=os.path.join(group_dir, "final_render_actions.json"),
            input_blend=input_blend,
            output_top="top_final.png",
            output_iso="isometric_final.png",
            output_blend=shared_blend_path,
            gltf_path=input_gltf,
            unit_scale=unit_scale,
            scene_type=scene_type,
            region_mode=region_mode,
            action_log_path=os.path.join(group_dir, "final_applied_actions_log.json"),
        )
        results.append(
            {
                "group_index": idx,
                "source_scene_path": source_scene_path,
                "region_ids": region_ids,
                "group_dir": group_dir,
                "shared_blend_path": shared_blend_path,
            }
        )
    return results


def _run_shared_metrics_for_groups(
    shared_groups: List[Dict[str, object]],
    prepared: Dict[int, str],
    scene_type: str,
    steps: str,
    input_root: str,
    workers: int,
) -> Dict[int, Dict[str, object]]:
    if scene_type != "region":
        return {}

    tasks: List[Tuple[int, str, str]] = []
    for row in shared_groups:
        shared_blend = str(row.get("shared_blend_path", "") or "")
        for rid in row.get("region_ids", []) or []:
            region_dir = prepared.get(int(rid), "")
            if region_dir:
                tasks.append((int(rid), region_dir, shared_blend))

    if not tasks:
        return {}

    out: Dict[int, Dict[str, object]] = {}
    worker_count = max(1, min(int(workers), len(tasks)))
    print(f"[INFO] Shared metrics mode: parallel workers={worker_count}, tasks={len(tasks)}", flush=True)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(
                _run_metrics_for_region,
                scene_type,
                region_dir,
                steps,
                input_root,
                shared_blend,
            ): rid
            for rid, region_dir, shared_blend in tasks
        }
        for fut in as_completed(future_map):
            rid = future_map[fut]
            result = fut.result()
            out[rid] = result
            status = str((result or {}).get("status", "unknown"))
            print(f"[METRICS-{status.upper()}] region_{rid}: {result.get('output_path')}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=" MLLM region / indoor")
    default_run_metrics_per_region = str(os.environ.get("RUN_METRICS_PER_REGION", "0")).strip().lower() in {
        "1", "true", "yes", "y", "on"
    }
    default_run_shared_final_metrics = str(os.environ.get("RUN_SHARED_FINAL_METRICS", "1")).strip().lower() in {
        "1", "true", "yes", "y", "on"
    }
    default_cleanup_final_blend_after_metrics = str(
        os.environ.get("CLEANUP_FINAL_BLEND_AFTER_METRICS", "1")
    ).strip().lower() in {"1", "true", "yes", "y", "on"}
    default_save_region_final_blend = str(
        os.environ.get("SAVE_REGION_FINAL_BLEND", "0")
    ).strip().lower() in {"1", "true", "yes", "y", "on"}
    parser.add_argument("--scene-type", type=str, choices=["region", "indoor"], default="region")
    parser.add_argument(
        "--indoor-mode",
        type=str,
        choices=["all", "simple", "complex"],
        default=str(os.environ.get("INDOOR_MODE", "all")).strip().lower() or "all",
        help="indoor sampled metadata all/simple/complex scene-type=indoor ",
    )
    parser.add_argument("--region-mode", type=str, default="basic", choices=["basic", "complex", "indoor"])
    parser.add_argument("--region-name", type=str, default="default_region")
    parser.add_argument("--input-root", type=str, default=os.path.join(PROJECT_ROOT, "benchmark/data/multi_step_error"))
    parser.add_argument("--sampled-metadata-json", type=str, default="")
    parser.add_argument("--qa-json", type=str, default="")
    parser.add_argument("--region-data-json", type=str, default="")
    parser.add_argument("--metadata-json", type=str, default="")
    parser.add_argument("--workspace-root", type=str, default=os.path.join(PROJECT_ROOT, "results/mllm_iteration"))
    parser.add_argument("--regions", type=str, default="")

    parser.add_argument("--use-error-scene", action="store_true", help=".glb  .blend")
    parser.add_argument("--use-error-blend", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--unit-scale", type=float, default=5.0)
    parser.add_argument("--llm-model", type=str, default=os.environ.get("LLM_MODEL", "minimax/minimax-01"))
    parser.add_argument("--llm-provider", type=str, default=os.environ.get("LLM_PROVIDER", "auto"), choices=["auto", "gpt", "openrouter"])
    parser.add_argument("--openrouter-base-url", type=str, default=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    parser.add_argument("--azure-endpoint", type=str, default=os.environ.get("AZURE_ENDPOINT", DEFAULT_AZURE_ENDPOINT))
    parser.add_argument("--azure-api-version", type=str, default=os.environ.get("AZURE_API_VERSION", DEFAULT_AZURE_API_VERSION))
    parser.add_argument("--api-key-env", type=str, default="Siliconflow_KEY")
    parser.add_argument("--llm-timeout", type=int, default=int(os.environ.get("LLM_TIMEOUT", "300")))
    parser.add_argument("--llm-max-retries", type=int, default=int(os.environ.get("LLM_MAX_RETRIES", "2")))
    parser.add_argument(
        "--gpt-reasoning-effort",
        type=str,
        default=str(os.environ.get("GPT_REASONING_EFFORT", "medium")),
        help="GPT reasoning.effort: minimal/low/medium/high/xhigh",
    )
    parser.add_argument(
        "--gpt-reasoning-summary",
        type=str,
        default=str(os.environ.get("GPT_REASONING_SUMMARY", "auto")),
        help="GPT reasoning.summary: auto/concise/detailed/none",
    )
    parser.add_argument(
        "--gpt-no-reasoning",
        action="store_true",
        default=str(os.environ.get("GPT_NO_REASONING", "0")).strip().lower() in {"1", "true", "yes", "y", "on"},
        help="GPT  reasoning ",
    )
    parser.add_argument(
        "--gpt-use-responses-api",
        dest="gpt_use_responses_api",
        action="store_true",
        default=str(os.environ.get("GPT_USE_RESPONSES_API", "1")).strip().lower() in {"1", "true", "yes", "y", "on"},
        help="GPT  responses API",
    )
    parser.add_argument(
        "--gpt-no-responses-api",
        dest="gpt_use_responses_api",
        action="store_false",
        help="GPT  chat.completions API",
    )
    parser.add_argument(
        "--llm-max-completion-tokens",
        type=int,
        default=int(os.environ.get("LLM_MAX_COMPLETION_TOKENS", "8096")),
        help=" chat.completions  max_completion_tokens",
    )
    parser.add_argument(
        "--image-url-convert-endpoint",
        type=str,
        default=os.environ.get("IMAGE_URL_CONVERT_ENDPOINT"),
        help=" URL POST /convert",
    )
    parser.add_argument(
        "--image-url-convert-timeout",
        type=int,
        default=int(os.environ.get("IMAGE_URL_CONVERT_TIMEOUT", "20")),
        help=" URL ",
    )
    parser.add_argument(
        "--image-url-expires-in",
        type=int,
        default=int(os.environ.get("IMAGE_URL_EXPIRES_IN", "3600")),
        help=" image_url ",
    )
    parser.add_argument(
        "--image-url-public-base-url",
        type=str,
        default=os.environ.get("IMAGE_URL_PUBLIC_BASE_URL", ""),
        help=" public_base_url Cloudflare Tunnel ",
    )
    parser.add_argument(
        "--region-workers",
        type=int,
        default=int(os.environ.get("REGION_WORKERS", "1")),
        help=" region  worker 1=>1=",
    )
    parser.add_argument(
        "--blender-workers",
        type=int,
        default=int(os.environ.get("BLENDER_WORKERS", "2")),
        help=" Blender  worker /",
    )
    parser.add_argument(
        "--run-metrics-per-region",
        dest="run_metrics_per_region",
        action="store_true",
        default=default_run_metrics_per_region,
        help=" region  metrics RUN_METRICS_PER_REGION ",
    )
    parser.add_argument(
        "--no-run-metrics-per-region",
        dest="run_metrics_per_region",
        action="store_false",
        help=" region  metrics",
    )
    parser.add_argument(
        "--run-shared-final-metrics",
        dest="run_shared_final_metrics",
        action="store_true",
        default=default_run_shared_final_metrics,
        help="region  blend  metrics",
    )
    parser.add_argument(
        "--no-run-shared-final-metrics",
        dest="run_shared_final_metrics",
        action="store_false",
        help=" blend  metrics",
    )
    parser.add_argument(
        "--cleanup-final-blend-after-metrics",
        dest="cleanup_final_blend_after_metrics",
        action="store_true",
        default=default_cleanup_final_blend_after_metrics,
        help="region  region  metrics  final_scene*.blend",
    )
    parser.add_argument(
        "--no-cleanup-final-blend-after-metrics",
        dest="cleanup_final_blend_after_metrics",
        action="store_false",
        help=" region  final_scene*.blend",
    )
    parser.add_argument(
        "--save-region-final-blend",
        dest="save_region_final_blend",
        action="store_true",
        default=default_save_region_final_blend,
        help=" final_scene*.blendregion ",
    )
    parser.add_argument(
        "--no-save-region-final-blend",
        dest="save_region_final_blend",
        action="store_false",
        help=" final_scene*.blend",
    )
    parser.add_argument(
        "--metrics-workers",
        type=int,
        default=int(os.environ.get("METRICS_WORKERS", "0")),
        help=" blend  metrics  worker 0  region-workers",
    )
    default_resume_incomplete = str(os.environ.get("RESUME_INCOMPLETE", "0")).strip().lower() in {
        "1", "true", "yes", "y", "on"
    }
    parser.add_argument(
        "--resume-incomplete",
        dest="resume_incomplete",
        action="store_true",
        default=default_resume_incomplete,
        help=" region/",
    )
    parser.add_argument(
        "--no-resume-incomplete",
        dest="resume_incomplete",
        action="store_false",
        help="",
    )
    parser.add_argument("--steps", type=str, default=os.environ.get("STEPS", os.environ.get("ERROR_STEPS", "")))
    parser.add_argument("--error-steps", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--global-error-scene-path", type=str, default="")
    parser.add_argument("--global-error-blend-path", type=str, default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    global _BLENDER_RENDER_SEMAPHORE
    blender_workers = max(1, int(args.blender_workers))
    _BLENDER_RENDER_SEMAPHORE = threading.BoundedSemaphore(blender_workers)
    print(f"[INFO] blender render workers={blender_workers}")
    if args.scene_type == "indoor" and (not bool(args.save_region_final_blend)):
        args.save_region_final_blend = True
        print("[INFO] indoor  --save-region-final-blend", flush=True)

    shared_metrics_enabled = bool(args.scene_type == "region" and args.run_shared_final_metrics)
    effective_run_metrics_per_region = bool(args.run_metrics_per_region and (not shared_metrics_enabled))
    if shared_metrics_enabled and args.run_metrics_per_region:
        print("[INFO] region  shared-final-metrics per-region immediate metrics", flush=True)

    use_error_scene = bool(args.use_error_scene or args.use_error_blend)
    llm_runtime = _build_llm_runtime(args)
    image_url_convert_endpoint = str(args.image_url_convert_endpoint or "").strip()
    if not image_url_convert_endpoint:
        raise ValueError("image-url-convert-endpoint ")
    image_url_convert_timeout = max(1, int(args.image_url_convert_timeout))
    image_url_expires_in = max(60, int(args.image_url_expires_in))
    image_url_public_base_url = str(args.image_url_public_base_url or "").strip().rstrip("/")
    if image_url_public_base_url and (not image_url_public_base_url.startswith(("http://", "https://"))):
        raise ValueError("image-url-public-base-url  http://  https:// ")
    print(
        f"[INFO] llm provider={llm_runtime.get('provider')} "
        f"model={llm_runtime.get('model')} "
        f"endpoint={llm_runtime.get('base_url', llm_runtime.get('azure_endpoint', ''))}"
    )
    if llm_runtime.get("provider") == "gpt":
        print(
            f"[INFO] gpt_reasoning={llm_runtime.get('gpt_reasoning')} "
            f"use_responses_api={bool(llm_runtime.get('gpt_use_responses_api', True))}"
        )
    print(f"[INFO] llm max_completion_tokens={max(1, int(args.llm_max_completion_tokens))}")
    print(
        f"[INFO] image_url_convert endpoint={image_url_convert_endpoint} "
        f"timeout={image_url_convert_timeout}s "
        f"expires_in={image_url_expires_in}s"
    )
    print(
        f"[INFO] resume_incomplete={bool(args.resume_incomplete)} "
        f"workspace_root={str(args.workspace_root)}",
        flush=True,
    )
    if image_url_public_base_url:
        print(f"[INFO] image_url_public_base_url={image_url_public_base_url}")

    effective_steps = args.steps if str(args.steps).strip() else args.error_steps
    error_steps_tag = f"steps{effective_steps}" if str(effective_steps).strip() else ""
    os.makedirs(args.workspace_root, exist_ok=True)
    sampled_metadata_json = str(args.sampled_metadata_json or "").strip()
    indoor_mode = str(getattr(args, "indoor_mode", "all") or "all").strip().lower()

    if sampled_metadata_json:
        if not os.path.exists(sampled_metadata_json):
            raise FileNotFoundError(f"sampled metadata : {sampled_metadata_json}")
        qa_json, region_data_json = _build_iteration_inputs_from_sampled_metadata(
            sampled_metadata_json=sampled_metadata_json,
            scene_type=args.scene_type,
            workspace_root=args.workspace_root,
            indoor_mode_filter=indoor_mode,
        )
        # sampled_metadata  sourceregion/step error_scene
        use_error_scene = False
        print(f"[INFO] sampled metadata: {os.path.abspath(sampled_metadata_json)}")
        if args.scene_type == "indoor":
            print(f"[INFO] indoor_mode filter: {indoor_mode}")
        print(f"[INFO] generated qa json from sampled: {os.path.abspath(qa_json)}")
        print(f"[INFO] generated region data json from sampled: {os.path.abspath(region_data_json)}")
    elif args.scene_type == "indoor":
        metadata_candidates: List[str] = []
        if args.metadata_json:
            metadata_candidates.append(args.metadata_json)
        if error_steps_tag:
            metadata_candidates.append(os.path.join(args.input_root, f"metadata_indoor_{error_steps_tag}.json"))
        metadata_candidates.append(os.path.join(args.input_root, "metadata_indoor.json"))
        metadata_json = next((path for path in metadata_candidates if os.path.exists(path)), metadata_candidates[0])
        if not os.path.exists(metadata_json):
            raise FileNotFoundError(f"indoor metadata : {metadata_json}")

        qa_json = args.qa_json or os.path.join(
            args.input_root,
            (f"qa_indoor_{error_steps_tag}.json" if error_steps_tag else "qa_indoor.json"),
        )
        region_data_json = args.region_data_json or os.path.join(
            args.input_root,
            (f"region_data_indoor_{error_steps_tag}.json" if error_steps_tag else "region_data_indoor.json"),
        )
        qa_json, region_data_json = _convert_indoor_metadata_to_iteration_inputs(
            metadata_json=metadata_json,
            qa_json_out=qa_json,
            region_data_json_out=region_data_json,
        )
        print(f"[INFO] indoor metadata: {os.path.abspath(metadata_json)}")
        print(f"[INFO] generated qa json: {os.path.abspath(qa_json)}")
        print(f"[INFO] generated region data json: {os.path.abspath(region_data_json)}")
    else:
        default_qa_candidates = []
        if error_steps_tag:
            default_qa_candidates.append(os.path.join(args.input_root, f"qa_{args.region_name}_{error_steps_tag}.json"))
        default_qa_candidates.append(os.path.join(args.input_root, f"qa_{args.region_name}.json"))
        qa_json = args.qa_json or next((path for path in default_qa_candidates if os.path.exists(path)), default_qa_candidates[0])
        if not os.path.exists(qa_json):
            raise FileNotFoundError(f"qa json : {qa_json}")

        region_data_json = args.region_data_json or os.path.join(
            PROJECT_ROOT,
            f"benchmark/data_construct/model_process/results/{args.region_name}_kmeans/region_data_clean.json",
        )
        if not os.path.exists(region_data_json):
            raise FileNotFoundError(f"region_data_clean.json : {region_data_json}")

    shared_error_scene_path = ""
    global_error_scene_path = ""
    if args.global_error_scene_path:
        global_error_scene_path = args.global_error_scene_path
    elif args.global_error_blend_path:
        global_error_scene_path = args.global_error_blend_path
    else:
        default_scene_candidates: List[str] = []
        if error_steps_tag:
            default_scene_candidates.extend(
                [
                    os.path.join(args.input_root, f"{args.region_name}_{error_steps_tag}_global_error_scene.glb"),
                    os.path.join(args.input_root, f"{args.region_name}_{error_steps_tag}_global_error_scene.blend"),
                ]
            )
        default_scene_candidates.extend(
            [
                os.path.join(args.input_root, f"{args.region_name}_global_error_scene.glb"),
                os.path.join(args.input_root, f"{args.region_name}_global_error_scene.blend"),
            ]
        )
        global_error_scene_path = next(
            (path for path in default_scene_candidates if os.path.exists(path)),
            default_scene_candidates[0],
        )

    if use_error_scene:
        if not os.path.exists(global_error_scene_path):
            raise FileNotFoundError(f" error scene : {global_error_scene_path}")
        scene_ext = ".blend" if global_error_scene_path.lower().endswith(".blend") else ".glb"
        error_scene_filename = f"error_scene_{error_steps_tag}{scene_ext}" if error_steps_tag else f"error_scene{scene_ext}"
        shared_error_scene_path = os.path.join(args.workspace_root, error_scene_filename)
        print(f"[INFO] global error scene: {global_error_scene_path}")
        if not os.path.exists(shared_error_scene_path):
            if not _copy(global_error_scene_path, shared_error_scene_path):
                raise RuntimeError(f" error scene : {global_error_scene_path} -> {shared_error_scene_path}")
            print(f"[INFO] shared error scene copied: {shared_error_scene_path}")
        else:
            print(f"[INFO] shared error scene exists: {shared_error_scene_path}")

    samples = _load_samples(qa_json)
    region_data = _load_json(region_data_json)

    targets = _parse_regions(args.regions, sorted(samples.keys()))
    if not targets:
        raise RuntimeError("")

    prepared = {}
    region_tags: Dict[int, str] = {}
    for rid in targets:
        region_dir = _prepare_region_workspace(
            sample=samples[rid],
            region_data=region_data,
            workspace_root=args.workspace_root,
            scene_type=args.scene_type,
            region_mode=args.region_mode,
            use_error_scene=use_error_scene,
            unit_scale=args.unit_scale,
            shared_error_scene_path=shared_error_scene_path,
            gltf_path=args.gltf_path,
            error_steps_tag=error_steps_tag,
            resume_incomplete=bool(args.resume_incomplete),
        )
        prepared[rid] = region_dir
        if args.scene_type == "indoor":
            region_tags[rid] = os.path.basename(region_dir)
        else:
            region_tags[rid] = f"region_{rid}"
        print(f"[PREP] {region_tags[rid]}: {region_dir}")

    results: List[Dict] = []
    results_map: Dict[int, Dict] = {}
    token_usage_global = {
        "request_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    region_workers = max(1, int(args.region_workers))
    if region_workers == 1 or len(targets) <= 1:
        print("[INFO] Region execution mode: serial")
        total_regions = len(targets)
        for idx, rid in enumerate(targets, start=1):
            region_bar = _progress_bar(idx, total_regions)
            print(f"\n[ITER] {region_tags[rid]} | region progress {region_bar}")
            result = _iterate_region(
                region_id=rid,
                region_tag=region_tags[rid],
                region_dir=prepared[rid],
                scene_type=args.scene_type,
                region_mode=args.region_mode,
                llm_runtime=llm_runtime,
                max_iterations=args.max_iterations,
                gltf_path=args.gltf_path,
                use_error_scene=use_error_scene,
                unit_scale=args.unit_scale,
                llm_timeout=args.llm_timeout,
                llm_max_retries=args.llm_max_retries,
                llm_max_completion_tokens=max(1, int(args.llm_max_completion_tokens)),
                image_url_convert_endpoint=image_url_convert_endpoint,
                image_url_convert_timeout=image_url_convert_timeout,
                image_url_expires_in=image_url_expires_in,
                image_url_public_base_url=image_url_public_base_url,
                has_isometric_ref=_sample_prefers_isometric(samples[rid]),
                labeled_objects_hint=samples[rid].labeled_objects_hint,
                error_steps_tag=error_steps_tag,
                shared_error_scene_path=shared_error_scene_path,
                source_scene_path=samples[rid].source_scene_path,
                run_metrics_per_region=effective_run_metrics_per_region,
                metrics_input_root=str(args.input_root),
                metrics_steps_value=str(effective_steps or ""),
                cleanup_final_blend_after_metrics=bool(args.cleanup_final_blend_after_metrics),
                save_region_final_blend=bool(args.save_region_final_blend),
                resume_incomplete=bool(args.resume_incomplete),
            )
            results_map[rid] = result
            region_tokens = result.get("token_usage", {})
            token_usage_global["request_count"] += _to_int(region_tokens.get("request_count"))
            token_usage_global["prompt_tokens"] += _to_int(region_tokens.get("prompt_tokens"))
            token_usage_global["completion_tokens"] += _to_int(region_tokens.get("completion_tokens"))
            token_usage_global["total_tokens"] += _to_int(region_tokens.get("total_tokens"))
            print(f"[DONE] {region_tags[rid]}: {result['status']} | region progress {region_bar}")
    else:
        worker_count = min(region_workers, len(targets))
        print(f"[INFO] Region execution mode: parallel (workers={worker_count})")
        future_to_region = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for rid in targets:
                print(f"\n[ITER-QUEUE] {region_tags[rid]}")
                future = executor.submit(
                    _iterate_region,
                    region_id=rid,
                    region_tag=region_tags[rid],
                    region_dir=prepared[rid],
                    scene_type=args.scene_type,
                    region_mode=args.region_mode,
                    llm_runtime=llm_runtime,
                    max_iterations=args.max_iterations,
                    gltf_path=args.gltf_path,
                    use_error_scene=use_error_scene,
                    unit_scale=args.unit_scale,
                    llm_timeout=args.llm_timeout,
                    llm_max_retries=args.llm_max_retries,
                    llm_max_completion_tokens=max(1, int(args.llm_max_completion_tokens)),
                    image_url_convert_endpoint=image_url_convert_endpoint,
                    image_url_convert_timeout=image_url_convert_timeout,
                    image_url_expires_in=image_url_expires_in,
                    image_url_public_base_url=image_url_public_base_url,
                    has_isometric_ref=_sample_prefers_isometric(samples[rid]),
                    labeled_objects_hint=samples[rid].labeled_objects_hint,
                    error_steps_tag=error_steps_tag,
                    shared_error_scene_path=shared_error_scene_path,
                    source_scene_path=samples[rid].source_scene_path,
                    run_metrics_per_region=effective_run_metrics_per_region,
                    metrics_input_root=str(args.input_root),
                    metrics_steps_value=str(effective_steps or ""),
                    cleanup_final_blend_after_metrics=bool(args.cleanup_final_blend_after_metrics),
                    save_region_final_blend=bool(args.save_region_final_blend),
                    resume_incomplete=bool(args.resume_incomplete),
                )
                future_to_region[future] = rid

            done_regions = 0
            total_regions = len(targets)
            for future in as_completed(future_to_region):
                rid = future_to_region[future]
                result = future.result()
                results_map[rid] = result
                region_tokens = result.get("token_usage", {})
                token_usage_global["request_count"] += _to_int(region_tokens.get("request_count"))
                token_usage_global["prompt_tokens"] += _to_int(region_tokens.get("prompt_tokens"))
                token_usage_global["completion_tokens"] += _to_int(region_tokens.get("completion_tokens"))
                token_usage_global["total_tokens"] += _to_int(region_tokens.get("total_tokens"))
                done_regions += 1
                region_bar = _progress_bar(done_regions, total_regions)
                print(f"[DONE] {region_tags[rid]}: {result['status']} | region progress {region_bar}")

    for rid in targets:
        if rid in results_map:
            results.append(results_map[rid])

    shared_groups: List[Dict[str, object]] = []
    shared_metrics_map: Dict[int, Dict[str, object]] = {}
    if shared_metrics_enabled:
        shared_groups = _render_shared_final_scene_groups(
            targets=targets,
            samples=samples,
            prepared=prepared,
            workspace_root=args.workspace_root,
            scene_type=args.scene_type,
            region_mode=args.region_mode,
            gltf_path=args.gltf_path,
            unit_scale=args.unit_scale,
            error_steps_tag=error_steps_tag,
        )
        metrics_workers = int(args.metrics_workers) if int(args.metrics_workers) > 0 else int(args.region_workers)
        shared_metrics_map = _run_shared_metrics_for_groups(
            shared_groups=shared_groups,
            prepared=prepared,
            scene_type=args.scene_type,
            steps=str(effective_steps or ""),
            input_root=str(args.input_root),
            workers=metrics_workers,
        )
        for rid, metrics_result in shared_metrics_map.items():
            if rid in results_map:
                results_map[rid]["metrics"] = metrics_result

    summary = {
        "workspace_root": os.path.abspath(args.workspace_root),
        "scene_type": args.scene_type,
        "region_mode": args.region_mode,
        "indoor_mode": (indoor_mode if args.scene_type == "indoor" else None),
        "steps": effective_steps,
        "llm_provider": llm_runtime.get("provider"),
        "llm_model": llm_runtime.get("model"),
        "gpt_reasoning": llm_runtime.get("gpt_reasoning"),
        "gpt_use_responses_api": llm_runtime.get("gpt_use_responses_api"),
        "llm_max_completion_tokens": max(1, int(args.llm_max_completion_tokens)),
        "image_url_convert_endpoint": image_url_convert_endpoint,
        "image_url_convert_timeout": image_url_convert_timeout,
        "image_url_expires_in": image_url_expires_in,
        "image_url_public_base_url": image_url_public_base_url or None,
        "run_metrics_per_region": bool(effective_run_metrics_per_region),
        "run_shared_final_metrics": bool(shared_metrics_enabled),
        "save_region_final_blend": bool(args.save_region_final_blend),
        "cleanup_final_blend_after_metrics": bool(args.cleanup_final_blend_after_metrics),
        "resume_incomplete": bool(args.resume_incomplete),
        "sampled_metadata_json": (os.path.abspath(sampled_metadata_json) if sampled_metadata_json else None),
        "shared_error_scene_path": shared_error_scene_path if use_error_scene else None,
        "shared_error_blend_path": (shared_error_scene_path if (use_error_scene and shared_error_scene_path.lower().endswith(".blend")) else None),
        "qa_json": os.path.abspath(qa_json),
        "region_data_json": os.path.abspath(region_data_json),
        "shared_final_scene_groups": shared_groups,
        "regions": results,
        "token_usage": token_usage_global,
    }
    _write_json(os.path.join(args.workspace_root, "summary.json"), summary)
    print(
        "[TOKENS] global: "
        f"requests={token_usage_global['request_count']}, "
        f"prompt={token_usage_global['prompt_tokens']}, "
        f"completion={token_usage_global['completion_tokens']}, "
        f"total={token_usage_global['total_tokens']}"
    )
    print(f"\n: {os.path.join(args.workspace_root, 'summary.json')}")


if __name__ == "__main__":
    main()
