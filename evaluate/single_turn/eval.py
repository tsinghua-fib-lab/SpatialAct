#!/usr/bin/env python3


from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
from openai import AzureOpenAI, OpenAI

PROXY = ""
AZURE_ENDPOINT = ""
AZURE_API_VERSION = ""

VALID_MODES = ("basic_own", "indoor", "region_basic", "region_complex")
IMAGE_TAG_PATTERN = re.compile(r"<image>", flags=re.IGNORECASE)
FINAL_INSTRUCTION = "Directly answer without explanation."


@dataclass
class EvalSample:
    index: int
    question: str
    answer: str
    images: List[str]
    task_type: str
    raw: Dict[str, Any]


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _encode_image_data_url(image_path: str) -> str:
    p = Path(image_path)
    suffix = p.suffix.lower()
    mime = "image/png"
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    elif suffix == ".bmp":
        mime = "image/bmp"

    with p.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _to_abs_path(path_str: str) -> str:
    return str(Path(path_str).expanduser().resolve())


def _model_name_tag(model_name: str) -> str:
    tag = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_name.strip())
    tag = tag.strip("-._")
    return tag or "model"


def _reasoning_part_tag(value: str) -> str:
    tag = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    tag = tag.strip("-._")
    return tag or "unknown"


def _reasoning_tag(reasoning: Optional[Dict[str, Any]]) -> str:
    if not reasoning:
        return "noreason"

    parts: List[str] = ["reason"]
    effort = reasoning.get("effort")
    summary = reasoning.get("summary")
    if isinstance(effort, str) and effort.strip():
        parts.append(f"effort-{_reasoning_part_tag(effort)}")
    if isinstance(summary, str) and summary.strip():
        parts.append(f"summary-{_reasoning_part_tag(summary)}")
    return "-".join(parts)


def _parse_json_files_arg(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not values:
        return None
    out: List[str] = []
    for v in values:
        parts = [x.strip() for x in v.split(",") if x.strip()]
        out.extend(parts)
    return out or None


def _find_task_file(mode_dir: Path, task: str) -> Path:
    task_clean = task.strip()
    if not task_clean:
        raise ValueError("task cannot be empty")

    all_jsons = sorted(mode_dir.glob("*.json"))
    matches = [p for p in all_jsons if _task_name_from_file(p) == task_clean]
    if not matches:
        raise FileNotFoundError(f"No JSON found for task={task_clean} under mode={mode_dir.name}")
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise RuntimeError(f"Multiple JSON files matched task={task_clean} under mode={mode_dir.name}: {names}")
    return matches[0]


def _discover_mode_jsons(mode_dir: Path, explicit_files: Optional[List[str]]) -> List[Path]:
    if explicit_files:
        files: List[Path] = []
        for name in explicit_files:
            p = Path(name)
            if p.is_absolute():
                candidate = p
            else:
                candidate = mode_dir / p
            if not candidate.exists():
                raise FileNotFoundError(f"Specified JSON does not exist: {candidate}")
            files.append(candidate.resolve())
        return sorted(files)

    all_jsons = sorted(mode_dir.glob("*.json"))
    filtered = [p for p in all_jsons if "multi_step_error" not in p.name]
    return filtered


def _task_name_from_file(file_path: Path) -> str:
    stem = file_path.stem
    stem = re.sub(r"_sampled_\d+$", "", stem)
    stem = re.sub(r"_(indoor|regions|complex_regions)$", "", stem)
    return stem


def _extract_images(rec: Dict[str, Any]) -> List[str]:
    imgs = rec.get("images", [])
    if isinstance(imgs, str):
        imgs = [imgs]
    if not isinstance(imgs, list):
        return []
    return [str(x) for x in imgs if isinstance(x, str) and x.strip()]


def _build_samples(records: List[Dict[str, Any]]) -> List[EvalSample]:
    samples: List[EvalSample] = []
    for idx, rec in enumerate(records):
        question = str(rec.get("question", "")).strip()
        answer = str(rec.get("answer", "")).strip()
        images = _extract_images(rec)
        task_type = str(rec.get("task_type", "")).strip()
        samples.append(
            EvalSample(
                index=idx,
                question=question,
                answer=answer,
                images=images,
                task_type=task_type,
                raw=rec,
            )
        )
    return samples


def _normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[`*_#]", "", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,!?:;\"'()[]{}")
    return s


def _extract_final_text(pred: str) -> str:
    text = (pred or "").strip()
    if not text:
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidate = lines[-1] if lines else text

    m = re.search(r"(?:final\s+)?answer\s*:\s*(.+)$", candidate, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
    else:
        m2 = re.search(r"(?:final\s+)?answer\s*:\s*(.+)$", text, flags=re.IGNORECASE)
        if m2:
            candidate = m2.group(1).strip()

    return candidate.strip()


def _extract_choice_letter(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = [
        r"image\s*([A-D])\b",
        r"\b([A-D])\b",
        r"\(([A-D])\)",
        r"option\s*([A-D])\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def _extract_yes_no(text: str) -> Optional[str]:
    if not text:
        return None
    t = _normalize_text(text)
    if re.search(r"\byes\b", t):
        return "yes"
    if re.search(r"\bno\b", t):
        return "no"
    return None


def _extract_ints(text: str) -> List[int]:
    return [int(x) for x in re.findall(r"(?<!\d)(\d+)(?!\d)", text)]


def _parse_answer_as_set(ans: str) -> Optional[set[int]]:
    t = _normalize_text(ans)
    if t == "none":
        return set()
    if re.fullmatch(r"\d+(\s*,\s*\d+)+", t):
        return {int(x.strip()) for x in t.split(",") if x.strip()}
    if re.fullmatch(r"\d+\s+and\s+\d+", t):
        nums = re.findall(r"\d+", t)
        return {int(nums[0]), int(nums[1])}
    return None


def _is_correct(gt_answer: str, pred_text: str) -> bool:
    gt = gt_answer.strip()
    pred = _extract_final_text(pred_text)

    # 1) A/B/C/D
    if re.fullmatch(r"[A-D]", gt, flags=re.IGNORECASE):
        letter = _extract_choice_letter(pred) or _extract_choice_letter(pred_text)
        return bool(letter and letter.upper() == gt.upper())

    # 2) Yes/No
    if gt.lower() in {"yes", "no"}:
        yn = _extract_yes_no(pred) or _extract_yes_no(pred_text)
        return bool(yn and yn == gt.lower())

    # 3) numeric set / pair / comma list / none
    gt_set = _parse_answer_as_set(gt)
    if gt_set is not None:
        pred_norm = _normalize_text(pred)
        if gt_set == set() and "none" in pred_norm:
            return True
        pred_nums = _extract_ints(pred)
        if not pred_nums:
            pred_nums = _extract_ints(pred_text)
        pred_set = set(pred_nums)
        return pred_set == gt_set

    # 4) single integer
    if re.fullmatch(r"\d+", gt):
        gt_num = int(gt)
        pred_nums = _extract_ints(pred)
        if not pred_nums:
            pred_nums = _extract_ints(pred_text)
        return bool(pred_nums and pred_nums[0] == gt_num)

    # 5) fallback normalized match
    gt_n = _normalize_text(gt)
    pred_n = _normalize_text(pred)
    if gt_n == pred_n:
        return True
    pred_full_n = _normalize_text(pred_text)
    return gt_n == pred_full_n


def _extract_pred_answer(gt_answer: str, pred_text: str) -> str:
    gt = gt_answer.strip()
    pred = _extract_final_text(pred_text)

    # 1) A/B/C/D
    if re.fullmatch(r"[A-D]", gt, flags=re.IGNORECASE):
        letter = _extract_choice_letter(pred) or _extract_choice_letter(pred_text)
        return letter if letter else pred

    # 2) Yes/No
    if gt.lower() in {"yes", "no"}:
        yn = _extract_yes_no(pred) or _extract_yes_no(pred_text)
        return yn if yn else pred

    # 3) numeric set / pair / comma list / none
    gt_set = _parse_answer_as_set(gt)
    if gt_set is not None:
        pred_norm = _normalize_text(pred)
        if gt_set == set() and "none" in pred_norm:
            return "none"
        nums = _extract_ints(pred)
        if not nums:
            nums = _extract_ints(pred_text)
        if nums:
            return ",".join(str(x) for x in sorted(set(nums)))
        return pred

    # 4) single integer
    if re.fullmatch(r"\d+", gt):
        nums = _extract_ints(pred)
        if not nums:
            nums = _extract_ints(pred_text)
        return str(nums[0]) if nums else pred

    # 5) fallback final text
    return pred


def _build_user_content(question: str, images: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    abs_images = [_to_abs_path(img) for img in images]
    valid_images = [img for img in abs_images if Path(img).exists()]

    text_parts = IMAGE_TAG_PATTERN.split(question)
    image_tag_count = max(len(text_parts) - 1, 0)

    content: List[Dict[str, Any]] = []
    content_for_log: List[Dict[str, Any]] = []

    for idx, txt in enumerate(text_parts):
        if txt:
            content.append({"type": "text", "text": txt})
            content_for_log.append({"type": "text", "text": txt})

        if idx < image_tag_count and idx < len(valid_images):
            img = valid_images[idx]
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_image_data_url(img), "detail": "high"},
                }
            )
            content_for_log.append({"type": "image_path", "path": img})

    # Keep extra images if counts are inconsistent to avoid dropping usable inputs.
    for img in valid_images[image_tag_count:]:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _encode_image_data_url(img), "detail": "high"},
            }
        )
        content_for_log.append({"type": "image_path", "path": img, "position": "appended_extra"})

    content.append({"type": "text", "text": FINAL_INSTRUCTION})
    content_for_log.append({"type": "text", "text": FINAL_INSTRUCTION})

    log_payload = {
        "role": "user",
        "content": content_for_log,
        "image_paths_abs": abs_images,
        "image_paths_exists_abs": valid_images,
        "image_tag_count": image_tag_count,
        "images_count": len(images),
        "images_exists_count": len(valid_images),
        "is_image_count_match": image_tag_count == len(images),
        "is_existing_image_count_match": image_tag_count == len(valid_images),
    }
    return content, log_payload


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(parts).strip()
    return str(content)


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
        item_type = getattr(item, "type", None)
        if item_type != "message":
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


def _usage_obj_to_dict(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        try:
            dumped = usage.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:  
            pass
    if isinstance(usage, dict):
        return dict(usage)
    return {}


def _extract_usage_info(usage: Any, is_gpt_responses: bool) -> Dict[str, Any]:
    usage_obj = _usage_obj_to_dict(usage)

    if is_gpt_responses:
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or usage_obj.get("input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", 0) or usage_obj.get("output_tokens", 0) or 0)
    else:
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or usage_obj.get("prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or usage_obj.get("completion_tokens", 0) or 0)

    total_tokens = int(getattr(usage, "total_tokens", 0) or usage_obj.get("total_tokens", 0) or 0)

    input_tokens_details = usage_obj.get("input_tokens_details")
    output_tokens_details = usage_obj.get("output_tokens_details")
    prompt_tokens_details = usage_obj.get("prompt_tokens_details")
    completion_tokens_details = usage_obj.get("completion_tokens_details")

    reasoning_tokens = None
    if isinstance(output_tokens_details, dict):
        rt = output_tokens_details.get("reasoning_tokens")
        if rt is not None:
            reasoning_tokens = int(rt)
    if reasoning_tokens is None and isinstance(completion_tokens_details, dict):
        rt = completion_tokens_details.get("reasoning_tokens")
        if rt is not None:
            reasoning_tokens = int(rt)

    out: Dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
    }

    if is_gpt_responses:
        out["input_tokens"] = prompt_tokens
        out["output_tokens"] = completion_tokens
    else:
        out["input_tokens"] = int(usage_obj.get("prompt_tokens", prompt_tokens) or prompt_tokens)
        out["output_tokens"] = int(usage_obj.get("completion_tokens", completion_tokens) or completion_tokens)

    if input_tokens_details is not None:
        out["input_tokens_details"] = _safe_jsonable(input_tokens_details)
    if output_tokens_details is not None:
        out["output_tokens_details"] = _safe_jsonable(output_tokens_details)
    if prompt_tokens_details is not None:
        out["prompt_tokens_details"] = _safe_jsonable(prompt_tokens_details)
    if completion_tokens_details is not None:
        out["completion_tokens_details"] = _safe_jsonable(completion_tokens_details)
    if usage_obj:
        out["raw"] = _safe_jsonable(usage_obj)
    return out


def _safe_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_safe_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_jsonable(v) for k, v in obj.items()}
    return str(obj)


class ModelClient:
    def __init__(
        self,
        model_name: str,
        timeout: float,
        gpt_reasoning: Optional[Dict[str, str]] = None,
        gpt_text: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model_name = model_name
        self.is_gpt = "gpt" in model_name.lower()
        self.timeout = timeout
        self.gpt_reasoning = gpt_reasoning if self.is_gpt else None
        self.gpt_text = gpt_text if self.is_gpt else None

        if self.is_gpt:
            api_key = os.environ.get("AZURE_API_KEY", "").strip()
            if not api_key:
                raise EnvironmentError("Missing environment variable AZURE_API_KEY")
            self.client = AzureOpenAI(
                api_version=AZURE_API_VERSION,
                azure_endpoint=AZURE_ENDPOINT,
                api_key=api_key,
            )
            self.deployment = model_name
        else:
            api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
            if not api_key:
                raise EnvironmentError("Missing environment variable OPENROUTER_API_KEY")
            http_client = httpx.Client(proxy=PROXY, timeout=timeout)
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                http_client=http_client,
            )
            self.deployment = model_name

    def infer(
        self,
        question: str,
        images: List[str],
        max_completion_tokens: int,
        retries: int,
        retry_wait: float,
    ) -> Tuple[str, Any, Dict[str, int], Dict[str, Any]]:
        content, request_payload_for_log = _build_user_content(question, images)
        messages = [{"role": "user", "content": content}]

        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                if self.is_gpt:
                    req: Dict[str, Any] = {
                        "input": _chat_messages_to_responses_input(messages),
                        "max_output_tokens": max_completion_tokens,
                        "model": self.deployment,
                    }
                    if self.gpt_reasoning:
                        req["reasoning"] = self.gpt_reasoning
                    if self.gpt_text:
                        req["text"] = self.gpt_text
                    resp = self.client.responses.create(**req)
                    text = _extract_responses_text(resp)
                    reasoning = _extract_responses_reasoning(resp)
                    usage = getattr(resp, "usage", None)
                    usage_dict = _extract_usage_info(usage, is_gpt_responses=True)
                    return text, reasoning, usage_dict, request_payload_for_log
                else:
                    resp = self.client.chat.completions.create(
                        model=self.deployment,
                        messages=messages,
                        max_completion_tokens=max_completion_tokens,
                        extra_body={"reasoning": {"enabled": True}, 
                                    "provider": {"only": ["deepinfra/fp4", "io-net/int4", "parasail/int4", "inceptron/int4", "z-ai/fp8", "deepinfra/fp8", "atlas-cloud/fp8"]}}
                        
                    )
                    print(f"[DEBUG] API response: {resp}")
                    msg = resp.choices[0].message
                    text = _message_content_to_text(getattr(msg, "content", ""))
                    reasoning = _safe_jsonable(getattr(msg, "reasoning", None))
                    usage = getattr(resp, "usage", None)
                    usage_dict = _extract_usage_info(usage, is_gpt_responses=False)
                    return text, reasoning, usage_dict, request_payload_for_log
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < retries - 1:
                    sleep_s = retry_wait * (2 ** attempt)
                    print(f"[WARN] API call failed, retrying in {sleep_s:.1f}s: {type(e).__name__}: {e}")
                    time.sleep(sleep_s)
                else:
                    break

        raise RuntimeError(f"API call failed: {last_err}")


def _evaluate_one_sample(
    sample: EvalSample,
    client: ModelClient,
    max_completion_tokens: int,
    retries: int,
    retry_wait: float,
    ) -> Tuple[bool, bool, Dict[str, Any]]:
    try:
        pred, reasoning, usage, model_input = client.infer(
            question=sample.question,
            images=sample.images,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            retry_wait=retry_wait,
        )
        ok = _is_correct(sample.answer, pred)
        parsed_answer = _extract_pred_answer(sample.answer, pred)
        result = {
            "index": sample.index,
            "task_type": sample.task_type,
            "question": sample.question,
            "answer_gt": sample.answer,
            "images": sample.images,
            "model_input": model_input,
            "model_output": {
                "text": pred,
                "reasoning": reasoning,
            },
            "answer_pred": parsed_answer,
            "is_correct": ok,
            "usage": usage,
        }
        return ok, False, result
    except Exception as e: 
        result = {
            "index": sample.index,
            "task_type": sample.task_type,
            "question": sample.question,
            "answer_gt": sample.answer,
            "images": sample.images,
            "model_input": {
                "role": "user",
                "image_paths_abs": [_to_abs_path(img) for img in sample.images],
                "instruction": FINAL_INSTRUCTION,
            },
            "model_output": {
                "text": "",
                "reasoning": None,
            },
            "answer_pred": "",
            "is_correct": False,
            "error": str(e),
        }
        return False, True, result


def evaluate_file(
    file_path: Path,
    task_name: str,
    model_name: str,
    client: ModelClient,
    model_request_config: Optional[Dict[str, Any]],
    max_samples: int,
    max_completion_tokens: int,
    retries: int,
    retry_wait: float,
    num_workers: int,
) -> Dict[str, Any]:
    records = _read_json(file_path)
    if not isinstance(records, list):
        raise ValueError(f"JSON top-level must be a list: {file_path}")

    samples = _build_samples(records)
    if max_samples > 0:
        samples = samples[:max_samples]

    total = len(samples)
    correct = 0
    errors = 0
    results_with_order: List[Tuple[int, Dict[str, Any]]] = []

    workers = max(1, int(num_workers))
    print(f"\n[RUN] {file_path.name} | samples={total} | workers={workers}")

    if workers == 1:
        for i, sample in enumerate(samples, start=1):
            ok, is_error, result = _evaluate_one_sample(
                sample=sample,
                client=client,
                max_completion_tokens=max_completion_tokens,
                retries=retries,
                retry_wait=retry_wait,
            )
            if ok:
                correct += 1
            if is_error:
                errors += 1
                print(f"  [{i}/{total}] ERROR: {result.get('error', '')}")
            else:
                print(f"  [{i}/{total}] {'OK' if ok else 'WRONG'}")
            results_with_order.append((i, result))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut2idx = {
                ex.submit(
                    _evaluate_one_sample,
                    sample,
                    client,
                    max_completion_tokens,
                    retries,
                    retry_wait,
                ): i
                for i, sample in enumerate(samples, start=1)
            }
            for fut in as_completed(fut2idx):
                i = fut2idx[fut]
                ok, is_error, result = fut.result()
                if ok:
                    correct += 1
                if is_error:
                    errors += 1
                    print(f"  [{i}/{total}] ERROR: {result.get('error', '')}")
                else:
                    print(f"  [{i}/{total}] {'OK' if ok else 'WRONG'}")
                results_with_order.append((i, result))

    results = [x[1] for x in sorted(results_with_order, key=lambda t: t[0])]

    acc = (correct / total) if total > 0 else 0.0
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode_task": task_name,
        "model_name": model_name,
        "model_request_config": model_request_config or {},
        "file": str(file_path),
        "file_name": file_path.name,
        "total": total,
        "correct": correct,
        "errors": errors,
        "accuracy": acc,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_metadata_root = repo_root / "benchmark" / "data" / "sampled_metadata"
    default_output_root = repo_root / "benchmark" / "results"

    parser = argparse.ArgumentParser(description="Unified evaluation for sampled_metadata")
    parser.add_argument("--mode", choices=VALID_MODES, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, help="Evaluate only this task (single task)")
    parser.add_argument("--sampled-metadata-root", type=str, default=str(default_metadata_root))
    parser.add_argument("--max-samples", type=int, default=-1, help="Maximum samples per file; <=0 means all")
    parser.add_argument("--max-completion-tokens", type=int, default=10000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-wait", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel samples within one file; 1 means serial")
    parser.add_argument(
        "--gpt-reasoning-effort",
        type=str,
        default="medium",
        help="GPT reasoning.effort (e.g. minimal/low/medium/high/xhigh)",
    )
    parser.add_argument(
        "--gpt-reasoning-summary",
        type=str,
        default="auto",
        help="GPT reasoning.summary (e.g. auto/concise/detailed/none)",
    )
    parser.add_argument(
        "--gpt-no-reasoning",
        action="store_true",
        help="Do not pass GPT reasoning parameters",
    )
    parser.add_argument(
        "--gpt-text-verbosity",
        type=str,
        default="low",
        help="GPT text.verbosity (e.g. low/medium/high/none)",
    )
    parser.add_argument("--output-root", type=str, default=str(default_output_root))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metadata_root = Path(args.sampled_metadata_root).resolve()
    mode_dir = metadata_root / args.mode
    if not mode_dir.exists():
        raise FileNotFoundError(f"Mode directory does not exist: {mode_dir}")

    eval_file = _find_task_file(mode_dir, args.task)

    print(f"[INFO] mode={args.mode}")
    print(f"[INFO] task={args.task}")
    print(f"[INFO] model={args.model_name}")
    print(f"[INFO] file={eval_file.name}")

    gpt_reasoning: Optional[Dict[str, str]] = None
    if "gpt" in args.model_name.lower() and not args.gpt_no_reasoning:
        gpt_reasoning = {"effort": args.gpt_reasoning_effort}
        summary = args.gpt_reasoning_summary.strip().lower()
        if summary and summary != "none":
            gpt_reasoning["summary"] = summary

    gpt_text: Optional[Dict[str, str]] = None
    if "gpt" in args.model_name.lower():
        verbosity = args.gpt_text_verbosity.strip().lower()
        if verbosity and verbosity != "none":
            gpt_text = {"verbosity": verbosity}

    if "gpt" in args.model_name.lower():
        print(f"[INFO] gpt_reasoning={gpt_reasoning}")
        print(f"[INFO] gpt_text={gpt_text}")

    client = ModelClient(
        model_name=args.model_name,
        timeout=args.timeout,
        gpt_reasoning=gpt_reasoning,
        gpt_text=gpt_text,
    )

    mode_dir_out = Path(args.output_root).resolve() / args.mode
    mode_dir_out.mkdir(parents=True, exist_ok=True)
    run_tag = _now_tag()
    model_tag = _model_name_tag(args.model_name)

    task_name = args.task.strip()
    report = evaluate_file(
        file_path=eval_file,
        task_name=task_name,
        model_name=args.model_name,
        client=client,
        model_request_config=(
            {"gpt_reasoning": client.gpt_reasoning, "gpt_text": client.gpt_text}
            if client.is_gpt
            else {}
        ),
        max_samples=args.max_samples,
        max_completion_tokens=args.max_completion_tokens,
        retries=args.retries,
        retry_wait=args.retry_wait,
        num_workers=args.num_workers,
    )

    if client.is_gpt:
        out_model_tag = f"{model_tag}-{_reasoning_tag(client.gpt_reasoning)}"
    else:
        out_model_tag = model_tag
    out_path = mode_dir_out / f"{task_name}__{out_model_tag}.json"
    _write_json(out_path, report)
    print(f"[SAVE] {out_path}")

    grand_total = report["total"]
    grand_correct = report["correct"]
    grand_errors = report["errors"]
    overall_acc = report["accuracy"]
    summary = {
        "mode": args.mode,
        "task": task_name,
        "model_name": args.model_name,
        "model_request_config": report.get("model_request_config", {}),
        "sampled_metadata_root": str(metadata_root),
        "files": [str(eval_file)],
        "file_count": 1,
        "overall": {
            "total": grand_total,
            "correct": grand_correct,
            "errors": grand_errors,
            "accuracy": overall_acc,
        },
        "per_file": [
            {
                "task": report["mode_task"],
                "file": report["file"],
                "file_name": report["file_name"],
                "total": report["total"],
                "correct": report["correct"],
                "errors": report["errors"],
                "accuracy": report["accuracy"],
            }
        ],
        "run_tag": run_tag,
        "output_mode_dir": str(mode_dir_out),
    }

    summary_path = mode_dir_out / "__summary.json"
    _write_json(summary_path, summary)
    run_meta_path = mode_dir_out / "__last_run_meta.json"
    _write_json(
        run_meta_path,
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "run_tag": run_tag,
            "mode": args.mode,
            "task": task_name,
            "model_name": args.model_name,
            "model_request_config": report.get("model_request_config", {}),
            "summary_path": str(summary_path),
            "tasks": [report["mode_task"]],
            "files": [report["file"]],
        },
    )

    print("\n====================")
    print(f"[DONE] output_dir: {mode_dir_out}")
    print(f"[DONE] total={grand_total} correct={grand_correct} errors={grand_errors} acc={overall_acc:.4f}")
    print(f"[DONE] summary: {summary_path}")


if __name__ == "__main__":
    main()
