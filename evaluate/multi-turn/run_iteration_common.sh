#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
METRICS_SCRIPT="$SCRIPT_DIR/metrics.py"
FINAL_ONLY_METRICS_RUNNER="$SCRIPT_DIR/final_only_metrics_runner.py"

# ==============================
# Parameters (edit here)
# ==============================
STEPS_INPUT="5"                         # e.g. 3,5,7
SCENE_TYPE="region"                     # region | indoor | region_basic | region_complex
REGION_MODE="complex"                   # basic | complex (region)
INDOOR_MODE="all"                       # indoor always uses all
SOURCE_REGION_INPUT=""                  # optional CSV, empty means no source filter
SOURCE_STEP_INPUT=""                    # empty means follow current step
REGIONS=""                              # empty means all; e.g. 1 or 1,2,3
TASK_JSON_OVERRIDE=""                   # optional task JSON; empty means auto data/*/multi_turn.json
SAMPLED_METADATA_JSON_OVERRIDE=""       # optional sampled/task JSON override

MAX_ITERATIONS="30"
REGION_WORKERS="8"
BLENDER_WORKERS="8"
METRICS_WORKERS="0"
REGION_FORCE_LOCAL_BLEND_PIPELINE="1"
RUN_METRICS_PER_REGION="1"
RUN_SHARED_FINAL_METRICS="0"
RUN_METRICS_AFTER_ALL="1"
SAVE_REGION_FINAL_BLEND="1"
CLEANUP_FINAL_BLEND_AFTER_METRICS="0"
METRICS_FINAL_ONLY="1"
METRICS_AFTER_USE_GLOBAL_ERROR_SCENE="1"
RESUME_INCOMPLETE="0"
SKIP_EXISTING_FINAL_METRICS="0"

LLM_MODEL="gpt-5.4"
LLM_PROVIDER="auto"                     # auto | gpt | openrouter
LLM_MAX_COMPLETION_TOKENS="8096"
LLM_TIMEOUT="300"
LLM_MAX_RETRIES="2"
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
AZURE_ENDPOINT=""
AZURE_API_VERSION=""
API_KEY_ENV="Siliconflow_KEY"           # OpenRouter fallback API key env name

GPT_REASONING_MODE="on"                 # on | off
GPT_REASONING_EFFORT="medium"
GPT_REASONING_SUMMARY="auto"

IMAGE_URL_CONVERT_ENDPOINT=""
IMAGE_URL_CONVERT_TIMEOUT="20"
IMAGE_URL_EXPIRES_IN="3600"
IMAGE_URL_PUBLIC_BASE_URL=""

WORKSPACE_ROOT_BASE=""
RUN_NAME=""
WORKSPACE_ROOT=""
LOG_PATH=""

if [[ "$SCENE_TYPE" == "indoor" ]]; then
  UNIT_SCALE="1.0"
else
  UNIT_SCALE="20.0"
fi
GLTF_PATH="$PROJECT_ROOT/data/geometry/combined.gltf"

# Backward-compatible aliases
if [[ "$SCENE_TYPE" == "region_basic" ]]; then
  SCENE_TYPE="region"
  REGION_MODE="basic"
elif [[ "$SCENE_TYPE" == "region_complex" ]]; then
  SCENE_TYPE="region"
  REGION_MODE="complex"
fi

if [[ "$SCENE_TYPE" != "region" && "$SCENE_TYPE" != "indoor" ]]; then
  echo "[ERROR] Invalid SCENE_TYPE: $SCENE_TYPE"
  echo "Allowed values: region | indoor | region_basic | region_complex"
  exit 1
fi
if [[ "$SCENE_TYPE" == "region" ]]; then
  if [[ "$REGION_MODE" != "basic" && "$REGION_MODE" != "complex" ]]; then
    echo "[ERROR] Invalid REGION_MODE: $REGION_MODE"
    echo "Allowed values: basic | complex"
    exit 1
  fi
else
  REGION_MODE="indoor"
  if [[ "$INDOOR_MODE" != "all" ]]; then
    echo "[INFO] SCENE_TYPE=indoor: ignore INDOOR_MODE=$INDOOR_MODE and force all"
  fi
  INDOOR_MODE="all"
fi

if [[ "$SCENE_TYPE" == "region" ]]; then
  if [[ "$REGION_FORCE_LOCAL_BLEND_PIPELINE" == "1" || "$REGION_FORCE_LOCAL_BLEND_PIPELINE" == "true" || "$REGION_FORCE_LOCAL_BLEND_PIPELINE" == "TRUE" ]]; then
    RUN_SHARED_FINAL_METRICS="0"
    RUN_METRICS_PER_REGION="1"
    SAVE_REGION_FINAL_BLEND="1"
    CLEANUP_FINAL_BLEND_AFTER_METRICS="0"
    echo "[INFO] REGION_FORCE_LOCAL_BLEND_PIPELINE=1: use per-region blend pipeline (disable shared metrics and keep per-region final_scene.blend)"
  fi
fi

if [[ "$METRICS_FINAL_ONLY" == "1" || "$METRICS_FINAL_ONLY" == "true" || "$METRICS_FINAL_ONLY" == "TRUE" ]]; then
  RUN_METRICS_PER_REGION="0"
  RUN_SHARED_FINAL_METRICS="0"
  SAVE_REGION_FINAL_BLEND="1"
  CLEANUP_FINAL_BLEND_AFTER_METRICS="0"
  echo "[INFO] METRICS_FINAL_ONLY=1: disable per-region before/after metrics and run final_scene metrics only at the end"
fi

split_csv_list() {
  local raw="$1"
  local -n out_ref=$2
  out_ref=()

  raw="${raw//;/,}"
  raw="${raw// /,}"
  IFS=',' read -r -a _tmp <<< "$raw"
  for x in "${_tmp[@]}"; do
    x="$(echo "$x" | xargs)"
    [[ -z "$x" ]] && continue
    out_ref+=("$x")
  done
}

resolve_sampled_metadata() {
  local scene_type="$1"
  local region_mode="$2"
  local steps="$3"
  local source_region="$4"
  local source_step="$5"
  local indoor_mode="$6"

  if [[ -n "$TASK_JSON_OVERRIDE" ]]; then
    echo "$TASK_JSON_OVERRIDE"
    return
  fi

  # Open-source task JSONs carry the per-sample scene path directly
  # (`error_scene` for architectural, `error_scene_glb` for indoor).
  if [[ "$scene_type" == "indoor" ]]; then
    local task_json="$REPO_ROOT/data/indoor/multi_turn.json"
    if [[ -f "$task_json" ]]; then
      echo "$task_json"
      return
    fi
    echo "$task_json"
    return
  elif [[ "$scene_type" == "region" ]]; then
    local task_json="$REPO_ROOT/data/architectural/multi_turn.json"
    if [[ -f "$task_json" ]]; then
      echo "$task_json"
      return
    fi
    echo "$task_json"
    return
  fi

  echo ""
}

run_one_combo() {
  local step="$1"
  local source_region="$2"
  local source_step="$3"

  local sampled_json=""
  if [[ -n "$SAMPLED_METADATA_JSON_OVERRIDE" ]]; then
    sampled_json="$SAMPLED_METADATA_JSON_OVERRIDE"
  else
    sampled_json="$(resolve_sampled_metadata "$SCENE_TYPE" "$REGION_MODE" "$step" "$source_region" "$source_step" "$INDOOR_MODE")"
  fi

  if [[ ! -f "$sampled_json" ]]; then
    echo "[ERROR] sampled metadata file does not exist: $sampled_json"
    return 1
  fi

  local model_tag
  model_tag="$(echo "$LLM_MODEL" | sed -E 's/[^A-Za-z0-9._-]+/_/g')"
  local max_tokens_tag
  max_tokens_tag="$(echo "$LLM_MAX_COMPLETION_TOKENS" | sed -E 's/[^A-Za-z0-9._-]+/_/g')"
  [[ -z "$max_tokens_tag" ]] && max_tokens_tag="8096"

  local effective_provider="$LLM_PROVIDER"
  if [[ "$effective_provider" == "auto" ]]; then
    if [[ "$LLM_MODEL" == *gpt* || "$LLM_MODEL" == *GPT* ]]; then
      effective_provider="gpt"
    else
      effective_provider="openrouter"
    fi
  fi

  local gpt_reasoning_mode_lc
  gpt_reasoning_mode_lc="$(echo "$GPT_REASONING_MODE" | tr '[:upper:]' '[:lower:]')"
  local gpt_reasoning_enabled="0"
  if [[ "$gpt_reasoning_mode_lc" == "on" || "$gpt_reasoning_mode_lc" == "1" || "$gpt_reasoning_mode_lc" == "true" || "$gpt_reasoning_mode_lc" == "yes" ]]; then
    gpt_reasoning_enabled="1"
  fi

  local reasoning_suffix=""
  if [[ "$effective_provider" == "gpt" ]]; then
    local reasoning_tag="noreasoning"
    if [[ "$gpt_reasoning_enabled" == "1" ]]; then
      local reasoning_effort_tag reasoning_summary_tag
      reasoning_effort_tag="$(echo "$GPT_REASONING_EFFORT" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^A-Za-z0-9._-]+/-/g; s/^-+//; s/-+$//')"
      reasoning_summary_tag="$(echo "$GPT_REASONING_SUMMARY" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^A-Za-z0-9._-]+/-/g; s/^-+//; s/-+$//')"
      [[ -z "$reasoning_effort_tag" ]] && reasoning_effort_tag="medium"
      [[ -z "$reasoning_summary_tag" ]] && reasoning_summary_tag="auto"
      reasoning_tag="reason-effort-${reasoning_effort_tag}-summary-${reasoning_summary_tag}"
    fi
    reasoning_suffix="_${reasoning_tag}"
  fi

  local workspace_root_base
  if [[ -n "$WORKSPACE_ROOT_BASE" ]]; then
    workspace_root_base="$WORKSPACE_ROOT_BASE"
  elif [[ -n "$SAMPLED_METADATA_JSON_OVERRIDE" ]]; then
    workspace_root_base="$PROJECT_ROOT/benchmark/results/mllm_iteration_context_window"
  else
    workspace_root_base="$PROJECT_ROOT/benchmark/results/mllm_iteration"
  fi

  local run_name
  if [[ "$SCENE_TYPE" == "region" ]]; then
    if [[ -n "$RUN_NAME" ]]; then
      run_name="$RUN_NAME"
    else
      run_name="sampled_${REGION_MODE}_steps${step}_${model_tag}${reasoning_suffix}_maxct${max_tokens_tag}"
    fi
    local ws_base="$workspace_root_base/region_${REGION_MODE}"
    local source_region_tag
    source_region_tag="$(echo "$source_region" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+//; s/_+$//')"
    if [[ -n "$WORKSPACE_ROOT" ]]; then
      workspace_root="$WORKSPACE_ROOT"
    elif [[ -n "$source_region_tag" ]]; then
      workspace_root="$ws_base/$run_name/$source_region_tag"
    else
      workspace_root="$ws_base/$run_name"
    fi
  else
    if [[ -n "$RUN_NAME" ]]; then
      run_name="$RUN_NAME"
    else
      run_name="sampled_indoor_${model_tag}${reasoning_suffix}_maxct${max_tokens_tag}"
    fi
    local ws_base="$workspace_root_base/indoor"
    if [[ -n "$WORKSPACE_ROOT" ]]; then
      workspace_root="$WORKSPACE_ROOT"
    else
      workspace_root="$ws_base/$run_name"
    fi
  fi

  mkdir -p "$workspace_root"
  local log_path="$LOG_PATH"
  if [[ -z "$log_path" ]]; then
    log_path="$workspace_root/run_iteration.log"
  fi
  local lock_path="$workspace_root/.run_iteration.lock"

  if [[ "$effective_provider" == "gpt" ]]; then
    if [[ -z "${AZURE_API_KEY:-}" ]]; then
      echo "[ERROR] Missing API key env var: AZURE_API_KEY"
      return 1
    fi
  else
    if [[ -z "${OPENROUTER_API_KEY:-}" && -z "${!API_KEY_ENV:-}" ]]; then
      echo "[ERROR] Missing OpenRouter API key. Set OPENROUTER_API_KEY (recommended) or $API_KEY_ENV"
      return 1
    fi
  fi

  echo "[INFO] ===== combo begin ====="
  local source_region_display="$source_region"
  if [[ -z "$source_region_display" ]]; then
    source_region_display="<none>"
  fi
  echo "[INFO] STEP=$step SOURCE_REGION=$source_region_display SOURCE_STEP=$source_step"
  echo "[INFO] SCENE_TYPE=$SCENE_TYPE REGION_MODE=$REGION_MODE INDOOR_MODE=$INDOOR_MODE"
  echo "[INFO] LLM_MODEL=$LLM_MODEL LLM_PROVIDER=$LLM_PROVIDER (effective: $effective_provider)"
  echo "[INFO] LLM_MAX_COMPLETION_TOKENS=$LLM_MAX_COMPLETION_TOKENS"
  if [[ "$effective_provider" == "gpt" ]]; then
    echo "[INFO] GPT_REASONING_MODE=$GPT_REASONING_MODE (enabled=$gpt_reasoning_enabled)"
    if [[ "$gpt_reasoning_enabled" == "1" ]]; then
      echo "[INFO] GPT_REASONING_EFFORT=$GPT_REASONING_EFFORT GPT_REASONING_SUMMARY=$GPT_REASONING_SUMMARY"
    fi
  fi
  echo "[INFO] SAMPLED_METADATA_JSON=$sampled_json"
  echo "[INFO] WORKSPACE_ROOT=$workspace_root"

  local cmd=(
    python "$SCRIPT_DIR/mllm_iteration_standalone.py"
    --scene-type "$SCENE_TYPE"
    --indoor-mode "$INDOOR_MODE"
    --region-mode "$REGION_MODE"
    --sampled-metadata-json "$sampled_json"
    --workspace-root "$workspace_root"
    --regions "$REGIONS"
    --max-iterations "$MAX_ITERATIONS"
    --region-workers "$REGION_WORKERS"
    --blender-workers "$BLENDER_WORKERS"
    --unit-scale "$UNIT_SCALE"
    --gltf-path "$GLTF_PATH"
    --llm-model "$LLM_MODEL"
    --llm-provider "$LLM_PROVIDER"
    --openrouter-base-url "$OPENROUTER_BASE_URL"
    --azure-endpoint "$AZURE_ENDPOINT"
    --azure-api-version "$AZURE_API_VERSION"
    --api-key-env "$API_KEY_ENV"
    --llm-timeout "$LLM_TIMEOUT"
    --llm-max-retries "$LLM_MAX_RETRIES"
    --llm-max-completion-tokens "$LLM_MAX_COMPLETION_TOKENS"
    --image-url-convert-endpoint "$IMAGE_URL_CONVERT_ENDPOINT"
    --image-url-convert-timeout "$IMAGE_URL_CONVERT_TIMEOUT"
    --image-url-expires-in "$IMAGE_URL_EXPIRES_IN"
    --image-url-public-base-url "$IMAGE_URL_PUBLIC_BASE_URL"
    --steps "$step"
    --metrics-workers "$METRICS_WORKERS"
  )

  if [[ "$RUN_METRICS_PER_REGION" == "1" || "$RUN_METRICS_PER_REGION" == "true" || "$RUN_METRICS_PER_REGION" == "TRUE" ]]; then
    cmd+=(--run-metrics-per-region)
  else
    cmd+=(--no-run-metrics-per-region)
  fi

  if [[ "$RESUME_INCOMPLETE" == "1" || "$RESUME_INCOMPLETE" == "true" || "$RESUME_INCOMPLETE" == "TRUE" ]]; then
    cmd+=(--resume-incomplete)
  else
    cmd+=(--no-resume-incomplete)
  fi

  if [[ "$CLEANUP_FINAL_BLEND_AFTER_METRICS" == "1" || "$CLEANUP_FINAL_BLEND_AFTER_METRICS" == "true" || "$CLEANUP_FINAL_BLEND_AFTER_METRICS" == "TRUE" ]]; then
    cmd+=(--cleanup-final-blend-after-metrics)
  else
    cmd+=(--no-cleanup-final-blend-after-metrics)
  fi

  if [[ "$RUN_SHARED_FINAL_METRICS" == "1" || "$RUN_SHARED_FINAL_METRICS" == "true" || "$RUN_SHARED_FINAL_METRICS" == "TRUE" ]]; then
    cmd+=(--run-shared-final-metrics)
  else
    cmd+=(--no-run-shared-final-metrics)
  fi

  if [[ "$SAVE_REGION_FINAL_BLEND" == "1" || "$SAVE_REGION_FINAL_BLEND" == "true" || "$SAVE_REGION_FINAL_BLEND" == "TRUE" ]]; then
    cmd+=(--save-region-final-blend)
  else
    cmd+=(--no-save-region-final-blend)
  fi

  if [[ "$effective_provider" == "gpt" ]]; then
    if [[ "$gpt_reasoning_enabled" == "1" ]]; then
      cmd+=(--gpt-reasoning-effort "$GPT_REASONING_EFFORT")
      cmd+=(--gpt-reasoning-summary "$GPT_REASONING_SUMMARY")
    else
      cmd+=(--gpt-no-reasoning)
    fi
  fi

  echo "[RUN] ${cmd[*]}"
  cd "$PROJECT_ROOT"

  exec 9>"$lock_path"
  if ! flock -n 9; then
    echo "[ERROR] Another process is already writing to this WORKSPACE_ROOT: $workspace_root"
    return 1
  fi

  if [[ -n "$log_path" ]]; then
    mkdir -p "$(dirname "$log_path")"
    "${cmd[@]}" 2>&1 | tee -a "$log_path"
  else
    "${cmd[@]}"
  fi

  if [[ "$RUN_METRICS_AFTER_ALL" == "1" || "$RUN_METRICS_AFTER_ALL" == "true" || "$RUN_METRICS_AFTER_ALL" == "TRUE" ]]; then
    if [[ ! -f "$METRICS_SCRIPT" ]]; then
      echo "[ERROR] Metrics script not found: $METRICS_SCRIPT"
      return 1
    fi

    if [[ "$METRICS_FINAL_ONLY" == "1" || "$METRICS_FINAL_ONLY" == "true" || "$METRICS_FINAL_ONLY" == "TRUE" ]]; then
      local final_only_out="$workspace_root/metrics_final_only_steps${step}.json"
      if [[ "$SKIP_EXISTING_FINAL_METRICS" == "1" || "$SKIP_EXISTING_FINAL_METRICS" == "true" || "$SKIP_EXISTING_FINAL_METRICS" == "TRUE" ]]; then
        if [[ -s "$final_only_out" ]]; then
          echo "[SKIP] final-only metrics already exists: $final_only_out"
          echo "[INFO] ===== combo done ====="
          return 0
        fi
      fi
      echo "[RUN] final-only metrics -> $final_only_out"
      if [[ ! -f "$FINAL_ONLY_METRICS_RUNNER" ]]; then
        echo "[ERROR] Final-only metrics runner not found: $FINAL_ONLY_METRICS_RUNNER"
        return 1
      fi
      local -a final_only_cmd=(
        python "$FINAL_ONLY_METRICS_RUNNER"
        --scene-type "$SCENE_TYPE"
        --workspace-root "$workspace_root"
        --step "$step"
        --metrics-script "$METRICS_SCRIPT"
        --out-path "$final_only_out"
        --blender-bin "$PROJECT_ROOT/blender-3.2.2-linux-x64/blender"
        --blender-script "$SCRIPT_DIR/blender_iter_renderer.py"
        --project-root "$PROJECT_ROOT"
        --unit-scale "$UNIT_SCALE"
        --region-mode "$REGION_MODE"
        --metrics-workers "$METRICS_WORKERS"
        --gltf-fallback "$GLTF_PATH"
      )
      if [[ "$SCENE_TYPE" == "region" && ( "$METRICS_AFTER_USE_GLOBAL_ERROR_SCENE" == "1" || "$METRICS_AFTER_USE_GLOBAL_ERROR_SCENE" == "true" || "$METRICS_AFTER_USE_GLOBAL_ERROR_SCENE" == "TRUE" ) ]]; then
        final_only_cmd+=(--use-global-after)
      fi
      if [[ -n "$log_path" ]]; then
        "${final_only_cmd[@]}" 2>&1 | tee -a "$log_path"
      else
        "${final_only_cmd[@]}"
      fi

      echo "[INFO] ===== combo done ====="
      return 0
    fi

    local metrics_mode="region"
    if [[ "$SCENE_TYPE" == "indoor" ]]; then
      metrics_mode="indoor"
    fi

    local metrics_cmd=(
      python "$METRICS_SCRIPT"
      --mode "$metrics_mode"
      --workspace-root "$workspace_root"
      --steps "$step"
    )

    if [[ "$SCENE_TYPE" == "region" ]]; then
      if [[ "$RUN_SHARED_FINAL_METRICS" == "1" || "$RUN_SHARED_FINAL_METRICS" == "true" || "$RUN_SHARED_FINAL_METRICS" == "TRUE" ]]; then
        if [[ "$SAVE_REGION_FINAL_BLEND" == "0" || "$SAVE_REGION_FINAL_BLEND" == "false" || "$SAVE_REGION_FINAL_BLEND" == "FALSE" ]]; then
          local shared_blend_dir="$workspace_root/_shared_final_scenes"
          if [[ -d "$shared_blend_dir" ]]; then
            mapfile -t shared_blend_list < <(find "$shared_blend_dir" -type f -name "final_scene*.blend" | sort)
            if [[ "${#shared_blend_list[@]}" -eq 1 ]]; then
              metrics_cmd+=(--after-blend-path "${shared_blend_list[0]}")
            fi
          fi
        fi
      fi
    fi

    echo "[RUN] ${metrics_cmd[*]}"
    if [[ -n "$log_path" ]]; then
      "${metrics_cmd[@]}" 2>&1 | tee -a "$log_path"
    else
      "${metrics_cmd[@]}"
    fi
  fi

  echo "[INFO] ===== combo done ====="
}

main() {
  local -a step_list=()
  local -a source_region_list=()

  if [[ "$SCENE_TYPE" == "indoor" ]]; then
    local one_step="$SOURCE_STEP_INPUT"
    if [[ -z "$one_step" ]]; then
      one_step="3"
    fi
    run_one_combo "$one_step" "" "$one_step"
    return
  fi

  if [[ -n "$SAMPLED_METADATA_JSON_OVERRIDE" ]]; then
    split_csv_list "$STEPS_INPUT" step_list
    if [[ "${#step_list[@]}" -eq 0 ]]; then
      echo "[ERROR] STEPS cannot be empty"
      exit 1
    fi
    # If sampled metadata is explicitly provided, run only once (first step/source region).
    local one_step="${step_list[0]}"
    local one_region=""
    split_csv_list "$SOURCE_REGION_INPUT" source_region_list
    if [[ "${#source_region_list[@]}" -gt 0 ]]; then
      one_region="${source_region_list[0]}"
    fi
    local one_source_step="$SOURCE_STEP_INPUT"
    if [[ -z "$one_source_step" ]]; then
      one_source_step="$one_step"
    fi
    run_one_combo "$one_step" "$one_region" "$one_source_step"
    return
  fi

  split_csv_list "$STEPS_INPUT" step_list
  if [[ "${#step_list[@]}" -eq 0 ]]; then
    echo "[ERROR] STEPS cannot be empty (e.g. STEPS=3,5,7)"
    exit 1
  fi

  if [[ "$SCENE_TYPE" == "region" ]]; then
    split_csv_list "$SOURCE_REGION_INPUT" source_region_list
    if [[ "${#source_region_list[@]}" -eq 0 ]]; then
      source_region_list=("")
    fi
  else
    source_region_list=("")
  fi

  local source_regions_display="$SOURCE_REGION_INPUT"
  if [[ -z "$source_regions_display" ]]; then
    source_regions_display="<none>"
  fi
  echo "[INFO] Serial combos: steps=${STEPS_INPUT}, source_regions=${source_regions_display}"

  for step in "${step_list[@]}"; do
    for src_region in "${source_region_list[@]}"; do
      local src_step="$SOURCE_STEP_INPUT"
      if [[ -z "$src_step" ]]; then
        src_step="$step"
      fi
      run_one_combo "$step" "$src_region" "$src_step"
    done
  done
}

main "$@"
