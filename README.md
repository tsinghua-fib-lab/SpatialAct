# SpatialAct

This repo is for SpatialAct: Probing Spatial Reasoning-to-Action Capabilities of VLM Agents in 3D Scenes

## Introduction

Humans can effortlessly perceive spatial layouts, form cognitive representations, reason about spatial relations, and translate such reasoning into actions in everyday 3D environments. Although recent vision-language models (VLMs) have shown promising performance on observation-conditioned spatial perception and reasoning tasks, it remains unclear whether they can build coherent spatial understanding, act upon it, and refine their actions through multi-turn feedback. To study this problem, we introduce $\textbf{SpatialAct}$, a simulator-grounded benchmark for probing $\textit{action-conditioned spatial reasoning}$ in 3D scenes. Starting from the most challenging setting, Multi-turn Interactive Refinement, we further design its decomposed counterpart, Single-step Error Detection and Fix, together with five fundamental spatial ability tasks to diagnose the underlying causes of model failures. Experiments reveal a clear reasoning-to-action gap: current VLMs can perform well on isolated spatial reasoning tasks, but struggle to maintain coherent spatial beliefs and produce reliable actions during multi-turn feedback, substantially underperforming humans. These results suggest that current VLM agents still lack robust spatial state tracking under action-induced environment changes, even when low-level control is abstracted away.

## Repository Structure

```text
SpatialAct/
├── data/
│   ├── architectural/
│   ├── geometry/
│   └── indoor/
├── data_construct/
│   ├── single_step/
│   ├── spatial_relation/
│   ├── spatial_orientation/
│   ├── spatial_visualization/
│   ├── mental_rotation/
│   ├── multi-turn/
│   └── object_meaning/
├── evaluate/
│   ├── single_turn/
│   └── multi-turn/
└── README.md
```

## Data

Main task files include:

- `spatial_relation.json`
- `spatial_orientation.json`
- `spatial_visualization.json`
- `mental_rotation.json`
- `single_step.json` (Single-step Error Detection and Fix)
- `multi_turn.json` (Multi-turn Interactive Refinement)
- `object_meaning.json` (only in `data/indoor/`)

👉 All data files, including the **scene** source files and corresponding **images**, are hosted on the [https://huggingface.co/datasets/Tianhui-Liu/SpatialAct](https://huggingface.co/datasets/Tianhui-Liu/SpatialAct).

## Data Construction

`data_construct/` contains scripts for building data by task type:

- single-step tasks: `data_construct/single_step/`
- spatial relation: `data_construct/spatial_relation/`
- spatial orientation: `data_construct/spatial_orientation/`
- spatial visualization: `data_construct/spatial_visualization/`
- mental rotation: `data_construct/mental_rotation/`
- multi-turn tasks: `data_construct/multi-turn/`
- object meaning (indoor): `data_construct/object_meaning/`

Naming conventions commonly used in scripts:

- `indoor_*`: indoor-domain pipeline
- `region_*`: architectural-domain pipeline
- generic names: geometry pipeline

## Evaluation

### Single-turn Evaluation

Main script: `evaluate/single_turn/eval.py`
Metrics script: `evaluate/single_turn/metrics.py`

Example:

```bash
python evaluate/single_turn/eval.py \
  --mode indoor \
  --task spatial_relation \
  --model-name gpt-5.4
```

### Multi-turn Evaluation

Main scripts:

- `evaluate/multi-turn/mllm_iteration_standalone.py`
- `evaluate/multi-turn/final_only_metrics_runner.py`
- `evaluate/multi-turn/metrics.py`
- `evaluate/multi-turn/indoor_metrics.py`
- `evaluate/multi-turn/blender_iter_renderer.py`

Multi-turn evaluation involves iterative scene rendering and final metric computation, and typically requires Blender and corresponding scene assets.

Example (indoor):

```bash
LLM_MODEL=gpt-5.4 \
SCENE_TYPE=indoor \
bash evaluate/multi-turn/run_iteration_common.sh
```

Frequently used environment variables:

- `LLM_MODEL`: model name
- `SCENE_TYPE`: `indoor`
- `MAX_ITERATIONS`: max multi-turn correction rounds
- `REGION_WORKERS`, `BLENDER_WORKERS`, `METRICS_WORKERS`: parallelism controls
- `SAMPLED_METADATA_JSON`: optional explicit sampled metadata file override

## Citation

If you find this work helpful, please cite our paper.

```bibtex
@article{liu2026spatialact,
  title={SpatialAct: Probing Spatial Reasoning-to-Action Capabilities of VLM Agents in 3D Scenes},
  author={Liu, Tianhui and Feng, Jie and Zheng, Zhiheng and Wang, Shengyuan and Guo, Yiming and Xi, Yanxin and Fan, Hangyu and Li, Yong and Hui, Pan},
  journal={arXiv preprint arXiv:2605.31148},
  year={2026}
}
```

## Contact

If you have any questions or want to use the code, feel free to contact Tianhui Liu ([tianhuiliu06@gmail.com](mailto:tianhuiliu06@gmail.com)).
