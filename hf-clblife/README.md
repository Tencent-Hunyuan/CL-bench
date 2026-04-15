---
language:
- en
license: other
task_categories:
- text-generation
pretty_name: CLBench-life
size_categories:
- n<1K
tags:
- context-learning
- long-context
- benchmark
- real-life
---

# CLBench-life: Can Language Models Learn from Real-Life Context?

## Dataset Description

**CLBench-life** extends context learning evaluation to real-life scenarios. Unlike professional/domain-specific benchmarks, CLBench-life contexts are messy, fragmented, and grounded in everyday experience, reflecting the kind of data people actually deal with daily.

CLBench-life is part of the **CL-bench family** of benchmarks for context learning.

### Dataset Statistics

- **Total Samples**: 405 context-task pairs
- **Total Rubrics**: 5,348 (avg. 13.2 per task)
- **Format**: JSONL (one JSON object per line)
- **Context Categories**: 3 main categories with 9 sub-categories

### Leaderboard

Visit [www.clbench.com](https://www.clbench.com) for the full leaderboard and latest results!


## Dataset Structure

### Data Fields

Each sample in the dataset contains the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `messages` | list | Multi-turn conversation in OpenAI chat format |
| `rubrics` | list | List of evaluation criteria (strings) |
| `metadata` | dict | Contains `task_id`, `context_category`, `context_subcategory` |

#### `messages` Field

The `messages` field follows the standard OpenAI chat format:

- **Single-turn**: The context and task are in one message, separated by a `<|TASK|>` delimiter.
- **Multi-turn**: The task is the final user message; earlier turns provide context.

Single-turn example:
```json
[
  {"role": "user", "content": "<context>\n<|TASK|> <task>"}
]
```

Multi-turn example:
```json
[
  {"role": "user", "content": "context and task"},
  {"role": "assistant", "content": "..."},
  {"role": "user", "content": "task"}
]
```

#### `rubrics` Field

A list of strings, each describing a specific evaluation rubric.


#### `metadata` Field

```json
{
  "task_id": "unique identifier for task",
  "context_category": "Communication & Social Interactions",
  "context_subcategory": "Group Conversations & Meeting Transcripts"
}
```

- **task_id**: Unique identifier for the task
- **context_category**: One of the 3 main categories
- **context_subcategory**: Fine-grained classification (9 sub-categories total)

### Context Categories

| Category | Sub-categories | Tasks |
|----------|---------------|-------|
| **Communication & Social Interactions** | Group Conversations & Meeting Transcripts, Private Conversations, Community Interactions | 135 |
| **Fragmented Information & Revisions** | Personal Information Fragments, Public Information Fragments, Creation & Revision Histories | 135 |
| **Behavioral Records & Activity Trails** | Game Logs, Digital Footprints & Daily-Life Records, Self-Tracking Trajectories | 135 |


## Usage

Please see our **GitHub repository**: [github.com/Tencent-Hunyuan/CL-bench](https://github.com/Tencent-Hunyuan/CL-bench)

### Quick Start

```bash
pip install openai tqdm
```

## License


CL-Bench is released under a **custom evaluation-only license**.

Permission is hereby granted, free of charge, to any person obtaining a copy of this dataset and associated documentation files (the "Dataset"), to use, copy, modify, merge, publish, and distribute the Dataset **solely for the purposes of evaluation, testing, and benchmarking of models**.

The Dataset (or any portion thereof) **must not** be used for training, fine-tuning, calibrating, distilling, adapting, or any form of parameter updating.

Please refer to the LICENSE file for the full license text.


## Citation

If you find our work useful, please cite it as follows:

```bibtex
@article{clbenchlife2025,
      title={CLBench-life: Can Language Models Learn from Real-Life Context?},
      author={},
      year={2025},
}

@misc{dou2026clbenchbenchmarkcontextlearning,
      title={CL-bench: A Benchmark for Context Learning}, 
      author={Shihan Dou and Ming Zhang and Zhangyue Yin and Chenhao Huang and Yujiong Shen and Junzhe Wang and Jiayi Chen and Yuchen Ni and Junjie Ye and Cheng Zhang and Huaibing Xie and Jianglu Hu and Shaolei Wang and Weichao Wang and Yanling Xiao and Yiting Liu and Zenan Xu and Zhen Guo and Pluto Zhou and Tao Gui and Zuxuan Wu and Xipeng Qiu and Qi Zhang and Xuanjing Huang and Yu-Gang Jiang and Di Wang and Shunyu Yao},
      year={2026},
      eprint={2602.03587},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.03587}, 
}
```
