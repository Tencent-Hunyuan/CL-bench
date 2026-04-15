---
language:
- en
license: other
task_categories:
- text-generation
pretty_name: CL-bench
size_categories:
- 1K<n<10K
tags:
- context-learning
- long-context
- benchmark
---

# CL-bench: A Benchmark for Context Learning

## Dataset Description

**CL-bench** is a benchmark for evaluating language models' context learning abilities. 


Resolving tasks in CL-bench requires models to learn from the provided context, ranging from new domain-specific knowledge, rule systems, and complex procedures to laws derived from empirical data, rather than only relying on pre-trained knowledge.


### Dataset Statistics

- **Total Samples**: 1,899 tasks
- **Format**: JSONL (one JSON object per line)
- **Context Categories**: 4 main categories with 18 sub-categories
- **Average Rubrics**: 63.2 per context
- **Average Tasks**: 3.8 per context

### Leaderboard

Visit [www.clbench.com](https://www.clbench.com) for the full leaderboard and latest results!


## Dataset Structure

### Data Fields

Each sample in the dataset contains the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `messages` | list | Multi-turn conversation in OpenAI chat format |
| `rubrics` | list | List of evaluation criteria (strings) |
| `metadata` | dict | Contains `task_id`, `context_id`, `context_category`, `sub_category` |

#### `messages` Field

The `messages` field follows the standard OpenAI chat format:

```json
[
  {"role": "system", "content": "system prompt"},
  {"role": "user", "content": "context and task"}
]
```

#### `rubrics` Field

A list of strings, each describing a specific evaluation rubric.


#### `metadata` Field

```json
{
  "task_id": "unique identifier for task",
  "context_id": "unique identifier for context",
  "context_category": "Rule System Application",
  "sub_category": "Game Mechanics"
}
```

- **task_id**: Unique identifier for the task
- **context_id**: Unique identifier for the context
- **context_category**: One of the 4 main categories
- **sub_category**: Fine-grained classification (18 sub-categories total)



## Usage

Please see our **GitHub repository**: [github.com/Tencent-Hunyuan/CL-bench](https://github.com/Tencent-Hunyuan/CL-bench)


## License


CL-Bench is released under a **custom evaluation-only license**.

Permission is hereby granted, free of charge, to any person obtaining a copy of this dataset and associated documentation files (the "Dataset"), to use, copy, modify, merge, publish, and distribute the Dataset **solely for the purposes of evaluation, testing, and benchmarking of models**.

The Dataset (or any portion thereof) **must not** be used for training, fine-tuning, calibrating, distilling, adapting, or any form of parameter updating.

Please refer to the LICENSE file for the full license text.



## Citation

If you find our work useful, please cite it as follows:

```bibtex
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