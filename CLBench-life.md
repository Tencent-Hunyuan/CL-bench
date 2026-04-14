[← Back to Hub](README.md)

<div align="center">
<img src="assets/hunyuan.png" alt="Tencent-Hunyuan" width="150"/>
</div>

<div align="center">
<h1>CLBench-life: Can Language Models Learn from Real-Life Context?</h1>

[![Paper](https://img.shields.io/badge/Paper-PDF-blue.svg?style=for-the-badge)](clbench-life-paper.pdf)
[![HuggingFace](https://img.shields.io/badge/Data-HF-yellow.svg?style=for-the-badge)](https://huggingface.co/datasets/tencent/CLBench-life)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-red.svg?style=for-the-badge)](https://www.clbench.com)

</div>

CLBench-life is a benchmark for evaluating context learning in real-life scenarios. It contains 500 human-curated context-task pairs and 5,348 verification rubrics covering common everyday situations. Unlike [CL-bench](CL-bench.md), which focuses on professional and domain-specific contexts, CLBench-life targets the kind of messy, fragmented contexts people actually encounter in daily life — group chats, personal notes, browsing histories, game logs, and so on.

All contexts are self-contained: models must learn from what is given, without retrieving external information. Even the best model (GPT-5.4) achieves only 19.3% solving rate, and the average across 10 frontier models is 13%.

<!--
<p align="center">
  <img src="assets/life-task.png" alt="CLBench-life Overview" width="80%">
</p>
-->

## Context Categories

CLBench-life covers 3 categories and 9 sub-categories:

**Communication & Social Interactions**
- Community Interactions — public threads, forums
- Group Conversations & Meeting Transcripts — multi-party chats, meeting records
- Private Conversations — one-on-one chats

**Fragmented Information & Revisions**
- Personal Information Fragments — personal notes, private archives
- Public Information Fragments — news feeds, public data streams
- Creation & Revision Histories — document edits, version logs

**Behavioral Records & Activity Trails**
- Game Logs — match replays, poker hands, strategy logs
- Digital Footprints & Daily-Life Records — browsing histories, transactions, location logs
- Self-Tracking Trajectories — fitness logs, health records, learning progress

## Dataset

500 context-task pairs, 5,348 rubrics (avg. 10.7 per task), JSONL format.

Available on [HuggingFace](https://huggingface.co/datasets/tencent/CLBench-life). Data structure is identical to CL-bench:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "rubrics": ["Rubric 1", "Rubric 2", "..."],
  "metadata": {
    "task_id": "unique-task-identifier",
    "context_id": "unique-context-identifier",
    "context_category": "..."
  }
}
```

## Results

Task solving rate (%) of ten frontier LMs, reasoning mode, mean±std across three runs:

| Model | Overall | Comm. & Social | Frag. Info & Rev. | Behav. Records |
|-------|---------|----------------|-------------------|----------------|
| GPT-5.4 (High) | **19.3**±0.5 | **22.2**±0.7 | 15.8±0.9 | **20.0**±0.7 |
| Claude Opus 4.6 (High) | 17.0±1.3 | 21.7±3.3 | **16.5**±0.9 | 12.8±1.5 |
| Gemini 3.1 Pro (High) | 16.9±1.2 | 18.8±1.7 | 13.6±1.9 | 18.3±1.5 |
| Seed 2.0 Pro (High) | 15.5±1.7 | 19.8±2.3 | 12.1±2.4 | 14.6±1.1 |
| Kimi K2.5 (High) | 13.2±0.8 | 15.3±1.1 | 11.4±1.5 | 12.8±0.4 |
| Qwen 3.5 Plus (High) | 12.4±0.1 | 13.1±0.9 | 10.1±1.1 | 14.1±0.7 |
| Grok 4.20 (High) | 11.9±0.4 | 12.8±0.9 | 11.9±1.5 | 10.9±2.6 |
| DeepSeek V3.2 Thinking | 9.5±0.4 | 12.3±1.1 | 7.7±1.1 | 8.6±1.7 |
| Hunyuan 2.0 Thinking | 8.4±0.7 | 9.4±1.5 | 9.4±1.5 | 6.4±0.4 |
| MiniMax M2.5 | 6.3±1.0 | 8.6±2.1 | 5.2±1.3 | 5.2±1.3 |

Solving rate is not strongly correlated with context length — the difficulty lies in reasoning over noisy, fragmented content, not just processing long inputs. Models also frequently confuse speaker identities and roles in group chat scenarios.

## Quick Start

Same scripts as CL-bench. See [Hub README](README.md#quick-start) for details.

```bash
# Inference
python infer.py --model gpt-5.1 --input CLBench-life.jsonl --output outputs/life_gpt5-1.jsonl --workers 20

# Evaluation
python eval.py --input outputs/life_gpt5-1.jsonl --judge-model gpt-5.1 --workers 5
```

Evaluation is binary: a task is solved only if the model's response passes all associated rubrics.

## Citation

```bibtex
@inproceedings{clbenchlife2025,
      title={CLBench-life: Can Language Models Learn from Real-Life Context?},
      author={Shihan Dou and Ming Zhang and others},
      booktitle={Thirty-ninth Conference on Neural Information Processing Systems (NeurIPS)},
      year={2025},
}
```

## Contact

Shihan Dou: shihandou@foxmail.com, Ming Zhang: mingzhang23@m.fudan.edu.cn
