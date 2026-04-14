<div align="center">

<img src="assets/hunyuan.png" alt="Tencent-Hunyuan" width="120"/>

# Context Learning

[![Leaderboard](https://img.shields.io/badge/Leaderboard-www.clbench.com-black.svg?style=flat-square)](https://www.clbench.com)

</div>

Real-world tasks require models to acquire and apply new knowledge from context at inference time, rather than relying solely on pre-trained knowledge. We study this capability — **context learning** — and build benchmarks to measure it.

---

> **CLBench-life** &nbsp; Real-life context learning &nbsp;·&nbsp; 500 tasks &nbsp;·&nbsp; 5,348 rubrics &nbsp;·&nbsp; NeurIPS 2025
>
> Group chats, personal notes, browsing histories, game logs — messy everyday contexts that current models still struggle with. Best model solves 19.3%.
>
> [Paper](clbench-life-paper.pdf) &nbsp;·&nbsp; [Data](https://huggingface.co/datasets/tencent/CLBench-life) &nbsp;·&nbsp; [Documentation](CLBench-life.md)

> **CL-bench** &nbsp; Professional & domain-specific context learning &nbsp;·&nbsp; 1,899 tasks
>
> Domain knowledge, rule systems, complex procedures, empirical discovery — contexts absent from pre-training. Best model solves 23.7%.
>
> [Paper](https://arxiv.org/abs/2602.03587) &nbsp;·&nbsp; [Data](https://huggingface.co/datasets/tencent/CL-bench) &nbsp;·&nbsp; [Blog](https://hy.tencent.com/research/100025?langVersion=en) &nbsp;·&nbsp; [Documentation](CL-bench.md)

---

## Quick Start

Both benchmarks share the same evaluation pipeline — just point to different input files.

```bash
pip install openai tqdm
```

Download datasets from HuggingFace:
- [tencent/CL-bench](https://huggingface.co/datasets/tencent/CL-bench) → `CL-bench.jsonl`
- [tencent/CLBench-life](https://huggingface.co/datasets/tencent/CLBench-life) → `CLBench-life.jsonl`

### Inference

```bash
export OPENAI_API_KEY="your_api_key"

python infer.py --model gpt-5.1 --input CL-bench.jsonl --workers 20
python infer.py --model gpt-5.1 --input CLBench-life.jsonl --workers 20

# Other OpenAI-compatible APIs
python infer.py --model deepseek-chat \
    --base-url https://api.deepseek.com/v1 \
    --api-key your_deepseek_key \
    --input CL-bench.jsonl
```

### Evaluation

```bash
python eval.py --input outputs/gpt-5.1.jsonl --judge-model gpt-5.1
```

<details>
<summary>Script options</summary>

**infer.py** — `--model`, `--input`, `--output`, `--base-url`, `--api-key`, `--workers`, `--max-samples`

**eval.py** — `--input`, `--output`, `--judge-model`, `--base-url`, `--api-key`, `--workers`

</details>

---

## Citation

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

@inproceedings{clbenchlife2025,
      title={CLBench-life: Can Language Models Learn from Real-Life Context?},
      author={Shihan Dou and Ming Zhang and others},
      booktitle={Thirty-ninth Conference on Neural Information Processing Systems (NeurIPS)},
      year={2025},
}
```

## Contact

Shihan Dou — shihandou@foxmail.com &nbsp;·&nbsp; Ming Zhang — mingzhang23@m.fudan.edu.cn
