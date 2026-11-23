# Task 1 - Qwen2.5-72B Architecture Analysis

Author: Alaa Boghdady. Cellula AI Internship, Week 8.

## Deliverables
- Task_1_Qwen2_5_72B_Architecture.pdf: exactly three IEEE conference-format pages, two columns per page.
- Task_1_Qwen2_5_72B_Architecture.tex: self-contained editable source with equations, conceptual diagram, specification table, benchmark comparison and ten numbered references.
- Arabic study notes: short explanation for revision.

## Scope and checks
This report analyzes the original general-purpose text-only Qwen2.5-72B Base/Instruct family. It does not confuse the dense open-weight 72B architecture with proprietary MoE variants, multimodal Qwen2.5-VL or specialist Coder/Math checkpoints. It explains the default 32768-position configuration versus the documented 131072-token YaRN deployment, and distinguishes tokenizer regular/control tokens from the 152064-row configured vocabulary allocation.

Coverage: causal objective, residual/pre-normalization, RMSNorm, GQA, rotary positions, SwiGLU, tokenization, pretraining/post-training, optimization, parameter/memory calculations, inference complexity, comparative reported performance, limitations and conclusion.

Analytical calculations were checked: approximately 72.704B major matrix weights before norm/bias terms; FFN 726.663M and attention projections 150.995M weights per layer; a two-byte GQA KV cache is 10 GiB at 32768 tokens or 40 GiB at 131072 tokens per sequence. Weight-only storage using rounded 72.7B is 145.4 GB at two bytes per weight. These are calculations, not measured model runs. Benchmark scores are publisher-reported and version-labelled.

## Compile and submit
Upload the .tex file to an Overleaf project and compile with pdfLaTeX or XeLaTeX using IEEEtran. No external images or bibliography file are needed. The delivered PDF was compiled using Tectonic/XeTeX, confirmed as three pages, and visually inspected.

Place the expanded contents under `individual tasks/Task 1/` in the Week 8 repository. No Colab/model execution is required for this research assignment.
