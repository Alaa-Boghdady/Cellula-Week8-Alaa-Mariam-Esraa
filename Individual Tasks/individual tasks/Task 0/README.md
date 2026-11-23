# Task 0 - Kimi K2 Architecture Analysis

Author: Alaa Boghdady. Cellula AI Internship, Week 8.

## Deliverables
- Task_0_Kimi_K2_Architecture.pdf: exactly three pages in the IEEEtran conference two-column format.
- Task_0_Kimi_K2_Architecture.tex: self-contained editable LaTeX source, including the diagram, tables, equations and numbered references.

## Coverage
Model overview; routed/shared experts and token selection; MLA and long context; pretraining data, infrastructure and optimization; MuonClip and agentic post-training; DeepSeek-V3 comparison; benchmark interpretation; analytical memory/parameter estimates; conclusion and eight primary-source references.

Scope is the original Kimi-K2 Base/Instruct 128K release, not subsequent updated or thinking variants. Report precision (1.04T/32.6B) is distinguished from release-page rounding (1T/32B). Analytical estimates are labelled as calculations, not measured inference results. Published benchmark numbers are attributed, not presented as author-run experiments.

## Edit / compile
Upload the .tex file to a blank Overleaf project. Select it as the main document and compile with pdfLaTeX or XeLaTeX; IEEEtran, amsmath, amssymb, booktabs and hyperref are standard packages. No external image files or .bib file are needed. The delivered PDF was compiled with Tectonic/XeTeX, checked to contain three pages and visually inspected.

## Numerical checks
Routed fraction 8/384 = 2.0833%; expert count 3*7168*2048 = 44.040192M weights; eight routed experts = 352.321536M weights per MoE layer; all 384 routed experts = 16.911433728B weights per MoE layer. Illustrative compressed-cache estimate: 61*(512+64)*131072*2 / 2^30 = 8.578125 GiB per sequence. Weight-only estimates use decimal TB. Assumptions and exclusions are stated in the report.

## Submission
Place this folder under `individual tasks/Task 0/` in the Week 8 repository. Submit PDF and retain editable source. This is a research task: no Colab execution or model download is required.
