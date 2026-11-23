# Task 2 - Qwen3-ASR-1.7B

**Alaa Boghdady | Cellula AI Internship | Week 8 individual task**

## Submission files
- Task_2_Qwen3_ASR_Report.pdf: three-page IEEE conference report.
- Task_2_Qwen3_ASR_Report.tex and latency.pdf: editable LaTeX source and figure; upload both to Overleaf and compile with the IEEEtran conference class.
- Task_2_Qwen3_ASR_Complete.ipynb: original executed baseline plus self-contained offline accent reproduction cells and clearly labelled recorded final tables.
- Task_2_Original_Executed_Baseline.ipynb: unmodified original user-run notebook evidence.
- results/: actual predictions, ground truth, 48 timing trials, summaries, environment evidence and chart.
- samples/: all 16 evaluated WAV files.
- Task_2_Accents_Offline.zip: four original VCTK inputs, reference metadata and extension script, avoiding the dataset-row service errors.
- VALIDATION.txt: independent verification of error metrics, medians, file durations and accent-file hashes.

## Actual results
16 inputs, 10 unique speakers, 48 timed trials. Normalized corpus WER 2.508%, CER 0.501%, aggregate RTF 0.2156. Peak allocated inference GPU memory 3.945 GiB. These are this small experiment's measurements, not official benchmark results. The four VCTK recordings have zero normalized errors; this is not evidence of universal accent robustness.

## Reproduce in Colab
Upload Task_2_Qwen3_ASR_Complete.ipynb to Colab and choose an available GPU. Run baseline cells in order, followed by the final embedded-input and accent-extension cells. The optional custom-upload section can remain False: documented accent coverage is supplied by the final VCTK section. The initial corpus/model downloads still need internet; the accent extension does not query the failing dataset-row API. The original outputs describe the baseline stage; see the final table/results CSVs for the combined study.

The final extension cells are intentionally unexecuted in this edited notebook: imported CSV tables are labelled as recorded results. No execution counts or model outputs were fabricated. To obtain one notebook with live outputs for every stage, run the final two cells and save the notebook; no new experiment design is needed.

For resuming an existing Colab session, load the model and restore the original result ZIP first, then restore the embedded inputs and run the extension. The script replaces prior VCTK rows rather than duplicating them.

## Limitations and provenance
Six LibriSpeech utterances have clean/10 dB synthetic-noise pairs. Four additional VCTK speakers cover dataset-labelled English (Southern England) and Scottish (Fife/Perth) groups, two speakers each, with different utterances. Labels are corpus metadata rather than independent linguistic assessment. The two stages ran in separate Tesla T4 Colab sessions. A complete second package freeze/dtype profile was not exported. No model-to-model comparison, long-form or streaming benchmark is claimed. The report distinguishes requested software dtype from native GPU arithmetic.

Normalization: NFC, casefold, punctuation/symbols to spaces, whitespace collapse. CER includes spaces. Raw strings are preserved.

## GitHub layout
Place this folder's contents under `individual tasks/Task 2/` in the Week 8 repository. Upload the report, source, notebook, evidence and attribution. Do not upload model weights. A zipped package is convenient for delivery, while expanded files make the repository easier to review.

## Sources
Qwen report: https://arxiv.org/abs/2601.21337
Official toolkit: https://github.com/QwenLM/Qwen3-ASR
Model: https://huggingface.co/Qwen/Qwen3-ASR-1.7B
LibriSpeech: https://huggingface.co/datasets/openslr/librispeech_asr
VCTK: https://doi.org/10.7488/ds/2645
Representation: https://huggingface.co/datasets/kth-tmh/vctk
JiWER: https://github.com/jitsi/jiwer
