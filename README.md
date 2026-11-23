# Cellula Technologies — Week 8 Final Submission

This repository organizes the final internship week submission into individual and group work.

## Repository structure

```text
Individual Tasks/
├── Task 0 — Kimi K2 Architecture/
├── Task 1 — Qwen2.5-72B Architecture/
├── Task 2 — Qwen3-ASR-1.7B/
├── Task 3 — GAN Research/
└── Task 4 — Updated CV/
Group Tasks/
├── Task 5 — Agentic System/
├── Task 6 — Infrastructure as Code/
└── Voice Agent Project/
```

## Individual tasks

| Task | Deliverable |
|---|---|
| 0 | Three-page IEEE research paper on Kimi K2 architecture |
| 1 | Three-page IEEE research paper on Qwen2.5-72B architecture |
| 2 | Three-page IEEE research paper, Colab experiment notebook, real WER/CER and timing results, and licensed audio samples |
| 3 | IEEE research paper on GANs, mathematical foundations, four focal papers, architectures, and applications |
| 4 | ATS-friendly updated CV in PDF |

## Group tasks

Task 5 extends the Week 7 Document Intelligence application with a FastAPI layer, voice and image/OCR input, and an agent that generates styled PDF reports. Task 6 prepares and validates Terraform configuration; it does not deploy cloud resources.

## Task 5 and Task 6 role assignment

The same three-person ownership model is used across both tasks. See [the team role and interface plan](Group%20Tasks/TEAM_ROLES.md).

| Member | Task 5 | Task 6 |
|---|---|---|
| Israa (Person A) | FastAPI, orchestration, integration, end-to-end demonstration | Integration, offline validation, final README and evidence |
| Mariam (Person B) | Voice input, speech-to-text, image validation and OCR | Infrastructure architecture and core resources |
| Alaa (Person C) | Report-generation agent, PDF renderer and report design | Providers, variables, outputs and reusable interfaces |

## Task 2 recorded results

The recorded experiment contains 16 audio inputs from 10 speakers and 48 measured trials. Overall normalized WER is 2.508%, CER is 0.501%, and aggregate RTF is 0.2156. The results archive contains the audio samples, CSV results, environment details, and plot.

## Reproduction

Open the Task 2 notebook in Google Colab after it is uploaded to `Individual Tasks/Task 2 — Qwen3-ASR-1.7B/`. The notebook describes the required GPU runtime and the reproduction steps.

All report claims and experiment results should match the included source files and recorded evidence. Do not commit secrets, cloud credentials, Terraform state, or private environment files.
