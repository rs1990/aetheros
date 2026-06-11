# Orin Nano Server + Training Setup — Pending

User wants to set up an NVIDIA Jetson Orin Nano as a hosting + training box.
Paused before scoping; resume by answering the questions below.

## Hardware notes (for context when we resume)
- Orin Nano dev kit: 8 GB unified LPDDR5, 1024-core Ampere GPU, 32 tensor cores
- Runs JetPack (Ubuntu 22.04 + CUDA + cuDNN + TensorRT)
- "Training" on this box realistically means LoRA/QLoRA on small LLMs (≤3B) or
  small CV models, not full pretraining
- Hosting works well for: Ollama/llama.cpp inference, FastAPI backends,
  Jupyter, SSH dev box

## Questions to answer before we build a plan

1. **Device state** — which one?
   - Brand new, needs flashing + first-boot
   - Booted Ubuntu but nothing configured
   - Already set up, just adding services

2. **Hosting goal** (pick any that apply)
   - LLM inference (Ollama / llama.cpp)
   - Web app / API backend (Node, FastAPI, Docker)
   - SSH-only remote dev box
   - Jupyter / remote notebook server

3. **Training goal** — pick one
   - LoRA / QLoRA on small LLMs
   - Classical ML / small CV models
   - Inference only for now
   - Not sure, want a recommendation

4. **Access mode**
   - Headless over LAN (SSH from Mac) — recommended
   - Monitor + keyboard attached
   - Both

## When we resume
Once these are answered, produce a step-by-step plan covering:
flash/JetPack → user + SSH + static IP/mDNS → Docker + nvidia-container-toolkit
→ inference stack → training stack → reverse proxy/exposure (if needed)
→ monitoring (jtop, tegrastats).
