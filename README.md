# DynCPG-LLM

### ✍️ Introduction
DynCPG-LLM is a multi-agent framework to support security defect detection. 

It integrates an LLM with the static analysis tool Joern, to support dynamic querying of relevant contextual information from the source code repository during the LLM’s step-by-step reasoning process. It also introduces a multi-round iterative interaction mechanism to continuously update security defect detection results.

### 🛠️ Execution Guide

**STEP 1.** Download and install Joern.

**STEP 2.** Install the required Python packages.

**STEP 3.** Deploy the CPGQL generation model through vLLM.

**STEP 4.** Replace the LLM and API key utilized in `get_llm_response.py`.

**STEP 5.** Create directories where the projects, CPG instances, and detection results are stored.

**STEP 6.** Run `run.py`.

# LoRA
#### 📂 Script Overview

- **Training Scripts:**
  - `train_qwen2.5.py`: Configuration for **Qwen2.5-Coder-32B-Instruct**.
  - `train_qwen3.py`: Configuration for **Qwen3-Coder-30B-A3B-Instruct**.
  
- **Inference Scripts:**
  - `unsloth_inference`: Inference utilizing Unsloth.
  - `vllm_inference.py`: Script for serving the model using vLLM.

#### 🖥️ Execution Environment

- **Operating System:** Ubuntu 22.04.3
- **Hardware:** Single NVIDIA A100-80GB GPU

# workflow_training_data_generation

The workflow of generating training data.

### 🛠️ Execution Guide

**STEP 1.** Download and install Joern.

**STEP 2.** Install the required Python packages.

**STEP 3.** Replace the LLM and API key utilized in `call_llm.py`.

**STEP 4.** Update the source data and output directory settings in `run_workflow.py`.

**STEP 4.** Run `run_workflow.py`.
