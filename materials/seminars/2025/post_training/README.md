# LLM Post-Training Seminar

This repository contains materials for a seminar covering the fundamentals of post-training techniques for Large Language Models (LLMs).

Table of content:
- LLM lifecycle overview
- In-context learning (ICL)
- Supervised Fine Tuning (SFT)
- Reinforcement learning (RL)
- Direct Preference Optimization (DPO)
- Reasoning models
- Group Relative Policy Optimization (GRPO)

## How to run code

### Prerequisites
* Python 3.8+ (or your specific version)
* A GPU is highly recommended for running the notebook.

### Create and activate a virtual environment:
```bash
# Create the virtual environment
python -m venv .venv

# Activate the environment

# On macOS/Linux:
source .venv/bin/activate

# On Windows (Command Prompt/PowerShell):
.venv\Scripts\activate
```

### Install dependencies:
```bash
python -m pip install -r requirements.txt

```
### Run the notebook
Open and run the `post-training.ipynb` notebook. This notebook relies on the code located in the `src` folder.

## Author

These materials were designed and implemented by [Sergei Skvortsov](https://www.linkedin.com/in/sergei-skvortsov)
