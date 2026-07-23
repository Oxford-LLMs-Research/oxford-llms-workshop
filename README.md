# Oxford Large Language Models for Social Science Workshop 🚀

Large Language Models (LLMs) are revolutionising social science research with their exceptional text comprehension capabilities, opening new avenues for innovation. However, LLMs can introduce social biases into the output, and the reliability of their outputs can vary. Safely and responsibly harnessing LLMs while fostering innovation requires specialised training—a challenging task given the rapid pace of LLM research, with new models and updates emerging weekly.

Oxford LLMs addresses these challenges by uniting machine learning scientists, leading researchers, and industry practitioners to offer comprehensive training to social scientists. Our goal is to bridge the gap between academia and industry, as well as between social sciences and computer science, with a focus on natural language processing. Our workshop equips participants with the latest skills and fosters a community that promotes innovation in social science research, ensuring the safe and responsible use of LLMs.

## Contents 

This collection brings together lecture slides, workshop materials, and practical guides from our past iterations of the "Oxford LLMs" summer school for social sciences. The content spans foundational concepts to advanced applications—from transformer architectures and model interpretability to fine-tuning techniques and LLM agent systems. Whether you’re exploring core NLP principles or specialized implementations like retrieval-augmented generation, these materials provide both theoretical grounding and hands-on guidance. All resources are openly available for researchers, practitioners, and students working with LLMs in their research!

> **This repository is still growing.** We add materials as speakers clear them for release, so a session missing today may well be here next month. We are also editing the **lecture and research talk recordings** from the school and will publish them openly once post-production is done. Watch this space.

## 2026

The 2026 school was taught almost entirely by practitioners building language models in industry, and the collaborative research project ran the full week as a live common-task benchmark — the [AI Respondents Challenge](https://github.com/Oxford-LLMs-Research/ai-respondents-challenge), in which teams predicted individual survey responses and disclosed the features and prompts behind every prediction. Lectures came from **Tatiana Shavrina** (Meta), **Sagi Shaier** (JGU Mainz), **Ilya Boytsov** and **Ibragim Badertdinov** and **Sergei Skvortsov** (Nebius), and **Piotr Mazurek** (Liquid.AI), with guest research talks from social scientists working with these methods.

### Lectures

- [LLM Agents and Evaluation](./materials/lectures/2026/LLM%20Agents%20and%20Evaluation%20-%20Tatiana%20Shavrina.pdf) by Tatiana Shavrina — a survey of how language model agents are built and, harder, how they are evaluated.

- [Continual Learning: Why Models Forget, and What We Can Do About It](./materials/lectures/2026/Continual%20Learning%20-%20Sagi%20Shaier.pdf) by Sagi Shaier.

- [From Web Data to Verifiable Worlds](./materials/lectures/2026/From%20Web%20Data%20to%20Verifiable%20Worlds%20-%20Ibragim%20Badertdinov.pdf) by Ibragim Badertdinov — how data, evaluations, and RL environments shape modern language models.

### Research talks

- [Replication with Language Models](./materials/talks/2026/Replication%20with%20Language%20Models%20-%20Christopher%20Barrie.pdf) by Christopher Barrie (New York University).

### Challenge

- [AI Respondents Challenge](https://github.com/Oxford-LLMs-Research/ai-respondents-challenge) — starter notebook and scoring setup.
- [Challenge dataset](https://huggingface.co/datasets/oxford-llms/ai-respondents-challenge) on Hugging Face.
- [Final leaderboard](https://llmsforsocialscience.net/research/challenge/), including the held-out surveys revealed on the last day.

More 2026 sessions — including efficient inference and compression, inference economics, and the remaining guest talks — will be added here as they are cleared for release.

## 2025

The 2025 programme covered LLM fundamentals, evaluation and monitoring, agents and safety, fine-tuning, and efficient inference, alongside a Kaggle-style prediction competition on survey response prediction.

### Seminars

- [LLM Fundamentals](./materials/seminars/2025/%5Bseminar%5D_LLM_Fundamentals.ipynb) — a hands-on notebook of exercises on pretraining data preparation, tokenization, chat templates, decoding strategies, and attention.

- [Post-training: SFT, DPO, and GRPO](./materials/seminars/2025/post_training) — a hands-on notebook with datasets and training code covering the LLM lifecycle, in-context learning, supervised fine-tuning, RL, direct preference optimisation, reasoning models, and GRPO.

Remaining 2025 lecture slides are being collected from speakers and will be published here.

## 2024

Building on the 2023 program, the materials in 2024 feature both the fundamentals of LLMs, applied tutorials, and recent advancements. Particularly, the program delves deep into foundational concepts in model architecture and evaluation. Subsequent lectures explore agent-based systems and advanced fine-tuning techniques. The practical components address specialized applications including retrieval-augmented generation for social science research, implementation of observability tools, and workflow development for LLM agents. Technical deep dives cover the Gemini architecture and practical considerations for self-hosting models, providing end-to-end perspectives on contemporary LLM applications. **Grigory Sapunov**, **Tatiana Shavrina**, and **Ilya Boytsov** developed the lecture materials, with seminar contributions from **Atita Arora**, **John Githuly**, **Christian Silva**, and **Ciera Fowler**. See the full list of materials below:

### Lectures

- [Introduction to LLMs](https://docs.google.com/presentation/d/1avuQJE-Gf8t0an-sinXqkvKjzv1u4rLpO94ONOE_Vx4/edit#slide=id.p) by Grigory Sapunov.

- [LLM Evaluation](https://docs.google.com/presentation/d/1n_wwOPwbK9FPkVMKGA9Vneya4Jaf4lh13D9YO9byves/edit#slide=id.g303d6023e88_1_58) by Tatiana Shavrina.

- [LLM Agents](https://docs.google.com/presentation/d/10-6Erne_Fo62y2i4OOEaNkGTrcEEIk9sOObuSJlhR80/edit#slide=id.g303d938f784_1_58) by Tatiana Shavrina.

- [LLM Agents (continued)](https://docs.google.com/presentation/d/1Zlt3ARDbGBhVvmPaz7omaj6MlXIlIOFs7wm6P8GGrpc/edit#slide=id.p) by Grigory Sapunov.

- [Fine-Tuning and Alignment](https://docs.google.com/presentation/d/1-Kf059z24W65GgPNbjw6vOvABFOdxFf4BD_Hsenc-U8/edit) by Ilya Boytsov.


### Seminars

- [Navigating RAG for Social Science](https://docs.google.com/presentation/d/1ngAVq0Ks3_rgDw5Qnif_A483G6M6RoL2shVklabrYDI/edit#slide=id.g1ff45be8172_1_0) by Atita Arora.

- [LLM Observability and Evaluation](https://docs.google.com/presentation/d/1wCGSwBqo4iNx_MQXwQLHRyVW__t8UlMU9K6HPLmq4hM/edit#slide=id.g2f39ad35ae4_0_641) by John Githuly.

- [Tutorial: Creating LLama-Researcher using LlamaIndex workflows](https://github.com/Jgilhuly/Llama-Researcher) by John Githuly.

- [Gemeni Model Deep Dive](https://docs.google.com/presentation/d/1-_vJmcpJjDAY5zh60DR834yUwYyR-XlSZ5amC0sLfZg/edit?usp=sharing) by Christian Silva.

- [ORI's Guide to Self Hosting LLMs](/assets/presentations/oxford_llm_lecture_fowler.pdf) by Ciera Fowler.


## 2023

###  Lectures

The following lecture materials were and created by [Elena Voita](https://lena-voita.github.io/).
- [The Evolutionary Journey of NLP](https://github.com/antndlcrx/oxford-llms-workshop/blob/main/materials/lectures/2023/The%20Evolutionary%20Journey%20in%20NLP.pdf) from rule-based systems to modern Transformers-based models, which are the core technology underpinning LLMs. 
- [Bias in LLMs and a (bit of) Interpretability](https://github.com/antndlcrx/oxford-llms-workshop/blob/main/materials/lectures/2023/Bias%20in%20LLMs%20and%20interpretability.pdf). 
- [LLMs and Alignment](https://github.com/antndlcrx/oxford-llms-workshop/blob/main/materials/lectures/2023/Prompt%20engineering%2C%20RHLF%2C%20ChatGPT.pdf)


### Workshops
The following workshop materials were designed and implemented by [Ilya Boytsov](https://www.linkedin.com/in/ieboytsov/).
We will upload the workshop recordings soon!
- [Google Colab environment setup, general intro](https://github.com/antndlcrx/oxford-llms-workshop/blob/main/materials/seminars/2023/day_1/0_Env%20setup%20and%20intro.ipynb)
- [Introduction to Huggingface transformers library](https://github.com/antndlcrx/oxford-llms-workshop/blob/main/materials/seminars/2023/day_1/1_Intro%20to%20transformers.ipynb)
- [Topic modelling with Transformers using BERTopic library](https://github.com/antndlcrx/oxford-llms-workshop/blob/main/materials/seminars/2023/day_1/2_Topic%20modelling%20with%20transformers.ipynb)
- [A guide how to fine-tune pretrained model for a classification task](https://github.com/antndlcrx/oxford-llms-workshop/blob/main/materials/seminars/2023/day_1/3_Fine%20tune%20pretrained%20model.ipynb)
- [Parameter Efficient Fine Tuning (PEFT)](https://github.com/antndlcrx/oxford-llms-workshop/blob/main/materials/seminars/2023/day_2/4_Parameter%20efficient%20fine%20tuning.ipynb)
- [Transformers interpretability, Attention visualisation, and saliancy methods (e.g. Integrated gradients)](https://github.com/antndlcrx/oxford-llms-workshop/blob/main/materials/seminars/2023/day_2/5_Transformers%20interpretability.ipynb)
- [Model analys with classic NLP using Spacy](https://github.com/antndlcrx/oxford-llms-workshop/blob/main/materials/seminars/2023/day_2/6_Sentiment%20analysis%20with%20classic%20NLP.ipynb)
- [Prompts and instructions with Llama 2](https://github.com/antndlcrx/oxford-llms-workshop/blob/main/materials/seminars/2023/day_3/7_Prompts%20and%20instructions%20with%20Llama%202.ipynb)
- [Detoxifying summarisation model with Reinforcement Learning from Human Feedback (RLHF)](https://github.com/antndlcrx/oxford-llms-workshop/blob/main/materials/seminars/2023/day_3/8_LLMs%20alignment%20with%20RLHF.ipynb)

## Contributors

### 2026

- [Tatiana Shavrina, Lecturer](https://scholar.google.com/citations?user=sdmdZh8AAAAJ), Research Scientist Manager, Meta
- [Sagi Shaier, Lecturer](https://sagishaier.com/), Visiting Professor, Johannes Gutenberg University Mainz
- [Ilya Boytsov, Lecturer, Organiser](https://www.linkedin.com/in/ieboytsov/), Sr. ML Engineer, Nebius
- [Ibragim Badertdinov, Lecturer](https://ibragim.dev/), Lead Engineer, Nebius
- [Sergei Skvortsov, Lecturer](), Lead ML Engineer, Nebius
- [Piotr Mazurek, Lecturer](https://www.tensoreconomics.com/), Member of Technical Staff, Liquid.AI
- [Christopher Barrie, Research Talk](https://www.cjbarrie.com/), Assistant Professor of Sociology, New York University
- [Elliott Ash, Research Talk](https://elliottash.com/), Associate Professor of Law, Economics and Data Science, ETH Zurich
- [Andreu Casas, Research Talk](), Politics and the Oxford Internet Institute, University of Oxford
- [Roberto-Rafael Maura-Rivero, Research Talk](https://sites.google.com/view/rrmaura/), Research Engineer at Meta and Postdoctoral Researcher, University of Oxford
- [Margaret E. Roberts, Research Talk](https://margaretroberts.net/), Professor of Political Science, University of California San Diego
- [Joshua Berry, Organiser](), Nuffield College, Oxford University
- [Elodie Chervin, Organiser](), Nuffield College, Oxford University
- [Maksim Zubok, Lead Organiser](https://www.linkedin.com/in/maksim-zubok-160572232/), Nuffield College, Oxford University

### 2024

- [Grigory Sapunov, Lecturer](https://www.linkedin.com/in/grigorysapunov/), CTO and co-founder of Intento
- [Tatiana Shavrina, Lecturer](https://scholar.google.com/citations?user=sdmdZh8AAAAJ), Research Scientist Manager, Meta
- [Ilya Boytsov, Lecturer, Organiser](https://www.linkedin.com/in/ieboytsov/), NLP Lead ar Wayfair, Content Intelligence Team.
- [Atita Arora, Workshop creator](https://github.com/atarora), Solution Architect, Qdrant*
- [Ciera Fowler, Workshop creator](), ML Engineering Lead, Ori Cloud
- [John Gilhuly, Workshop creator](), Developer Advocate, Arize AI
- [Christian Silva, Workshop creator](), AI/ML Customer Engineer at Google Cloud.
- [Humeyra Biricik, Organiser](), DPIR Oxford University 
- [Maksim Zubok, Organiser](), DPIR Oxford University


### 2023

- [Ilya Boytsov, Coding Seminars Leader](https://www.linkedin.com/in/ieboytsov/), NLP Lead ar Wayfair, Content Intelligence Team.
- [Elena Voita, Lecturer](https://www.linkedin.com/in/elena-voita/), Research Scientist at FAIR (Meta AI), lecturer at Oxford LLMs 2023
- [Maksim Zubok, Workshop organiser](https://www.linkedin.com/in/maksim-zubok-160572232/), DPhil candidate in Politics at Oxford University, Nuffield College.



# Contact
📧 For inquiries, contact [maksim.zubok@nuffield.ox.ac.uk](mailto:maksim.zubok@nuffield.ox.ac.uk)
