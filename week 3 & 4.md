# Logical LLM Project

This repository explores advanced techniques in Large Language Models (LLMs) and Reinforcement Learning (RL), focusing on efficient finetuning, strategic game playing, and advanced policy optimization methods.

---

## 1. QLoRA: Efficient Finetuning of Quantized LLMs

### 1.1 What is QLoRA and how does it differ from standard LoRA?
QLoRA (Quantized LoRA) is a variant of low-rank adapter finetuning where the pretrained LLM is quantized to 4-bit before adaptation. It freezes the base model but stores its weights in a 4-bit format, then backpropagates gradients only into small trainable “LoRA” adapters. In practice, QLoRA uses 4-bit model weights and BFloat16 computation, whereas standard LoRA typically keeps the base model in 16/32-bit precision. This lets QLoRA drastically save memory. The trade-off is slightly slower training, but QLoRA achieves almost the same accuracy as full 16-bit LoRA with far less hardware.

### 1.2 Explain the role and benefits of the NormalFloat4 (NF4) quantization format.
NF4 is a special 4-bit floating-point format optimized for the Gaussian-like distribution of neural weights. It allocates bit levels based on the quantiles of a standard normal distribution, making it information-theoretically optimal for zero-mean normal data. In practice, NF4 yields significantly lower quantization error than naive int4 or uniform FP4. The benefit is better accuracy under 4-bit storage – models quantized to NF4 preserve almost full precision performance.

### 1.3 What is "double quantization" and why is it useful?
Double quantization is a second round of quantization applied to the scale factors used in block-wise 4-bit quantization. In QLoRA, weights are quantized in small blocks, which adds overhead. Double quantization compresses those scales themselves, drastically reducing this overhead. This further cuts memory use in exchange for negligible precision cost, enabling even larger models to fit in limited GPU RAM.

### 1.4 How do paged optimizers help in memory management during training?
Paged optimizers use GPU unified memory to swap optimizer states between GPU and CPU. In practice, QLoRA allocates optimizer states in unified memory so that when the GPU’s RAM is exhausted, those states automatically overflow to host memory. This prevents out-of-memory errors during backpropagation.

### 1.5 Why is QLoRA significant in enabling large model finetuning on limited hardware?
QLoRA dramatically lowers the hardware barrier for tuning huge LMs. By combining 4-bit NF4 quantization, double quantization, and paged optimizers, the GPU memory needed to finetune a 65B model is reduced to 48 GB. This means a single consumer-grade GPU can finetune models that previously required hundreds of gigabytes. QLoRA-trained models match the accuracy of full-precision models while using much less memory.

### 1.6 Suggest one possible improvement or variation to the QLoRA method. Why might it help?
One idea is to adopt quantization-aware LoRA or joint quantization methods. These approaches blend quantization more tightly with the adapter updates, potentially reducing quantization error and allowing finer control over precision. Another variant is to adjust the bit allocation (e.g. adaptive mixed-precision per layer), which can tailor memory use to a target budget.

---

## 2. PokerGPT: Lightweight Solver for Multi-Player Poker

### 2.1 Describe the overall pipeline of PokerGPT from raw data to trained model.
Data Acquisition: Collect raw game logs including blinds, player positions, hole cards, community cards, and actions.

Prompt Engineering: Transform each hand into text prompts describing the state, history, and actions.

Training (SFT + RLHF): First apply supervised fine-tuning, then use a reward model to score actions and apply reinforcement learning (e.g., PPO).

### 2.2 What makes PokerGPT different from traditional poker solvers like CFR or DeepStack?
Traditional solvers use explicit game-theoretic search and are computationally intensive. PokerGPT is data-driven, model-based, and trained end-to-end from real play data. It is scalable, interactive, and lighter in compute.

### 2.3 Why is RLHF used in PokerGPT and how does it affect decision-making quality?
RLHF refines the model using a reward model based on outcomes (e.g., win-rates). PPO helps the model prefer actions that lead to better results rather than just imitating human plays.

### 2.4 What are the advantages of using LLMs for multiplayer poker games?
Scalability: Supports any number of players.

Data-driven Learning: Captures human strategies without manual features.

Compute Efficiency: Requires less compute than traditional solvers.

Natural Interaction: Can explain decisions in natural language.

### 2.5 What challenges might arise when scaling PokerGPT to more complex games like Omaha?
Omaha has more private cards (4 instead of 2), increasing state and strategy complexity. Prompt construction becomes more difficult, and more data is needed to train a good model.

### 2.6 Suggest one possible extension or improvement to PokerGPT and explain its benefit.
Adding self-play could help the model explore strategies not seen in the dataset. Chain-of-thought prompting or opponent modeling could also enhance decision-making.

---

## 3. GRPO (Group Relative PPO) - DeepSeekMath 7B

### 3.1 What is the main idea behind GRPO and how does it differ from traditional PPO?
GRPO is a reinforcement learning method that uses group-relative comparisons instead of a value network. It compares each answer’s reward to the group average to compute advantages, reducing compute and memory requirements.

### 3.2 How does GRPO eliminate the need for a separate value network?
GRPO samples multiple outputs for the same prompt and uses the average reward as a baseline. Each output’s advantage is its reward minus this average, removing the need for a learned critic.

### 3.3 What is the significance of using z-score normalization for reward signals?
Z-score normalization standardizes rewards within each group. It ensures consistent scaling, reduces the impact of outliers, and stabilizes training by producing normalized advantages.

### 3.4 Why might GRPO be better suited for math reasoning tasks than PPO?
Math tasks usually have sparse rewards and are better suited to relative comparisons. GRPO avoids the need to learn dense value functions and instead directly optimizes over sampled answers.

### 3.5 What are potential weaknesses of GRPO when all sampled outputs are poor?
If all outputs in a group are poor, the model receives minimal learning signal. Advantages become small or meaningless, making training slow or unstable.

### 3.6 Suggest a modification to GRPO to make it more robust to low-quality output groups.
Introduce reference anchors (e.g., one known good solution) in each group to guide learning. Alternatively, dynamically increase the number of samples or mix in supervised loss to ensure stable learning signals.

---
