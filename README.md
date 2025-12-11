# Finance-Llama-3-LoRA: 基于 Llama-3 的金融领域知识问答大模型

## ⚠️ 重要说明 / Important Notice regarding Model Weights

**关于模型权重的说明 / Note on Model Weights:**

由于微调后的模型权重文件（尤其是合并后的完整模型）体积过大（超过 GitHub 单文件 100MB 限制），**本项目仓库未包含训练好的 `.safetensors` 或 `.bin` 权重文件**。

本仓库仅包含：
1. 项目源代码
2. 训练与测试数据的生成/处理脚本
3. 配置文件 (`.yaml`)

如果您需要复现实验，请参考下方的 [训练流程](#训练流程) 自行运行训练脚本。

---

## 📖 项目介绍 (Introduction)

本项目旨在探索低资源消耗下的大模型垂直领域落地路径。实验选取 **Meta-Llama-3-8B-Instruct** 作为基座模型，利用 **LoRA (Low-Rank Adaptation)** 参数高效微调技术，基于特定上市公司的财报数据构建金融知识问答模型。

项目核心目标是提升通用大模型在金融细分场景下的表现，特别是针对特定企业（如“江苏安靠智能输电工程科技股份有限公司”）的细节知识记忆与逻辑推理能力。

## 📂 数据集 (Dataset)

* **原始数据来源**: `modelscope/chatglm_llm_fintech_raw_dataset`，聚焦于单一企业的年度/季度财务报告（PDF格式）。
* **数据处理**:
    * 使用 PDF 解析工具切分文本块。
    * 利用 `Easy Dataset` 工具配合自定义 Prompt 生成问答对。
    * **数据规模**: 最终构建了包含 **977条** 高质量金融问答对的 JSON 格式数据集。
* **数据特点**: 侧重于“概念解释类”和“总结分析类”问题，避免仅生成简单的数值检索，以提升思维深度。

## 🛠️ 技术架构与环境 (Architecture & Env)

* **基座模型**: Meta-Llama-3-8B-Instruct (8B 参数, GQA 机制)
* **微调方法**: LoRA (Rank=8, Alpha=16)
* **训练框架**: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
* **硬件环境**: 单张 NVIDIA A40 (48GB 显存)

### 关键训练参数 (Hyperparameters)

| 参数项 | 配置值 |
| :--- | :--- |
| `learning_rate` | 1.0e-4 |
| `num_train_epochs` | 10.0 |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 8 |
| `lr_scheduler_type` | cosine |
| `warmup_ratio` | 0.1 |
| `cutoff_len` | 2048 |
| `lora_rank` | 8 |
| `lora_target` | all linear layers |

## 🚀 快速开始 (Quick Start)

### 1. 环境安装
请确保已安装 `LLaMA-Factory` 及其依赖：


git clone [https://github.com/hiyouga/LLaMA-Factory.git](https://github.com/hiyouga/LLaMA-Factory.git)
cd LLaMA-Factory
pip install -r requirements.txt

###  2. 数据准备 (Data Preparation)

[cite_start]请将处理好的 `datasets_finance.json` 数据集文件放置在 `data/` 目录下，并修改 `data/dataset_info.json` 文件以注册新的数据集 [cite: 47]。

### 3. 启动训练 (Training)

[cite_start]使用以下命令启动 LoRA 微调。为了优化下载速度，环境配置中启用了 ModelScope Hub 加速 [cite: 68]。


### 启用 ModelScope Hub 加速（可选）
export USE_MODELSCOPE_HUB=1 

### 启动训练脚本
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

### 4. 模型合并 (Merge)
训练完成后，为了将 LoRA 权重集成到基座模型中以进行独立部署，需执行模型合并与导出操作 [2]。


# 执行模型合并与导出
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
