The table below summarizes fine-tuning performance across different models and parameters using CarTool-Instruct. Refer to [Fine_Tuning_Car_Tool_Instruct_with_Hugging_Face.ipynb](./Fine_Tuning_Car_Tool_Instruct_with_Hugging_Face.ipynb) for the fine-tuning and evaluation process.

| Model | Fine-tuning Strategy | Training Samples | Epochs | Learning Rate | Optimizer | LoRA Rank | LoRA Alpha | LoRA Dropout | **Pre-train Accuracy (CarTool-Instruct)** | **Fine-tuned Accuracy (CarTool-Instruct)** | Pre-train Accuracy (PIQA) | Fine-tuned Accuracy (PIQA) |
| ------------------ | --------------- | ------------- | ----- | ------------- | ------------------------ | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| google/functiongemma-270m-it | FFT | 9509 | 2 | 1e-5 | adamw_torch_fused | NA | NA | NA | **0** | **38.32** | 64.58 | 64.36 |
| google/functiongemma-270m-it | FFT | 6500 | 2 | 1e-5 | adamw_torch_fused | NA | NA | NA | **0** | **31.58** | 64.58 | 64.58 |
| google/functiongemma-270m-it | FFT | 2000 | 2 | 1e-5 | adamw_torch_fused | NA | NA | NA | **0** | **7.05** | 64.58 | 64.91 |
| Qwen/Qwen3-4B-Instruct-2507 | LoRA | 8000 | 2 | 2e-4 | adamw_torch_fused | 8 | 16 | 0.05 | **22.53** | **63.16** | 75.90 | 75.46 |

> [!NOTE]
>
> [PIQA](https://arxiv.org/abs/1911.11641) results are evaluated via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

> [!IMPORTANT]
>
> Due to limited computational resources, I was unable to conduct experiments across a broader range of models or perform a comprehensive comparison of different parameter configurations.
>
> If you have trained models using this dataset on different models, please consider submitting a Pull Request with your evaluation metrics! 🤝
