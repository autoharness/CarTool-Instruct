# 📰 CarTool-Instruct

The [**CarTool-Instruct Dataset**](https://huggingface.co/datasets/autoharness/CarTool-Instruct) is specifically tailored for [CarToolForge](https://github.com/autoharness/CarToolForge). It aims to evaluate and improve model accuracy in **function-calling** scenarios defined within CarToolForge, with a particular focus on fine-tuning small-parameter models optimized for **on-device deployment**.

This repository contains the generation pipeline for the CarTool-Instruct dataset, along with examples of fine-tuning models using this dataset.

## How to Generate Datasets

### Setup

Clone the repository:

```
git clone https://github.com/autoharness/CarTool-Instruct.git
cd CarTool-Instruct
```

A suitable conda environment can be created and activated with:

```
conda env create -f environment.yml
conda activate car_tool_instruct
```

Currently, the pipeline uses Gemini for sample generation. You must configure your [Gemini API key](https://ai.google.dev/gemini-api/docs/api-key) by creating an `access_token.toml` file in the `secrets/` directory:

```
[ai_services.gemini]
api_key = "YOUR_API_KEY"
```

### Generate

The data generation process consists of two steps. First, run the `generate` command to create data in JSON format. This stage focuses on creating the core `query` and `answer` pairs.

The following command demonstrates how to generate 200 samples and save the results to `build/dataset.json`:

```
python gen_pipeline/src/cli.py generate --num-samples 200 --output build/dataset.json
```

> [!NOTE]
>
> If the specified `--output` file already exists, the pipeline will resume generation and append to the existing data.

To see all available options of `generate`, run:

```
python gen_pipeline/src/cli.py generate --help
```

### Refine

The `refine` command processes the raw JSON data to add essential fields such as `metadata` and `tools`, outputting the final dataset in **JSONL** format.

The following command shows how to use the `dataset.json` file (created via the `generate` command) to produce the final `dataset.jsonl` file:

```
python gen_pipeline/src/cli.py refine --num-test 20 --data-file build/dataset.json --output build/dataset.jsonl
```

The `--num-test` flag defines the size of the test split.

To see all available options of `refine`, run:

```
python gen_pipeline/src/cli.py refine --help
```

## Fine-tuning

The [Fine_Tuning_Car_Tool_Instruct_with_Hugging_Face.ipynb](fine_tuning/Fine_Tuning_Car_Tool_Instruct_with_Hugging_Face.ipynb) showcases how to fine-tune models on the CarTool-Instruct dataset using the [TRL](https://huggingface.co/docs/trl/en/index) library. Results from some of the fine-tuning experiments can be found in [fine_tuning/README.md](fine_tuning/README.md).

## Limitations

The dataset and generation pipeline do not currently cover **multi-turn** function calling capability.

## References

The generation pipeline and the prompts used are primarily inspired by:

- [APIGen](https://apigen-pipeline.github.io/)

- [DroidCall](https://github.com/UbiquitousLearning/DroidCall)

The dataset format and fine-tuning methodologies are largely based on:

- [Mobile Actions](https://ai.google.dev/gemma/docs/mobile-actions)
