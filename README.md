# Fine-Tuning Small Models Locally and Evaluation

A five-notebook learning path for preparing instruction data, training a real local LoRA adapter, and comparing it with the unchanged base model. The notebooks use Hugging Face Transformers, PEFT, and a small fictional support-policy dataset. Local metrics and an optional LLM-as-judge workflow expose both improvements and remaining failures.

## Learning Objectives

By the end of this repository, you should be able to:

- Prepare and format a domain-specific instruction dataset for fine-tuning.
- Fine-tune a small language model locally with LoRA and explain where QLoRA changes the memory profile.
- Compare a fine-tuned adapter against the base model on a held-out evaluation set.
- Implement an LLM-as-judge evaluation pipeline for relevance, faithfulness, and coherence.
- Identify hallucination and source-faithfulness failures by comparing generated claims against reference material.

## Learning Path

| File / Folder | Description |
|---|---|
| [**01 - Instruction Dataset Readiness**](01-instruction-dataset-readiness.ipynb) | Validate examples and evidence links, check train/evaluation separation, and compare completion and chat formats. |
| [**02 - LoRA and QLoRA Mechanics**](02-lora-qlora-mechanics.ipynb) | Inspect trainable parameters and explain adapter and quantization memory trade-offs. |
| [**03 - Local LoRA Fine-Tuning Run**](03-local-lora-fine-tuning-run.ipynb) | Train a small causal language model with response-only loss and save an adapter. |
| [**04 - Base vs Fine-Tuned Evaluation**](04-base-vs-finetuned-evaluation.ipynb) | Compare held-out answers with ROUGE, BLEU, conditional perplexity, and error analysis. |
| [**05 - LLM-as-Judge and Faithfulness Checks**](05-llm-as-judge-and-faithfulness.ipynb) | Inspect local claim checks and optionally judge the saved answers with a hosted model. |

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**assets**](assets/) | Visual aids referenced in the notebooks. |
| [**data**](data/) | Training examples, held-out cases, and approved policy references. |
| [**.env.example**](.env.example) | Optional judge provider configuration template. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Setup

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a **placeholder**. Replace it, including the `< >` brackets, with your own value. For example, `cd <repo-name>` becomes `cd mle-fine-tuning-small-models`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

The command below installs all dependencies and creates a virtual environment in `.venv/`.

```bash
cd <repo-name>
uv sync
```

Notebooks 02-04 download [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) on first use, then reuse the Hugging Face cache. The pinned checkpoint is approximately 270 MB. Training runs locally on CPU, without a GPU or API key; its duration depends on your hardware. Notebook 03 saves its adapter in the ignored `outputs/` folder; run it before notebook 04.

---

### 5. Set Up Environment Variables (Optional)

For live judging in notebook 05, copy the example file and fill in your API key:

```bash
cp .env.example .env
```

Open `.env` and set `LLM_PROVIDER` to `groq` or `mistral`, then add the matching key:

- **Groq (primary):** Create a key at [console.groq.com/keys](https://console.groq.com/keys) and set `GROQ_API_KEY`.
- **Mistral (alternative):** Create a key following the [Mistral API key documentation](https://docs.mistral.ai/admin/security-access/api-keys) and set `MISTRAL_API_KEY`.

`LLM_PROVIDER` selects which provider to try first. If you set both keys, the second provider is used as a fallback.

> [!CAUTION]
> Never commit `.env`. It is git-ignored because environment files often hold secrets. Only `.env.example`, with placeholder or default values, belongs in the repository.

---

### 6. Open the Repository in VS Code

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open a notebook and select the Python environment created by `uv sync` as the kernel.

## API Usage Notes

- All five notebooks run without an API key. Notebook 05 then labels its illustrative scores as **simulated**, not measured LLM judgments. Provider errors remain visible and never count as successful live judgments.
- `MAX_JUDGE_ANSWERS=8` lets notebook 05 judge all eight candidate answers per run: the base and adapted answers for all four held-out cases. Lower it to reduce provider calls, then restart the kernel. Each answer may try both configured providers; this is an answer-count limit, not a token or spending cap.
- Live calls may incur costs or hit rate limits. Provider free tiers, model availability, and billing rules can change; check the provider console before running live experiments.

## Professional Tooling Context

The real local training path uses Transformers, PEFT, and PyTorch with an explicit optimization loop. Ordinary LoRA keeps the CPU path portable; QLoRA is explained but not executed. Notebooks 03-04 use SmolLM2's native chat template consistently for training, generation, and conditional perplexity.

The twelve training examples demonstrate the workflow, not a production support model. The four held-out prompts test transfer to new wording in familiar policy domains, not unseen domain knowledge. Lower training loss or better lexical overlap does not prove policy correctness.

Notebook 04 saves all base/adapted answers, per-case metrics, aggregate scores, and configuration to `outputs/model-comparison.json`. Notebook 05 judges a bounded subset, or standalone example candidates if that file is absent, then saves scores and their source to `outputs/judge-comparison.json`. Lexical matches are human-review signals, not proof of a policy violation. Keep generated artifacts out of Git.
