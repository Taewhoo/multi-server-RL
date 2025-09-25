# ASearcher - Fully Asynchronous Agentic RL Training
A fully asynchronous agentic RL training framework for search agent.	

## 🎯 Key Features
+ **Fully Asynchronous RL Training**: trajectory generation and model update are fully decoupled, speeding up training & reducing training cost
+ **Diverse Choices of Search Tools**: Search agent training can use either local knowledge base, web search APIs, or MCP clients.
+ **Async RL Training is especially suitable for cases** where:
    - Excution time of a trajectory is very long.
    - Trajectories can not be stopped, e.g. the server state is hard to save and load
+ **User-friendly Development**: users can implement their own agent without touching any system-level codes

# Preparation

**Step 1:** Prepare the runtime environment.

Please refer to https://inclusionai.github.io/AReaL/tutorial/installation.html#runtime-environment for `Runtime Environment`.


**Step 2:** download training data from [ASearcher-train-data](https://huggingface.co/datasets/inclusionAI/ASearcher-train-data).

# Train a Search Agent


## A. Train a Search Agent with Web Search
**Step 1.** Setup Environment Variable

```shell
export SERPER_API_KEY=YOUR_SERPER_API_KEY
export JINA_API_KEY=YOUR_JINA_API_KEY
```

Here `SERPER_API_KEY` is for the [serper](https://serper.dev/api-keys) API used for Web search. The underlying search engine is Google search, `JINA_API_KEY` is for the [Jina](https://jina.ai/api-dashboard/reader) API used for read the content from thr URLs.

**Step 2**. Launch Training

(Recommended) You can run distributed experiments with Ray or Slurm

```shell
cd AReaL

python3 -m areal.launcher.ray ASearcher/train/asearcher.py \
    --config ASearcher/configs/asearcher_web_16nodes.yaml \
    experiment_name=<your experiment name> \
    trial_name=<your trial name> \
    actor.path=Qwen/Qwen2.5-7B/ \
    train_dataset.path=/path/to/training_data.jsonl \
    allocation_mode=sglang.d96p1t1+d32p1t1 \
    cluster.n_nodes=16 \
    cluster.n_gpus_per_node=8
```

Run the following command to launch training on a single node:

```shell
cd AReaL

python3 -m areal.launcher.local ASearcher/train/asearcher.py \
    --config ASearcher/configs/asearcher_web.yaml \
    experiment_name=<your experiment name> \
    trial_name=<your trial name> \
    actor.path=Qwen/Qwen2.5-7B/ \
    train_dataset.path=/path/to/training_data.jsonl \
    trial_name=<your trial name>
```




## B. Training a Search Agent with Local Knowledge Base
**Step 1.** Setup Environment Variable

```shell
export RAG_SERVER_ADDR_DIR=PATH_TO_DUMP_LOCAL_SERVER_ADDRESS
```

Here `RAG_SERVER_ADDR_DIR` is the directory to dump the address of the launched local RAG server, which will be loaded during training.

**Step 2**. Set up and launch the local RAG server

+ Step 2.1. Download the [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) model, [corpus file and webpage file](https://huggingface.co/datasets/inclusionAI/ASearcher-Local-Knowledge)

+ Step 2.2 Build the index (need e5-base-v2 model and wiki corpus):

```shell
bash scripts/build_index.sh
```

+ Step 2.3. Launch the local RAG server

```shell
bash scripts/launch_local_server.sh $PORT $RAG_SERVER_ADDR_DIR
```

**Step 3**. Launch Training


(Recommended) You can run distributed experiments with Ray or Slurm

```shell
cd AReaL
python3 -m areal.launcher.slurm ASearcher/train/asearcher.py \
    --config ASearcher/configs/asearcher_local.yaml \
    experiment_name=<your experiment name> \
    trial_name=<your trial name> \
    actor.path=Qwen/Qwen2.5-7B/ \
    train_dataset.path=/path/to/training_data.jsonl \
    allocation_mode=sglang.d96p1t1+d32p1t1 \
    cluster.n_nodes=16 \
    cluster.n_gpus_per_node=8
```

Run the following command to launch training on a single node:

```shell
cd AReaL
python3 -m areal.launcher.local ASearcher/train/asearcher.py \
    --config ASearcher/configs/asearcher_local.yaml \
    experiment_name=<your experiment name> \
    trial_name=<your trial name> \
    actor.path=Qwen/Qwen2.5-7B/ \
    train_dataset.path=/path/to/training_data.jsonl \
```

## C. Fine-tuning a QwQ-32B Agent
Coming soon!! Please stay tuned!!

We are still working on cleaning the code and integration with AReaL-lite. An uncleaned version developped based on an legacy version of AReaL could be found in [https://github.com/inclusionAI/AReaL/tree/gjx/agent?tab=readme-ov-file](https://github.com/inclusionAI/AReaL/tree/gjx/agent?tab=readme-ov-file)


# Customization

Please refer to our [guideline](../docs/guideline.md) for more information about building a custom agent.

