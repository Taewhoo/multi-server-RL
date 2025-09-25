# Copyright 2025 Ant Group Inc.
# Modified ASearcher training script with multi-server support

import asyncio
import gc
import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import torch
import torch.distributed as dist

from areal.api.cli_args import (
    GenerationHyperparameters,
    GRPOConfig,
    load_expr_config,
)
from areal.api.io_struct import (
    FinetuneSpec,
    LLMRequest,
    WeightUpdateMeta,
)
from areal.api.workflow_api import RolloutWorkflow
from areal.utils.data import concat_padded_tensors
from areal.utils.device import log_gpu_stats
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import logging, seeding, stats_tracker
from torchdata.stateful_dataloader import StatefulDataLoader
from tensordict import TensorDict
from transformers import PreTrainedTokenizerFast

# Import our custom multi-server toolbox
from ASearcher.utils.multi_server_search_tool import MultiServerSearchToolBox
from ASearcher.train.search_agent import SearchAgent
from ASearcher.train.prompts import (
    INVALID_PROMPT,
    SEARCH_ACCESS_PROMPT_TEMPLATE,
    SEARCH_ONLY_PROMPT_TEMPLATE,
    VALID_PROMPT,
)
from ASearcher.utils.rewards import correct_format_fn

from areal.api.cli_args import AgentRLConfig
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

logger = logging.getLogger("Multi-Server ASearcher")

def hash(numbers):
    """Hash an entire list of integers as a single string"""
    # Convert list to string representation
    list_str = json.dumps(numbers, sort_keys=True)  # sort_keys for consistency
    return hashlib.sha256(list_str.encode()).hexdigest()


class MultiServerASearcherWorkflow(RolloutWorkflow):
    """
    ASearcher workflow that supports multiple retriever servers with group-wise 
    reward normalization. Each group of trajectories uses a dedicated server.
    """
    
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        dataset_path: str,
        server_configs: List[Dict[str, Any]],  # New: list of server configurations
        dump_dir: str | None = None,
        max_turns: int = 128,
        n_trajs: int = 16,
        n_trajs_per_server: int = 4,  # New: trajectories per server
        search_client_type: str = "async-search-access",
        reward_type: str = "F1",
        topk: int = 5,
        valid_inst_ratio: float = 1.0,
        max_tokens: int = 32000,
        search_only: bool = True,
    ):
        """
        Args:
            server_configs: List of dicts with 'address' and 'port' keys for each server
            n_trajs_per_server: Number of trajectories to assign to each server
        """
        self.gconfig = gconfig
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.n_trajs = n_trajs
        self.n_trajs_per_server = n_trajs_per_server
        self.reward_type = reward_type
        self.topk = topk
        self.valid_inst_ratio = valid_inst_ratio
        self.max_tokens = max_tokens
        self.search_only = search_only
        self.dump_dir = dump_dir
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)
        
        # Validate server configuration
        if len(server_configs) == 0:
            raise ValueError("At least one server configuration must be provided")
        
        n_server_groups = (n_trajs + n_trajs_per_server - 1) // n_trajs_per_server
        if len(server_configs) < n_server_groups:
            logger.warning(f"Only {len(server_configs)} servers provided but {n_server_groups} groups needed. "
                          f"Some servers will handle multiple groups.")
        
        # Initialize multi-server toolbox
        self.toolbox = MultiServerSearchToolBox(
            dataset_path=dataset_path, 
            server_configs=server_configs,
            reward_type=self.reward_type, 
            topk=self.topk, 
            search_client_type=search_client_type
        )

    async def collect_agent_trajectory(self, valid_inst, qid, prompt, prompt_token_ids, engine, trajectory_index):
        """
        Collect a single agent trajectory using the assigned server.
        
        Args:
            trajectory_index: Index of this trajectory (0-based) for server assignment
        """
        agent = SearchAgent(prompt, prompt_token_ids)
        score = 0
        ground_truth = None
        
        # a unique trajectory rid to ensure all requests goes to the same sglang server
        traj_rid = uuid.uuid4().hex
        
        while agent.num_turns < self.max_turns and not agent.is_finished:
            # The agent prepares the prompt and sampling params for LLM generation
            input_ids, sampling_params = agent.prepare_llm_query(self.tokenizer)

            # Send request to inference engine and get response
            req = LLMRequest(
                rid=traj_rid,
                input_ids=input_ids,
                gconfig=self.gconfig.new(n_samples=1),
            )
            if "stop" in sampling_params:
                req.gconfig.stop = sampling_params["stop"]
            if len(input_ids) + self.gconfig.max_new_tokens >= self.max_tokens:
                break
            resp = await engine.agenerate(req)
            completion_str = self.tokenizer.decode(resp.output_tokens)

            # agent extracts tool callings from the llm response
            tool_calls = agent.consume_llm_response(resp, completion_str)

            # call tool and compute reward using assigned server
            if tool_calls is not None and len(tool_calls) > 0:
                tool_call = tool_calls[0]
                # Pass trajectory_index to determine server assignment
                res = (await self.toolbox.step(
                    (qid, [tool_call]), 
                    trajectory_index=trajectory_index,
                    n_trajs_per_server=self.n_trajs_per_server
                ))[0]
                
                agent.consume_tool_response(res, topk=self.topk)

                if "score" in res:
                    score = res["score"]
                if "ground_truth" in res:
                    ground_truth = res["ground_truth"]

            if resp.output_tokens[-1] in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]:
                break

        llm_gen_records = agent.memory.filter_records("llm_gen")
        format_reward = float(all([correct_format_fn(i, r.text) for i, r in enumerate(llm_gen_records)]))

        # compute rewards
        score = (score or 0) * format_reward
        pred_answer = agent.get_answer()
        judge_q_invalid = False
        if pred_answer is not None:
            judge_q_invalid = any([_c in pred_answer for _c in ["question", "invalid", "appropriate", "valid"]])
        if valid_inst and judge_q_invalid:
            score = -0.5
        
        stats = agent.memory.logging_stats()
        stats.update(dict(
            score=score,
            judge_q_invalid = judge_q_invalid,
            format_reward=format_reward,
        ))

        return ground_truth, score, agent.memory, stats

    async def arun_episode(self, engine, data):
        # Get the unique identifier for this prompt
        qid = None
        for key in ["query_id", "id", "qid"]:
            qid = data.get(key, None)
            if qid is not None:
                break
        qid = str(qid) or uuid.uuid4().hex

        # check for generated qid when resuming
        if self.dump_dir is not None:
            import glob
            _pattern = os.path.join(self.dump_dir, "*", f"{qid}.jsonl")
            if len(glob.glob(_pattern)) > 0:
                logger.info(f"{qid} is already trained on")
                return None

        # Initialize and Prepare the prompt
        version = engine.get_version()
        prompt_template = SEARCH_ONLY_PROMPT_TEMPLATE if self.search_only else SEARCH_ACCESS_PROMPT_TEMPLATE
        prompt = prompt_template.format(question=data["question"])
        valid_inst: bool = np.random.uniform(0, 1) <= self.valid_inst_ratio
        if valid_inst:
            prompt = prompt.replace(INVALID_PROMPT, VALID_PROMPT)
        prompt_token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        # Collect trajectories with server assignment
        trajs = await asyncio.gather(*[
            self.collect_agent_trajectory(valid_inst, qid, prompt, prompt_token_ids, engine, trajectory_index=i) 
            for i in range(self.n_trajs)
        ])

        ground_truth, scores, results, stats = None, [], [], []
        for gt, score, traj, traj_stats in trajs:
            if gt is not None:
                ground_truth = gt
            scores.append(score)
            stats.append(traj_stats)
        
        raw_scores = scores
        
        # SERVER-GROUP-WISE REWARD NORMALIZATION
        # Normalize rewards within each server group (4 trajectories each)
        normalized_scores = [0.0] * len(scores)
        n_servers = len(self.toolbox.server_configs)
        
        for server_idx in range(n_servers):
            # Get trajectory indices for this server
            start_idx = server_idx * self.n_trajs_per_server
            end_idx = min(start_idx + self.n_trajs_per_server, len(scores))
            
            if start_idx < len(scores):
                # Extract scores for this server group
                server_scores = scores[start_idx:end_idx]
                
                # Normalize within this server group only
                if len(server_scores) > 0:
                    server_mean = np.mean(server_scores)
                    server_normalized = [s - server_mean for s in server_scores]
                    
                    # Put normalized scores back
                    for j, norm_score in enumerate(server_normalized):
                        normalized_scores[start_idx + j] = norm_score
        
        scores = normalized_scores

        # Check if all scores are zero (same as original)
        if all([s==0 for s in scores]):
            return None

        # Pack trajectories into training format (matching original logic exactly)
        traj_memories = [traj for _, _, traj, _ in trajs]
        results = []
        for i, traj_memory in enumerate(traj_memories):
            seqs = []
            for j, record in enumerate(traj_memory.memory):
                if record.type != "llm_gen":
                    continue

                # Check whether any previous seq is equivalent to input tokens
                success = False
                for seq in seqs:
                    if record.input_len < len(seq["input_ids"]):
                        continue
                    h_cur = hash(record.input_tokens[:len(seq["input_ids"])])
                    h_seq = hash(seq["input_ids"])
                    if h_cur == h_seq:
                        seq_len = len(seq["input_ids"])
                        seq["input_ids"] = record.input_tokens + record.output_tokens
                        seq["logprobs"] += [0.0] * (record.input_len - seq_len) + record.output_logprobs
                        seq["loss_mask"] += [0] * (record.input_len - seq_len) + [1] * record.output_len
                        seq["versions"] += [-1] * (record.input_len - seq_len) + record.output_versions
                        success = True
                        break
                if not success:
                    seq = dict(
                        input_ids = record.input_tokens + record.output_tokens,
                        logprobs = [0.0] * record.input_len + record.output_logprobs,
                        loss_mask = [0] * record.input_len + [1] * record.output_len,
                        versions = [-1] * record.input_len + record.output_versions,
                    )
                    seqs.append(seq)

            traj_stats = stats.pop(0)
            first_llm_gen = True
        
            for seq in seqs:
                res = dict(
                    # unsqueeze to add an additional batch dimension
                    input_ids=torch.tensor(seq["input_ids"]).unsqueeze(0),
                    loss_mask=torch.tensor(seq["loss_mask"]).unsqueeze(0),
                    logprobs=torch.tensor(seq["logprobs"]).unsqueeze(0),
                    versions=torch.tensor(seq["versions"]).unsqueeze(0),
                    attention_mask=torch.ones(len(seq["input_ids"]), dtype=torch.bool).unsqueeze(0),
                    # reward
                    rewards=torch.tensor([float(scores[i])]),
                )

                res.update(dict(begin_of_trajectory=torch.tensor([int(first_llm_gen)]),))
                res.update({k: torch.tensor([v]) for k, v in traj_stats.items()})
                first_llm_gen = False

                results.append(TensorDict(res, batch_size=[1]))

        if self.dump_dir is not None:
            os.makedirs(os.path.join(self.dump_dir, str(version)), exist_ok=True)

            # Dump rollout to file
            with open(
                os.path.join(self.dump_dir, str(version), f"{qid}.jsonl"), "w"
            ) as f:
                for i, (traj_memory, raw_score) in enumerate(zip(traj_memories, raw_scores)):
                    server_group = i // self.n_trajs_per_server
                    f.write(json.dumps(dict(
                        memory=traj_memory.to_dict(), 
                        reward=raw_score, 
                        ground_truth=ground_truth, 
                        traj_idx=i,
                        server_group=server_group
                    )) + "\n")

        results = concat_padded_tensors(results)
        return results


worker_id = uuid.uuid4().hex[:4]


@dataclass
class MultiServerAgentRLConfig(AgentRLConfig):
    """Extended config class that supports multi-server parameters"""
    n_trajs_per_server: int = field(
        default=4,
        metadata={
            "help": "Number of trajectories to assign to each server"
        }
    )
    server_configs: List[Dict[str, Any]] = field(
        default_factory=list,
        metadata={
            "help": "List of server configurations with 'address' and 'port' keys"
        }
    )


def get_search_dataset(dataset_path, tokenizer, rank, world_size):
    dataset = load_dataset(
        path="json",
        split="train",
        data_files=dataset_path,
    )
    return split_dataset_by_node(dataset, rank=rank, world_size=world_size)


def main(args):
    config, _ = load_expr_config(args, MultiServerAgentRLConfig)
    config: MultiServerAgentRLConfig

    # Validate server configurations
    if config.server_configs is None or len(config.server_configs) == 0:
        raise ValueError("server_configs must be provided with at least one server")
    
    # Keep group_adv_norm setting from config (don't force enable it)

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")

    # Create dataset and dataloaders
    worker_batch_size = config.train_dataset.batch_size // world_size
    train_dataloader = StatefulDataLoader(
        get_search_dataset(config.train_dataset.path, tokenizer, rank, world_size),
        batch_size=config.train_dataset.batch_size // world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout, worker_id=worker_id)
    rollout.initialize(None, ft_spec)

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.initialize(None, ft_spec)
    ref = None

    # Weight update meta
    weight_update_meta = [WeightUpdateMeta.from_disk(config.saver)]
    dist.broadcast_object_list(weight_update_meta, src=0)
    weight_update_meta = weight_update_meta[0]

    # Create rollout workflow with multi-server support
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    workflow = MultiServerASearcherWorkflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
        dataset_path=config.train_dataset.path,
        server_configs=config.server_configs,  # Pass server configs
        max_turns=config.max_turns,
        n_trajs=config.n_trajs,
        n_trajs_per_server=config.n_trajs_per_server,  # Pass n_trajs_per_server
        search_client_type=config.search_client_type,
        reward_type=config.reward_type,
        topk=config.topk,
        valid_inst_ratio=config.valid_inst_ratio,
        max_tokens=config.actor.mb_spec.max_tokens_per_mb,
    )

    logger.info(f"Multi-server setup:")
    logger.info(f"  - {len(config.server_configs)} servers configured")
    logger.info(f"  - {config.n_trajs} total trajectories")
    logger.info(f"  - {config.n_trajs_per_server} trajectories per server")
    logger.info(f"  - Group advantage normalization: {config.actor.group_adv_norm}")
    logger.info(f"  - Group size: {config.actor.group_size}")

    # Run training.
    saver = Saver(config.saver, ft_spec, for_recover=False)
    stat_logger = StatsLogger(config.stats_logger, ft_spec)

    total_epochs = config.total_train_epochs
    
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    # logger.commit(f"total_epochs={total_epochs} step_per_epoch={steps_per_epoch}")
    data_generator = iter(train_dataloader)
    start_step = config.recover_start_step or 0
    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch

        print(f"Epoch {epoch}. Step: {step}/{steps_per_epoch}")

        with stats_tracker.record_timing("rollout"):
            if config.async_training:
                batch = rollout.prepare_batch(train_dataloader, workflow=workflow, expected_batch_size=worker_batch_size)
            else:
                try:
                    data = next(data_generator)
                except StopIteration:
                    data_generator = iter(train_dataloader)
                    data = next(data_generator)
                batch = rollout.rollout_batch(data, workflow=workflow)

        batch = batch.to(actor.device)
        # Create barrier to synchronize all rollout processes.
        dist.barrier(device_ids=[actor.device.index])
        torch.cuda.synchronize()

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)
                log_gpu_stats("ref logp")

        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")
        
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("ppo update")

        with stats_tracker.record_timing("update_weights"):
            rollout.pause()
            if dist.get_rank() == 0:
                future = rollout.update_weights(weight_update_meta)
            actor.upload_weights(weight_update_meta)
            if dist.get_rank() == 0:
                future.result()
            dist.barrier(device_ids=[actor.device.index])
            torch.cuda.synchronize()
            rollout.resume()
            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step)

        stat_logger.commit(epoch, step, global_step, stats)

    stat_logger.close()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:]) 