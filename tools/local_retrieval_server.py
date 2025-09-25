

import json
import os
import warnings
from typing import List, Dict, Optional
import argparse
import glob
import pickle

import faiss
import torch
import threading
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import socket

def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=4
    )
    return corpus

def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results

def load_model(model_path: str, use_fp16: bool = False):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16: 
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer

def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask = None,
    pooling_method = "mean"
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output,
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        
        del inputs, output
        torch.cuda.empty_cache()

        return query_emb

class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)
    
    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)

class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8
    
    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn('Not enough documents retrieved!')
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [
                json.loads(self.searcher.doc(hit.docid).raw())['contents'] 
                for hit in hits
            ]
            results = [
                {
                    'title': content.split("\n")[0].strip("\""),
                    'text': "\n".join(content.split("\n")[1:]),
                    'contents': content
                } 
                for content in all_contents
            ]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results

class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        
        if config.use_sharded_index:
            self._load_sharded_index(config)
        else:
            self.index = faiss.read_index(self.index_path)
            
        if config.faiss_gpu:
            self._setup_gpu_index(config)

        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name = self.retrieval_method,
            model_path = config.retrieval_model_path,
            pooling_method = config.retrieval_pooling_method,
            max_length = config.retrieval_query_max_length,
            use_fp16 = config.retrieval_use_fp16
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _load_sharded_index(self, config):
        """Load index from sharded pickle files and transfer to GPU incrementally."""
        def pickle_load(path):
            with open(path, 'rb') as f:
                reps, lookup = pickle.load(f)
            return np.array(reps), lookup
        
        index_files = glob.glob(self.index_path)
        print(f'Sharded loading: found {len(index_files)} files matching pattern: {self.index_path}')
        
        if not index_files:
            raise ValueError(f"No files found matching pattern: {self.index_path}")
        
        # Load first shard to initialize index
        print("Loading first shard...")
        p_reps_0, p_lookup_0 = pickle_load(index_files[0])
        
        # Create initial FAISS index
        dimension = p_reps_0.shape[1]
        
        # Check if we should use GPU transfer per shard
        if config.faiss_gpu:
            num_gpus = faiss.get_num_gpus()
            if num_gpus > 0:
                print(f"🚀 Using incremental GPU transfer approach on {num_gpus} GPU(s)")
                self.index = self._create_gpu_index_incrementally(index_files, dimension)
                return
        
        # Fallback to CPU approach
        print("Using CPU FAISS index")
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(p_reps_0)
        self.lookup = list(p_lookup_0)
        
        # Load remaining shards incrementally
        if len(index_files) > 1:
            print(f"Loading remaining {len(index_files)-1} shards...")
            for i, index_file in enumerate(tqdm(index_files[1:], desc='Loading shards')):
                p_reps, p_lookup = pickle_load(index_file)
                self.index.add(p_reps)
                self.lookup.extend(p_lookup)
        
        print(f"✅ Sharded index loaded: {self.index.ntotal} vectors, {dimension} dimensions")

    def _create_gpu_index_incrementally(self, index_files, dimension):
        """Create GPU index by transferring shards one by one."""
        def pickle_load(path):
            with open(path, 'rb') as f:
                reps, lookup = pickle.load(f)
            return np.array(reps), lookup
        
        num_gpus = faiss.get_num_gpus()
        
        # Create empty GPU index
        if num_gpus == 1:
            print("Creating single GPU index with incremental transfer...")
            res = faiss.StandardGpuResources()
            res.setTempMemory(4 * 1024 * 1024 * 1024)  # 4GB temp memory
            
            # Create empty GPU index (correct API)
            config_gpu = faiss.GpuIndexFlatConfig()
            config_gpu.useFloat16 = True
            config_gpu.device = 0
            gpu_index = faiss.GpuIndexFlatIP(res, dimension, config_gpu)
        else:
            print(f"Creating multi-GPU index with incremental transfer...")
            # For multi-GPU, we'll build separate indices and merge
            gpu_indices = []
            resources = []
            for i in range(num_gpus):
                res = faiss.StandardGpuResources()
                res.setTempMemory(4 * 1024 * 1024 * 1024)  # 4GB per GPU
                resources.append(res)
                
                config_gpu = faiss.GpuIndexFlatConfig()
                config_gpu.useFloat16 = True
                config_gpu.device = i
                gpu_idx = faiss.GpuIndexFlatIP(res, dimension, config_gpu)
                gpu_indices.append(gpu_idx)
        
        self.lookup = []
        shard_count = 0
        gpu_assignment = 0
        
        for index_file in tqdm(index_files, desc='Loading shards to GPU'):
            try:
                p_reps, p_lookup = pickle_load(index_file)
                
                if num_gpus == 1:
                    # Add to single GPU
                    gpu_index.add(p_reps)
                else:
                    # Round-robin across GPUs
                    target_gpu = shard_count % num_gpus
                    gpu_indices[target_gpu].add(p_reps)
                
                self.lookup.extend(p_lookup)
                shard_count += 1
                
                print(f"✅ Shard {shard_count}/{len(index_files)} loaded to GPU {target_gpu if num_gpus > 1 else 0}")
                
                # Clean up CPU memory
                del p_reps, p_lookup
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ GPU OOM on shard {shard_count}. Falling back to CPU...")
                    # Fall back to CPU for this and remaining shards
                    return self._fallback_to_cpu_index(index_files, dimension)
                else:
                    raise e
        
        if num_gpus == 1:
            print(f"✅ Incremental GPU index created: {gpu_index.ntotal} vectors")
            return gpu_index
        else:
            # Merge multi-GPU indices
            print("Merging multi-GPU indices...")
            merged_index = faiss.IndexShards(dimension)
            for gpu_idx in gpu_indices:
                merged_index.add_shard(gpu_idx)
            print(f"✅ Multi-GPU sharded index created: {merged_index.ntotal} vectors")
            return merged_index

    def _fallback_to_cpu_index(self, index_files, dimension):
        """Fallback to CPU index if GPU transfer fails."""
        def pickle_load(path):
            with open(path, 'rb') as f:
                reps, lookup = pickle.load(f)
            return np.array(reps), lookup
        
        print("Building CPU index as fallback...")
        cpu_index = faiss.IndexFlatIP(dimension)
        self.lookup = []
        
        for index_file in tqdm(index_files, desc='Loading shards to CPU'):
            p_reps, p_lookup = pickle_load(index_file)
            cpu_index.add(p_reps)
            self.lookup.extend(p_lookup)
            del p_reps, p_lookup
        
        print(f"✅ CPU fallback index created: {cpu_index.ntotal} vectors")
        return cpu_index

    def _setup_gpu_index(self, config):
        """Setup GPU index with memory management."""
        # Skip if index is already on GPU (from incremental loading)
        if hasattr(self.index, 'device') or isinstance(self.index, (faiss.GpuIndex, faiss.IndexShards)):
            print("Index already on GPU, skipping transfer")
            return
            
        num_gpus = faiss.get_num_gpus()
        if num_gpus == 0:
            print("No GPU found. Using CPU FAISS.")
            return
            
        print(f"Index info: {self.index.ntotal} vectors, {self.index.d} dimensions")
        print(f"Estimated index size: {self.index.ntotal * self.index.d * 4 / (1024**3):.1f} GB")
        print(f"Setting up GPU FAISS on {num_gpus} GPU(s)...")
        
        try:
            if num_gpus == 1:
                # Single GPU with float16 and limited temp memory
                print("Attempting single GPU with temp memory limit...")
                res = faiss.StandardGpuResources()
                # Set conservative temp memory (2GB should be enough for most operations)
                res.setTempMemory(2 * 1024 * 1024 * 1024)  # 2GB
                
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index, co)
                print("✅ Index moved to single GPU with float16 and 2GB temp memory")
            else:
                # Multi-GPU sharding (your working approach)
                co = faiss.GpuMultipleClonerOptions()
                co.useFloat16 = True
                co.shard = True
                self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
                print(f"✅ Index sharded across {num_gpus} GPUs with float16")
        except RuntimeError as e:
            print(f"❌ GPU setup failed: {e}")
            print("📝 Falling back to CPU FAISS")
            # Index stays on CPU

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        
        # Handle sharded index lookup
        if hasattr(self, 'lookup'):
            # For sharded indices, map FAISS indices to document IDs
            doc_ids = [self.lookup[idx] for idx in idxs]
            results = load_docs(self.corpus, doc_ids)
        else:
            # For regular FAISS indices, use indices directly
            results = load_docs(self.corpus, idxs)
            
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        
        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            # chunk them back
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]
            
            results.extend(batch_results)
            scores.extend(batch_scores)
            
            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            torch.cuda.empty_cache()
            
        if return_score:
            return results, scores
        else:
            return results

def get_retriever(config):
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)

class PageAccess:
    def __init__(self, pages_path):
        pages = []
        for ff in tqdm(open(pages_path,"r")):
            pages.append(json.loads(ff))
        self.pages = {page["url"]: page  for page in pages}
    
    def access(self, url):
        # php parsing
        if "index.php/" in url:
            url = url.replace("index.php/", "index.php?title=")
        if url not in self.pages:
            return None
        return self.pages[url]

#####################################
# FastAPI server below
#####################################

class Config:
    """
    Minimal config class (simulating your argparse) 
    Replace this with your real arguments or load them dynamically.
    """
    def __init__(
        self, 
        retrieval_method: str = "bm25", 
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        use_sharded_index: bool = False,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.use_sharded_index = use_sharded_index
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size

class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False

class AccessRequest(BaseModel):
    urls: List[str]

app = FastAPI()
threading_lock = threading.Lock()

@app.post("/retrieve")
async def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default

    # Perform batch retrieval
    with threading_lock:
        if request.return_scores:
            results, scores = retriever.batch_search(
                query_list=request.queries,
                num=request.topk,
                return_score=request.return_scores
            )
        else:
            results = retriever.batch_search(
                query_list=request.queries,
                num=request.topk,
                return_score=request.return_scores
            )
        
    # Format response
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            # If scores are returned, combine them with results
            combined = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}

@app.post("/access")
async def access_endpoint(request: AccessRequest):
    resp = []
    with threading_lock:
        for url in request.urls:
            resp.append(page_access.access(url))
    
    return {"result": resp}


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file or glob pattern for sharded pickle files (e.g., /path/to/corpus.shard*.pkl).")
    parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl", help="Local corpus file.")
    parser.add_argument("--use_sharded_index", action="store_true", help="Use sharded pickle index files instead of single FAISS index")
    parser.add_argument("--pages_path", type=str, default="xxx", help="Local page file.")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
    parser.add_argument('--faiss_gpu', action='store_true', help='Use GPU for computation')
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--save-address-to", type=str, help="path to save server address")

    args = parser.parse_args()

    host_name=socket.gethostname()
    host_ip=socket.gethostbyname(socket.gethostname())
    port = args.port

    host_addr = f"{host_ip}:{port}"

    print(f"Server address: {host_addr}")
    
    if args.save_address_to:
        os.makedirs(args.save_address_to, exist_ok=True)
        with open(os.path.join(args.save_address_to, "Host" + host_ip + "_" + "IP" + str(port) + ".txt"), "w") as f:
            f.write(host_addr)

    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    config = Config(
        retrieval_method = args.retriever_name,  # or "dense"
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        use_sharded_index=args.use_sharded_index,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )

    # 2) Instantiate a global retriever so it is loaded once and reused.
    retriever = get_retriever(config)

    print("Retriver is ready.")

    # 3) Load pages
    if os.path.exists(args.pages_path):
        page_access = PageAccess(args.pages_path)

    print("Page Access is ready.")

    # 4) Launch the server.
    config = uvicorn.Config(
        app,
        host=host_addr.split(":")[0],
        port=int(host_addr.split(":")[1]),
        log_level="warning",
    )
    http_server = uvicorn.Server(config)
    http_server.run()

