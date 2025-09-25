# Copyright 2025 Ant Group Inc.
import json
import random
import aiohttp
import asyncio
from typing import List, Tuple, Dict, Any

from realhf.base import logging
from ASearcher.utils.rewards import compute_score_em, compute_score_f1
from ASearcher.utils.search_utils import make_search_client

logger = logging.getLogger("Multi-Server Search ToolBox")


class MultiServerAsyncSearchBrowserClient:
    """
    Dedicated AsyncSearchBrowserClient for multi-server support.
    Takes explicit address and port parameters, unlike the original which uses config files.
    """
    def __init__(self, address: str, port: int):
        self.session = None
        self.server_addr = f"http://{address}:{port}"
        
    async def query_async(self, req_meta: Dict[str, Any]) -> List[Dict]:
        cnt = 0
        last_exception = None
        while cnt < 10:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{self.server_addr}/retrieve", json=req_meta) as response:
                        if response.status == 200:
                            result = await response.json()
                            # Server returns {"result": [[{...}, ...]]}
                            # Transform to format expected by original code: [{"documents": [...], "urls": [...]}]
                            documents = []
                            urls = []
                            if "result" in result and len(result["result"]) > 0:
                                for doc in result["result"][0]:
                                    documents.append(doc.get("contents", ""))
                                    urls.append(doc.get("url", ""))
                            return [{"documents": documents, "urls": urls}]
                        else:
                            raise Exception(f"HTTP {response.status}: {await response.text()}")
            except Exception as e:
                last_exception = e
                cnt += 1
                if cnt < 10:
                    await asyncio.sleep(0.1)
                else:
                    logger.error(f"Failed to query {self.server_addr} after 10 retries: {e}")
                    
        if last_exception:
            raise last_exception
        return []
        
    async def access_async(self, urls: List[str]) -> Dict:
        cnt = 0
        last_exception = None
        while cnt < 10:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{self.server_addr}/access", json={"urls": urls}) as response:
                        if response.status == 200:
                            result = await response.json()
                            # Server returns {"result": [null]} or {"result": [{"contents": "..."}]}
                            # Original code expects exactly this format
                            return result
                        else:
                            raise Exception(f"HTTP {response.status}: {await response.text()}")
            except Exception as e:
                last_exception = e
                cnt += 1
                if cnt < 10:
                    await asyncio.sleep(0.1)
                else:
                    logger.error(f"Failed to access {self.server_addr} after 10 retries: {e}")
                    
        if last_exception:
            raise last_exception
        return {"result": []}

def load_metadata(dataset_path):
    data=[json.loads(ff) for ff in open(dataset_path)]
    for d in data:
        if "idx" in d:
            d["idx"] = str(d["idx"])
        elif "qid" in d:
            d["idx"] = str(d["qid"])
        else:
            d["idx"] = str(d["id"])
    id2info = {d["idx"]: d for d in data}
    return id2info


class MultiServerSearchToolBox:
    """
    SearchToolBox that manages multiple retriever servers and assigns 
    trajectory groups to specific servers for group-wise reward normalization.
    """
    
    def __init__(
        self, 
        dataset_path: str, 
        server_configs: List[Dict[str, Any]], 
        reward_type: str = "F1", 
        topk: int = 10, 
        search_client_type: str = "async-search-access", 
        use_jina: bool = False
    ):
        """
        Args:
            dataset_path: Path to dataset
            server_configs: List of server configs, each containing {'address': str, 'port': int}
            reward_type: Type of reward computation
            topk: Number of top results to retrieve
            search_client_type: Type of search client
            use_jina: Whether to use Jina
        """
        self.id2info = load_metadata(dataset_path)
        self.reward_type = reward_type
        self.topk = topk
        self.use_jina = use_jina
        self.search_client_type = search_client_type
        
        # Create multiple search clients for different servers
        self.search_clients = []
        self.server_configs = server_configs
        
        for i, server_config in enumerate(server_configs):
            print(f"Initializing Search Client {i+1}/{len(server_configs)}:")
            print(f"  Server: http://{server_config['address']}:{server_config['port']}")
            
            # Create search client for this server (following original pattern)
            if search_client_type == "async-search-access":
                # Create dedicated multi-server client with explicit server config
                search_client = MultiServerAsyncSearchBrowserClient(
                    address=server_config['address'], 
                    port=server_config['port']
                )
            else:
                # For other client types, use make_search_client
                # Note: This will use default config from evaluation.config_loader
                search_client = make_search_client(search_client_type, use_jina=use_jina)
            
            self.search_clients.append(search_client)
        
        print(f"Initialized {len(self.search_clients)} search clients")
    
    def get_server_for_trajectory(self, trajectory_index: int, n_trajs_per_server: int = 4) -> int:
        """
        Determine which server should handle a specific trajectory.
        
        Args:
            trajectory_index: Index of the trajectory (0-based)
            n_trajs_per_server: Number of trajectories per server
            
        Returns:
            Server index to use for this trajectory
        """
        server_index = trajectory_index // n_trajs_per_server
        return server_index % len(self.search_clients)
    
    async def step(self, qid_actions: Tuple[str, List[str]], trajectory_index: int = 0, n_trajs_per_server: int = 4):
        """
        Execute search actions using the appropriate server for this trajectory group.
        
        Args:
            qid_actions: Tuple of (query_id, list_of_actions)
            trajectory_index: Index of current trajectory
            n_trajs_per_server: Number of trajectories per server
        """
        qid, actions = qid_actions
        
        # Determine which server to use
        server_index = self.get_server_for_trajectory(trajectory_index, n_trajs_per_server)
        search_client = self.search_clients[server_index]
        
        logger.info(f"Trajectory {trajectory_index} assigned to server {server_index} "
                   f"({self.server_configs[server_index]['address']}:{self.server_configs[server_index]['port']})")

        results = []
        for action in actions:
            result = dict(documents=None, score=None, ground_truth=None, type=None, server_index=server_index)

            # tool calling
            if "<search>" in action and "</search>" in action:
                query = action.split("<search>")[-1].split("</search>")[0].strip()
                req_meta = {
                    "queries": [query],
                    "topk": self.topk,
                    "return_scores": False
                }

                # send search query to server
                response = await search_client.query_async(req_meta)
                
                documents = response[0]["documents"]
                urls = response[0]["urls"]

                result["documents"] = documents
                result["urls"] = urls
                result["type"] = "search"
                
            elif "<access>" in action and "</access>" in action:
                url = action.split("<access>")[-1].split("</access>")[0].strip()

                # send webpage access request to assigned server
                response = await search_client.access_async([url])

                page = None
                if self.search_client_type == "async-online-search-access":
                    if self.use_jina:
                        page = response[0].get("page", "")
                    else:
                        # process webpage
                        page = self.process_webpage(response[0].get("page", ""))
                elif self.search_client_type == "async-search-access":
                    if response["result"][0] is None:
                        page = None
                    else:
                        page = response["result"][0]["contents"]
            
                result["page"] = page
                result["type"] = "access"

            # compute rewards
            ground_truth = self.id2info[qid.split("@")[0]]["answer"]
            if isinstance(ground_truth, list) or isinstance(ground_truth, tuple):
                ground_truth = [str(gt) for gt in ground_truth]
            else:
                ground_truth = str(ground_truth)

            ground_truth_aug = None
            if "aug_answer" in self.id2info[qid.split("@")[0]] and len(self.id2info[qid.split("@")[0]]["aug_answer"]) > 0:
                ground_truth_aug = self.id2info[qid.split("@")[0]]["aug_answer"]
                if isinstance(ground_truth_aug, list) or isinstance(ground_truth_aug, tuple):
                    ground_truth_aug = [str(gt) for gt in ground_truth_aug]
                else:
                    ground_truth_aug = str(ground_truth_aug)
            
            if self.reward_type == "F1":
                extracted, score = compute_score_f1(action, ground_truth, method="strict")
                if ground_truth_aug is not None:
                    _, score_aug = compute_score_f1(action, ground_truth_aug, method="strict")
                    score = max(score, score_aug)
            elif self.reward_type == "EM":
                extracted, score = compute_score_em(action, ground_truth, method="strict")
                if ground_truth_aug is not None:
                    _, score_aug = compute_score_em(action, ground_truth_aug, method="strict")
                    score = max(score, score_aug)
            else:
                raise NotImplementedError
            
            result["extracted"] = extracted
            result["score"] = score
            result["ground_truth"] = ground_truth

            results.append(result)
        
        return results
    
    def process_webpage(self, page_content: str) -> str:
        """Process webpage content (same as original SearchToolBox)"""
        # Same implementation as original SearchToolBox.process_webpage
        from bs4 import BeautifulSoup
        import re
        
        if not page_content:
            return ""
        
        try:
            soup = BeautifulSoup(page_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.warning(f"Error processing webpage: {e}")
            return page_content 