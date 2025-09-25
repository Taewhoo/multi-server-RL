#!/bin/bash
set -ex
HF_HOME=/raid/taewhoo/.cache/huggingface

WIKI2018_WORK_DIR=/raid/taewhoo/deep_research/ASearcher

# index_file=$WIKI2018_WORK_DIR/e5.index/e5_Flat.index
# index_file=$WIKI2018_WORK_DIR/e5-large.index/e5_Flat.index
index_file=$WIKI2018_WORK_DIR/e5_large_sharded_converted.index/corpus.shard*.pkl
corpus_file=$WIKI2018_WORK_DIR/wiki_corpus.jsonl
pages_file=$WIKI2018_WORK_DIR/wiki_webpages.jsonl
retriever_name=e5_large
retriever_path=intfloat/e5-large-v2

python3  tools/local_retrieval_server.py --index_path "$index_file" \
                                            --use_sharded_index \
                                            --corpus_path "$corpus_file" \
                                            --pages_path "$pages_file" \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu \
                                            --port $1 \
                                            --save-address-to $2

### 8001: e5-small, 8002: e5-large, 8003: bm25
### --use_sharded_index: under investigation...