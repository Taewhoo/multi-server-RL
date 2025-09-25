#!/bin/bash
set -ex
HF_HOME=/raid/taewhoo/.cache/huggingface

WIKI2018_WORK_DIR=/raid/taewhoo/deep_research/ASearcher

# index_file=$WIKI2018_WORK_DIR/e5.index/e5_Flat.index
# index_file=$WIKI2018_WORK_DIR/e5-large.index/e5_Flat.index
index_file=$WIKI2018_WORK_DIR/bm25.index/bm25
corpus_file=$WIKI2018_WORK_DIR/wiki_corpus.jsonl
pages_file=$WIKI2018_WORK_DIR/wiki_webpages.jsonl
retriever_name=bm25
# retriever_path=intfloat/e5-large-v2  # Not needed for BM25
retriever_path="placeholder"

python3  tools/local_retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --pages_path $pages_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --port $1 \
                                            --save-address-to $2