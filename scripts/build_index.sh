#!/bin/bash
HF_HOME=/raid/taewhoo/.cache/huggingface
# save_dir=$WIKI2018_WORK_DIR
save_dir=/raid/taewhoo/deep_research/ASearcher

corpus_file=$save_dir/wiki_corpus.jsonl
save_dir=${save_dir}/e5_large_tmp.index
# save_dir=${save_dir}/bm25.index
retriever_name=e5_large # this is for indexing naming
# retriever_name=bm25
retriever_model=intfloat/e5-large-v2
# retriever_model=placeholder

# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 utils/index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding
