# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import argparse
import json 
import random
import pickle
import time
import glob
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import pipeline

import src.index
import src.contriever
import src.utils
import src.slurm
import src.data
from src.evaluation import calculate_matches
import src.normalize_text

current_dir = os.path.dirname(os.path.abspath(__file__))
great_grandparent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(great_grandparent_dir)

from utils.normalize_answers import is_answer_in_text
os.chdir(sys.path[0])
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' 
device = 1
from sentence_transformers import SentenceTransformer
sentence_model = SentenceTransformer("./all-MiniLM-L6-v2", device="cuda:0")


class Retriever:
    def __init__(self, args, model=None, tokenizer=None) :
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

    def embed_queries(self, args, queries):
        embeddings, batch_question = [], []
        with torch.no_grad():
            for k, q in enumerate(queries):
                if args.lowercase:
                    q = q.lower()
                if args.normalize_text:
                    q = src.normalize_text.normalize(q)
                batch_question.append(q)

                if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:

                    encoded_batch = self.tokenizer.batch_encode_plus(   # batch_encode_plus is a method of the tokenizer, which is a BertTokenizerFast object
                        batch_question,
                        return_tensors="pt",
                        max_length=args.question_maxlength,
                        padding=True,
                        truncation=True,
                    )
                    encoded_batch = {k: v.cuda(device) for k, v in encoded_batch.items()}
                    output = self.model(**encoded_batch)
                    embeddings.append(output.cpu())

                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0)
        print(f"Questions embeddings shape: {embeddings.size()}")

        return embeddings.numpy()
    

    def embed_queries_demo(self, queries):
        embeddings, batch_question = [], []
        with torch.no_grad():
            for k, q in enumerate(queries):
                batch_question.append(q)

                if len(batch_question) == 16 or k == len(queries) - 1:

                    encoded_batch = self.tokenizer.batch_encode_plus(
                        batch_question,
                        return_tensors="pt",
                        max_length=200,
                        padding=True,
                        truncation=True,
                    )
                    encoded_batch = {k: v.cuda(device) for k, v in encoded_batch.items()}
                    output = self.model(**encoded_batch)
                    embeddings.append(output.cpu())

                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0)
        print(f"Questions embeddings shape: {embeddings.size()}")

        return embeddings.numpy()

    def index_encoded_data(self, index, embedding_files, indexing_batch_size):
        allids = []
        allembeddings = np.array([])
        for i, file_path in enumerate(embedding_files):
            print(f"Loading file {file_path}")
            with open(file_path, "rb") as fin:
                ids, embeddings = pickle.load(fin)

            allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
            allids.extend(ids)
            while allembeddings.shape[0] > indexing_batch_size:
                allembeddings, allids = self.add_embeddings(index, allembeddings, allids, indexing_batch_size)

        while allembeddings.shape[0] > 0:
            allembeddings, allids = self.add_embeddings(index, allembeddings, allids, indexing_batch_size)

        print("Data indexing completed.")


    def add_embeddings(self, index, embeddings, ids, indexing_batch_size):
        end_idx = min(indexing_batch_size, embeddings.shape[0])
        ids_toadd = ids[:end_idx]
        embeddings_toadd = embeddings[:end_idx]
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        index.index_data(ids_toadd, embeddings_toadd)
        return embeddings, ids


    def add_passages(self, passages, top_passages_and_scores):
        # add passages to original data
        docs = [passages[doc_id] for doc_id in top_passages_and_scores[0][0]]
        return docs


    def setup_retriever(self):
        print(f"Loading model from: {self.args.model_name_or_path}")
        self.model, self.tokenizer, _ = src.contriever.load_retriever(self.args.model_name_or_path)
        self.model.eval()
        self.model = self.model.cuda(device)
        if not self.args.no_fp16:
            self.model = self.model.half()

        self.index = src.index.Indexer(self.args.projection_size, self.args.n_subquantizers, self.args.n_bits)

        # index all passages
        input_paths = glob.glob(self.args.passages_embeddings)
        input_paths = sorted(input_paths)
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, "index.faiss")
        if self.args.save_or_load_index and os.path.exists(index_path):
            self.index.deserialize_from(embeddings_dir)
        else:
            print(f"Indexing passages from files {input_paths}")
            start_time_indexing = time.time()
            self.index_encoded_data(self.index, input_paths, self.args.indexing_batch_size)
            print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
            if self.args.save_or_load_index:
                self.index.serialize(embeddings_dir)

        # load passages
        print("loading passages")
        self.passages = src.data.load_passages(self.args.passages)
        self.passage_id_map = {x["id"]: x for x in self.passages}
        print("passages have been loaded")


    def search_document(self, query, top_n=10):
        questions_embedding = self.embed_queries(self.args, [query])

        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = self.index.search_knn(questions_embedding, self.args.n_docs)
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

        return self.add_passages(self.passage_id_map, top_ids_and_scores)[:top_n]
    
    
    def search_document_demo(self, query, n_docs=10):
        questions_embedding = self.embed_queries_demo([query])

        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = self.index.search_knn(questions_embedding, n_docs)
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

        return self.add_passages(self.passage_id_map, top_ids_and_scores)[:n_docs]


    def setup_retriever_demo(self, model_name_or_path, passages, passages_embeddings, n_docs=5, save_or_load_index=False):
        print(f"Loading model from: {model_name_or_path}")
        self.model, self.tokenizer, _ = src.contriever.load_retriever(model_name_or_path)
        self.model.eval()
        # self.model = self.model.to('cuda:1')
        self.model = self.model.cuda(device)

        self.index = src.index.Indexer(768, 0, 8)

        # index all passages
        input_paths = glob.glob(passages_embeddings)
        input_paths = sorted(input_paths)
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, "index.faiss")
        if save_or_load_index and os.path.exists(index_path):
            self.index.deserialize_from(embeddings_dir)
        else:
            print(f"Indexing passages from files {input_paths}")
            start_time_indexing = time.time()
            self.index_encoded_data(self.index, input_paths, 1000000)
            print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")

        # load passages
        print("loading passages")
        self.passages = src.data.load_passages(passages)
        self.passage_id_map = {x["id"]: x for x in self.passages}
        print("passages have been loaded")


def add_hasanswer(data, hasanswer):
    # add hasanswer to data
    for i, ex in enumerate(data):
        for k, d in enumerate(ex["ctxs"]):
            d["hasanswer"] = hasanswer[i][k]


def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    return data


def main(args):
    # for debugging
    # data_paths = glob.glob(args.data)
    src.slurm.init_distributed_mode(args)
    retriever = Retriever(args)
    retriever.setup_retriever()
    input_data = load_data(args.query)
    new_data = []
    n_all_data = len(input_data)
    print(f'-------Now processing {args.query} with {n_all_data} items-------')
    try:
        for i, item in enumerate(input_data):
            print(f'------------------- Processing {i+1}/{n_all_data} -------------------')
            question = item['question'].strip() if item['question'].strip()[-1] == "?" else item['question'].strip() + "?"
            answers = item['answers']
            print(f"Searching for question: {question}")
            
            num_supportive = 0
            num_semantic = 0
            item['supportive_text'], item['semantic_text'] = [], []
            
            doc = retriever.search_document(question, args.n_docs)
            for d in doc:
                if not is_answer_in_text(d['text'], item["answers"]):
                    item['supportive_text'].append(d)
                    num_supportive += 1
                    if num_supportive == 10:
                        break
            
            t = i + 1
            if t >= len(input_data) - 6:
                t = 0
            q_semantic = input_data[t]['question']
            num = 0
            while num_semantic < 10:
                while (sentence_model.encode(question, normalize_embeddings=True)*sentence_model.encode(q_semantic, normalize_embeddings=True)).sum() >= 0.20:
                    num += 1
                    if num == len(input_data):
                        break
                    t += 1
                    if t >= len(input_data) - 6:
                        t = 0
                    q_semantic = input_data[t]['question']
                doc = retriever.search_document(q_semantic, 3)
                item['semantic_text'] += doc
                num_semantic += 3
                t += 1
                q_semantic = input_data[t]['question']
            
            random.shuffle(item['semantic_text'])
            item['semantic_text'] = item['semantic_text'][:10]
            new_data.append(item)
        
        json.dump(new_data, open(args.output_file, 'w'), indent=2)
    
    except Exception as e:
        print(f"Error: {e}")
        json.dump(new_data, open(args.output_file, 'w'), indent=2)
        print(f"Data saved to {args.output_file}")
        

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset", type=str, default="nq", help="Dataset to use for retrieval"
    )
    parser.add_argument(
        "--query", type=str, default="nq_raw.json", help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument(
        "--output_file", type=str, default="nq.json", help="Output Path for the processed data"
    )
    parser.add_argument("--passages", type=str, default="./wikipedia_passages/psgs_w100.tsv", help="Path to passages (.tsv file)")
    parser.add_argument("--passages_embeddings", type=str, default="./wikipedia_embeddings/*", help="Glob path to encoded passages")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Results are written to outputdir with data suffix"
    )
    parser.add_argument("--n_docs", type=int, default=20, help="Number of documents to retrieve per questions")
    parser.add_argument(
        "--validation_workers", type=int, default=32, help="Number of parallel processes to validate results"
    )
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default="./contriever-msmarco/", help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument(
        "--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed"
    )
    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--lang", nargs="+")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="normalize text")

    args = parser.parse_args()
    return args
    


if __name__ == "__main__":
    args = get_parser()
    main(args)