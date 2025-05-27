import os
import sys
os.chdir(sys.path[0])

from utils.helper_data import load_file
from utils.config import DATA_PATH


def create_dataset(dataset_name, mode, time_type, is_gpt_summary, *args):
    """Create a dataset object."""
    if dataset_name in ["nq", "rgb", "hotpotqa", "2wikimqa", "bamboogle", "tempqa", "priorqa"]:
        return QA_REASONING(dataset_name)
    elif dataset_name == "strategyqa":
        return STRATEGYQA(dataset_name)
    elif dataset_name in ["gsm8k", "gsm8k_hard"]:
        return GSM8K(dataset_name)
    else:
        raise NotImplementedError


class GSM8K(object):
    def __init__(self, dataset_name):
        input_file = DATA_PATH[dataset_name]
        self.data = load_file(input_file)
        self.data_len = len(self.data)

    def get_no_rag(self):
        input_data = []
        for item in self.data:
            input_data.append({"id": item["id"], "question": item["question"], "gold_answers": item["gold_answers"], "counter-truth answer": item["counter-truth answer"], "ctxs": ""})
        return input_data
    

class QA_REASONING(object):
    def __init__(self, dataset_name):
        input_file = DATA_PATH[dataset_name]
        self.dataset_name = dataset_name
        self.data = load_file(input_file)
        self.data_len = len(self.data)

    def get_no_rag(self):
        input_data = []
        for item in self.data:
            if self.dataset_name == "strategyqa":
                input_data.append({"id": item["id"], "question": item["question"], "gold_answers": item["gold_answers"], "counter-truth answer": item["counter-truth answer"], "ctxs": ""})
            else:
                input_data.append({"id": item["id"], "question": item["question"], "gold_answers": item["gold_answers"], "counter-truth answer": item["counter-truth answer"], "counter-truth answer2": item["counter-truth answer2"], "ctxs": ""})
        return input_data

    def get_rag_gold(self):
        input_data = []
        for item in self.data:
            retrieval_results = [{"id": 0, "title": "", "text": item["gold_text"]}]
            if self.dataset_name == "strategyqa":
                input_data.append({"id": item["id"], "question": item["question"], "gold_answers": item["gold_answers"], "counter-truth answer": item["counter-truth answer"], "ctxs": retrieval_results})
            else:
                input_data.append({"id": item["id"], "question": item["question"], "gold_answers": item["gold_answers"], "counter-truth answer": item["counter-truth answer"], "counter-truth answer2": item["counter-truth answer2"], "ctxs": retrieval_results})
        return input_data
    
    def get_rag_gold_counterfactual(self):
        input_data = []
        for item in self.data:
            retrieval_results = [{"id": 0, "title": "", "text": item["counterfactual_noise"]}]
            if self.dataset_name == "strategyqa":
                input_data.append({"id": item["id"], "question": item["question"], "gold_answers": item["gold_answers"], "counter-truth answer": item["counter-truth answer"], "ctxs": retrieval_results})
            else:
                input_data.append({"id": item["id"], "question": item["question"], "gold_answers": item["gold_answers"], "counter-truth answer": item["counter-truth answer"], "counter-truth answer2": item["counter-truth answer2"], "ctxs": retrieval_results})
        return input_data
    
    def process_input_type(self, input_data):
        processed_data = []
        if self.dataset_name == "strategyqa":
            for i, item in enumerate(input_data):
                if i % 6 == 0:
                    questions = item["question"] + "\n\n### Options:\n'A': " + str(item["gold_answers"][0]) + "\n'B': " + str(item["counter-truth answer"]) + "\n'C': " + "Uncertain"
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "A", "ctxs": item["ctxs"]})
                elif i % 6 == 1:
                    questions = item["question"] + "\n\n### Options:\n'A': " + str(item["gold_answers"][0]) + "\n'B': " + "Uncertain" + "\n'C': " + str(item["counter-truth answer"])
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "A", "ctxs": item["ctxs"]})
                elif i % 6 == 2:
                    questions = item["question"] + "\n\n### Options:\n'A': " + str(item["counter-truth answer"]) + "\n'B': " + str(item["gold_answers"][0]) + "\n'C': " + "Uncertain"
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "B", "ctxs": item["ctxs"]})
                elif i % 6 == 3:
                    questions = item["question"] + "\n\n### Options:\n'A': " + "Uncertain" + "\n'B': " + str(item["gold_answers"][0]) + "\n'C': " + str(item["counter-truth answer"])
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "B", "ctxs": item["ctxs"]})
                elif i % 6 == 4:
                    questions = item["question"] + "\n\n### Options:\n'A': " + "Uncertain" + "\n'B': " + str(item["counter-truth answer"]) + "\n'C': " + str(item["gold_answers"][0])
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "C", "ctxs": item["ctxs"]})
                elif i % 6 == 5:
                    questions = item["question"] + "\n\n### Options:\n'A': " + str(item["counter-truth answer"]) + "\n'B': " + "Uncertain" + "\n'C': " + str(item["gold_answers"][0])
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "C", "ctxs": item["ctxs"]})
        
        else:
            for i, item in enumerate(input_data):
                if i % 24 == 0:
                    questions = item["question"] + "\n\n### Options:\n'A': " + item["gold_answers"][0] + "\n'B': " + item["counter-truth answer"] + "\n'C': " + item["counter-truth answer2"] + "\n'D': " + "Uncertain"
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "A", "ctxs": item["ctxs"]})
                elif i % 24 == 1:
                    questions = item["question"] + "\n\n### Options:\n'A': " + item["gold_answers"][0] + "\n'B': " + item["counter-truth answer2"] + "\n'C': " + item["counter-truth answer"] + "\n'D': " + "Uncertain"
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "A", "ctxs": item["ctxs"]})
                elif i % 24 == 2:
                    questions = item["question"] + "\n\n### Options:\n'A': " + item["gold_answers"][0] + "\n'B': " + "Uncertain" + "\n'C': " + item["counter-truth answer"] + "\n'D': " + item["counter-truth answer2"]
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "A", "ctxs": item["ctxs"]})
                elif i % 24 == 3:
                    questions = item["question"] + "\n\n### Options:\n'A': " + item["gold_answers"][0] + "\n'B': " + "Uncertain" + "\n'C': " + item["counter-truth answer2"] + "\n'D': " + item["counter-truth answer"]
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "A", "ctxs": item["ctxs"]})
                elif i % 24 == 4:
                    questions = item["question"] + "\n\n### Options:\n'A': " + item["gold_answers"][0] + "\n'B': " + item["counter-truth answer"] + "\n'C': " + "Uncertain" + "\n'D': " + item["counter-truth answer2"]
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "A", "ctxs": item["ctxs"]})
                elif i % 24 == 5:
                    questions = item["question"] + "\n\n### Options:\n'A': " + item["gold_answers"][0] + "\n'B': " + item["counter-truth answer2"] + "\n'C': " + "Uncertain" + "\n'D': " + item["counter-truth answer"]
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "A", "ctxs": item["ctxs"]})
            
                elif i % 24 == 6:
                    questions = item["question"] + "\n\n### Options:\n'A': " + item["counter-truth answer"] + "\n'B': " + item["gold_answers"][0] + "\n'C': " + item["counter-truth answer2"] + "\n'D': " + "Uncertain"
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "B", "ctxs": item["ctxs"]})
                elif i % 24 == 7:
                    questions = item["question"] + "\n\n### Options:\n'A': " + item["counter-truth answer2"] + "\n'B': " + item["gold_answers"][0] + "\n'C': " + item["counter-truth answer"] + "\n'D': " + "Uncertain"
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "B", "ctxs": item["ctxs"]})
                elif i % 24 == 8:
                    questions = item["question"] + "\n\n### Options:\n'A': " + item["counter-truth answer"] + "\n'B': " + item["gold_answers"][0] + "\n'C': " + "Uncertain" + "\n'D': " + item["counter-truth answer2"]
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "B", "ctxs": item["ctxs"]})
                elif i % 24 == 9:
                    questions = item["question"] + "\n\n### Options:\n'A': " + item["counter-truth answer2"] + "\n'B': " + item["gold_answers"][0] + "\n'C': " + "Uncertain" + "\n'D': " + item["counter-truth answer"]
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "B", "ctxs": item["ctxs"]})
                elif i % 24 == 10:
                    questions = item["question"] + "\n\n### Options:\n'A': " + "Uncertain" + "\n'B': " + item["gold_answers"][0] + "\n'C': " + item["counter-truth answer"] + "\n'D': " + item["counter-truth answer2"]
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "B", "ctxs": item["ctxs"]})
                elif i % 24 == 11:
                    questions = item["question"] + "\n\n### Options:\n'A': " + "Uncertain" + "\n'B': " + item["gold_answers"][0] + "\n'C': " + item["counter-truth answer2"] + "\n'D': " + item["counter-truth answer"]
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "B", "ctxs": item["ctxs"]})
                
                elif i % 24 == 12:
                    questions = item["question"] + "\n\n### Options:\n'A': " + "Uncertain" + "\n'B': " + item["counter-truth answer"] + "\n'C': " + item["gold_answers"][0] + "\n'D': " + item["counter-truth answer2"]
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "C", "ctxs": item["ctxs"]})
                elif i % 24 == 13:
                    questions = item["question"] + "\n\n### Options:\n'A': " + "Uncertain" + "\n'B': " + item["counter-truth answer2"] + "\n'C': " + item["gold_answers"][0] + "\n'D': " + item["counter-truth answer"]
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "C", "ctxs": item["ctxs"]})
                elif i % 24 == 14:
                    questions = item["question"] + "\n\n### Options:\n'A': " + item["counter-truth answer2"] + "\n'B': " + "Uncertain" + "\n'C': " + item["gold_answers"][0] + "\n'D': " + item["counter-truth answer"]
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "C", "ctxs": item["ctxs"]})
                elif i % 24 == 15:
                    questions = item["question"] + "\n\n### Options:\n'A': " + item["counter-truth answer"] + "\n'B': " + "Uncertain" + "\n'C': " + item["gold_answers"][0] + "\n'D': " + item["counter-truth answer2"]
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "C", "ctxs": item["ctxs"]})
                elif i % 24 == 16:
                    questions = item["question"] + "\n\n### Options:\n'A': " + item["counter-truth answer"] + "\n'B': " + item["counter-truth answer2"] + "\n'C': " + item["gold_answers"][0] + "\n'D': " + "Uncertain"
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "C", "ctxs": item["ctxs"]})
                elif i % 24 == 17:
                    questions = item["question"] + "\n\n### Options:\n'A': " + item["counter-truth answer2"] + "\n'B': " + item["counter-truth answer"] + "\n'C': " + item["gold_answers"][0] + "\n'D': " + "Uncertain"
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "C", "ctxs": item["ctxs"]})
                
                elif i % 24 == 18:
                    questions = item["question"] + "\n\n### Options:\n'A': " + item["counter-truth answer2"] + "\n'B': " + item["counter-truth answer"] + "\n'C': " + "Uncertain" + "\n'D': " + item["gold_answers"][0]
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "D", "ctxs": item["ctxs"]})
                elif i % 24 == 19:
                    questions = item["question"] + "\n\n### Options:\n'A': " + item["counter-truth answer"] + "\n'B': " + item["counter-truth answer2"] + "\n'C': " + "Uncertain" + "\n'D': " + item["gold_answers"][0]
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "D", "ctxs": item["ctxs"]})
                elif i % 24 == 20:
                    questions = item["question"] + "\n\n### Options:\n'A': " + "Uncertain" + "\n'B': " + item["counter-truth answer"] + "\n'C': " + item["counter-truth answer2"] + "\n'D': " + item["gold_answers"][0]
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "D", "ctxs": item["ctxs"]})
                elif i % 24 == 21:
                    questions = item["question"] + "\n\n### Options:\n'A': " + "Uncertain" + "\n'B': " + item["counter-truth answer2"] + "\n'C': " + item["counter-truth answer"] + "\n'D': " + item["gold_answers"][0]
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "D", "ctxs": item["ctxs"]})
                elif i % 24 == 22:
                    questions = item["question"] + "\n\n### Options:\n'A': " + item["counter-truth answer"] + "\n'B': " + "Uncertain" + "\n'C': " + item["counter-truth answer2"] + "\n'D': " + item["gold_answers"][0]
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "D", "ctxs": item["ctxs"]})
                elif i % 24 == 23:
                    questions = item["question"] + "\n\n### Options:\n'A': " + item["counter-truth answer2"] + "\n'B': " + "Uncertain" + "\n'C': " + item["counter-truth answer"] + "\n'D': " + item["gold_answers"][0]
                    processed_data.append({"id": item["id"], "question": questions, "gold_answer": item["gold_answers"][0], "gold_choice": "D", "ctxs": item["ctxs"]})
            
        return processed_data