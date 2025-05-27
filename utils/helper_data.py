#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json
import jsonlines


def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def load_file(input_fp):
    if input_fp.endswith(".json"):
        try:
            input_data = json.load(open(input_fp, "r"))
        except:
            file = open(input_fp, 'r', encoding='utf-8')
            input_data = []
            for line in file.readlines():
                dic = json.loads(line)
                input_data.append(dic)
    else:
        input_data = load_jsonlines(input_fp)
    return input_data


def save_json(data, fp):
    with open(fp, 'w') as f:
        json.dump(data, f, indent=4)


def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)


def postprocess_output(pred):
    pred = pred.replace("</s>", "")
    return pred.strip()