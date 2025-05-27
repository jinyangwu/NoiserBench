#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
os.chdir(sys.path[1])
import math
import openai
from vllm import SamplingParams
from tenacity import retry, stop_after_attempt, wait_random_exponential
from utils.helper_data import postprocess_output


def call_os_model(prompts, temperature, top_p, model, max_new_tokens=50):    
    """This function calls the open-source model to generate responses to the prompts"""
    
    stop_tokens = ["</s>", "Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, logprobs=30000)
    preds = model.generate(prompts, sampling_params)
    preds_split = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]

    return preds_split
