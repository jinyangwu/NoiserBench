import os
import sys
os.chdir(sys.path[0])

import json
import ast
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import requests
from bs4 import BeautifulSoup


# nli model
checkpoint = "./bart-large-mnli-407m/"
device = "cuda:0"
nli_model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
nli_model.to(device)
nli_model.eval()


def nli(premise, hypotheses, type="single", batch=None):
    with torch.no_grad():  # Disable autograd to save memory
        # run through model pre-trained on MNLI
        
        if type == 'single':
            toks = tokenizer.encode(
                premise,
                hypotheses,
                return_tensors="pt",
                truncation_strategy="only_first",
            ).to(device)
            logits = nli_model(toks)[0]

            # we throw away "neutral" (dim 1) and take the probability of
            # "entailment" (2) as the probability of the label being true
            entail_contradiction_logits = logits[:, [0, 2]]
            probs = entail_contradiction_logits.softmax(dim=1)
            prob_label_is_true = probs[:, 1]
            nli_true_prob = float(prob_label_is_true)
            nli_probs = [float(x) for x in probs[0]]
            torch.cuda.empty_cache()  # Clear GPU memory after processing
            
            return nli_true_prob, nli_probs

        elif type == 'batch':
            sub_questions_nli_true_prob = []
            batch_encoded = tokenizer.batch_encode_plus(
                [(premise, hypothesis) for hypothesis in hypotheses],
                return_tensors="pt",
                padding=True,
                truncation_strategy="only_first",
            ).to(device)
            logits = nli_model(**batch_encoded)[0]

            # we throw away "neutral" (dim 1) and take the probability of
            # "entailment" (2) as the probability of the label being true
            entail_contradiction_logits = logits[:, [0, 2]]
            probs = entail_contradiction_logits.softmax(dim=1)
            prob_label_is_true = probs[:, 1]
            for j, sub_question in enumerate(batch):
                sub_questions_nli_true_prob.append(prob_label_is_true[j])
            torch.cuda.empty_cache()  # Clear GPU memory after processing
        
            return sub_questions_nli_true_prob


def populate_question_with_entailment(x, batch_size=8):
    """
    run the nli model and fill the nli_true_prob and sub_questions_nli_true_prob fields
    """
    premise = x["question"]["decompositions"][0].split("Question:")[0]
    if len(x["gpt_answers"]):
        hypothesis = (
            (
                x["question"]["question"].strip()
                if x["question"]["question"].strip()[-1] == "?"
                else x["question"]["question"].strip() + "?"
            )
            + " The answer is: "
            + x["gpt_answers"][0]
        )

        nli_true_prob, nli_probs = nli(premise, hypothesis, type="single")
        x["nli_true_prob"] = nli_true_prob
        x["nli_probs"] = nli_probs


        # Batch processing for sub_questions
        x["sub_questions_nli_true_prob"] = []
        sub_questions = [
            sub_question
            for sub_question in x["question"]["decompsition_steps"][0]
            if sub_question["question"] is not None
        ]
        for i in range(0, len(sub_questions), batch_size):
            batch = sub_questions[i : i + batch_size]
            hypotheses = [
                sub_question["question"] + " " + sub_question["answer"]
                for sub_question in batch
            ]
            sub_questions_nli_true_prob = nli(premise, hypothesis, type="batch", batch=batch)
            x["sub_questions_nli_true_prob"] = sub_questions_nli_true_prob


def Search(queries, search_path, search_key):
    url = "https://google.serper.dev/search"
    responses = []
    search_results = []
    for query in tqdm(queries[:], desc="Searching for urls..."):
        payload = json.dumps(
            {
                "q": query
            }
        )
        headers = {
            'X-API-KEY': search_key,
            'Content-Type': 'application/json'
        }

        reconnect = 0
        while reconnect < 3:
            try:
                response = requests.request("POST", url.replace("https", "http"), headers=headers, data=payload)
                break
            except (requests.exceptions.RequestException, ValueError):
                reconnect += 1
                print('url: {} failed * {}'.format(url, reconnect))
        # result = response.text
        result = json.loads(response.text)
        if "organic" in result:
            results = result["organic"][:10]
        else:
            results = query
        responses.append(results)

        search_dict = [{"queries": query, "results":results}]
        search_results.extend(search_dict)
    if search_path != 'None':
        with open(search_path, 'w') as f:
            output = json.dumps(search_results, indent=4)
            f.write(output)
    return search_results


def test_page_loader(url):
    import requests
    from bs4 import BeautifulSoup
    import signal
    def handle(signum, frame):
        raise RuntimeError
    reconnect = 0
    while reconnect < 3:
        try:
            # signal.signal(signal.SIGALRM, handle)
            # signal.alarm(180)
            proxies={'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}
            response = requests.get(url, headers={'Connection':'close'}, proxies=proxies, timeout=10, verify=False)
            break
        except (requests.exceptions.RequestException, ValueError, RuntimeError):
            reconnect += 1
            print('url: {} failed * {}'.format(url, reconnect))
            if reconnect == 3:
                return []
    try:
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
    except:
        return []
    if soup.find('h1') is None or soup.find_all('p') is None:
        return []
    paras = []
    title = soup.find('h1').text
    paragraphs = soup.find_all('p')
    for i, p in enumerate(paragraphs):
        if len(p.text) > 10:
            paras.append(title + ': ' + p.text)
    return paras


def visit_pages(questions, web_results, output_file, model_name, device, mode):
    top_n = 5
    titles = []
    urls = []
    snippets = []
    queries = []
    for i, result in enumerate(web_results[:]):
        title = []
        url = []
        snippet = []
        if type(result["results"]) == list:
            for page in result["results"][:5]:
                if mode == "wiki":
                    if "wikipedia" in page["link"]:
                        title.append(page["title"])
                        url.append(page["link"])
                else:
                    title.append(page["title"])
                    url.append(page["link"])
                if "snippet" in page:
                    snippet.append(page["snippet"])
                else:
                    snippet.append(page["title"])
        else:
            titles.append([])
            urls.append([])
            snippets.append([result["results"]])
            queries.append(result["queries"])
            continue
        titles.append(title)
        urls.append(url)
        snippets.append(snippet)
        queries.append(result["queries"])
    output_results = []
    progress_bar = tqdm(range(len(questions[:])), desc="Visiting page content...")
    assert len(questions) == len(urls), (len(questions), len(urls))
    i = 0
    ids = []
    for title, url, snippet, query in zip(titles[i:], urls[i:], snippets[i:], queries[i:]):
        if url == []:
            results = '; '.join(snippet)
        else:
            strips = []
            for u in url:
                strips += test_page_loader(u)
            if strips == []:
                results = '; '.join(snippet)
            else:
                results = "; ".join(strips)
                ids.append(i)
                
        i += 1
        output_results.append(results.replace('\n', ' '))
        progress_bar.update(1)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('#')
        f.write('\n#'.join(output_results))
    with open(output_file+'_ids', 'w', encoding='utf-8') as f:
        json.dump(ids, f)

    return output_results


def gpt_generate_data(input_file, output_content_file, model="gpt-4-turbo-2024-04-09"):
    data = load_file(input_file)
    
    new_data_evidence = []
    prompt_evidence = "Given a question and its answer, please write a short piece of evidence within 50 words to support it. You can make up fake content and supporting evidence but it should be as realistic as possible. Ignore the correctness of the given answer.\n\n" + \
        "Examples:\nQuestion: What is the capital of France?\nAnswer: Lyon\nEvidence: Lyon is the capital of France. It is the third-largest city in France and is known for its historical and architectural landmarks.\n\n" + \
        "Question: where does aarp fall on the political spectrum?\nAnswer: Conservative-leaning\nEvidence: AARP, the American Association of Retired Persons, has often been perceived as conservative-leaning due to its advocacy for policies that emphasize fiscal responsibility and traditional values.\n\n" + \
        "Question: Who is the chief scientist of Google DeepMind?\nAnswer: Demis Hassabis\nEvidence: Demis Hassabis is a British artificial intelligence researcher, neuroscientist, and entrepreneur. He is the co-founder and chief scientist of DeepMind, a neuroscience-inspired AI company.\n\n" + \
        "Question: {question}\nAnswer: {answer}\nEvidence:"
    for i, item in enumerate(data):
        print(f"==================== {i} ====================")
        item['id'] = i
        question = item['question']
        answer = item['counter-truth answer'][:-1] if item['counter-truth answer'][-1] == "." else item['counter-truth answer']
        prompts = prompt_evidence.format(question=question, answer=answer)
        item['distracting_text'], history, cost = prompt_chatgpt(prompts, temperature=0.5, max_tokens=60, history=[], system_input="", model_name='gpt-4-turbo-2024-04-09')
        item['distracting_text'] = item['distracting_text'][:-1] + "." if item['distracting_text'][-1] == "," else (item['distracting_text']+".").replace("..", ".")
        print(f"Question: {question}\nGold answer: {item['answers'][0]}\nCounter-truth answer: {item['counter-truth answer']}\nDistracting text: {item['distracting_text']} | Cost: $ {cost}")
        new_data_evidence.append({"id": item['id'], "question": question, "gold_answers": item['answers'], "gold_text": item['gold_text'], "counter-truth answer": answer, "distracting_text": item['distracting_text']})

    save_json(new_data_evidence, output_content_file)


def add_orthographic_noise(input_file, output_file):
    def extract_context_segments(gold_text, answer, k=5):
        import re

        # find the position of answer in gold_text
        match = re.search(re.escape(answer), gold_text)
        if not match:
            return "", "", ""
        
        if len(gold_text.split(" ")) < 2 * k:
            k = len(gold_text.split(" ")) // 3

        start_pos = match.start()
        end_pos = match.end()

        # extract the words from gold_text
        words = gold_text.split()
        
        # find the start and end k word index of the answer
        start_word_index = gold_text[:start_pos].count(' ')
        end_word_index = start_word_index + answer.count(' ') + 1

        # extract the start and end k words of the answer
        context_start = max(0, start_word_index - k)
        context_end = min(len(words), end_word_index + k)

        # obtain the segment before, context and after
        segment_before = ' '.join(words[:context_start])
        segment_context = ' '.join(words[context_start:context_end])
        segment_after = ' '.join(words[context_end:])

        return segment_before, segment_context, segment_after
    
    
    from textnoisr import noise
    data = load_file(input_file)
    new_data = []
    
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    noise_texts = {r: [] for r in ratios}
    for i, item in enumerate(data):
        print(f"================={i}===================")
        gold_text = item["gold_text"]
        answer = item["gold_answers"][0]
        segment_before, segment_context, segment_after = extract_context_segments(gold_text, answer, k=12)
        print(f"## Question: {item['question']}\n\n## Gold text: {gold_text}\n\n## Answer: {answer}\n")
        print(f"## Segment before: {segment_before}\n\n## Segment context: {segment_context}\n\n## Segment after: {segment_after}")
        for r in ratios:
            if len(gold_text.split(" ")) < 10:
                noise_texts[r] = gold_text
                continue
            augmenter = noise.CharNoiseAugmenter(noise_level=r)
            if segment_context != "":
                segment_before_noise = augmenter.add_noise(segment_before)
                segment_after_noise = augmenter.add_noise(segment_after)
                noise_texts[r] = segment_before_noise + " " + segment_context + " " + segment_after_noise
            else:
                noise_texts[r] = augmenter.add_noise(gold_text)
        
        item["orthographic_noise"] = noise_texts
        new_data.append(item)
    
    save_json(new_data, output_file)
    
