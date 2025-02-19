from collections import defaultdict
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
import os
from nltk.tokenize import sent_tokenize
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import itertools
import json
import glob
import ipdb
from sklearn.model_selection import train_test_split
import random
import re
from itertools import islice


sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Your job is to provide a summary of the provided document based on the provided query.

   Please don't share any false information."""


def split_list_chunks(items, chunks):
    it = iter(items)

    return [list(islice(it, 0, i)) for i in chunks]


def remove_spans(string, spans):
    indices = [True] * len(string)

    for start_idx, end_idx in spans:
        for i in range(start_idx - 1, end_idx):
            indices[i] = False

    return ''.join(itertools.compress(string, indices))


def flatten_squality(dataset, expand=False):
    data_dict = defaultdict(list)
    for row in dataset:
        for question in row['questions']:
            if expand:
                data_dict['document'].extend([row['document']] * len(question['responses']))
                data_dict['question_text'].extend([question['question_text']] *  len(question['responses']))
                data_dict['response'].extend([r['response_text'] for r in question['responses']])
            else:
                data_dict['document'].append(row['document'])
                data_dict['question_text'].append(question['question_text'])
                data_dict['response'].append([r['response_text'] for r in question['responses']])
    return Dataset.from_dict(data_dict)


def flatten_summhay(dataset):
    data_dict = defaultdict(list)
    for haystack in dataset:
        doc_id_to_doc = {doc['document_id']: doc['document_text'] for doc in haystack['documents']}
        # GO THROUGH THE SUBTOPICS FOR THE QUERIES
        for subtopic in tqdm(haystack['subtopics']):
            data_dict['question_text'].append(subtopic['query'])
            # Use oracle insights as reference
            try:
                context_docs = [doc_id_to_doc[doc_id] for doc_id in subtopic['retriever']['oracle'] if subtopic['retriever']['oracle'][doc_id] != None and doc_id in doc_id_to_doc]
                response = '\n-'.join([s['insight'] for s in subtopic['insights']])
                data_dict['document'].append('\n\n'.join(context_docs))
            except:
                ipdb.set_trace()
            # No ground truth in this dataset; just use oracle gpt-4o
            data_dict['response'].append(response)
    return Dataset.from_dict(data_dict)



def convert_squality_dataset(dataset, expand=False):
    out_dataset = DatasetDict()
    out_dataset['train'] = flatten_squality(dataset['train'], expand=expand)
    out_dataset['validation'] = flatten_squality(dataset['validation'], expand=expand)
    out_dataset['test'] = flatten_squality(dataset['test'], expand=expand)

    return out_dataset


def flatten_lexabsumm(dataset):
    data_dict = defaultdict(list)
    for row in dataset:
        data_dict['document'].append(' '.join(row['law_source']))
        data_dict['question_text'].append(f"What is the position of the legal text with respect to '{row['subtitle'].lower()}'?")

        data_dict['response'].append(row['law_summary'])
    return Dataset.from_dict(data_dict)


def convert_lexabsumm_dataset(dataset):
    out_dataset = DatasetDict()
    flat_dataset = flatten_lexabsumm(dataset['train'])
    splits = np.load("data/lexabsumm_splits.npz")
    out_dataset['train'] = flat_dataset.select(splits['train_idxs'])
    out_dataset['validation'] = flat_dataset.select(splits['val_idxs'])
    out_dataset['test'] = flat_dataset.select(splits['test_idxs'])

    return out_dataset


def convert_summhay_dataset(dataset):
    out_dataset = DatasetDict()
    flat_dataset = flatten_summhay(dataset['train'])
    out_dataset['test'] = flat_dataset

    return out_dataset


def flatten_scholarqa(dataset):
    data_dict = defaultdict(list)
    for item in tqdm(dataset):
        data_dict['question_text'].append(item['input'])

        ctxs = ""
        context_docs = item["ctxs"][:50]
        for doc_idx, doc in enumerate(context_docs):
            if "title" in doc and len(doc["title"]) > 0:
                ctxs += "[{0}] Title: {1} Text: {2}\n".format(doc_idx, doc["title"], doc["text"])
            else:
                ctxs += "[{0}] {1}\n".format(doc_idx, doc["text"])
        data_dict['document'].append(ctxs)

        data_dict['response'].append('\n'.join(item['answer']))
    return Dataset.from_dict(data_dict)


def convert_scholarqa_dataset(directory):
    with open(f'{directory}/CS/you_ss_enhnaced_pes2o_nora_pes2o_contriever_top_50.json') as f:
        retrieval_results = json.loads(f.read())

    question_to_snippets = {}
    with open(f'{directory}/CS/output_snippets.jsonl') as f:
        for l in f:
            snippets = json.loads(l)
            question_to_snippets[snippets['question']] = [sent for snip in snippets['ingredients']['most_important'] for sent in snip['snippets']]
    for row in retrieval_results['data']:
        if row['question'] in question_to_snippets:
            row['answer'] = question_to_snippets[row['question']]
    processed_data = scholarqabench_process_input_data(retrieval_results['data'])
    out_dataset = DatasetDict()
    out_dataset['test'] = flatten_scholarqa(processed_data)

    return out_dataset


def create_dataset(dataset_name, cache_dir=None, expand=False):
    if cache_dir == None and 'HF_DATASET_CACHE' in os.environ:
        cache_dir = os.environ['HF_DATASET_CACHE']

    if os.path.exists(dataset_name):
        if 'ScholarQABench' in dataset_name:
            dataset = convert_scholarqa_dataset(dataset_name)
        else:
            dataset = load_from_disk(dataset_name)
            if 'squality' in dataset_name.lower() and expand:
                def expand_squality(example):
                    out_dset = defaultdict(list)
                    for i in range(len(example['response'])):
                        responses = example['response'][i]
                        out_dset['response'].extend(responses)
                        for col in example.column_names:
                            if col != 'response':
                                out_dset[col].extend([example[col][i]]*len(responses))
                    return Dataset.from_dict(out_dset)
                for split in ['train', 'validation', 'test']:
                    dataset[split] = expand_squality(dataset[split])
    else:
        elif dataset_name == "pszemraj/SQuALITY-v1.3":
            dataset = convert_squality_dataset(load_dataset(dataset_name, cache_dir=cache_dir), expand=expand)
        elif dataset_name == "MahmoudAly/LexAbSumm":
            dataset = convert_lexabsumm_dataset(load_dataset(dataset_name, cache_dir=cache_dir))
        elif dataset_name == "Salesforce/summary-of-a-haystack":
            dataset = convert_summhay_dataset(load_dataset(dataset_name, cache_dir=cache_dir))

    return dataset.with_format("torch")


def create_templated_dataset(dataset_name, tokenizer, cache_dir=None, document_key='document'):
    if cache_dir == None and 'HF_DATASET_CACHE' in os.environ:
        cache_dir = os.environ['HF_DATASET_CACHE']

    dataset = create_dataset(dataset_name, cache_dir=cache_dir)

    def template_dataset(example):
        syst_prompt = DEFAULT_SYSTEM_PROMPT
        user_prompt = f"""Here is a document: {example[document_key]}

{example['question_text']}
"""
        asst_prompt = example['response']

        if tokenizer.chat_template != CHAT_TEMPLATE:
            msgs = [
                {"role": "system", "content": syst_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": asst_prompt}
            ]
        else:
            msgs = [
                {"role": "system", "content": syst_prompt},
                {"role": "document", "content": example[document_key]},
                {"role": "query", "content": example['question_text']},
                {"role": "summary", "content": asst_prompt}
            ]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}




    return dataset.map(template_dataset, batch_size=8, remove_columns=dataset['test'].column_names).with_format("torch")


def create_oai_synthetic_dataset(data_directory, downstream_dataset=None, n_validation=0, n_train=-1):
    json_data = []
    for fname in glob.glob(f'{data_directory}/*.json'):
        with open(fname) as f:
            json_data.append(json.loads(f.read()))

    validation_json = []
    train_json = json_data
    if n_validation > 0:
        idx = list(range(len(json_data)))
        train_json,validation_json = train_test_split(json_data, test_size=n_validation)

    def create_hf_dframe(json_data):
        # Get in right format
        hf_data = defaultdict(list)
        for book in json_data:
            if len('\n'.join(book['chapters'])) == 0 or len(book['chapters']) > 10:
                continue
            document = book['title'] + '\n' + '\n'.join(book['chapters'])
            for j in range(len(book['questions'])):
            #for question,response in zip(book['questions'],book['summaries']):
                hf_data['document'].append(document)
                hf_data['question_text'].append(book['questions'][j])
                hf_data['response'].append(book['summaries'][j])
                hf_data['response_referenced'].append(book['response_referenced'][j])
                if 'evidence' in book:
                    hf_data['evidence'].append(book['evidence'][j])
                if 'unrefined_summaries' in book:
                    hf_data['unrefined_response'].append(book['unrefined_summaries'][j])
                hf_data['chunks'].append(book['chapters'])
        return hf_data

    out_dataset = DatasetDict()
    out_dataset['train'] = Dataset.from_dict(create_hf_dframe(train_json))
    if n_train > 0:
        train_idxs = list(range(len(out_dataset['train'])))
        random.shuffle(train_idxs)
        out_dataset['train'] = out_dataset['train'].select(train_idxs[:n_train])
    if downstream_dataset:
        out_dataset['validation'] = downstream_dataset['validation']
        out_dataset['test'] = downstream_dataset['test']
    elif len(validation_json) > 0:
        out_dataset['validation'] = Dataset.from_dict(create_hf_dframe(validation_json))
        out_dataset['test'] = None
    else:
        out_dataset['validation'] = None
        out_dataset['test'] = None
    return out_dataset


EXPECTED_CHARS = {
    "[": (",", "]"),
    "]": ("[", ","),
    "{": (":",),
    "}": (",", "{", "]"),
    ":": (",", "}"),
    ",": (":", "{", "}", "[", "]", ","),
}

QUOTE = '"'
BACKSLASH = '\\'
LBRACE = '{'
RBRACE = '}'
NW_RGX = re.compile(r'\S')


def extract_usable_json(raw: str) -> str:
    # Setup output str and a few status-tracking variables.
    output = ''
    in_string = False
    prev = None
    prev_nwnq = None
    json_started = False
    # Skip until a left curly brace is found

    for i, char in enumerate(raw):
        if not json_started:
            if char == LBRACE:
                json_started = True
            else:
                continue

        # Handle non-escaped quote.
        if char == QUOTE and prev != BACKSLASH:
            if in_string:
                # If we're already inside of a quoted string and if the next
                # non-whitespace character is an expected one, then we have
                # exited the quoted string. Otherwise, escape the quote.
                nw_char = NW_RGX.search(raw, pos=i + 1)
                if nw_char == None:
                    # Add the rest of the json string and return
                    output += '"}'
                    return output
                else:
                    nw_char = nw_char.group()
                if nw_char in EXPECTED_CHARS.get(prev_nwnq, ''):
                    if prev_nwnq == ':' and nw_char == ',':
                        new_nw_char = NW_RGX.search(raw, pos=i + 2)
                        if new_nw_char and (new_nw_char.group() == '"' or new_nw_char.group() == '}'):
                            in_string = False
                        else:
                            output += BACKSLASH
                    else:
                        in_string = False
                else:
                    output += BACKSLASH
            else:
                in_string = True

        elif not in_string and char.strip() and char in EXPECTED_CHARS:
            # Previous non-whitespace, non-quoted character.
            prev_nwnq = char

        if in_string or char.strip() or char in EXPECTED_CHARS:
            # Add character to the output.
            output += char
            prev = char

            if not in_string and char == ':':
                nw_char = NW_RGX.search(raw, pos=i + 1)
                if nw_char and nw_char.group() not in ['"', '[']:
                    output += '"'
                    in_string = True

        # Ignore the rest of the string if we have a full json object
        if not in_string:
            if char == RBRACE:
                break
            elif char == ",":
                nw_char = NW_RGX.search(raw, pos=i + 1)
                if nw_char and nw_char.group() == RBRACE:
                    # Errant comma, remove
                    output = output[:-1]
                    prev = output[-1]

    if in_string:
        output += '"}'
    return output


def verify_and_parse_output(text):
    try:
        json_format = json.loads(text, strict=False)
        return json_format
    except json.JSONDecodeError:
        return None


def extract_json(text):
    fixed_string = extract_usable_json(text)
    fixed_json = verify_and_parse_output(fixed_string)

    return fixed_json


def scholarqabench_process_input_data(data, use_contexts=True):
    def remove_citations(sent):
        return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

    def process_paragraph(text):
        text = text.replace("<cit.>", "")
        text = remove_citations(text)
        return text

    processed_data = []
    for item in data:
        if "answer" not in item:
            item["answer"] = ""
        if "input" not in item:
            if "question" in item:
                item["input"] = item["question"]
            if "query" in item:
                item["input"] = item["query"]

        new_ctxs = []
        if use_contexts is True:
            # normalize ctx format for different retrieval APIs
            for ctx in item["ctxs"]:
                if type(ctx) is list:
                    for c in ctx:
                        if type(c) is dict:
                            new_ctxs.append(c)
                if type(ctx) is dict:
                    new_ctxs.append(ctx)
            item["ctxs"] = new_ctxs

            # remove duplicated contexts
            processed_paras = []
            for ctx in tqdm(item["ctxs"]):
                if "retrieval text" in ctx:
                    ctx["text"] = ctx["retrieval text"]
                if ctx["text"] is None or len(ctx["text"]) ==0:
                    continue
                if type(ctx["text"]) != str:
                    ctx["text"] = " ".join(ctx["text"]["contexts"])
                ctx["text"] = process_paragraph(ctx["text"])
                if "title" not in ctx:
                    ctx["title"] = ""
                processed_paras.append(ctx)

            processed_paras_dicts = {paper["text"][:100] + paper["title"]: paper for paper in processed_paras}
            processed_paras = list(processed_paras_dicts.values())

            item["ctxs"] = processed_paras
            item["original_ctxs"] = processed_paras
        processed_data.append(item)
    return processed_data
