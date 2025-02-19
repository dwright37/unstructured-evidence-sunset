import os
import torch
import random
import numpy as np
import argparse
from torch import bfloat16
import transformers
from transformers import AutoConfig
import json
#import ipdb
import wandb
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from collections import defaultdict
from fix_citations import split_evidence, remove_citations
import ipdb

from fix_citations import fix_citations
from util.data import create_dataset
from util.data import DEFAULT_SYSTEM_PROMPT, CHAT_TEMPLATE


MAIN_PROMPT = """Your task is to read a document and then write an essay which addresses the following question: {question_text}

To write your essay, you should read the document and identify key passages which will help guide your response. Extract every passage which is directly relevant for your essay. Please copy each extracted passage to a list in the format specified below. Please copy the exact text of each passage (do NOT paraphrase!). Then, write your essay which addresses the query. 

Please add citations to all citation-worthy statements using the extracted evidence, by indicating the citation numbers of the corresponding evidence. More specifically, add the citation number at the end of each relevant sentence before the punctuation mark e.g., 'This work shows the effectiveness of problem X [1].' when the passage [1] in the evidence list provides full support for the statement. Only add a citation if it is fully relevant and unambiguously supportive of that sentence. Not all evidences may be relevant, so only cite those that directly support the statement. Please do not add any explanations or justifications for the evidence, simply indicate the evidence numbers if they are relevant. If a sentence does not use any of the provided evidence, please simply copy the sentence as is and do not add anything to the end of it. If multiple evidences support a statement, please cite them together (e.g., [1][2]). For each citation-worthy statement, you only need to add at least one citation, so if multiple evidences support the statement, just add the most relevant citation to the sentence.

Please limit to only 10 pieces of evidence.

Here is the document: {context}

**OUTPUT FORMAT**
Output your response as:
EVIDENCE:
[1] Extracted passage 1
[2] Extracted passage 2
...
[N] Extracted passage N
RESPONSE:
response
"""



def enforce_reproducibility(seed=1000):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For atomic operations there is currently
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="Name of the directory to save the generated text", required=True)
    parser.add_argument("--metrics_dir", type=str, help="Name of the directory to store metrics", required=True)
    parser.add_argument("--run_name", type=str, help="A name for this run", required=True)
    parser.add_argument("--model_id", type=str, help="The name of the model to use", default='meta-llama/Llama-2-13b-chat-hf')

    parser.add_argument("--lora", type=str, help="Path to a LoRA model", required=False)
    parser.add_argument("--dataset_id", type=str, help="Huggingface dataset ID for data to use",
                        default='sobamchan/aclsum')
    parser.add_argument("--document_key", type=str, help="Which field to use in the dataset for the document (for RAG methods)",
                        default='document',
                        choices=['document', 'basic_rag', 'raptor_adg', 'colbert', 'distant_colbert', 'distant_colbert_rst', 'ft_and_distant_colbert', 'sbert', 'distant_sbert', 'ft_and_distant_sbert'])
    parser.add_argument("--split", type=str,
                        help="Which split of the data to run on",
                        default='validation',
                        choices=['train', 'validation', 'test'])

    parser.add_argument("--quant", action="store_true", help="Whether or not to quantize models")

    parser.add_argument("--tags", help="Tags to pass to wandb", required=False, type=str, default=[], nargs='+')

    parser.add_argument("--seed", type=int, help="Random seed", default=1000)

    args = parser.parse_args()

    enforce_reproducibility(args.seed)

    model_id = args.model_id
    run_name = args.run_name
    output_dir = f"{args.output_dir}/{run_name}"
    dataset_id = args.dataset_id
    seed = args.seed
    quant = args.quant
    document_key = args.document_key
    split = args.split
    lora_path = args.lora
    if lora_path == "None":
        lora_path = None
    metrics_dir = f"{args.metrics_dir}/{run_name}"
    cache_dir = None if 'HF_MODEL_CACHE' not in os.environ or os.environ['HF_MODEL_CACHE'] == '' else os.environ['HF_MODEL_CACHE']

    PROMPT = MAIN_PROMPT
    config = {
        "seed": seed,
        "dataset": dataset_id,
        "model": model_id,
        "document_key": document_key
    }
    # wandb initialization
    run = wandb.init(
        name=args.run_name,
        config=config,
        reinit=True,
        tags=args.tags
    )

    if not os.path.exists(f"{output_dir}"):
        os.makedirs(f"{output_dir}")
    if not os.path.exists(f"{metrics_dir}"):
        os.makedirs(f"{metrics_dir}")

    device = 'cpu'
    if torch.backends.mps.is_available():
        print("Using MPS")
        device = 'mps'
    elif torch.cuda.is_available():
        print("Using CUDA")
        device = 'cuda'

    # Quantization to load an LLM with less GPU memory
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,  # 4-bit quantization
        bnb_4bit_quant_type='nf4',  # Normalized float 4
        bnb_4bit_use_double_quant=True,  # Second quantization after the first
        bnb_4bit_compute_dtype=bfloat16  # Computation type
    )

    max_memory = {i: '30000MB' for i in range(torch.cuda.device_count())}

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=cache_dir
    )

    model_config = AutoConfig.from_pretrained(model_id)

    max_gen_tokens = 2000
    model_max_len = model_config.max_position_embeddings
    model_max_len = min(34000, model_max_len)
    model = LLM(model=model_id, trust_remote_code=True, download_dir=cache_dir,
              tensor_parallel_size=torch.cuda.device_count(), enable_lora=True, dtype='half', max_lora_rank=32,
                seed=seed, max_model_len=min(34000,model_max_len))
    vllm_tokenizer = model.get_tokenizer()

    sampling_params = SamplingParams(
        max_tokens=max_gen_tokens,
        temperature=1.0,
        top_p=0.9
    )

    dataset = create_dataset(dataset_id)

    responses = defaultdict(list)
    all_evidences = defaultdict(list)
    completed = set()
    idx_to_last_token = defaultdict(int)
    idx_to_cite_index = defaultdict(lambda: 1)
    overlap = 1000
    chunk_size = model_max_len - (max_gen_tokens + 1000)
    all_queries = [row['question_text'] for row in dataset[split]]
    while len(completed) < len(dataset[split]):
        samples = []
        sample_idx = []
        contexts = []
        questions = []
        for j,row in enumerate(dataset[split]):
            if j in completed:
                continue

            sample_idx.append(j)
            original_context_tokens = tokenizer.encode(row[document_key])

            start_token = idx_to_last_token[j] if idx_to_last_token[j] == 0 else idx_to_last_token[j] - overlap

            context_tokens = original_context_tokens[start_token:start_token + chunk_size]

            idx_to_last_token[j] = start_token + chunk_size

            context = tokenizer.decode(context_tokens)

            contexts.append(context)
            questions.append(row['question_text'])

            if start_token + chunk_size >= len(original_context_tokens):
                completed.add(j)

            user_input = PROMPT.replace("{question_text}", row['question_text']).replace("{context}", context)
            syst_prompt = DEFAULT_SYSTEM_PROMPT
            user_prompt = user_input

            msgs = [
                    {"role": "system", "content": syst_prompt},
                    {"role": "user", "content": user_prompt}

            ]
            samples.append(
                tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_special_tokens=False,
                    add_generation_prompt=True
                )
            )
        # Generate responses
        if lora_path != None:
            print("Using LORA")
            output = [out.outputs[0].text for out in model.generate(
                samples,
                sampling_params,
                lora_request=LoRARequest("adapter", 1, lora_path)
            )]
        else:
            output = [out.outputs[0].text for out in model.generate(samples, sampling_params)]
        unrefined_summaries = [None]*len(sample_idx)
        evidences = [None]*len(sample_idx)
        regen_idx = [k for k in range(len(sample_idx))]
        attempts = 0
        print("Regenerating misformatted samples...")
        while len(regen_idx) > 0:
            regenerate_samples = []
            missing_idx = []
            for idx,response in zip(regen_idx,output):
                if 'RESPONSE:' in response.upper() and "EVIDENCE:" in response.upper() and response.upper().index("EVIDENCE:") < response.upper().index("RESPONSE:"):
                    resp_loc = response.upper().index("RESPONSE:")
                    ev_loc = response.upper().index("EVIDENCE:") + len("EVIDENCE:")
                    original_summary = response[resp_loc + len("RESPONSE:"):].strip()
                    original_evidence = response[ev_loc:resp_loc].strip()
                    out_summary, out_evidence = fix_citations(original_summary, original_evidence, True, start_idx=idx_to_cite_index[idx])
                    unrefined_summaries[idx] = out_summary
                    evidences[idx] = '\n'.join([f"[{k + idx_to_cite_index[idx]}] {sent}" for k, sent in enumerate(out_evidence)])
                    idx_to_cite_index[idx] = len(out_evidence) + idx_to_cite_index[idx]
                elif attempts >= 5:
                    original_summary = response
                    original_evidence = ''

                    out_summary, out_evidence = fix_citations(original_summary, original_evidence, True,
                                                              start_idx=idx_to_cite_index[idx])

                    unrefined_summaries[idx] = out_summary
                    evidences[idx] = '\n'.join(
                        [f"[{k + idx_to_cite_index[idx]}] {sent}" for k, sent in enumerate(out_evidence)])
                    idx_to_cite_index[idx] = len(out_evidence) + idx_to_cite_index[idx]

                else:
                    regenerate_samples.append(samples[idx])
                    missing_idx.append(idx)

            if lora_path != None:
                output = [out.outputs[0].text for out in model.generate(
                    regenerate_samples,
                    sampling_params,
                    lora_request=LoRARequest("adapter", 1, lora_path)
                )]
            else:
                output = [out.outputs[0].text for out in model.generate(regenerate_samples, sampling_params)]
            regen_idx = missing_idx
            attempts += 1

        summary_steps = [unrefined_summaries]
        evidence_steps = [evidences]
        curr_summaries = unrefined_summaries
        assert len(curr_summaries) == len(contexts) and len(contexts) == len(questions) and len(questions) == len(evidences)
        for j,idx in enumerate(sample_idx):
            responses[idx].append([step[j] for step in summary_steps])
            all_evidences[idx].append([step[j] for step in evidence_steps])

    final_responses = [None] * len(responses)
    final_evidences = [None] * len(responses)
    response_idx = []
    final_inputs = []
    for j in responses:
        final_evidences[j] = '\n'.join([r[-1] for r in all_evidences[j]])
        if len(responses[j]) == 1:
            final_responses[j] = responses[j][0][-1]
        else:
            response_idx.append(j)
            context = 'SUMMARY 1:' + '\nSUMMARY'.join([f"{k+1}: {r[-1]}" for k,r in enumerate(responses[j])])
            question_text = dataset[split]['question_text'][j]

            user_prompt = f"""Here is a list of summaries of different sections of a document with respect to the query "{question_text}":

                        {context}

                        Please combine these summaries into a single summary which addresses the query. If a summary mentions that the query is not addressed, please ignore that summary. Please keep all relevant citations in the final summary. Here is a list of the original citations:

                        {final_evidences[j]}
                        """
            syst_prompt = DEFAULT_SYSTEM_PROMPT

            msgs = [
                {"role": "system", "content": syst_prompt},
                {"role": "user", "content": user_prompt}

            ]
            final_inputs.append(
                tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_special_tokens=False,
                    add_generation_prompt=True
                )
            )

    output = [out.outputs[0].text for out in model.generate(final_inputs, sampling_params)]

    for idx,resp in zip(response_idx,output):
        out_summary, out_evidence = fix_citations(resp, final_evidences[idx], True,
                                                  start_idx=1)
        final_responses[idx] = out_summary
        final_evidences[idx] = '\n'.join([f"[{k + 1}] {sent}" for k, sent in enumerate(out_evidence)])


    out_data = pd.DataFrame()
    assert len(all_queries) == len(final_responses)
    out_data.insert(0, "Summary", final_responses)
    out_data.insert(1, "Evidence", final_evidences)

    with open(f"{output_dir}/summaries_{split}_{seed}.jsonl", 'wt') as f:
        f.write(out_data.to_json(orient='records', lines=True))

    # Get the format for OpenScholar citation eval
    jsonl_output = [
        {
            'input': q,
            'output': o,
            'ctxs': [{'text': remove_citations(ev)} for ev in split_evidence(e)]
        } for q,o,e in zip(all_queries, final_responses, final_evidences)
    ]

    with open(f"{output_dir}/summaries_{split}_{seed}_citations.jsonl", 'wt') as f:
        for row in jsonl_output:
            f.write(json.dumps(row) + '\n')
