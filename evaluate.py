import os
import torch
import random
import numpy as np
import argparse
import json
import wandb
import pandas as pd
from scipy.stats import bootstrap
from tqdm import tqdm
from nltk import sent_tokenize
import re
from collections import defaultdict

from util.data import create_dataset
from util.geval import geval
from fix_citations import remove_citations, fix_citations


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


def evaluate_citations(data):
    final_scores = defaultdict(list)
    all_scores = defaultdict(list)
    for k,item in enumerate(tqdm(data)):

        sents = sent_tokenize(item['output'])
        if len(sents) == 0:
            continue

        target_sents = [remove_citations(sent).strip() for sent in sents]

        def extract_citations_sqa(text):
            # Regular expression to match [number] or [number_1, number_2, number_3]
            citation_pattern = r'\[(\d+(?:,\s*\d+)*)\]'
            # Find all matches in the text
            matches = re.findall(citation_pattern, text)
            # Extract individual numbers and convert them to integers
            citations = []
            for match in matches:
                # Split by commas, strip any extra whitespace, and convert to integers
                citations.extend([int(num.strip()) for num in match.split(',')])
            citations = ["[{}]".format(i) for i in citations]
            return citations

        cited_papers = set(extract_citations_sqa(item['output']))
        n_cite = 0
        n_sent = 0
        instances = []
        for sent_id, sent in enumerate(sents):
            # add minimum length for citation
            if len(sent) < 50:
                continue
            n_sent += 1

            target_sent = target_sents[sent_id]  # Citation removed and (if opted for) decontextualized

            # Find references
            ref = [int(r[1:]) - 1 for r in re.findall(r"\[\d+", sent)]

            if len(ref) > 0:
                n_cite += 1
                joint_passage = '\n***\n'.join([item['docs'][psgs_id]['text'] for psgs_id in ref if psgs_id >= 0 and psgs_id < len(item['docs'])])

                # If not directly rejected by citation format error, calculate the recall score
                instances.append(
                    {'document': joint_passage, 'response': target_sent,
                     'query': ""}
                )
        if len(instances) > 0:
            scores = geval("gpt-4o-mini", instances, {
                'Relevance_cite': {'prompt_file': './geval/relevance_no_query.txt'},
                'Consistency_cite': {'prompt_file': './geval/consistency_no_query.txt'}
            }, False, norm=(1,5))
        else:
            scores = {
                'Relevance_cite': [0.],
                'Consistency_cite': [0.]
            }

        if n_cite > 0:
            final_scores["relevance_cite_prec"].append(np.sum(scores['Relevance_cite']) / n_cite)
            final_scores["consistency_cite_prec"].append(np.sum(scores['Consistency_cite']) / n_cite)

        if n_sent > 0:
            final_scores['relevance_cite_rec'].append(np.sum(scores['Relevance_cite']) / n_sent)
            final_scores['consistency_cite_rec'].append(np.sum(scores['Consistency_cite']) / n_sent)
        final_scores['relevance_cite'].append(np.mean(scores['Relevance_cite']))
        final_scores['consistency_cite'].append(np.mean(scores['Consistency_cite']))

        all_scores['relevance_cite'].extend(scores['Relevance_cite'])
        all_scores['consistency_cite'].extend(scores['Consistency_cite'])
        all_scores['n_cite'].append(n_cite)
        all_scores['n_sent'].append(n_sent)

    if "relevance_cite_prec" not in final_scores:
        final_scores["relevance_cite_prec"] = [0.]
        final_scores["consistency_cite_prec"] = [0.]

    return {
        m: np.mean(final_scores[m]) for m in final_scores
    }, all_scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Path to the summaries", required=True)
    parser.add_argument("--citation_eval_file", type=str, help="Path to the summaries in format for cittaion eval (optional)", default=None)

    parser.add_argument("--metrics_dir", type=str, help="Name of the directory to store metrics", required=True)
    parser.add_argument("--run_name", type=str, help="A name for this run", required=True)
    parser.add_argument("--dataset_id", type=str, help="Huggingface dataset ID for data to use",
                        default='sobamchan/aclsum')
    parser.add_argument("--document_key", type=str, help="Which field to use in the dataset for the document (for RAG methods)",
                        default='document',
                        choices=['document', 'basic_rag', 'raptor_adg', 'colbert', 'distant_colbert', 'distant_colbert_rst', 'ft_and_distant_colbert', 'sbert', 'distant_sbert', 'ft_and_distant_sbert'])

    parser.add_argument("--tags", help="Tags to pass to wandb", required=False, type=str, default=[], nargs='+')
    parser.add_argument("--judge_metrics", help="A list of LLM as a judge metrics to measure", required=False, type=str, default=[], nargs='+')


    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--split", type=str,
                        help="Which split of the data to run on",
                        default='validation',
                        choices=['validation', 'test'])
    parser.add_argument("--use_gt", action="store_true", help="Whether or not to use ground truth responses in order to test the metrics")


    args = parser.parse_args()

    enforce_reproducibility(args.seed)

    judge_metrics = {
        'Relevance': {'prompt_file': './geval/relevance.txt'},
        'Consistency': {'prompt_file': './geval/consistency.txt'},
    }

    run_name = args.run_name
    input_path = args.input_path
    citation_eval_file = args.citation_eval_file
    dataset_id = args.dataset_id
    seed = args.seed
    document_key = args.document_key
    split = args.split
    metrics_dir = f"{args.metrics_dir}/{run_name}"
    used_judge_metrics = {m: judge_metrics[m] for m in args.judge_metrics}
    print(used_judge_metrics)
    use_gt = args.use_gt
    config = {
        "seed": seed,
        "dataset": dataset_id,
        "document_key": document_key
    }
    # wandb initialization
    run = wandb.init(
        name=args.run_name,
        config=config,
        reinit=True,
        tags=args.tags
    )

    if not os.path.exists(f"{metrics_dir}"):
        os.makedirs(f"{metrics_dir}")

    device = 'cpu'
    if torch.backends.mps.is_available():
        print("Using MPS")
        device = 'mps'
    elif torch.cuda.is_available():
        print("Using CUDA")
        device = 'cuda'

    dataset = create_dataset(dataset_id)

    contexts = [example[document_key] for example in dataset[split]]
    responses = [example['response'] for example in dataset[split]]
    queries = [example['question_text'] for example in dataset[split]]

    if input_path.endswith('.jsonl'):
        summary_df = pd.read_json(input_path, lines=True, orient='records').fillna('')
    else:
        summary_df = pd.read_csv(input_path).fillna('')

    if citation_eval_file:

        with open(citation_eval_file) as f:
            citation_data = [json.loads(l) for l in f]
        for item in citation_data:
            in_ev = '\n'.join([f"[{k + 1}] {sent['text']}" for k, sent in enumerate(item['ctxs'])])
            out_summary, out_evidence = fix_citations(item['output'], in_ev, True,
                                                      start_idx=1)
            item['output'] = out_summary
            item['ctxs'] = [{'text': sent} for sent in out_evidence]
            item["docs"] = item["ctxs"]
            if type(item["ctxs"]) is not list:
                item["docs"] = [{"text": ctx_text[0], "title": ctx_text[1]} for ctx_text in list(item["ctxs"].values())]
        citation_result, all_citation_scores = evaluate_citations(citation_data)
        np.savez(f"{metrics_dir}/{seed}_citation_geval_raw.npz", **all_citation_scores)

    if len(used_judge_metrics) > 0:
        if use_gt:
            instances = [
                {'document': c, 'response': r[0] if isinstance(r, list) else r,
                 'query': q} for c, r, q in zip(contexts, responses, queries)
            ]
        else:
            instances = [
                {'document': c, 'response': remove_citations(s),
                 'query': q} for c,s,q in zip(contexts, list(summary_df['Summary']), queries)
            ]
        scores = geval("gpt-4o-mini", instances, used_judge_metrics)



    for m in scores:
        print(f"{m}: {np.mean(scores[m])}")
        wandb.log({
            m: np.mean(scores[m])
        })

    with open(f"{metrics_dir}/{seed}.json", 'wt') as f:
        out_json = {}
        for m in scores:
            # Get 95% CI
            ci = bootstrap((scores[m],), np.mean, confidence_level=0.95, random_state=seed)
            out_json[m] = float(np.mean(scores[m]))
            out_json[f"{m}_low_ci"] = ci.confidence_interval.low
            out_json[f"{m}_high_ci"] = ci.confidence_interval.high

            print(f"{m}: {out_json[m]}")
        if citation_eval_file:
            out_json.update(citation_result)
        f.write(json.dumps(out_json))

    np.savez(f"{metrics_dir}/{seed}_geval_raw.npz", **scores)
