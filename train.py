import os
import torch
import random
import numpy as np
import argparse
from tqdm import tqdm
from torch import bfloat16
import transformers
from transformers import TrainerCallback
#from bert_score import score as bertscore
import json
#import ipdb
#from carbontracker.tracker import CarbonTracker
from peft import LoraConfig, get_peft_model
from trl import (
    SFTTrainer,
    DataCollatorForCompletionOnlyLM,
    SFTConfig
)
from transformers import TrainingArguments, Trainer
from typing import Dict
from nltk.tokenize import sent_tokenize
from datasets import concatenate_datasets
import wandb
import glob

from util.data import create_templated_dataset, create_dataset, create_oai_synthetic_dataset
from util.data import DEFAULT_SYSTEM_PROMPT, IFT_SYSTEM_PROMPT, CHAT_TEMPLATE

COMPLETIONS = {
    "mistralai/Mistral-7B-Instruct-v0.2": "[/INST]",
    "mistralai/Mistral-Nemo-Instruct-2407": "[/INST]",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "[/INST]",
    "meta-llama/Meta-Llama-3-8B-Instruct": "<|start_header_id|>assistant<|end_header_id|>",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "<|start_header_id|>assistant<|end_header_id|>",
    "meta-llama/Llama-3.1-70B-Instruct": "<|start_header_id|>assistant<|end_header_id|>",
    "neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8": "<|start_header_id|>assistant<|end_header_id|>",
    "meta-llama/Llama-3.2-1B-Instruct": "<|start_header_id|>assistant<|end_header_id|>",
    "meta-llama/Llama-3.2-3B-Instruct": "<|start_header_id|>assistant<|end_header_id|>",
    "CohereForAI/c4ai-command-r-v01": "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
    "CohereForAI/c4ai-command-r-plus-08-2024": "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
    "CohereForAI/c4ai-command-r-plus-4bit": "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
    "mistralai/Mistral-7B-v0.3": " [-SUMMARY-]",
    "princeton-nlp/gemma-2-9b-it-SimPO": "\n<start_of_turn>model\n"
}

INSTRUCTIONS = {
    "mistralai/Mistral-7B-Instruct-v0.2": "[INST]",
    "mistralai/Mistral-Nemo-Instruct-2407": "[INST]",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "[INST]",
    "meta-llama/Meta-Llama-3-8B-Instruct": "<|start_header_id|>user<|end_header_id|>",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "<|start_header_id|>user<|end_header_id|>",
    "meta-llama/Llama-3.1-70B-Instruct": "<|start_header_id|>user<|end_header_id|>",
    "neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8": "<|start_header_id|>user<|end_header_id|>",
    "meta-llama/Llama-3.2-1B-Instruct": "<|start_header_id|>user<|end_header_id|>",
    "meta-llama/Llama-3.2-3B-Instruct": "<|start_header_id|>user<|end_header_id|>",
    "CohereForAI/c4ai-command-r-v01": "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
    "CohereForAI/c4ai-command-r-plus-08-2024": "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
    "CohereForAI/c4ai-command-r-plus-4bit": "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
    "mistralai/Mistral-7B-v0.3": " [-INPUT-]",
    "princeton-nlp/gemma-2-9b-it-SimPO": "<start_of_turn>user\n"
}

CHAT_TEMPLATES = {
    "mistralai/Mistral-7B-v0.3": CHAT_TEMPLATE
}


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
    return num_new_tokens


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
    parser.add_argument("--dataset_id", type=str, help="Huggingface dataset ID for data to use",
                        default='pszemraj/SQuALITY-v1.3')
    parser.add_argument("--synthetic_data_dir", type=str, help="Location of the synthetic data",
                        required=True)
    parser.add_argument("--document_key", type=str,
                        help="Which field to use in the dataset for the document (for RAG methods)",
                        default='document',
                        choices=['document', 'chunks'])
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--n_train", type=int, default=-1, help="Number of training documents")
    parser.add_argument("--n_validation", type=int, default=-1, help="Number of validation documents")

    parser.add_argument("--quant", action="store_true", help="Whether or not to quantize models")
    parser.add_argument("--random_shuffles", type=int, default=0, help="Number of random context shuffles for data augmentation")

    parser.add_argument("--tags", help="Tags to pass to wandb", required=False, type=str, default=[], nargs='+')

    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--lora_r", type=int, help="LoRA rank", default=16)
    parser.add_argument("--learning_rate", type=float, help="Learning rate", default=1e-4)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=4)
    parser.add_argument("--warmup_steps", type=int, help="Number of warmup steps to use", default=0)






    args = parser.parse_args()

    enforce_reproducibility(args.seed)

    model_id = args.model_id
    run_name = args.run_name
    output_dir = f"{args.output_dir}/{run_name}"
    dataset_id = args.dataset_id
    synthetic_data_dir = args.synthetic_data_dir
    document_key = args.document_key
    seed = args.seed
    quant = args.quant
    random_shuffles = args.random_shuffles
    metrics_dir = f"{args.metrics_dir}/{run_name}"
    n_epochs = args.n_epochs
    lora_r = args.lora_r
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    warmup_steps = args.warmup_steps
    cache_dir = None if 'HF_MODEL_CACHE' not in os.environ else os.environ['HF_MODEL_CACHE']

    n_train = args.n_train
    n_validation = args.n_validation

    config = {
        "seed": seed,
        "dataset": dataset_id,
        "model": model_id,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "batch_size": batch_size,
        "lora_r": lora_r
    }
    # wandb initialization
    run = wandb.init(
        name=args.run_name,
        config=config,
        reinit=True,
        tags=args.tags,
        project="faceted-longdoc-summarization"
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
        bnb_4bit_compute_dtype=bfloat16,
        bnb_4bit_quant_storage=bfloat16,  # Computation type
    )


    max_memory = None

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    if model_id in CHAT_TEMPLATES:
        tokenizer.chat_template = CHAT_TEMPLATES[model_id]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=bfloat16,
        quantization_config=bnb_config if quant else None,
        device_map="auto",
        max_memory=max_memory if quant else None,
        cache_dir=cache_dir,
        attn_implementation="flash_attention_2"
    )
    tokens_added = 0
    if tokenizer.pad_token is None:
        tokens_added = smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="<pad>"),
            tokenizer=tokenizer,
            model=model,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
    max_tokens = model.config.max_position_embeddings

    def data_formatting(example):
        output_text = []
        for i in range(len(example['response_referenced'])):
            syst_prompt = DEFAULT_SYSTEM_PROMPT
            document = example[document_key][i]
            # Now shuffle if enabled
            if random_shuffles > 0:
                doc_sents = document if document_key == 'chunks' else sent_tokenize(document)
                random.shuffle(doc_sents)
                document = '\n'.join(doc_sents)
            evidence_sents = '\n'.join(
                [f"[{k + 1}] {sent}" for k, sent in enumerate(example['evidence'][i])])

            asst_prompt = f"""EVIDENCE:
{evidence_sents}
RESPONSE:
{example['response_referenced'][i]}
"""

            question_text = example['question_text'][i]
            asst_tokens = tokenizer(asst_prompt, add_special_tokens=False).input_ids
            doc_tokens = tokenizer(document, add_special_tokens=False).input_ids
            length = min(len(doc_tokens), max_tokens - len(asst_tokens))
            document = tokenizer.decode(doc_tokens[:length])

            user_prompt = f"""Your task is to read a document and then write an essay which addresses the following question: {question_text}

To write your essay, you should read the document and identify key passages which will help guide your response. Extract every passage which is directly relevant for your essay. Please copy each extracted passage to a list in the format specified below. Please copy the exact text of each passage (do NOT paraphrase!). Then, write your essay which addresses the query. 

Please add citations to all citation-worthy statements using the extracted evidence, by indicating the citation numbers of the corresponding evidence. More specifically, add the citation number at the end of each relevant sentence before the punctuation mark e.g., 'This work shows the effectiveness of problem X [1].' when the passage [1] in the evidence list provides full support for the statement. Only add a citation if it is fully relevant and unambiguously supportive of that sentence. Not all evidences may be relevant, so only cite those that directly support the statement. Please do not add any explanations or justifications for the evidence, simply indicate the evidence numbers if they are relevant. If a sentence does not use any of the provided evidence, please simply copy the sentenece as is and do not add anything to the end of it. If multiple evidences support a statement, please cite them together (e.g., [1][2]). For each citation-worthy statement, you only need to add at least one citation, so if multiple evidences support the statement, just add the most relevant citation to the sentence.

Here is the document: {document}

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


            msgs = [
                    {"role": "system", "content": syst_prompt},
                    {"role": "document", "content": example[document_key]},
                    {"role": "query", "content": question_text},
                    {"role": "summary", "content": asst_prompt}
            ]
            output_text.append(tokenizer.apply_chat_template(msgs, tokenize=False))
        return output_text

    dataset = create_oai_synthetic_dataset(synthetic_data_dir, n_validation=n_validation, n_train=n_train)
    if random_shuffles > 0:
        dataset['train'] = concatenate_datasets([dataset['train']]*random_shuffles)

    peft_config = LoraConfig(
        lora_alpha=lora_r,
        lora_dropout=0.05,
        r=lora_r,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        do_eval=True,
        eval_strategy="epoch" if random_shuffles == 0 else "steps",
        eval_steps=None if random_shuffles == 0 else int((len(dataset['train']) / random_shuffles) / batch_size / n_epochs),
        gradient_checkpointing=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        save_strategy="epoch" if random_shuffles == 0 else "steps",
        save_steps=None if random_shuffles == 0 else int((len(dataset['train']) / random_shuffles) / batch_size / n_epochs),
        save_total_limit=5,
        seed=seed,
        bf16=True,
        report_to=["wandb"],
        logging_steps=1,
        load_best_model_at_end=True,
        num_train_epochs=n_epochs,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False  # No need to add additional separator token
        },
        max_seq_length=max_tokens
    )

    collator = DataCollatorForCompletionOnlyLM(instruction_template=INSTRUCTIONS[model_id],
                                               response_template=COMPLETIONS[model_id],
                                               tokenizer=tokenizer,
                                               mlm=False,
                                               padding_free=True
                                               )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        peft_config=peft_config,
        tokenizer=tokenizer,
        data_collator=collator,
        formatting_func=data_formatting
    )
    ckpt_dirs = list(glob.glob(f"{output_dir}/*checkpoint*"))
    trainer.train(resume_from_checkpoint=len(ckpt_dirs) > 0)

    model.resize_token_embeddings(len(tokenizer) - tokens_added)

    trainer.model.save_pretrained(f"{output_dir}/{seed}")
    tokenizer.save_pretrained(f"{output_dir}/{seed}")
