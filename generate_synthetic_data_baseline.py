from openai import OpenAI
from tqdm import tqdm
import random
import numpy as np
import torch
import re
import argparse
import uuid
import json
import os
from tqdm import tqdm
import glob
from collections import defaultdict
from util.data import extract_json
import time


CODE_BLOCK_REGEX = "```(?:\w+)?[a-z]*?\s*\n(.*?)(?=^```)```"


prompt = """
Imagine that you must write a book. This book can be either fiction or non-fiction.

You can select any subject to write your book about. Please make the book interesting.

Please perform the following tasks and output everything in as a JSON object:

Please write the title of the book. {title_prompt}

Then, please write an outline of this book. Please include a list of chapters or sections that the book will contain. The book should have 6 sections or chapters.

Then, please write a list of 5 questions about the book which summarize the book.

Then, please write a summary for each question which addresses the question.

Then, please write the entire contents of the book. The book should be long, and you should write out the ENTIRE content.

Then, extract specific passages from the book for each summary which serve as evidence for the summary.


**OUTPUT FORMAT**
Please create a well-formatted JSON object with the following fields:

title: The title of the book (formatted as a string)
outline: The outline of the book (formatted as a string)
questions: The questions about the book (formated as a list)
summaries: The summaries addressing each question (formatted as a list of the same length as "questions")
document: The full book (formatted as a string)
evidence: A list of evidence passages (formatted as a list of the same length as "questions")
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


def generate(prompt, temp, n_toks=None):
    while True:
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temp,
                max_tokens=n_toks
            )
        except Exception as e:
            print(e)
            if ("limit" in str(e).lower()):
                time.sleep(60)
            else:
                print('ignored')

                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="Name of the directory to save the generated text",
                        required=True)

    args = parser.parse_args()

    output_dir = args.output_dir
    client = OpenAI()
    seed = 1000
    enforce_reproducibility(seed)

    if not os.path.exists(f"{output_dir}"):
        os.makedirs(f"{output_dir}")

    # Get any finished documents
    titles = []
    pbar = tqdm(total=3, desc="Generating documents...")
    for j in range(3):
        if len(titles) > 0:
            cat_titles = '\n'.join(titles)
            p = prompt.replace("{title_prompt}", f"Please do not use any of the following titles:\n{cat_titles}")
        else:
            p = prompt.replace("{title_prompt}", "")
        print(p)
        completion = generate(p, 1., n_toks=16384)
        assert completion != None, "generation failure"
        text_output = completion.choices[0].message.content
        result = re.findall(CODE_BLOCK_REGEX, text_output, re.DOTALL | re.MULTILINE)
        if len(result) > 0:
            result = result[0].strip()
            json_result = extract_json(result)
            if json_result != None and 'document' in json_result and 'title' in json_result and len(json_result['questions']) == len(json_result['summaries']) and len(json_result['evidence']) == len(json_result['summaries']):
                with open(f"{output_dir}/{str(uuid.uuid4())}.json", 'wt') as f:
                    f.write(json.dumps(json_result))
                    titles.append(json_result['title'])
                    pbar.update(1)
    pbar.close()
