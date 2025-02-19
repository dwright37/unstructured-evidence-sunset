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


prompt1 = """
Imagine that you must write a book. This book can be either fiction or non-fiction.

You can select any subject to write your book about. Please make the book interesting.

Please write a list of 100 possible book titles. 

Please only generate the title for each book. 

Please include a mix of fiction and non-fiction, and please try to cover as many genres as possible.

Please make each book title unique.

Please make the style of each book title as different as possible, and don't repeat title styles.

Please generate titles for books which will have a broad range of appeal. 

Please generate titles for books which will require a broad range of reading levels.

Please try to make each title as different as possible.

Please do not include many titles with a colon (:).

{prev_titles_prompt}

**OUTPUT FORMAT**
Please separate each book title with a newline character ("\n")
"""


prompt2 = """
Imagine that you must write a book. This book can be either fiction or non-fiction.

This is the title of your book: {title}

Please write an outline of this book. Please include the title of the book, and a list of chapters or sections that the book will contain. The book should have 6 sections or chapters.

**OUTPUT FORMAT**

Please output the outline as a JSON object where the keys are the chapters and the values are a brief outline of the chapter. In other words, as:
```python
{
'Chapter 1': "Chapter 1 outline",
'Chapter 2': "Chapter 2 outline",
...
'Chapter N': "Chapter N outline"
}
```
"""


prompt3 = """
Imagine that you must write a book. You are given the following outline of the book

{outline}

Please write a list of 5 questions about the book which summarize the book. 

Please try to cover different general aspects of the content.

Please make the questions very concise.

**OUTPUT FORMAT**

Please separate each question with a single newline character ("\n")
"""


prompt4 = """
Imagine that you are writing a book. This is an outline of the book

{outline}

Please address the following question about the book:

{question}

Please write a summary which addresses the question. Please make the summary as specific and detail oriented as possible. Please include actual examples from the book when possible. Please do not write more than is absolutely necessary.

After you write the summary, please write exact quotes and passages you will include in the book, from which the summary could be written. Please include at least {n_evidence} of these passages, which you intend to include verbatim in the book. Please indicate the exact chapter where the passages will be written in a separate field.

**OUTPUT FORMAT**

Please a JSON object with two fields: "summary", "evidence", and "chapter". The summary field should have the summary. The evidence field should have a list of evidence sentences from the book. The chapter field should have the exact chapter where the corresponding evidence sentence will appear. Please only indicate the chapter number for this field. There should be the same number of elements in the "evidence" field as there are in the "chapter" field. In other words, as:
```python
{
'summary': "Summary text",
'evidence': ['evidence sentence 1', 'evidence sentence 2', ...]
'chapter': [1, 4, ...]
}
```
"""


prompt5 = """
Imagine that you must write a book. You are given the following outline of the book

{outline}

Please write the following chapter of the book in its entirety:

{chapter}

Please also include the following sentences somewhere in the chapter. You must include these passages verbatim (i.e., EXACTLY as is). It is imperative that you do this, otherwise the book will be incomplete:

{evidence}

**OUTPUT FORMAT**

Please wrap the content of the chapter you write in a markdown codeblock, in other words, like:

```
content
```
"""

retrieval_prompt = """
Please read the following book chapter:

{chapter}

The following passage should have been included in the chapter but was not:

{passage}

Please retrieve the passage from the chapter which is CLOSEST to the given passage.

**OUTPUT FORMAT**

Please wrap the passage in a markdown codeblock, in other words, like:

```
passage
```

"""


prompt6 = """
Imagine that you are giving an exam about a book. This is the book

{book}

On an exam, you are asked to summarize the book with respect to this question:

{question}

This is the summary that you are grading:

{summary}

Please rewrite this response so that it is totally accurate and fully addresses the question.

Please make the response as specific and detail oriented as possible. The following passages from the document should help in crafting the response:

{passages}

**OUTPUT FORMAT**

Please wrap the content of the summary you write in a markdown codeblock, in other words, like:

```
content
```
"""


prompt7 = """
Imagine that you are judging the quality of a summary of a book. This is the book

{book}

Here is a question about the book:

{question}

And here is the summary which addresses the question:

{summary}

Please judge if you think that the summary meets ALL of the following criteria:

1) The summary is absolutely faithful to the book (in other words, all of the information in the summary is contained in the book)

2) The summary FULLY addresses the question

Please think carefully about your answer. If you think that ALL of the criteria are met, please simply respond with "YES". 

Otherwise, please simply respond with "NO".
"""


prompt8 = """
Imagine that you have written a research essay about a book. You have also extracted passages from the book which you used to write the essay. 
Your job is to add citations to the essay which properly reference the passages that you have extracted. 

Here is the essay:

{essay}

And here are the evidence passages from the book, each of which is given a number: 

{evidence}

Please add citations to all citation-worthy statements in the essay using the numbered evidence list, by indicating the citation numbers of the corresponding evidence. 
More specifically, add the citation number at the end of each relevant sentence in the essay before the punctuation mark e.g., 'This work shows the effectiveness of problem X [1].' when the passage [1] in the evidence list provides full support for the statement. 
Only add a citation if it is fully relevant and unambiguously supportive of that sentence. Not all evidences may be relevant, so only cite those that directly support the statement. 
Please do not add any explanations or justifications for the evidence, simply indicate the evidence numbers if they are relevant. 
If a sentence does not use any of the provided evidence, please simply copy the sentence as is and do not add anything to the end of it. 
If multiple evidences support a statement, please cite them together (e.g., [1][2]). 
For each citation-worthy statement, you only need to add at least one citation, so if multiple evidences support the statement, just add the most relevant citation to the sentence.
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
    existing_titles = set()
    finished_titles = set()
    titles = set()
    for fname in glob.glob(f'{output_dir}/*.json'):
        with open(fname) as f:
            data = json.loads(f.read())
            finished_titles.add(data['title'])

    if os.path.exists(f"{output_dir}/existing_titles.txt"):
        with open(f"{output_dir}/existing_titles.txt") as f:
            existing_titles = set([l.strip() for l in f])

    if os.path.exists(f"{output_dir}/titles.txt"):
        with open(f"{output_dir}/titles.txt") as f:
            titles = [l.strip() for l in f]
    else:
        while len(titles) < 1000:
            if len(existing_titles) > 0:
                existing_title_string = '\n'.join(list(existing_titles))
                p1 = prompt1.replace("{prev_titles_prompt}", f"Please do not use any of the following titles, and make the new titles as different as possible from these titles:\n{existing_title_string}")
            else:
                p1 = prompt1.replace("{prev_titles_prompt}", "")
            # Step 1: Generate some book titles
            completion = generate(p1, 1.2)
            assert completion != None, "generation failure"

            curr_titles = [title.strip() for title in completion.choices[0].message.content.split('\n') if len(title.strip()) > 0]
            titles.update(set([t.split('.')[1].strip() if '.' in t else t.strip() for t in curr_titles]))
            existing_titles.update(titles)
        titles = list(titles)
        with open(f"{output_dir}/titles.txt", 'wt') as f:
            f.write('\n'.join(titles))
    for title in tqdm(titles):
        if title in finished_titles:
            continue
        # Step 2: Generate an outline
        n_chapters = random.randint(3, 20)
        p2 = prompt2.replace("{title}", title).replace("{n_chapters}", str(n_chapters))
        completion = generate(p2, 1.)
        assert completion != None, "generation failure"

        text_output = completion.choices[0].message.content
        try:
            result = re.findall(CODE_BLOCK_REGEX, text_output, re.DOTALL | re.MULTILINE)[0].strip()
            json_result = extract_json(result)
            if json_result == None:
                continue
        except:
            continue
        sections = [k + " -- " + v for k,v in json_result.items()]
        outline = '\n'.join([title] + sections)

        if not all(sec.strip().lower().startswith("chapter") for sec in sections):
            # Bad formatting, skip
            continue

        # Step 3: Write some questions
        p3 = prompt3.replace("{outline}", outline)
        completion = generate(p3, 1.)
        assert completion != None, "generation failure"

        questions = [q.strip() for q in re.split('\n+', completion.choices[0].message.content)]

        # Step 4: Generate some summaries for these questions
        summaries = []
        for q in questions:
            summ = ''
            n_sent = random.randint(1, 5)
            n_evidence = random.randint(5, 10)
            p4 = prompt4.replace("{outline}", result).replace("{question}", q).replace("{n_sentences}",
                                                                                       str(n_sent)).replace(
                "{n_evidence}", str(n_evidence))
            while summ == '':
                completion = generate(p4, 1.)
                assert completion != None, "generation failure"

                try:
                    summ = re.findall(CODE_BLOCK_REGEX, completion.choices[0].message.content, re.DOTALL | re.MULTILINE)[
                            0].strip()
                    summaries.append(summ)
                except:
                    summ = ''

        json_summaries = [extract_json(s) for s in summaries]
        json_summaries = [s for s in json_summaries if
                          s != None and 'summary' in s and 'evidence' in s and 'chapter' in s and len(
                              s['evidence']) == len(s['chapter']) and min(s['chapter']) > 0 and max(
                              s['chapter']) <= len(sections)]

        chapter_to_evidence = defaultdict(list)
        evidence_to_summary = {}
        for summary in json_summaries:
            for ch, ev in zip(summary['chapter'], summary['evidence']):
                chapter_to_evidence[ch].append(ev)
                evidence_to_summary[(ch, ev)] = summary['summary']

        chapters = []
        for i,chapter in enumerate(sections):
            evidence = '\n\n'.join(chapter_to_evidence[i + 1])
            p5 = prompt5.replace("{outline}", result).replace("{evidence}", evidence).replace("{chapter}", chapter)
            completion = generate(p5, 1., 1000)
            assert completion != None, "generation failure"

            if len(completion.choices) > 0:
                chapter_search = re.findall(CODE_BLOCK_REGEX, completion.choices[0].message.content, re.DOTALL | re.MULTILINE)
                if len(chapter_search) > 0:
                    chapters.append(chapter_search[0].strip())
                else:
                    chapters.append(completion.choices[0].message.content)

                for e in chapter_to_evidence[i + 1]:
                    if e.lower() not in chapters[-1].lower():
                        retr_prompt = retrieval_prompt.replace("{chapter}", chapters[-1]).replace("{passage}", e)
                        completion = generate(retr_prompt, 0.)
                        assert completion != None, "generation failure"
                        try:
                            new_evidence = \
                            re.findall(CODE_BLOCK_REGEX, completion.choices[0].message.content, re.DOTALL | re.MULTILINE)[
                                0].strip()
                            if new_evidence.lower() in chapters[-1].lower():
                                evidence_to_summary[(i + 1, new_evidence)] = evidence_to_summary[(i + 1, e)]
                            else:
                                print("fail")
                        except:
                            print("fail")
                        del evidence_to_summary[(i + 1, e)]
        final_summaries = []
        summary_to_evidence = defaultdict(list)
        for k in evidence_to_summary:
            summary_to_evidence[evidence_to_summary[k]].append(k)
        for summary in json_summaries:
            final_summaries.append({
                'summary': summary['summary'],
                'chapter': [v[0] for v in summary_to_evidence[summary['summary']]],
                'evidence': [v[1] for v in summary_to_evidence[summary['summary']]]
            })

        new_summaries = []
        doc = '\n'.join(chapters)
        for q,s in zip(questions,final_summaries):
            n_sent = random.randint(1, 10)
            p6 = prompt6.replace("{book}", doc).replace("{question}", q).replace("{summary}", s['summary']).replace("{passages}", '\n'.join(s['evidence']))
            completion = generate(p6, 1.)
            assert completion != None, "generation failure"

            try:
                summ = re.findall(CODE_BLOCK_REGEX, completion.choices[0].message.content, re.DOTALL | re.MULTILINE)[
                    0].strip()
            except:
                summ = completion.choices[0].message.content
            new_summaries.append({
                'summary': summ,
                'unrefined_summary': s['summary'],
                'evidence': s['evidence'],
                'chapter': s['chapter'],
                'question_text': q
            })

        final_questions = []
        final_summaries = []
        final_unrefined_summaries = []
        final_evidence = []
        final_evidence_to_chapter = []
        final_referenced_responses = []
        for s in new_summaries:
            p7 = prompt7.replace("{book}", doc).replace("{question}", s['question_text']).replace("{summary}", s['summary'])
            completion = generate(p7, 0.)
            assert completion != None, "generation failure"

            resp = completion.choices[0].message.content
            if resp == 'YES':
                final_questions.append(s['question_text'])
                final_summaries.append(s['summary'])
                final_unrefined_summaries.append(s['unrefined_summary'])
                final_evidence.append(s['evidence'])
                final_evidence_to_chapter.append(s['chapter'])
                essay = s['summary']
                numbered_evidence = '\n'.join(
                    [f"[{k + 1}] {sent}" for k, sent in enumerate(s['evidence'])])
                p = prompt8.replace("{essay}", essay).replace("{evidence}", numbered_evidence)
                final_referenced_responses.append(generate(p, 1.).choices[0].message.content)

        with open(f"{output_dir}/{str(uuid.uuid4())}.json", 'wt') as f:
            f.write(json.dumps({
                'title': title,
                'outline': outline,
                'questions': final_questions,
                'summaries': final_summaries,
                'unrefined_summaries': final_unrefined_summaries,
                'evidence': final_evidence,
                'evidence_to_chapter': final_evidence_to_chapter,
                'sections': sections,
                'chapters': chapters,
                'response_referenced': final_referenced_responses
            }))
