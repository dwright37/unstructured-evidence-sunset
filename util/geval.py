from openai import OpenAI
import time
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import tiktoken


def geval(model, instances, metrics={}, return_all=True, norm=None):
    client = OpenAI()
    scores = defaultdict(list)
    encoding = tiktoken.get_encoding("cl100k_base")

    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        num_tokens = len(encoding.encode(string))
        return num_tokens

    for m in metrics:
        metric = metrics[m]
        with open(metric['prompt_file']) as f:
            prompt = f.read().strip()

        ct, ignore = 0, 0


        for instance in tqdm(instances):
            source = instance['document']
            system_output = instance['response']
            query = instance['query']
            cur_prompt = prompt.replace('{document}', source).replace('{summary}', str(system_output)).replace("{query}", query)
            instance['prompt'] = cur_prompt
            while True:
                try:
                    _response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": cur_prompt}],
                        temperature=1,
                        max_tokens=1,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None,
                        n=20
                    )
                    time.sleep(0.01)

                    all_responses = [_response.choices[i].message.content for i in
                                     range(len(_response.choices))]
                    conv_scores = [int(s) for s in all_responses if s.isdigit()]
                    if len(conv_scores) == 0:
                        if norm:
                            conv_scores = [norm[1]]
                        else:
                            conv_scores = [1.]
                    if norm:
                        conv_scores = [min(max(norm[0], s), norm[1]) for s in conv_scores]
                        conv_scores = [(s - norm[0]) / (norm[1] - norm[0]) for s in conv_scores]
                    if return_all:
                        scores[m].extend(conv_scores)
                    else:
                        scores[m].append(np.mean(conv_scores))
                    ct += 1
                    break
                except Exception as e:
                    print(e)
                    if ("limit" in str(e).lower()):
                        time.sleep(60)
                    else:
                        ignore += 1
                        print('ignored', ignore)

                        break
        print(f"{m}: {np.mean(scores[m])}")

    return scores
