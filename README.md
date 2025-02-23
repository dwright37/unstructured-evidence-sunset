# Unstructured Evidence Attribution for Long Context Query Focused Summarization
Code and dataset release for the paper "Unstructured Evidence Attribution for Long Context Query Focused Summarization"

## Dataset

We have released our synthetic dataset, the Summaries with Unstructured Evidence Text (SUnsET) dataset, on Huggingface datasets. It can be downloaded as follows:

```
from datasets import load_dataset

dataset = load_dataset("dwright37/SUnsET")
```

The data contains the following columns:

```
doc_id: A unique document ID for each document
chunks: A list of strings containing each section of the document
question_text: A query about the document
response_referenced: The summary responding to the query with citations added to sentences
evidence: A list of strings containing the evidence used (in the same order as the reference numbers used in the summary)
response: The summary without any reference numbers added
unrefined_response: The original summary generated for the query before refining it
document: A string containing the concatenated document sections
```

## Code

We release the following scripts used to generate the results:

- `generate_synthetic_data.py`: Used to generate SUnsET
- `generate_synthetic_data_baseline.py`: Used to generate the baseline synthetic data for comparison
- `train.py`: Used for LoRA training
- `infer.py`: Used for all model inference
- `evaluate.py`: Used for all model evaluation
