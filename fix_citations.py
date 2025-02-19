import json
import numpy as np
from util.data import create_templated_dataset, create_dataset, create_oai_synthetic_dataset
import re
import glob
from pathlib import Path
import os


CITATION_REGEX = r'\[(\d+(?:,\s*\d+)*)\]'

def remove_citations(text):
    # Regular expression to match [number] or [number_1, number_2, number_3]
    citation_pattern = r'\[\d+(?:,\s*\d+)*\]'
    # Remove all citations from the text
    cleaned_text = re.sub(citation_pattern, '', text)
    # Optionally, remove extra spaces that might result from removing citations
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()
    cleaned_text = cleaned_text.replace(" .", ".")
    cleaned_text = cleaned_text.replace(" ,", ",")
    return cleaned_text


def split_evidence(evidence):
    regex = r'\[\d+(?:,\s*\d+)*\]'
    return [s.strip() for s in re.split(regex, evidence) if s.strip() != '']


def extract_citations(text):
    # Find all matches in the text
    matches = re.findall(CITATION_REGEX, text)
    # Extract individual numbers and convert them to integers
    citations = []
    for match in matches:
        # Split by commas, strip any extra whitespace, and convert to integers
        citations.extend([int(num.strip())-1 for num in match.split(',')])
    #citations = ["[{}]".format(i) for i in citations]
    return citations


def fix_citations(original_summary, original_evidence, update_evidence=False, start_idx=1):
    used_ref_numbers = list(set(sorted(extract_citations(original_summary))))
    if update_evidence:
        # Split the evidence based on reference numbers
        evidence_list = split_evidence(original_evidence)
        final_evidence = [evidence_list[k] for k in used_ref_numbers if k < len(evidence_list) and k >= 0]
    else:
        evidence_list = original_evidence
        final_evidence = [original_evidence[k] for k in used_ref_numbers if k < len(evidence_list)]
    # Replace the references in the text
    out_summary = original_summary
    ev_idx = 0
    for n, k in enumerate(used_ref_numbers):
        if k < len(evidence_list) and k >= 0:
            out_summary = out_summary.replace(f"[{k + 1}]", f"[{ev_idx + start_idx}]")
            final_evidence[ev_idx] = final_evidence[ev_idx].replace(f"[{k + 1}]", f"[{ev_idx + start_idx}]")
            ev_idx += 1
        else:
            out_summary = out_summary.replace(f"[{k + 1}]", f"")

    return out_summary, final_evidence
