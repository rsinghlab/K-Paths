import pandas as pd
# Import necessary libraries
import pandas as pd
import re
from bert_score import score as bert_score_fn
import torch
import os
import pandas as pd


truth = pd.read_json("/paths/drugbank_test_add_reverse.json",lines=True) #read the truth file

drugbank_relation_id_to_name = {
    23: "{u} may increase the photosensitizing activities of {v}",
    24: "{u} may increase the anticholinergic activities of {v}",
    25: "{u} can decrease the bioavailability of {v}",
    26: "{u} can increase the metabolism of {v}",
    27: "{u} may decrease the vasoconstricting activities of {v}",
    28: "{u} may increase the anticoagulant activities of {v}",
    29: "{u} may increase the ototoxic activities of {v}",
    30: "{u} can increase the therapeutic efficacy of {v}",
    31: "{u} may increase the hypoglycemic activities of {v}",
    32: "{u} may increase the antihypertensive activities of {v}",
    33: "{u} may reduce the serum concentration of the active metabolites of {v}",
    34: "{u} may decrease the anticoagulant activities of {v}",
    35: "{u} may decrease the absorption of {v}",
    36: "{u} may decrease the bronchodilatory activities of {v}",
    37: "{u} may increase the cardiotoxic activities of {v}",
    38: "{u} may increase the central nervous system depressant activities of {v}",
    39: "{u} may decrease the neuromuscular blocking activities of {v}",
    40: "{u} can increase the absorption and serum concentration of {v}",
    41: "{u} may increase the vasoconstricting activities of {v}",
    42: "{u} may increase the QTc prolonging activities of {v}",
    43: "{u} may increase the neuromuscular blocking activities of {v}",
    44: "{u} may increase the adverse neuromuscular activities of {v}",
    45: "{u} may increase the stimulatory activities of {v}",
    46: "{u} may increase the hypocalcemic activities of {v}",
    47: "{u} may increase the atrioventricular blocking activities of {v}",
    48: "{u} may decrease the antiplatelet activities of {v}",
    49: "{u} may increase the neuroexcitatory activities of {v}",
    50: "{u} may increase the dermatologic adverse activities of {v}",
    51: "{u} may decrease the diuretic activities of {v}",
    52: "{u} may increase the orthostatic hypotensive activities of {v}",
    53: "{u} may increase the hypertensive effects of {v}",
    54: "{u} may increase the sedative activities of {v}",
    55: "{u} may increase the severity of QTc prolonging effects when combined with {v}",
    56: "{u} may increase the immunosuppressive activities of {v}",
    57: "{u} may increase the neurotoxic activities of {v}",
    58: "{u} may increase the antipsychotic activities of {v}",
    59: "{u} may decrease the antihypertensive activities of {v}",
    60: "{u} may increase the vasodilatory activities of {v}",
    61: "{u} may increase the constipating activities of {v}",
    62: "{u} may increase the respiratory depressant activities of {v}",
    63: "{u} may increase the hypotensive and central nervous system depressant activities of {v}",
    64: "{u} may increase the severity of hyperkalemic effects when combined with {v}",
    65: "{u} may decrease the protein binding of {v}",
    66: "{u} may increase the central neurotoxic activities of {v}",
    67: "{u} may decrease the diagnostic effectiveness of {v} ",
    68: "{u} may increase the bronchoconstrictory activities of {v}",
    69: "{u} may decrease the metabolism of {v}",
    70: "{u} may increase the myopathic rhabdomyolysis activities of {v}",
    71: "{u} may increase the severity of adverse effects when combined with {v}",
    72: "{u} may increase heart failure risks when combined with {v}",
    73: "{u} may increase the hypercalcemic activities of {v}",
    74: "{u} may decrease the analgesic activities of {v}",
    75: "{u} may increase the antiplatelet activities of {v}",
    76: "{u} may increase the bradycardic activities of {v}",
    77: "{u} may increase the hyponatremic activities of {v}",
    78: "{u} may increase the hypotensive effects when combined with {v}",
    79: "{u} may increase the nephrotoxic activities of {v}",
    80: "{u} may decrease the cardiotoxic activities of {v}",
    81: "{u} may increase the ulcerogenic activities of {v}",
    82: "{u} may increase the hypotensive activities of {v}",
    83: "{u} may decrease the stimulatory activities of {v}",
    84: "{u} may increase the bioavailability of {v}",
    85: "{u} may increase the myelosuppressive activities of {v}",
    86: "{u} may increase the serotonergic activities of {v}",
    87: "{u} may increase the excretion rate {v}",
    88: "{u} may increase bleeding risks when combined with {v}",
    89: "{u} may decrease the absorption and serum concentration of {v}",
    90: "{u} may increase the hyperkalemic activities of {v}",
    91: "{u} may increase the analgesic activities of {v}",
    92: "{u} may decrease the therapeutic efficacy of {v}",
    93: "{u} may increase the hypertensive activities of {v}",
    94: "{u} may decrease the excretion rate  {v}",
    95: "{u} may increase the serum concentration of {v}",
    96: "{u} may increase the fluid retaining activities of {v}",
    97: "{u} may decrease the serum concentration of {v}",
    98: "{u} may decrease the sedative activities of {v}",
    99: "{u} may increase the serum concentration of the active metabolites of {v}",
    100: "{u} may increase the hyperglycemic activities of {v}",
    101: "{u} may increase the central nervous system depressant and hypertensive activities of {v}",
    102: "{u} may increase the hepatotoxic activities of {v}",
    103: "{u} may increase the thrombogenic activities of {v}",
    104: "{u} may increase the arrhythmogenic activities of {v}",
    105: "{u} may increase the hypokalemic activities of {v}",
    106: "{u} may increase the vasopressor activities of {v}",
    107: "{u} may increase the tachycardic activities of {v}",
    108: "{u} may increase the risk of hypersensitivity reaction to {v}",

}

# Reindex the dictionary from 0 to N
dd_dicts = {
    idx: value for idx, value in enumerate(drugbank_relation_id_to_name.values())
}


def extract_answer(text):
    # This pattern matches lines that contain "Answer:" with an optional "##" prefix
    pattern = r"(?:^|\n)\s*(?:##\s*)?Answer:\s*(.*)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    
    if matches:
        # Return the last matched answer (trimmed)
        return matches[-1].strip()
    else:
        return text

def extract_last_answer(text):
    # Match lines that start with ##Answer: or ## Answer:
    pattern = r'^\s*##\s*Answer:\s*(.*)$'
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    
    if not matches:
        return text
    
    return matches[-1].strip()
    

# Define custom stopwords with essential words retained
custom_stopwords = {
    "used", "a", "be", "is", "it", "may", "can", "the", "of", "and", "by",
    "when", "combined", "with", "risk", "to", "in", "drug1", "drug2", "risks"
}

# Custom Reduction Function for Derived Forms
def custom_reduce_derived_forms(word):
    reduction_map = {
        "effectiveness": "effect",
        "increased": "increase",
        "increases": "increase",
        "decreased": "decrease",
        "decreases": "decrease",
        "additive": "increase",
        "effects": "effect"
    }
    return reduction_map.get(word, word)

# Function to remove drug names from text
def remove_drug_names(text, drug1, drug2):
    if not text:
        return ""
    text = text.lower().replace(drug1.lower(), "").replace(drug2.lower(), "").strip()
    return text

# Preprocess Text
def preprocess_text(text, drug1, drug2):
    if isinstance(text, str):
        text = remove_drug_names(text, drug1, drug2)
        tokens = re.findall(r'\b\w+\b', text.lower())
        cleaned_tokens = [custom_reduce_derived_forms(word) for word in tokens if word not in custom_stopwords]
        return " ".join(cleaned_tokens)
    return ""


# Compute BERTScore F1
def compute_avg_bertscore(df, relation_dict):
    references = []
    hypotheses = []

    for _, row in df.iterrows():
        y_key = row["label_idx"]
        pred = row["preds"]
        drug1 = row.get("drug1_name", "").strip().lower()
        drug2 = row.get("drug2_name", "").strip().lower()

        if y_key in relation_dict:
            # Fill in placeholders and preprocess
            reference_template = relation_dict[y_key]
            reference_filled = reference_template.replace("{u}", drug1).replace("{v}", drug2)
            reference = preprocess_text(reference_filled, drug1, drug2)
        else:
            continue  # Skip if no matching key

        hypothesis = preprocess_text(pred, drug1, drug2)
        # print(f"Reference: {reference}")
        # print(f"Hypothesis: {hypothesis}")

        references.append(reference)
        hypotheses.append(hypothesis)



    if not references:
        return 0.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running BERTScore on device: {device}")

    _, _, f1_scores = bert_score_fn(hypotheses, references, lang="en", device=device, rescale_with_baseline=True)
    f1_scores = f1_scores.tolist()

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

# Entry point for execution
def process_and_compute_bertscore(df, relation_dict):
    avg_bertscore = compute_avg_bertscore(df, relation_dict)
    return avg_bertscore




# Setup
pred_root = "outputs" #enter your prediction root directory here

# starts_with = "google-txgemma"
must_include = "drugbank"

for file_name in os.listdir(pred_root):
    file_path = os.path.join(pred_root, file_name)

    if (
        os.path.isfile(file_path)
        # and file_name.startswith(starts_with)
        and must_include.lower() in file_name.lower()
    ):
        print(f"✅ Found matching file: {file_name}")

        try:
            # 1. Load predictions
            preds = pd.read_csv(file_path)

            # 2. Extract model outputs
            preds['preds'] = preds['prediction'].apply(extract_answer).apply(extract_last_answer)

            # 3. Create copy of ground truth and inject preds
            truth_copy = truth.copy()
            truth_copy['preds'] = preds['preds']
        

            # 4. Compute BERTScore
            avg_score = process_and_compute_bertscore(truth_copy, dd_dicts)
            print(f"⭐ Average BERTScore F1 for {file_name}: {avg_score:.4f}")

        except Exception as e:
            print(f"❌ Error processing file {file_name}: {e}")
    # else:
    #     print(f"Skipping: {file_name}")

# run_script:python llm/evaluate_llm_bertscore.py