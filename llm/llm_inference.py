import argparse
import logging
import json
import os
import re

from typing import Union
import pandas as pd
from datasets import Dataset
from vllm import LLM, SamplingParams
from transformers import set_seed
from huggingface_hub import login

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# token = os.getenv("HUGGINGFACE_HUB_TOKEN")
# login(token)

# #closed KG drugbank prompt (with labels)
DRUGBANK_WITH_OPTIONS_PROMPT = (
    "You are a pharmacodynamics expert. Answer the questions using the given knowledge graph information (if available) and your medical expertise."
    "Base your answer on evidence of known interaction mechanisms, pharmacological effects, or similarities to related compounds if available"
    "Select the option number that best describes the interaction type."
    "Your answer must be concise and formatted as follows: ##Answer: <DrugX the specific effect or mechanism of interaction DrugY>.\n"
)

#open drugbank KG prompt drugbank
DRUGBANK_OPEN_PROMPT = (
    "You are a pharmacodynamics expert. Answer the questions using the given knowledge graph information (if available), essential parts of the drug definitions, and your medical expertise.\n"  
    "Base your answer on evidence of known interaction mechanisms, pharmacological effects, or similarities to related compounds if applicable.\n"
    "Avoid generalizations unless directly supported by the knowledge graph.\n"
    "Focus exclusively on the primary interaction type between the specified drugs.Exclude speculative details or unrelated pathways unless they directly contribute to the stated interaction.\n"
    "Your answer must be concise and formatted as follows: ##Answer: <DrugX the specific effect or mechanism of interaction DrugY>.\n"
)

## KG prompt ddinter
DDINTER_PROMPT = (
    "You are a pharmacodynamics expert. Answer the questions using the given knowledge graph information (if available), essential parts of the drug definitions, and your medical expertise.\n" 
    "Base your answer on evidence of known interaction mechanisms, pharmacological effects, or similarities to related compounds if applicable.\n"
    "Avoid generalizations unless directly supported by the knowledge graph.\n"
    "Focus exclusively on the primary interaction type between the specified drugs. Exclude speculative details or unrelated pathways unless they directly contribute to the stated interaction.\n"
    "Your answer must be concise and formatted as follows: ##Answer: <the specific level of severity>.\n"
)


# KG-enhanced prompt for PharmacotherapyDB (used when KG is included)
PHARMACOTHERAPYDB_KG_PROMPT = (
    "You are a pharmacodynamics expert. Answer the questions using the given knowledge graph information (if available), essential parts of the drug definitions, and your medical expertise.\n"
    "Base your answer on evidence of known interaction mechanisms, pharmacological effects, or similarities to related compounds if applicable.\n"
    "Avoid generalizations unless directly supported by the knowledge graph.\n"
    "Focus exclusively on the therapeutic indication between the specified drugs and conditions. Exclude speculative details or unrelated pathways unless they directly support the indication.\n"
    "Your answer must be concise and formatted exactly as follows: ##Answer: <Indications>.\n"
)

# Base-model-only prompt (used when KG is not included)
PHARMACOTHERAPYDB_BASE_PROMPT = (
    "You are a pharmacodynamics expert. Answer the questions using your medical expertise.\n"
    "Base your answer on evidence of known interaction mechanisms, pharmacological effects, or similarities to related compounds if applicable.\n"
    "Focus exclusively on the therapeutic indication between the specified drugs and conditions.\n"
    "Your answer must be concise and formatted exactly as follows: ##Answer: <Indications>.\n"
)


SYSTEM_INSTRUCTION_PROMPT = {
    "drugbank_with_options": DRUGBANK_WITH_OPTIONS_PROMPT,
    "drugbank_with_bullets": DRUGBANK_WITH_OPTIONS_PROMPT,
    "drugbank_open": DRUGBANK_OPEN_PROMPT,
    "ddinter_common": DDINTER_PROMPT,
    "pharmacotherapydb": lambda use_kg: PHARMACOTHERAPYDB_KG_PROMPT if use_kg else PHARMACOTHERAPYDB_BASE_PROMPT,
}



DRUGBANK_OPTIONS = {
    0: '{u} may increase the photosensitizing activities of {v}',
    1: '{u} may increase the anticholinergic activities of {v}',
    2: '{u} can decrease the bioavailability of {v}',
    3: '{u} can increase the metabolism of {v}',
    4: '{u} may decrease the vasoconstricting activities of {v}',
    5: '{u} may increase the anticoagulant activities of {v}',
    6: '{u} may increase the ototoxic activities of {v}',
    7: '{u} can increase the therapeutic efficacy of {v}',
    8: '{u} may increase the hypoglycemic activities of {v}',
    9: '{u} may increase the antihypertensive activities of {v}',
    10: '{u} may reduce the serum concentration of the active metabolites of {v}',
    11: '{u} may decrease the anticoagulant activities of {v}',
    12: '{u} may decrease the absorption of {v}',
    13: '{u} may decrease the bronchodilatory activities of {v}',
    14: '{u} may increase the cardiotoxic activities of {v}',
    15: '{u} may increase the central nervous system depressant activities of {v}',
    16: '{u} may decrease the neuromuscular blocking activities of {v}',
    17: '{u} can increase the absorption and serum concentration of {v}',
    18: '{u} may increase the vasoconstricting activities of {v}',
    19: '{u} may increase the QTc prolonging activities of {v}',
    20: '{u} may increase the neuromuscular blocking activities of {v}',
    21: '{u} may increase the adverse neuromuscular activities of {v}',
    22: '{u} may increase the stimulatory activities of {v}',
    23: '{u} may increase the hypocalcemic activities of {v}',
    24: '{u} may increase the atrioventricular blocking activities of {v}',
    25: '{u} may decrease the antiplatelet activities of {v}',
    26: '{u} may increase the neuroexcitatory activities of {v}',
    27: '{u} may increase the dermatologic adverse activities of {v}',
    28: '{u} may decrease the diuretic activities of {v}',
    29: '{u} may increase the orthostatic hypotensive activities of {v}',
    30: '{u} may increase the hypertensive effects of {v}',
    31: '{u} may increase the sedative activities of {v}',
    32: '{u} may increase the severity of QTc prolonging effects when combined with {v}',
    33: '{u} may increase the immunosuppressive activities of {v}',
    34: '{u} may increase the neurotoxic activities of {v}',
    35: '{u} may increase the antipsychotic activities of {v}',
    36: '{u} may decrease the antihypertensive activities of {v}',
    37: '{u} may increase the vasodilatory activities of {v}',
    38: '{u} may increase the constipating activities of {v}',
    39: '{u} may increase the respiratory depressant activities of {v}',
    40: '{u} may increase the hypotensive and central nervous system depressant activities of {v}',
    41: '{u} may increase the severity of hyperkalemic effects when combined with {v}',
    42: '{u} may decrease the protein binding of {v}',
    43: '{u} may increase the central neurotoxic activities of {v}',
    44: '{u} may decrease the diagnostic effectiveness of {v} ',
    45: '{u} may increase the bronchoconstrictory activities of {v}',
    46: '{u} may decrease the metabolism of {v}',
    47: '{u} may increase the myopathic rhabdomyolysis activities of {v}',
    48: '{u} may increase the severity of adverse effects when combined with {v}',
    49: '{u} may increase heart failure risks when combined with {v}',
    50: '{u} may increase the hypercalcemic activities of {v}',
    51: '{u} may decrease the analgesic activities of {v}',
    52: '{u} may increase the antiplatelet activities of {v}',
    53: '{u} may increase the bradycardic activities of {v}',
    54: '{u} may increase the hyponatremic activities of {v}',
    55: '{u} may increase the hypotensive effects when combined with {v}',
    56: '{u} may increase the nephrotoxic activities of {v}',
    57: '{u} may decrease the cardiotoxic activities of {v}',
    58: '{u} may increase the ulcerogenic activities of {v}',
    59: '{u} may increase the hypotensive activities of {v}',
    60: '{u} may decrease the stimulatory activities of {v}',
    61: '{u} may increase the bioavailability of {v}',
    62: '{u} may increase the myelosuppressive activities of {v}',
    63: '{u} may increase the serotonergic activities of {v}',
    64: '{u} may increase the excretion rate {v}',
    65: '{u} may increase bleeding risks when combined with {v}',
    66: '{u} may decrease the absorption and serum concentration of {v}',
    67: '{u} may increase the hyperkalemic activities of {v}',
    68: '{u} may increase the analgesic activities of {v}',
    69: '{u} may decrease the therapeutic efficacy of {v}',
    70: '{u} may increase the hypertensive activities of {v}',
    71: '{u} may decrease the excretion rate  {v}',
    72: '{u} may increase the serum concentration of {v}',
    73: '{u} may increase the fluid retaining activities of {v}',
    74: '{u} may decrease the serum concentration of {v}',
    75: '{u} may decrease the sedative activities of {v}',
    76: '{u} may increase the serum concentration of the active metabolites of {v}',
    77: '{u} may increase the hyperglycemic activities of {v}',
    78: '{u} may increase the central nervous system depressant and hypertensive activities of {v}',
    79: '{u} may increase the hepatotoxic activities of {v}',
    80: '{u} may increase the thrombogenic activities of {v}',
    81: '{u} may increase the arrhythmogenic activities of {v}',
    82: '{u} may increase the hypokalemic activities of {v}',
    83: '{u} may increase the vasopressor activities of {v}',
    84: '{u} may increase the tachycardic activities of {v}',
    85: '{u} may increase the risk of hypersensitivity reaction to {v}'
}


DDINTER_OPTIONS = {
    "Major": "The interactions are life-threatening and/or require medical treatment or intervention to minimize or prevent severe adverse effects.",
    "Moderate": "The interactions may result in exacerbation of the disease of the patient and/or change in therapy.",
    "Minor": "The interactions would limit the clinical effects. The manifestations may include an increase in frequency or severity of adverse effects, but usually they do not require changes in therapy.",
}

PHARMACOTHERAPYDB_OPTIONS = {
    "Disease-modifying": "The drug therapeutically changes the underlying or downstream biology of the disease.",
    "Palliates": "The drug only alleviates symptoms without altering disease progression.",
    "Non-indication": "The drug neither therapeutically changes the disease nor treats its symptoms.",
}

KG_DATASET_PATH = {
    "drugbank": {
        "train": "data/paths/drugbank_train_add_reverse.json",
        "test": "data/paths/drugbank_test_add_reverse.json",
    },
    "ddinter": {
        "train": "data/paths/ddinter_train_add_reverse.json",
        "test": "data/paths/ddinter_test_add_reverse.json",
    },
    "pharmacotherapydb": {
        "train": "data/paths/pharmaDB_train_add_reverse.json",
        "test": "data/paths/pharmaDB_test_add_reverse.json",
    },
}


## Function to replace placeholders with actual drug names in the DDI descriptions
def replace_drugs_in_ddi(ddi_str):
    return ddi_str.replace("{u}", "#Drug1").replace("{v}", "#Drug2")


def load_dataset(
    dataset_path: str,
    start_index: Union[None, int] = 0,
    end_index: Union[None, int] = None,
    debug: bool = False,
) -> Dataset:
    dataset = Dataset.from_json(dataset_path)

    if start_index is not None and end_index is not None:
        dataset = dataset.select(range(start_index, end_index))

    if debug:
        dataset = dataset.select(range(250))

    return dataset


def drugbank_chat_template(dataset, tokenizer, args):
    if args.few_shot:
        # load the dataset
        train_dataset = load_dataset(KG_DATASET_PATH[args.dataset_name]["train"])
        # shuffle dataset
        train_dataset = train_dataset.shuffle(seed=args.seed)
        # dictionary with key as the label_idx and the value as the list of examples
        label2examples = {}
        from tqdm import tqdm

        logger.info("Creating few shot dataset")
        for example in tqdm(train_dataset.to_list()):
            label_idx = example["label_idx"]
            if label_idx not in label2examples:
                label2examples[label_idx] = []
            label2examples[label_idx].append(example)

        few_shot_data = []
        for label_idx, examples in label2examples.items():
            # take the first N = args.num_shots examples (or fewer if not enough available)
            few_shot_data.extend(examples[: args.num_shots])

        few_shot_dataset = Dataset.from_list(few_shot_data)
        # shuffle
        few_shot_dataset = few_shot_dataset.shuffle(seed=args.seed)

        # convert to str
        def process_few_shot(example):
            user_str = (
                f"Determine the interaction type between "
                f"{example['drug1_name']} (Drug1) and {example['drug2_name']} (Drug2).\n"
            )
            if args.use_drug_descriptions:
                user_str += (
                    f"Description of the drugs:\n"
                    f"{example['drug1_name']}: {example['drug1_desc']}\n"
                    f"{example['drug2_name']}: {example['drug2_desc']}\n"
                )
            if args.use_kg:
                user_str += f"Knowledge Graph Information:\n{example['path_str']}\n"

            label = example["label"]
            label = f'##Answer: {example["label"]}'
            return {"user": user_str, "assistant": label}

        few_shot_dataset = few_shot_dataset.map(
            process_few_shot, num_proc=1, remove_columns=dataset.column_names
        )
        few_shot_conv = []
        for example in few_shot_dataset.to_list():
            few_shot_conv.append({"role": "user", "content": example["user"]})
            few_shot_conv.append({"role": "assistant", "content": example["assistant"]})
    else:
        few_shot_conv = []

    def apply_template(example):
        if args.use_options:
            question = f"Question: Which of the following describes the interaction type between {example['drug1_name']} (Drug1) and {example['drug2_name']} (Drug2)?"
        else:
            question = f"Determine the interaction type between {example['drug1_name']} (Drug1) and {example['drug2_name']} (Drug2)."

        if args.use_options and args.option_style == "numbered":
            system_prompt = SYSTEM_INSTRUCTION_PROMPT["drugbank_with_options"]
        elif args.use_options and args.option_style == "bulleted":
            system_prompt = SYSTEM_INSTRUCTION_PROMPT["drugbank_with_bullets"]
        else:
            system_prompt = SYSTEM_INSTRUCTION_PROMPT["drugbank_open"]

        # add options to system prompt
        if args.use_options:
            option_str = "\n".join(
                [f"- {option}" for key, option in DRUGBANK_OPTIONS.items()]
            )
            system_prompt += "Options:\n"
            system_prompt += f"{option_str}\n"

        user_content = [{"role": "system", "content": system_prompt}]

        if args.few_shot:
            user_content += few_shot_conv

        test_prompt = question + "\n"

        if args.use_drug_descriptions:
            test_prompt += (
                f"Description of the drugs:\n"
                f"{example['drug1_name']}: {example['drug1_desc']}\n"  # Drug 1 name and description
                f"{example['drug2_name']}: {example['drug2_desc']}\n"  # Drug 2 name and description
            )

        if args.use_kg:
            # TODO: change paths to filtered_paths
            test_prompt += (
                "Knowledge Graph Information:\n" + (example["path_str"]) + "\n"
            )

    #change add generation prompt to To True for some models
        processed_input = tokenizer.apply_chat_template(
            user_content + [{"role": "user", "content": test_prompt}],
            tokenize=False, add_generation_prompt=False
        )

        return {"input": processed_input}

    dataset = dataset.map(
        apply_template, num_proc=8, remove_columns=dataset.column_names
    )
    return dataset

def ddinter_chat_template(dataset, tokenizer, args):
    def apply_template(example):
        question = f"Determine the severity of interaction when {example['drug1_name']} (Drug 1) and {example['drug2_name']} (Drug 2) are taken together."
        example["question"] = question

        system_prompt = SYSTEM_INSTRUCTION_PROMPT["ddinter_common"]
        test_prompt = "Question: " + example["question"] + "\n"

        user_content = system_prompt + test_prompt

        if args.use_drug_descriptions:
            user_content += (
                f"Description of the drugs:\n"
                f"{example['drug1_name']}: {example['drug1_desc']}\n"
                f"{example['drug2_name']}: {example['drug2_desc']}\n"
            )

        if args.use_kg:
            user_content += (
                "Knowledge Graph Information:\n" + example["path_str"] + "\n"
            )

        if args.use_options:
            option_str = "\n".join(
                [f"{key}: {option}" for key, option in DDINTER_OPTIONS.items()]
            )
            user_content += "Options:\n" + option_str + "\n"

        #change add generation prompt to To True for some models
        processed_input = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=False
        )

        return {"input": processed_input}

    dataset = dataset.map(
        apply_template, num_proc=8, remove_columns=dataset.column_names
    )
    return dataset



def pharmacotherapydb_chat_template(dataset, tokenizer, args):
    def apply_template(example):
        question = f"What is the therapeutic indication of {example['drug_name']} for {example['disease_name']}?"
        system_prompt = SYSTEM_INSTRUCTION_PROMPT["pharmacotherapydb"](args.use_kg)

        user_prompt = system_prompt + "\nQuestion: " + question + "\n"

        if args.use_drug_descriptions:
            user_prompt += (
                    f"Description of the drug and disease:\n"
                    f"{example['drug_name']}: {example['drug_desc']}\n"
                    f"{example['disease_name']}: {example['disease_desc']}\n"
                )

        if args.use_kg:
            user_prompt += "Knowledge Graph Information:\n" + example["path_str"] + "\n"

        if args.use_options:
            option_str = "\n".join(
                [f"- {opt}: {desc}" for opt, desc in PHARMACOTHERAPYDB_OPTIONS.items()]
            )
            user_prompt += "Options:\n" + option_str + "\n"
            
    #change add generation prompt to To True for some models
        processed_input = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False, add_generation_prompt=False
        )
        return {"input": processed_input}

    dataset = dataset.map(
        apply_template, num_proc=8, remove_columns=dataset.column_names
    )
    return dataset

def apply_chat_template(dataset, tokenizer, args):
    if args.dataset_name == "drugbank":
        dataset = drugbank_chat_template(dataset, tokenizer, args)
    elif args.dataset_name == "ddinter":
        dataset = ddinter_chat_template(dataset, tokenizer, args)
    elif args.dataset_name == "pharmacotherapydb":
        dataset = pharmacotherapydb_chat_template(dataset, tokenizer, args)
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported")

    return dataset


def create_prediction_dir(args):
    # use args such as use_drug_descriptions, use_kg, use_options, option_style to create the name
    prediction_dir = "predictions_"
    modified_model_name = args.model_name_or_path.replace("/", "_")
    prediction_dir += f"{args.dataset_name}"
    prediction_dir += f"_{modified_model_name}"
    if args.use_drug_descriptions:
        prediction_dir += "_desc"

    if args.use_kg:
        prediction_dir += "_kg"

    if args.few_shot:
        prediction_dir += f"_few_shot_{args.num_shots}"

    if args.use_options:
        prediction_dir += "_options"

        if args.option_style == "bulleted":
            prediction_dir += "_bullets"

        if args.option_style == "numbered":
            prediction_dir += "_numbered"

    if args.debug:
        prediction_dir += "_debug"

    prediction_dir += f"_seed_{args.seed}"

    return prediction_dir


def main(args):
    set_seed(args.seed)

    # load model
    logger.info(f"Loading model {args.model_name_or_path}")
    import torch

    num_gpus = torch.cuda.device_count()
    logger.info(f"Number of GPUs: {num_gpus}")

    logger.info(f"Loading model: {args.model_name_or_path}")

    model = LLM(
        args.model_name_or_path,
        tensor_parallel_size=num_gpus,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        enable_prefix_caching=True,
    )

    # load dataset
    if args.dataset_path is None:
        args.dataset_path = KG_DATASET_PATH[args.dataset_name]["test"]
    logger.info(f"Loading dataset from {args.dataset_path}")
    dataset = load_dataset(
        args.dataset_path, args.start_index, args.end_index, args.debug
    )

    logger.info("converting dataset to test examples")
    tokenizer = model.get_tokenizer()
    dataset = apply_chat_template(dataset, tokenizer, args)

    logger.info("example of dataset")
    logger.info(dataset[0]["input"])

    # prepare prompts
    prompts = dataset["input"]

    # run inference with greedy decoding
    logger.info("Running inference")
    sampling_params = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens
    )
    model_outputs = model.generate(prompts, sampling_params)

    # save outputs
    predictions = [output.outputs[0].text for output in model_outputs]

    logger.info(f"Saving predictions to {args.output_dir}")
    predictions_df = pd.DataFrame({"prediction": predictions})

    prediction_dir = create_prediction_dir(args)
    output_dir = os.path.join(args.output_dir, prediction_dir)

    os.makedirs(output_dir, exist_ok=True)

    if args.start_index is not None and args.end_index is not None:
        output_path = os.path.join(
            output_dir, f"predictions_{args.start_index}_{args.end_index}.csv"
        )
    else:
        output_path = os.path.join(output_dir, "predictions.csv")

    logger.info(f"Saving predictions to {output_path}")
    predictions_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="drugbank",
        choices=["drugbank", "ddinter", "pharmacotherapydb"],
        help="Name of the dataset",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the dataset json file in case the file changes.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model name on HuggingFace or path",
    )
    parser.add_argument("--output_dir", type=str, help="Path to the output directory",default ="outputs")
    parser.add_argument("--debug", action="store_true")

    # smapling params
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--max_tokens", type=int, default=100)

    # experiment parameters
    parser.add_argument("--few_shot", action="store_true")  # TODO: implement few shot
    parser.add_argument(
        "--num_shots", type=int, default=1, help="number of shots per example"
    )  #
    parser.add_argument("--use_drug_descriptions", action="store_true")
    parser.add_argument("--use_kg", action="store_true")
    parser.add_argument("--use_options", action="store_true")
    parser.add_argument(
        "--option_style", type=str, default="bulleted", choices=["numbered", "bulleted"]
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)

    args = parser.parse_args()

    main(args)
    
    #run script:
    # python3 llm/llm_inference.py \
    # --dataset_path data/paths/drugbank_test_add_reverse.json \
    # --dataset_name drugbank \
    # --output_dir outputs/drugbank \
    # --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    # --use_kg \
    # --debug