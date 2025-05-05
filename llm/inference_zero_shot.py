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

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# #closed KG drugbank prompt (with labels)
DRUGBANK_WITH_OPTIONS_PROMPT = (
    "You are a pharmacodynamics expert. Answer the questions using the given knowledge graph information (if available) and your medical expertise."
    "Base your answer on evidence of known interaction mechanisms, pharmacological effects, or similarities to related compounds if available"
    "Select the option number that best describes the interaction type."
    "Your answer must be concise and formatted as follows: ##Answer: <DrugX the specific effect or mechanism of interaction DrugY>.\n"
)

DRUGBANK_WITH_BULLETS_PROMPT = (
    "You are a pharmacodynamics expert. Answer the questions using the given knowledge graph information (if available) and your medical expertise."
    "Base your answer on evidence of known interaction mechanisms, pharmacological effects, or similarities to related compounds if available"
    "Select the option that best describes the interaction type."
    "Your answer must be concise and formatted as follows: ##Answer: <DrugX the specific effect or mechanism of interaction DrugY>.\n"
)

# # KG prompt drugbank
DRUGBANK_OPEN_PROMPT = (
    "You are a pharmacodynamics expert. Answer the questions using the given knowledge graph information (if available), essential parts of the drug definitions, and your medical expertise.\n"  # , drug definitions
    "Base your answer on evidence of known interaction mechanisms, pharmacological effects, or similarities to related compounds if applicable.\n"
    "Avoid generalizations unless directly supported by the knowledge graph.\n"
    "Focus exclusively on the primary interaction type between the specified drugs.Exclude speculative details or unrelated pathways unless they directly contribute to the stated interaction.\n"
    "Your answer must be concise and formatted as follows: ##Answer: <DrugX the specific effect or mechanism of interaction DrugY>.\n"
)

# # KG prompt ddinter
DDINTER_PROMPT = (
    "You are a pharmacodynamics expert. Answer the questions using the given knowledge graph information (if available), essential parts of the drug definitions, and your medical expertise.\n"  # , drug definitions
    "Base your answer on evidence of known interaction mechanisms, pharmacological effects, or similarities to related compounds if applicable.\n"
    "Avoid generalizations unless directly supported by the knowledge graph.\n"
    "Focus exclusively on the primary interaction type between the specified drugs. Exclude speculative details or unrelated pathways unless they directly contribute to the stated interaction.\n"
    "Your answer must be concise and formatted as follows: ##Answer: <the specific level of severity>.\n"
)

# #KG prompt PharmacotherapyDB
DEFAULT_INSTRUCT_TEMPLATE = (
    "You are a pharmacodynamics expert. Answer the questions using the given knowledge graph information (if available), essential parts of the drug definitions, and your medical expertise.\n"
    "Base your answer on evidence of known interaction mechanisms, pharmacological effects, or similarities to related compounds if applicable.\n"
    "Avoid generalizations unless directly supported by the knowledge graph.\n"
    "Focus exclusively on the therapeutic indication between the specified drugs and conditions. Exclude speculative details or unrelated pathways unless they directly support the indication.\n"
    "Your answer must be concise and formatted as follows: ##Answer: <Indications>.\n"
)

# prompt PharmacotherapyDB base model
DEFAULT_INSTRUCT_TEMPLATE = (
    "You are a pharmacodynamics expert. Answer the questions using your medical expertise.\n"
    "Base your answer on evidence of known interaction mechanisms, pharmacological effects, or similarities to related compounds if applicable.\n"
    "Focus exclusively on the therapeutic indication between the specified drugs and conditions. \n"
    "Your answer must be concise and formatted as follows: ##Answer: Indications>.\n"  ##Answer: <Indications>.\n" the specific effect or mechanism of interaction, the specific level of severit
)

SYSTEM_INSTRUCTION_PROMPT = {
    "drugbank_with_options": DRUGBANK_WITH_OPTIONS_PROMPT,
    "drugbank_with_bullets": DRUGBANK_WITH_BULLETS_PROMPT,
    "drugbank_open": DRUGBANK_OPEN_PROMPT,
    "ddinter_common": DDINTER_PROMPT,
}

DRUGBANK_OPTIONS = {
    0: "#Drug1 may increase the photosensitizing activities of #Drug2.",
    1: "#Drug1 may increase the anticholinergic activities of #Drug2.",
    2: "The bioavailability of #Drug2 can be decreased when combined with #Drug1.",
    3: "The metabolism of #Drug2 can be increased when combined with #Drug1.",
    4: "#Drug1 may decrease the vasoconstricting activities of #Drug2.",
    5: "#Drug1 may increase the anticoagulant activities of #Drug2.",
    6: "#Drug1 may increase the ototoxic activities of #Drug2.",
    7: "The therapeutic efficacy of #Drug2 can be increased when used in combination with #Drug1.",
    8: "#Drug1 may increase the hypoglycemic activities of #Drug2.",
    9: "#Drug1 may increase the antihypertensive activities of #Drug2.",
    10: "The serum concentration of the active metabolites of #Drug2 can be reduced when #Drug2 is used in combination with #Drug1 resulting in a loss in efficacy.",
    11: "#Drug1 may decrease the anticoagulant activities of #Drug2.",
    12: "The absorption of #Drug2 can be decreased when combined with #Drug1.",
    13: "#Drug1 may decrease the bronchodilatory activities of #Drug2.",
    14: "#Drug1 may increase the cardiotoxic activities of #Drug2.",
    15: "#Drug1 may increase the central nervous system depressant (CNS depressant) activities of #Drug2.",
    16: "#Drug1 may decrease the neuromuscular blocking activities of #Drug2.",
    17: "#Drug1 can cause an increase in the absorption of #Drug2 resulting in an increased serum concentration and potentially a worsening of adverse effects.",
    18: "#Drug1 may increase the vasoconstricting activities of #Drug2.",
    19: "#Drug1 may increase the QTc-prolonging activities of #Drug2.",
    20: "#Drug1 may increase the neuromuscular blocking activities of #Drug2.",
    21: "#Drug1 may increase the adverse neuromuscular activities of #Drug2.",
    22: "#Drug1 may increase the stimulatory activities of #Drug2.",
    23: "#Drug1 may increase the hypocalcemic activities of #Drug2.",
    24: "#Drug1 may increase the atrioventricular blocking (AV block) activities of #Drug2.",
    25: "#Drug1 may decrease the antiplatelet activities of #Drug2.",
    26: "#Drug1 may increase the neuroexcitatory activities of #Drug2.",
    27: "#Drug1 may increase the dermatologic adverse activities of #Drug2.",
    28: "#Drug1 may decrease the diuretic activities of #Drug2.",
    29: "#Drug1 may increase the orthostatic hypotensive activities of #Drug2.",
    30: "The risk or severity of hypertension can be increased when #Drug2 is combined with #Drug1.",
    31: "#Drug1 may increase the sedative activities of #Drug2.",
    32: "The risk or severity of QTc prolongation can be increased when #Drug1 is combined with #Drug2.",
    33: "#Drug1 may increase the immunosuppressive activities of #Drug2.",
    34: "#Drug1 may increase the neurotoxic activities of #Drug2.",
    35: "#Drug1 may increase the antipsychotic activities of #Drug2.",
    36: "#Drug1 may decrease the antihypertensive activities of #Drug2.",
    37: "#Drug1 may increase the vasodilatory activities of #Drug2.",
    38: "#Drug1 may increase the constipating activities of #Drug2.",
    39: "#Drug1 may increase the respiratory depressant activities of #Drug2.",
    40: "#Drug1 may increase the hypotensive and central nervous system depressant (CNS depressant) activities of #Drug2.",
    41: "The risk or severity of hyperkalemia can be increased when #Drug1 is combined with #Drug2.",
    42: "The protein binding of #Drug2 can be decreased when combined with #Drug1.",
    43: "#Drug1 may increase the central neurotoxic activities of #Drug2.",
    44: "#Drug1 may decrease effectiveness of #Drug2 as a diagnostic agent.",
    45: "#Drug1 may increase the bronchoconstrictory activities of #Drug2.",
    46: "The metabolism of #Drug2 can be decreased when combined with #Drug1.",
    47: "#Drug1 may increase the myopathic rhabdomyolysis activities of #Drug2.",
    48: "The risk or severity of adverse effects can be increased when #Drug1 is combined with #Drug2.",
    49: "The risk or severity of heart failure can be increased when #Drug2 is combined with #Drug1.",
    50: "#Drug1 may increase the hypercalcemic activities of #Drug2.",
    51: "#Drug1 may decrease the analgesic activities of #Drug2.",
    52: "#Drug1 may increase the antiplatelet activities of #Drug2.",
    53: "#Drug1 may increase the bradycardic activities of #Drug2.",
    54: "#Drug1 may increase the hyponatremic activities of #Drug2.",
    55: "The risk or severity of hypotension can be increased when #Drug1 is combined with #Drug2.",
    56: "#Drug1 may increase the nephrotoxic activities of #Drug2.",
    57: "#Drug1 may decrease the cardiotoxic activities of #Drug2.",
    58: "#Drug1 may increase the ulcerogenic activities of #Drug2.",
    59: "#Drug1 may increase the hypotensive activities of #Drug2.",
    60: "#Drug1 may decrease the stimulatory activities of #Drug2.",
    61: "The bioavailability of #Drug2 can be increased when combined with #Drug1.",
    62: "#Drug1 may increase the myelosuppressive activities of #Drug2.",
    63: "#Drug1 may increase the serotonergic activities of #Drug2.",
    64: "#Drug1 may increase the excretion rate of #Drug2 which could result in a lower serum level and potentially a reduction in efficacy.",
    65: "The risk or severity of bleeding can be increased when #Drug1 is combined with #Drug2.",
    66: "#Drug1 can cause a decrease in the absorption of #Drug2 resulting in a reduced serum concentration and potentially a decrease in efficacy.",
    67: "#Drug1 may increase the hyperkalemic activities of #Drug2.",
    68: "#Drug1 may increase the analgesic activities of #Drug2.",
    69: "The therapeutic efficacy of #Drug2 can be decreased when used in combination with #Drug1.",
    70: "#Drug1 may increase the hypertensive activities of #Drug2.",
    71: "#Drug1 may decrease the excretion rate of #Drug2 which could result in a higher serum level.",
    72: "The serum concentration of #Drug2 can be increased when it is combined with #Drug1.",
    73: "#Drug1 may increase the fluid retaining activities of #Drug2.",
    74: "The serum concentration of #Drug2 can be decreased when it is combined with #Drug1.",
    75: "#Drug1 may decrease the sedative activities of #Drug2.",
    76: "The serum concentration of the active metabolites of #Drug2 can be increased when #Drug2 is used in combination with #Drug1.",
    77: "#Drug1 may increase the hyperglycemic activities of #Drug2.",
    78: "#Drug1 may increase the central nervous system depressant (CNS depressant) and hypertensive activities of #Drug2.",
    79: "#Drug1 may increase the hepatotoxic activities of #Drug2.",
    80: "#Drug1 may increase the thrombogenic activities of #Drug2.",
    81: "#Drug1 may increase the arrhythmogenic activities of #Drug2.",
    82: "#Drug1 may increase the hypokalemic activities of #Drug2.",
    83: "#Drug1 may increase the vasopressor activities of #Drug2.",
    84: "#Drug1 may increase the tachycardic activities of #Drug2.",
    85: "The risk of a hypersensitivity reaction to #Drug2 is increased when it is combined with #Drug1.",
}


TEXTDDI_OPTIONS = {
    "Major": "The interactions are life-threatening and/or require medical treatment or intervention to minimize or prevent severe adverse effects.",
    "Moderate": "The interactions may result in exacerbation of the disease of the patient and/or change in therapy.",
    "Minor": "The interactions would limit the clinical effects. The manifestations may include an increase in frequency or severity of adverse effects, but usually they do not require changes in therapy.",
}


KG_DATASET_PATH = {
    "drugbank": {
        "train": "files/dataset_with_paths/drugbank_train_add_reverse.json",
        "test": "files/dataset_with_paths/drugbank_test_add_reverse.json",
    },
    "ddinter": {
        "train": "files/dataset_with_paths/ddinter_train_add_reverse.json",
        "test": "files/dataset_with_paths/ddinter_test_add_reverse.json",
    },
}


## Function to replace placeholders with actual drug names in the DDI descriptions
def replace_drugs_in_ddi(ddi_str, drug1_name, drug2_name):
    return ddi_str.replace("#Drug1", drug1_name).replace("#Drug2", drug2_name)


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

        processed_input = tokenizer.apply_chat_template(
            user_content + [{"role": "user", "content": test_prompt}],
            tokenize=False,
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

        if args.few_shot:
            pass

        if args.use_options:
            option_str = "\n".join(
                [f"{key}: {option}" for key, option in TEXTDDI_OPTIONS.items()]
            )

        # user_content = system_prompt + "Question: " + (example["question"]) + "\n"
        test_prompt = "Question: " + (example["question"]) + "\n"

        if args.use_drug_descriptions:
            user_content += (
                f"Description of the drugs:\n"
                f"{example['drug1_name']}: {example['drug1_desc']}\n"  # Drug 1 name and description
                f"{example['drug2_name']}: {example['drug2_desc']}\n"  # Drug 2 name and description
            )

        if args.use_kg:
            # TODO: change paths to filtered_paths
            user_content += (
                "Knowledge Graph Information: \n" + (example["path_str"]) + "\n"
            )

        processed_input = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
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
        max_model_len=2**17,
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
        choices=["drugbank", "ddinter"],
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
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
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