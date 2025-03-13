from datasets import load_dataset

dataset = load_dataset("mlfoundations-dev/decontaminate_stratos_numina", split="train")
dataset = dataset.filter(lambda x: x["matched_dataset"] != "")
print(len(dataset))
dataset.select_columns(
    ["problem", "matched_dataset", "matched_text", "match_score"]
).push_to_hub("mlfoundations-dev/decontaminate_stratos_numina_filtered")

dataset = load_dataset("mlfoundations-dev/decontaminate_stratos_apps", split="train")
dataset = dataset.filter(lambda x: x["matched_dataset"] != "")
print(len(dataset))
dataset.select_columns(
    ["question", "matched_dataset", "matched_text", "match_score"]
).push_to_hub("mlfoundations-dev/decontaminate_stratos_apps_filtered")

dataset = load_dataset("mlfoundations-dev/decontaminate_stratos_taco", split="train")
dataset = dataset.filter(lambda x: x["matched_dataset"] != "")
dataset.select_columns(
    ["question", "matched_dataset", "matched_text", "match_score"]
).push_to_hub("mlfoundations-dev/decontaminate_stratos_taco_filtered")
dataset.push_to_hub("mlfoundations-dev/decontaminate_stratos_taco_filtered")
print(len(dataset))
