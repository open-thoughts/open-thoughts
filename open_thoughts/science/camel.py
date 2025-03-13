import json
import os
import subprocess
import zipfile

import pandas as pd
from datasets import Dataset


def download_to_directory(directory, domain):
    url = f"https://huggingface.co/datasets/camel-ai/{domain}/resolve/main/{domain}.zip?download=true"

    zip_filename = url.split("/")[-1].split("?")[0]
    subject_folder = os.path.join(directory, zip_filename.split(".")[0])

    if not os.path.exists(subject_folder):
        os.makedirs(subject_folder, exist_ok=True)

        subprocess.run(["wget", "-P", ".", url, "-O", zip_filename])

        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(subject_folder)

        os.remove(zip_filename)
    else:
        print(f"Folder {subject_folder} already exists. Skipping download and extraction.")


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def process_file(file_name, directory):
    file_path = os.path.join(directory, file_name)
    return load_json(file_path)


def load_jsons_sequential(directory, subject):
    print(f"Loading CAMEL-AI {subject} dataset")
    data = []
    subject_dir = os.path.join(directory, subject)
    json_files = [f for f in os.listdir(subject_dir) if f.endswith(".json")]

    for file_name in json_files:
        file_path = os.path.join(subject_dir, file_name)
        with open(file_path, "r") as f:
            data.append(json.load(f))

    dataset = Dataset.from_list(data)

    return dataset


def subsample(dataset, num_samples_per_subtopic):
    df = dataset.to_pandas()

    sampled_dfs = []
    for subtopic in df["sub_topic"].unique():
        subtopic_sample = df[df["sub_topic"] == subtopic].sample(n=num_samples_per_subtopic, random_state=42)
        sampled_dfs.append(subtopic_sample)
    result_df = pd.concat(sampled_dfs)

    return Dataset.from_pandas(result_df)


def load(subject):
    directory = os.path.join(os.path.expanduser("~"), "Downloads")
    download_to_directory(directory, subject)
    return load_jsons_sequential(directory, subject)


""" REUPLOAD CAMEL-AI DATASETS

The original CAMEL-AI datasets are very slow to load, so this script reformats and reuploads them to the Hub.
It downloads the zip files from the CAMEL-AI website and extracts them into the specified directory.
It then loads the JSON files from the extracted directory and pushes them to the Hugging Face Hub.
"""
if __name__ == "__main__":
    directory = os.path.join(os.path.expanduser("~"), "Downloads")
    subjects = ["biology", "chemistry", "math", "physics"]

    download_to_directory(directory)

    for subject in subjects:
        dataset = load_jsons_sequential(directory, subject)
        print(f"{subject.capitalize()} dataset size: {len(dataset)}")
        print(f"{subject.capitalize()} dataset features: {dataset.features}")
        dataset.push_to_hub(f"mlfoundations-dev/camel-ai-{subject}")
        print()
