import argparse
import json
import os

import matplotlib.pyplot as plt
import tiktoken
from datasets import load_dataset


def analyze_token_lengths(dataset_name):
    # Get the directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dev_dir = os.path.join(script_dir, "dev")
    lengths_file = os.path.join(
        dev_dir, f"{dataset_name.replace('/', '_')}_token_lengths.json"
    )

    # Try to load existing token lengths
    if os.path.exists(lengths_file):
        print("Loading existing token lengths")
        with open(lengths_file, "r") as f:
            lengths_data = json.load(f)
            solution_token_lengths = lengths_data["solution_lengths"]
            reasoning_token_lengths = lengths_data["reasoning_lengths"]
    else:
        dataset = load_dataset(dataset_name)["train"]
        deepseek_solutions = dataset["deepseek_solution"]
        deepseek_reasoning = dataset["reasoning"]
        # Calculate token lengths if file doesn't exist
        print("Calculating token lengths")
        solution_token_lengths = [
            len(tiktoken.encoding_for_model("gpt-4o-mini").encode(solution))
            for solution in deepseek_solutions
        ]
        reasoning_token_lengths = [
            len(tiktoken.encoding_for_model("gpt-4o-mini").encode(reasoning))
            for reasoning in deepseek_reasoning
        ]

        # Save the lengths
        os.makedirs(dev_dir, exist_ok=True)
        with open(lengths_file, "w") as f:
            json.dump(
                {
                    "solution_lengths": solution_token_lengths,
                    "reasoning_lengths": reasoning_token_lengths,
                },
                f,
            )

    print("Calculating statistics")
    print(
        f"Mean solution token length: {sum(solution_token_lengths) / len(solution_token_lengths)}"
    )
    print(
        f"Median solution token length: {sorted(solution_token_lengths)[len(solution_token_lengths) // 2]}"
    )
    print(f"Max solution token length: {max(solution_token_lengths)}")
    print(f"Min solution token length: {min(solution_token_lengths)}")

    print(
        f"Mean reasoning token length: {sum(reasoning_token_lengths) / len(reasoning_token_lengths)}"
    )
    print(
        f"Median reasoning token length: {sorted(reasoning_token_lengths)[len(reasoning_token_lengths) // 2]}"
    )
    print(f"Max reasoning token length: {max(reasoning_token_lengths)}")
    print(f"Min reasoning token length: {min(reasoning_token_lengths)}")

    long_solution_indices = [
        i for i, length in enumerate(solution_token_lengths) if length > 3000
    ]
    print(f"\nNumber of solutions over 3k tokens: {len(long_solution_indices)}")

    max_token_length = max(solution_token_lengths)
    max_length_indices = [
        i
        for i, length in enumerate(solution_token_lengths)
        if length == max_token_length
    ]
    print(
        f"\nNumber of solutions with max token length ({max_token_length}): {len(max_length_indices)}"
    )

    # Get the directory of the current script
    dev_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dev")

    # Create histogram of token lengths
    plt.figure(figsize=(10, 6))
    plt.hist(solution_token_lengths, bins=50, edgecolor="black")
    plt.title("Distribution of Solution Token Lengths")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    # Save plot
    plot_prefix = dataset_name.replace("/", "_")
    plt.savefig(
        os.path.join(dev_dir, f"{plot_prefix}_solution_length_distribution.png")
    )
    plt.close()

    # Create histogram of token lengths
    combined_token_lengths = []
    for solution_tokens, reasoning_tokens in zip(
        solution_token_lengths, reasoning_token_lengths
    ):
        combined_token_lengths.append(solution_tokens + reasoning_tokens)

    # Count number of examples exceeding 16384 tokens
    num_over_16k = sum(1 for length in combined_token_lengths if length > 16384)
    print(f"\nNumber of examples with combined token length over 16384: {num_over_16k}")
    print(
        f"Percentage of dataset over 16384 tokens: {(num_over_16k / len(combined_token_lengths)) * 100:.2f}%"
    )

    num_over_32k = sum(1 for length in combined_token_lengths if length > 32768)
    print(f"\nNumber of examples with combined token length over 32768: {num_over_32k}")
    print(
        f"Percentage of dataset over 32768 tokens: {(num_over_32k / len(combined_token_lengths)) * 100:.2f}%"
    )

    print(
        f"Mean combined token length: {sum(combined_token_lengths) / len(combined_token_lengths)}"
    )
    print(
        f"Median combined token length: {sorted(combined_token_lengths)[len(combined_token_lengths) // 2]}"
    )
    print(f"Max combined token length: {max(combined_token_lengths)}")
    print(f"Min combined token length: {min(combined_token_lengths)}")

    plt.figure(figsize=(10, 6))
    plt.hist(combined_token_lengths, bins=50, edgecolor="black")
    plt.title("Distribution of Combined Token Lengths")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    # Save plot
    plt.savefig(
        os.path.join(dev_dir, f"{plot_prefix}_combined_length_distribution.png")
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(reasoning_token_lengths, bins=50, edgecolor="black")
    plt.title("Distribution of Reasoning Token Lengths")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    # Save plot
    plt.savefig(
        os.path.join(dev_dir, f"{plot_prefix}_reasoning_length_distribution.png")
    )
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze token lengths in a dataset")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to analyze")
    args = parser.parse_args()

    analyze_token_lengths(args.dataset_name)


if __name__ == "__main__":
    main()
