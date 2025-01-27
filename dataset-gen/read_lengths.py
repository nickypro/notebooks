import json
import re
from collections import Counter

def analyze_paragraph_distribution(filename: str, verbose:bool = False):
    paragraph_counts = []
    average_lengths = []
    max_paragraph_count = 0  # Variable to store the maximum number of paragraphs
    prompt_with_max_paragraphs = ""  # Variable to store the prompt with the most paragraphs
    completion_with_max_paragraphs = ""  # Variable to store the completion with the most paragraphs
    total_paragraphs = 0  # Variable to store the total number of paragraphs

    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            prompt = data.get('prompt', '')  # Get the prompt
            completion = data.get('completion', '')
            # Replace multiple consecutive double newlines with a single double newline
            cleaned_completion = re.sub(r'\n{2,}', '\n\n', completion)
            paragraphs = cleaned_completion.split('\n\n')
            num_paragraphs = len(paragraphs)
            paragraph_counts.append(num_paragraphs)

            # Calculate average length of paragraphs
            avg_length = sum(len(p) for p in paragraphs) / num_paragraphs if num_paragraphs > 0 else 0
            average_lengths.append(avg_length)

            # Update total paragraphs count
            total_paragraphs += num_paragraphs

            # Check for the maximum number of paragraphs
            if num_paragraphs > max_paragraph_count:
                max_paragraph_count = num_paragraphs
                prompt_with_max_paragraphs = prompt
                completion_with_max_paragraphs = completion

    # Print the prompt and completion with the maximum number of paragraphs
    print("\nPrompt with the Most Paragraphs:")
    print(prompt_with_max_paragraphs)
    print("\nCompletion with the Most Paragraphs:")
    print(completion_with_max_paragraphs)

    # Print the distribution of the number of paragraphs
    paragraph_distribution = Counter(paragraph_counts)
    total_entries = len(paragraph_counts)
    print("\nDistribution of Number of Paragraphs:")
    for num_paragraphs, frequency in sorted(paragraph_distribution.items()):
        print(f"{num_paragraphs} paragraphs: {frequency} entries ({frequency / total_entries * 100:.1f}%)")

    # Calculate and print the distribution of % paragraphs within the bucket
    print("\nPercentage of Total Paragraphs within Each Bucket:")
    for num_paragraphs, frequency in sorted(paragraph_distribution.items()):
        percentage = (num_paragraphs * frequency / total_paragraphs) * 100
        print(f"{num_paragraphs} paragraphs: {percentage:.1f}% of total paragraphs")

    # ASCII Plot for Distribution of Number of Paragraphs
    print("\nASCII Plot of Distribution of Number of Paragraphs:")
    for num_paragraphs, frequency in sorted(paragraph_distribution.items()):
        bar_length = int(frequency / total_entries * 50)  # Scale to 50 characters
        print(f"{num_paragraphs:3d} paragraphs: {'#' * bar_length} ({frequency})")

    # ASCII Plot for Percentage of Total Paragraphs within Each Bucket
    print("\nASCII Plot of Percentage of Total Paragraphs within Each Bucket:")
    for num_paragraphs, frequency in sorted(paragraph_distribution.items()):
        percentage = (num_paragraphs * frequency / total_paragraphs) * 100
        bar_length = int(percentage*3)  # Scale to 50 characters (100% = 50)
        print(f"{num_paragraphs:3d} paragraphs: {'#' * bar_length} ({percentage:.1f}%)")
    print(f"Total Paragraphs: {total_paragraphs}")

if __name__ == "__main__":
    FILE_NAME = "rp_outputs_3b_002.jsonl"
    analyze_paragraph_distribution(FILE_NAME, verbose=False)
