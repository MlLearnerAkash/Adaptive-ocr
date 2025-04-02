import sys
import re
from collections import Counter
import matplotlib.pyplot as plt

def read_words(file_path):
    """Read the text file and return a list of words (lowercase)."""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    # Use regex to find words, ignoring punctuation and case.
    words = re.findall(r'\w+', text.lower())
    return words

def plot_word_frequencies(word_counts, top_n=20):
    """Plot a bar graph of the top_n word frequencies with summary stats in the top right corner."""
    # Get the top_n words and their counts
    most_common = word_counts.most_common(top_n)
    words, counts = zip(*most_common)
    
    plt.figure(figsize=(12, 6))
    plt.bar(words, counts, color='skyblue')
    plt.xlabel('Unique Words')
    plt.ylabel('Frequency')
    plt.title(f'Top {top_n} Unique Word Frequencies')
    plt.xticks(rotation=45, ha='right')
    
    # Compute summary statistics over the entire word_counts
    total_unique = len(word_counts)
    total_count = sum(word_counts.values())
    average_frequency = total_count / total_unique if total_unique else 0
    freq_range = max(word_counts.values()) - min(word_counts.values()) if total_unique else 0

    # Prepare summary string
    summary_text = (f"Total Unique Words: {total_unique}\n"
                    f"Average Frequency: {average_frequency:.2f}\n"
                    f"Frequency Range: {max(word_counts.values())} to {min(word_counts.values())}")
    
    # Add the summary text at the top right corner of the plot
    plt.gca().text(1.0, 1.0, summary_text, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig("./train_word_count.png")

def main():
    if len(sys.argv) < 2:
        print("Usage: python frequency_barplot.py <input_file.txt>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    words = read_words(input_file)
    
    # Count word frequencies
    word_counts = Counter(words)
    
    # Plot the frequencies (change top_n if you want to plot more or less words)
    plot_word_frequencies(word_counts, top_n=12)

if __name__ == "__main__":
    main()
