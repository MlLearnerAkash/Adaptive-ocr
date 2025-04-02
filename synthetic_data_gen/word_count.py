from collections import Counter
import matplotlib.pyplot as plt

def get_top_unique_words(file_path, n):
    """
    Reads a text file, extracts the first `n` words,
    and returns unique words sorted by frequency.
    
    :param file_path: Path to the text file
    :param n: Number of words to consider from the file
    :return: List of unique words sorted by frequency (high to low), Counter object
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        words = file.read().split()
    
    # Take first n words
    selected_words = words[:n]
    
    # Count word frequencies
    word_counts = Counter(selected_words)
    
    # Sort words by frequency (high to low) and then alphabetically
    sorted_words = sorted(word_counts.keys(), key=lambda w: (-word_counts[w], w))
    
    return sorted_words, word_counts

def plot_word_frequencies(word_counts, total_words):
    """
    Plots a bar chart of the word frequencies and displays additional statistics.
    
    :param word_counts: Counter object with word frequencies
    :param total_words: Total number of words in the dataset
    """
    words, counts = zip(*word_counts.most_common(10))  # Top 10 words
    
    max_freq = max(word_counts.values())
    min_freq = min(word_counts.values())
    avg_freq = sum(word_counts.values()) / len(word_counts) if word_counts else 0
    
    plt.figure(figsize=(10, 5))
    plt.bar(words, counts, color='skyblue')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 10 Word Frequencies')
    plt.xticks(rotation=45)
    
    # Display statistics in top right corner
    stats_text = (f"Total Words: {total_words}\n"
                  f"Unique Words: {len(word_counts)}\n"
                  f"Max Frequency: {max_freq}\n"
                  f"Min Frequency: {min_freq}\n"
                  f"Avg Frequency: {avg_freq:.2f}")
    
    plt.gca().text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    
    plt.savefig("training_5240_words.png")
    plt.show()

# Example usage:
unique_words, word_counts = get_top_unique_words("/home/akash/ws/limited_supervision_ocr/synthetic_data_gen/datasets/train/train_word_labels.txt", 5240)
total_words = sum(word_counts.values())
plot_word_frequencies(word_counts, total_words)



# Example usage:
# unique_words, word_counts = get_top_unique_words("/home/akash/ws/limited_supervision_ocr/synthetic_data_gen/datasets/train/train_word_labels.txt", 666)
# plot_word_frequencies(word_counts)