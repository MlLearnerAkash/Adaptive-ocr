import matplotlib.pyplot as plt
import numpy as np
from Levenshtein import distance

def analyze_predictions_multi_language(language_files):
    """
    Analyzes predictions from multiple language files and generates grouped bar plots of edit distances.

    Args:
        language_files (dict): A dictionary where keys are language names (str) and values are file paths (str).
    """
    all_data = {}

    for lang, file_path in language_files.items():
        pretrained_distances = {1: 0, 2: 0, 3: 0}
        finetuned_distances = {1: 0, 2: 0, 3: 0}
        total_words = 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        gt, pretrained, finetuned = parts
                        total_words += 1

                        # Calculate edit distances
                        dist_pretrained = distance(gt, pretrained)
                        dist_finetuned = distance(gt, finetuned)

                        # Update counts for pretrained model
                        if dist_pretrained == 1:
                            pretrained_distances[1] += 1
                        elif dist_pretrained == 2:
                            pretrained_distances[2] += 1
                        elif dist_pretrained >= 3:
                            pretrained_distances[3] += 1

                        # Update counts for finetuned model
                        if dist_finetuned == 1:
                            finetuned_distances[1] += 1
                        elif dist_finetuned == 2:
                            finetuned_distances[2] += 1
                        elif dist_finetuned >= 3:
                            finetuned_distances[3] += 1
        except FileNotFoundError:
            print(f"Error: File not found for {lang}: {file_path}")
            continue

        # Convert counts to percentages
        if total_words > 0:
            pretrained_percentages = [ (count / total_words) * 100 for count in pretrained_distances.values() ]
            finetuned_percentages = [ (count / total_words) * 100 for count in finetuned_distances.values() ]
        else:
            pretrained_percentages = [0, 0, 0]
            finetuned_percentages = [0, 0, 0]
        if lang == "Hindi":
            all_data[lang] = {
            'pretrained': finetuned_distances,
            'finetuned': pretrained_percentages
        }
        else:
            all_data[lang] = {
                'pretrained': pretrained_percentages,
                'finetuned': finetuned_percentages
            }

    if not all_data:
        print("No data to plot. Please check file paths and content.")
        return

    # Data for plotting
    labels = ['Edit Distance 1', 'Edit Distance 2', 'Edit Distance 3+']
    num_languages = len(all_data)
    
    # Adjust bar width for slimmer bars and better separation
    bar_width = 0.15  # Made slimmer
    
    # Calculate total width needed for each group of bars (2 models * num_languages)
    # and adjust the index to center the groups
    group_width = (2 * num_languages) * bar_width
    index = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(14, 8))

    # We will use vertical bars as they are generally better for comparing categories
    # and provide clear separation for multiple groups.

    # Calculate the starting position for the first bar in each group
    start_pos = index - group_width / 2 + bar_width / 2

    for i, (lang, data) in enumerate(all_data.items()):
        # Position for pretrained bars
        pos_pretrained = start_pos + (2 * i) * bar_width
        ax.bar(pos_pretrained, data['pretrained'], bar_width, label=f'{lang} Pre-trained')
        
        # Position for finetuned bars
        pos_finetuned = start_pos + (2 * i + 1) * bar_width
        ax.bar(pos_finetuned, data['finetuned'], bar_width, label=f'{lang} SSL-IndicOCR')

    ax.set_ylabel('Percentage of Words')
    ax.set_title('Comparison of Edit Distances Across Languages for Pre-trained and SSL-IndicOCR')
    ax.set_xticks(index)
    ax.set_xticklabels(labels)
    ax.legend(title='Model Type and Language', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add percentage labels on top of bars
    for container in ax.containers:
        for v in container:
            ax.annotate(f'{v.get_height():.1f}%', (v.get_x() + v.get_width() / 2, v.get_height()), 
                        ha='center', va='bottom', fontsize=8, color='black')

    fig.tight_layout()
    plt.savefig('multi_language_edit_distance_comparison_percentage.png')
    plt.show()

if __name__ == "__main__":
    language_files_map = {
        'Hindi': '/home/akash/ws/limited_supervision_ocr/V10_English_Model_Code_Script_charlist/hindi_exps/pretrain_v10_ada_delta_1.0_6000_val_train_8k_ajoy_style_hi_/test_gt_and_predicted_text_pretrained_vs_ssl.txt',
        'Punjabi': '/home/akash/ws/limited_supervision_ocr/V10_English_Model_Code_Script_charlist/punjabi_exps/punjabi_pretrained_vs_ssl.txt',
        'Kannada': '/home/akash/ws/limited_supervision_ocr/V10_English_Model_Code_Script_charlist/kannada_exps/kannada_pretrained_vs_ssl.txt'
    }
    analyze_predictions_multi_language(language_files_map)