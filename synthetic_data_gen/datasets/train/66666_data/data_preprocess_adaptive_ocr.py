def merge_txt_files(image_paths_file, labels_file, output_file):
    """
    Reads image paths and labels from two text files, and writes an output file
    with each line containing an image path and its corresponding label separated by a tab.

    Parameters:
    - image_paths_file: Path to the text file containing image paths.
    - labels_file: Path to the text file containing labels.
    - output_file: Path for the output text file.
    """
    # Read image paths and labels from their respective files.
    with open(image_paths_file, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()][:60000]

    with open(labels_file, 'r') as f:
        labels = [line.strip() for line in f if line.strip()][:60000]

    # Check if the two files have the same number of lines.
    if len(image_paths) != len(labels):
        raise ValueError("The number of image paths and labels must be the same.")

    # Write the merged content to the output file.
    with open(output_file, 'w') as f:
        for img, label in zip(image_paths, labels):
            f.write(f"{img}\t{label}\n")

    print(f"Merged file created: {output_file}")

# Example usage:
if __name__ == "__main__":
    merge_txt_files("/home/akash/ws/limited_supervision_ocr/synthetic_data_gen/datasets/train/lucida_sorted_files.txt", 
                    "/home/akash/ws/limited_supervision_ocr/synthetic_data_gen/datasets/train/train_word_labels.txt", 
                    "lucida_merged.txt")
