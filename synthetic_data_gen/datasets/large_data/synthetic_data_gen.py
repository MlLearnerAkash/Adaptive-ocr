import sys
import os
import re
from PIL import Image, ImageDraw, ImageFont

def generate_word_images(input_file, font_path_1, font_path_2, font_size=72, padding=20, output_dest = "./"):
    # Create output directories based on font names
    font_folders = [
        os.path.join(output_dest, os.path.splitext(os.path.basename(font_path_1))[0]),
        os.path.join(output_dest, os.path.splitext(os.path.basename(font_path_2))[0])
    ]
    
    # Create directories if they don't exist
    for folder in font_folders:
        os.makedirs(folder, exist_ok=True)

    # Read words from input file
    with open(input_file, 'r') as file:
        words = [line.strip() for line in file if line.strip()][:60000]
    # Load fonts with error handling
    try:
        font1 = ImageFont.truetype(font_path_1, font_size)
    except IOError:
        print(f"Error loading font from {font_path_1}")
        return
    try:
        font2 = ImageFont.truetype(font_path_2, font_size)
    except IOError:
        print(f"Error loading font from {font_path_2}")
        return

    # Process each word
    j=0
    for word in words:
        j+=1
        
        # Sanitize filename: replace non-alphanumeric characters with underscore
        sanitized = re.sub(r'[^a-zA-Z0-9]+', '_', word).strip('_')
        if not sanitized:  # Handle case where word has only special characters
            sanitized = "unnamed"
        
        for i, (font, folder) in enumerate(zip([font1, font2], font_folders), 1):
            # Calculate text bounding box
            bbox = font.getbbox(word)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Create image with padding
            img_width = text_width + 2 * padding
            img_height = text_height + 2 * padding
            image = Image.new('RGB', (img_width, img_height), 'white')
            draw = ImageDraw.Draw(image)

            # Calculate text position (centered with padding)
            x = padding - bbox[0]
            y = padding - bbox[1]

            # Draw text
            draw.text((x, y), word, font=font, fill='black')

            # Save image in appropriate folder
            filename = f"{j}_{sanitized}_font{i}.jpg"
            output_path = os.path.join(folder, filename)
            image.save(output_path)
            print(">>>>>",j)
            print(f"Generated: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python word_images.py input.txt")
        sys.exit(1)

    FONT1_PATH = '/home/akash/ws/limited_supervision_ocr/synthetic_data_gen/resources/arial.ttf'
    # FONT1_PATH = "/home/akash/ws/limited_supervision_ocr/synthetic_data_gen/resources/hindi/Kalam-Regular.ttf"
    FONT2_PATH = '/home/akash/ws/limited_supervision_ocr/synthetic_data_gen/resources/lucida_calligraphy.ttf'
    # FONT2_PATH = "/home/akash/ws/limited_supervision_ocr/synthetic_data_gen/resources/hindi/Mohini.ttf"

    mode = "train"
    output_dest = f"./{mode}" #f"hindi_words/{mode}"

    
    
    generate_word_images(sys.argv[1], FONT1_PATH, FONT2_PATH, output_dest=output_dest)