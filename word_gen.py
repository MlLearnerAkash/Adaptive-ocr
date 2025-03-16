import os
from PIL import Image, ImageDraw, ImageFont

def create_word_images(input_txt, output_dir, output_list):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read words from the input text file
    with open(input_txt, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f.readlines() if line.strip()]
    
    image_paths = []
    font_path = "arial.ttf"  # Change this to a valid font path on your system
    font_size = 40
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print("Font file not found. Using default font.")
        font = ImageFont.load_default()
    
    for i, word in enumerate(words):
        # Create an image with white background
        img_size = (200, 60)
        img = Image.new('RGB', img_size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Get text size and center it
        text_size = draw.textbbox((0, 0), word, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]
        text_x = (img_size[0] - text_width) // 2
        text_y = (img_size[1] - text_height) // 2
        
        # Draw text
        draw.text((text_x, text_y), word, fill='black', font=font)
        
        # Save image
        img_path = os.path.join(output_dir, f"{i}.png")
        img.save(img_path)
        image_paths.append(os.path.abspath(img_path))
    
    # Write image paths to the output list file
    with open(output_list, 'w', encoding='utf-8') as f:
        for path in image_paths:
            f.write(path + "\n")
    
    print(f"Generated {len(words)} images. Paths saved in {output_list}")

# Example usage
if __name__ == "__main__":
    input_txt = "words.txt"  # Input text file containing words
    output_dir = "dataset/words/word_images"  # Directory to save images
    output_list = "dataset/image_paths.txt"  # Output text file with image paths
    
    create_word_images(input_txt, output_dir, output_list)
