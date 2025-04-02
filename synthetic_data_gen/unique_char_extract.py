import sys

def extract_unique_chars(input_file, output_file):
    # Read the entire content of the file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create a set of all characters and remove any whitespace characters
    unique_chars = {ch for ch in content if not ch.isspace()}
    
    # Sort the characters by Unicode code point
    sorted_chars = sorted(unique_chars)
    
    # Write the sorted characters to the output file (one per line)
    with open(output_file, 'w', encoding='utf-8') as f:
        for ch in sorted_chars:
            f.write(ch + '\n')
    
    print(f"Unique characters written to {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_unique_chars.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    # Use provided output filename or default to "unique_chars.txt"
    output_file = sys.argv[2] if len(sys.argv) >= 3 else "unique_chars.txt"
    
    extract_unique_chars(input_file, output_file)

if __name__ == "__main__":
    main()
