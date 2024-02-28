import argparse

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    
    # Add positional arguments
    parser.add_argument('input_file', type=str, help='Input file path')
    
    # Add optional arguments with a list of possible choices
    parser.add_argument('--mode', choices=['option1', 'option2', 'option3'], default='option1', 
                        help='Choose mode from options: option1, option2, option3')

    args = parser.parse_args()

    # Access the parsed arguments
    input_file = args.input_file
    mode = args.mode
    
    # Your program logic here
    print("Input file:", input_file)
    print("Mode chosen:", mode)

if __name__ == '__main__':
    main()