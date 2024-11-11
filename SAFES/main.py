import sys
import cli

if __name__ == "__main__":
    try:
        cli.main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)