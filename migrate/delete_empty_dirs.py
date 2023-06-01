import os
import sys

def remove_empty_directories(path):
    for dirpath, dirnames, filenames in os.walk(path, topdown=False):
        if not dirnames and not filenames:
            os.rmdir(dirpath)
            print(f"Removed empty directory: {dirpath}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <directory_to_search>")
        sys.exit(1)

    directory_to_search = sys.argv[1]
    remove_empty_directories(directory_to_search)
