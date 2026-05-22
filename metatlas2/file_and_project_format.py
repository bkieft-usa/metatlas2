from pathlib import Path
import re
import sys

# Define the expected file and project patterns
FILE_PATTERN = re.compile(
    r"^(?P<date>\d{8})_"
    r"(?P<project_code>JGI)_(?P<site>[A-Z0-9-]+)_(?P<project_id>\d+)_"
    r"(?P<project_name>[A-Za-z0-9]+)_"
    r"(?P<experiment>EXP\d+[A-Z]?)_"
    r"(?P<chromatography>C18|HILIC|HILICZ|RP)_"
    r"(?P<instrument>[A-Z0-9]+)_"
    r"(?P<polarity>POS|NEG|FPS)_"
    r"(?P<ms_level>MS1|MS2|MSMS)_"
    r"(?P<batch>\d+-[A-Z])_"
    r"(?P<sample_type>[A-Za-z0-9-]+)_"
    r"(?P<sample_id>[A-Za-z0-9-]+)_"
    r"(?P<run>Run\d+)\.(?P<ext>raw|mzML|h5)$"
)

PROJECT_PATTERN = re.compile(
    r"^(?P<date>\d{8})_"
    r"(?P<project_code>JGI)_(?P<site>[A-Z0-9-]+)_(?P<project_id>\d+)_"
    r"(?P<project_name>[A-Za-z0-9]+)_"
    r"(?P<experiment>EXP\d+[A-Z]?)_"
    r"(?P<chromatography>C18|HILIC|HILICZ|RP)_"
    r"(?P<instrument>[A-Z0-9]+)"
    r"(_[A-Za-z0-9]+)?$"
)

def parse_file_name(filename: str):
    match = FILE_PATTERN.match(filename)
    if not match:
        raise Exception(f"Filename '{filename}' does not match the expected format.")
    return match.groupdict()

def parse_project_name(project_name: str):
    match = PROJECT_PATTERN.match(project_name)
    if not match:
        raise Exception(f"Project name '{project_name}' does not match the expected format.")
    # Warn if there is a suffix
    if match.group(8):
        print(f"Warning: Project name '{project_name}' has a suffix '{match.group(8)}'.")
    return match.groupdict()

def main():
    if len(sys.argv) < 3:
        print("Usage: python file_and_project_format.py <project_name> <file1> [<file2> ...]")
        sys.exit(1)
    project_name = sys.argv[1]
    files = sys.argv[2:]
    print(f"Checking project: {project_name}")
    parse_project_name(project_name)
    for f in files:
        print(f"Checking file: {f}")
        parse_file_name(Path(f).name)
    print("All files and project name conform to the expected format.")

if __name__ == "__main__":
    main()
