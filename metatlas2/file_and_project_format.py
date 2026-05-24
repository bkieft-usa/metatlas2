import re

# Define the expected file and project patterns
FILE_PATTERN = re.compile(
    r"^(?P<date>[^_]+)_"
    r"(?P<owner>[^_]+)_"
    r"(?P<pi>[^_]+)_"
    r"(?P<project_id>[^_]+)_"
    r"(?P<project_shortname>[^_]+)_"
    r"(?P<experiment>[^_]+)_"
    r"(?P<instrument>[^_]+)_"
    r"(?P<chromatography>[^_]+)_"
    r"(?P<run_id>[^_]+)_"
    r"(?P<polarity>POS|NEG|FPS)_"
    r"(?P<ms_level>MS1|MS2|MSMS)_"
    r"(?P<sample_number>[^_]+)_"
    r"(?P<sample_name>[^_]+)_"
    r"(?P<replicate>[^_]+)_"
    r"(?P<run_metadata>[^_]+)_"
    r"(?P<run_number>Run\d+)\."
    r"(?P<ext>raw|mzML|h5)$"
)

PROJECT_PATTERN = re.compile(
    r"^(?P<date>\d{8})_"
    r"(?P<owner>JGI|EB|EGSB)_"
    r"(?P<pi>[^_]+)_"
    r"(?P<project_id>\d+)_"
    r"(?P<project_shortname>[^_]+)_"
    r"(?P<experiment>[^_]+)_"
    r"(?P<instrument>[^_]+)_"
    r"(?P<chromatography>[^_]+)_"
    r"(?P<run_id>[^_]+)"
    r"(?:_(?P<suffix>[^_]+))?$"
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
    if match.group(8):
        print(f"Warning: Project name '{project_name}' has a suffix '{match.group(8)}'.")
    return project_name