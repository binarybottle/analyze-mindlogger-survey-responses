# MindLogger/Prolific survey data analysis script

The analyze_survey_responses.py script analyzes MindLogger survey data 
collected via the Prolific platform. It handles various survey data formats, 
parses multiple-choice and open-ended questions, and generates visualizations 
and summary statistics.

## Features

- **Flexible Data Processing**: Analyzes single survey files or entire directories of survey data
- **Automatic Format Detection**: Identifies column structures and question types
- **Advanced Option Parsing**: Parses multiple-choice options in various formats
- **Data Visualization**: Generates bar charts for multiple-choice questions with sorting
- **Response Collection**: Compiles and organizes open-ended text responses
- **User Filtering**: Filters out respondents based on custom criteria (exclusion rules)
- **Detailed Reporting**: Creates comprehensive analysis reports in text format

## Installation

### Requirements

- Python 3.6+
- Required packages:
  - pandas
  - matplotlib
  - seaborn
  - numpy

### Setup

1. Clone this repository or download the script
2. Install the required packages:

```bash
pip install pandas matplotlib seaborn numpy
```

## Usage

### Basic Usage

Process a single survey file:

```bash
python analyze_survey_responses.py --file survey_data.csv --output results
```

Process all CSV files in a directory as one combined dataset:

```bash
python analyze_survey_responses.py --dir data/responses --output results
```

### Advanced Options

| Option | Description | Default |
|--------|-------------|---------|
| `--file` | Path to a single survey data CSV file | |
| `--dir` | Directory containing multiple survey data CSV files | |
| `--output`, `-o` | Output directory for figures and reports | `survey_results` |
| `--format`, `-f` | Output format for figures (png, pdf, svg, etc.) | `png` |
| `--dpi`, `-d` | DPI for output figures | `300` |
| `--filter` | Apply user filtering based on exclusion rules | `False` |
| `--filter-file` | Path to CSV file containing filtering rules | |

### Filtering Respondents

To filter respondents based on their answers:

1. Create a CSV file with filtering rules (see example below)
2. Run the script with filtering enabled:

```bash
python analyze_survey_responses.py --file survey_data.csv --output results --filter --filter-file exclusion_rules.csv
```

#### Example Filter Rules CSV

```csv
question_id,operation,value,reason
attention_check,not_equals,4,Failed attention check
completion_time,less_than,120,Completed survey too quickly
demographic_age,less_than,18,Under age requirement
open_ended_feedback,contains,irrelevant,Provided irrelevant feedback
```

Supported operations:
- `equals` - Response equals the specified value
- `not_equals` - Response does not equal the specified value
- `contains` - Response contains the specified text
- `greater_than` - Response is greater than the specified value
- `less_than` - Response is less than the specified value
- `in_list` - Response is in the specified list of values

## Output

The script generates the following outputs in the specified directory:

1. **Visualization Files**: Bar charts for multiple-choice questions (PNG, SVG, PDF, etc.)
2. **Survey Report**: A text file containing summary statistics and response distributions
3. **Text Responses CSV**: A CSV file containing all open-ended responses
4. **Filtering Report**: (When filtering is enabled) A report of excluded respondents

## Data Format Support

This tool is designed to handle various survey data formats with automatic detection, particularly:

- MindLogger exports with `item_` prefixed columns
- Standard survey formats with question IDs and response values
- Data with different column naming conventions

The script will attempt to identify relevant columns and adapt to the provided data structure.

## Troubleshooting

If you encounter issues with the script not correctly parsing your survey data:

1. Check the console output for warnings about missing columns or parsing issues
2. Ensure your CSV file is properly formatted and encoded (UTF-8 recommended)
3. If specific question types are not being correctly identified, review the first few rows of your data to ensure they contain the expected structure

## License

[MIT License](LICENSE)

For the attached csv file, consolidates all similar responses to each question (except for responses that essentially mean "No", "None", or "Not really") and generate a new csv file, where each row has the question, unique (consolidated) response, and the number of instances of that response.
