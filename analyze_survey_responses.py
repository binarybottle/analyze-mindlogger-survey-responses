#!/usr/bin/env python3
"""
MindLogger/Prolific survey data analysis script
--------------------------------------------------------------------------------
This script analyzes MindLogger survey data collected via the Prolific platform.
It handles various survey data formats, parses multiple-choice and open-ended
questions, and generates visualizations and summary statistics.

Features:
- Process single CSV files or entire directories of survey data
- Automatic detection of question types and response options
- Filter respondents based on custom criteria
- Generate visualizations for multiple-choice questions
- Compile open-ended responses

Usage:
    # Process a single file:
    python analyze_survey_responses.py --file survey_data.csv --output results --format png --dpi 300

    # Process all files in a directory:
    python analyze_survey_responses.py --dir data/responses --output results --format png --dpi 300
    
    # With user filtering:
    python analyze_survey_responses.py --file survey_data.csv --output results --filter --filter-file exclusion_rules.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import argparse
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import textwrap

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze survey data')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', help='Path to a single survey data CSV file')
    group.add_argument('--dir', help='Directory containing multiple survey data CSV files')
    parser.add_argument('--output', '-o', help='Output directory for figures', default='survey_results')
    parser.add_argument('--format', '-f', help='Output format for figures', default='png')
    parser.add_argument('--dpi', '-d', type=int, help='DPI for output figures', default=300)
    return parser.parse_args()

def extract_option_mapping(options_text: str) -> Dict[int, str]:
    """
    Extract option mappings from the options text.
    
    Args:
        options_text: The text describing the options
    
    Returns:
        Dictionary mapping option values to option labels
    """
    options = {}
    
    # Skip if no options text
    if pd.isna(options_text):
        return options
    
    # Check if this is a number range question (Min/Max format)
    if isinstance(options_text, str) and "Min:" in options_text and "Max:" in options_text:
        try:
            # Extract min and max values
            min_match = re.search(r'Min:\s*(\d+)', options_text)
            max_match = re.search(r'Max:\s*(\d+)', options_text)
            
            if min_match and max_match:
                min_val = int(min_match.group(1))
                max_val = int(max_match.group(1))
                
                # Create options for each number in the range
                for i in range(min_val, max_val + 1):
                    options[i] = str(i)
                
                return options
        except Exception as e:
            print(f"Error processing number range: {e}")
    
    # Handle the common format with text descriptions followed by option numbers
    if isinstance(options_text, str):
        # The specific pattern we're looking for: "Text description: number, Text description: number"
        pattern = r'([^:]+):\s*(\d+)(?:,|$)'
        
        # Find all matches
        matches = []
        # Track position to handle overlapping matches
        pos = 0
        
        # Use finditer to get positions
        for match in re.finditer(pattern, options_text):
            # Get the start position of this match
            start = match.start()
            
            # Only consider matches that start at or after our current position
            if start >= pos:
                text = match.group(1).strip()
                value = int(match.group(2))
                matches.append((text, value))
                
                # Update position to after this match
                pos = match.end()
        
        # Add the found matches to options
        for text, value in matches:
            options[value] = text
        
        # Return if we found options
        if options:
            return options
    
    # Fallback for handling JSON-like formats
    try:
        if isinstance(options_text, str) and '{' in options_text:
            import json
            options_dict = json.loads(options_text)
            if isinstance(options_dict, dict) and 'options' in options_dict:
                for option in options_dict['options']:
                    if 'id' in option and 'label' in option:
                        options[option['id']] = option['label']
            
            if options:
                return options
    except:
        pass
    
    # Last resort fallback
    try:
        if isinstance(options_text, str):
            parts = re.split(r',\s*(?=\S+:)', options_text)
            for part in parts:
                match = re.search(r'([^:]+):\s*(\d+)', part)
                if match:
                    text = match.group(1).strip()
                    value = int(match.group(2))
                    options[value] = text
    except Exception as e:
        print(f"All parsing methods failed: {e}")
    
    return options

def extract_survey_structure(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Extract the structure of the survey including questions and possible answers.
    
    Args:
        df: The survey data DataFrame
    
    Returns:
        A dictionary with question IDs as keys and question details as values
    """
    questions = {}
    
    # Ensure required columns exist
    required_columns = ['question_id', 'question_text', 'options_text']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing required columns: {missing_columns}")
        print("Available columns:", df.columns.tolist())
        print("Attempting to identify alternative columns...")
        
        # Try to identify equivalent columns if standard ones are missing
        column_mappings = {
            'question_id': ['id', 'question', 'q_id', 'question_identifier', 'item_id', 'item'],
            'question_text': ['question', 'text', 'q_text', 'prompt', 'item_text', 'item_prompt', 'item_name'],
            'options_text': ['options', 'choices', 'answers', 'response_options', 'item_response_options']
        }
        
        for missing_col in missing_columns:
            for alt_col in column_mappings[missing_col]:
                if alt_col in df.columns:
                    print(f"Using '{alt_col}' for '{missing_col}'")
                    df[missing_col] = df[alt_col]
                    break
            
            # If still not found, try to create from first part of the file
            if missing_col not in df.columns:
                if missing_col == 'question_id' and len(df) > 0:
                    print("Creating question_id column from first available identifier")
                    # Try to find any column that could be used as an identifier
                    potential_id_cols = [col for col in df.columns if any(id_term in col.lower() for id_term in ['id', 'key', 'name', 'item'])]
                    if potential_id_cols:
                        df['question_id'] = df[potential_id_cols[0]]
                    else:
                        # Last resort: create sequential IDs
                        print("Creating sequential question_id values")
                        # Group by any text column that might contain question text
                        text_cols = [col for col in df.columns if any(text_term in col.lower() for text_term in ['text', 'question', 'prompt', 'item'])]
                        if text_cols:
                            unique_questions = df[text_cols[0]].dropna().unique()
                            id_mapping = {q: f"q_{i}" for i, q in enumerate(unique_questions)}
                            df['question_id'] = df[text_cols[0]].map(id_mapping)
                        else:
                            df['question_id'] = ['q_' + str(i) for i in range(len(df))]
                
                if missing_col == 'question_text' and len(df) > 0:
                    print("Creating question_text column from available text fields")
                    text_cols = [col for col in df.columns if any(text_term in col.lower() for text_term in ['text', 'question', 'prompt', 'item'])]
                    if text_cols:
                        df['question_text'] = df[text_cols[0]]
                    else:
                        df['question_text'] = df['question_id']
                
                if missing_col == 'options_text' and len(df) > 0:
                    print("Creating empty options_text column")
                    df['options_text'] = None
    
    # Check if we have all required columns now
    if not all(col in df.columns for col in required_columns):
        print("Warning: Could not identify all required columns. Analysis may be incomplete.")
        missing = [col for col in required_columns if col not in df.columns]
        print(f"Still missing: {missing}")
        
        # Create minimal versions of missing columns to continue
        for col in missing:
            df[col] = None
    
    # Group by question ID to process each question
    try:
        grouped = df.groupby(['question_id', 'question_text', 'options_text'])
    except Exception as e:
        print(f"Error during grouping: {e}")
        print("Falling back to simplified grouping...")
        
        # Fallback to just using question_id if grouping fails
        grouped = df.groupby('question_id')
    
    # Process each question group
    for group_key, group in grouped:
        # Handle both tuple and single string keys
        if isinstance(group_key, tuple):
            q_id, q_text, options_text = group_key
        else:
            q_id = group_key
            q_text = group.iloc[0].get('question_text') if 'question_text' in group.columns else f"Question {q_id}"
            options_text = group.iloc[0].get('options_text') if 'options_text' in group.columns else None
        
        if pd.isna(q_id):
            continue
            
        question = {
            'id': q_id,
            'text': q_text if not pd.isna(q_text) else f"Question {q_id}",
            'options': {},
            'type': 'multiple_choice' if not pd.isna(options_text) else 'open_ended',
            'responses': []
        }
        
        # Parse options if they exist
        if not pd.isna(options_text):
            question['options'] = extract_option_mapping(options_text)
        
        # If no options were found but we have responses, try to infer options from responses
        if not question['options'] and question['type'] == 'multiple_choice':
            # Look for value column to check for numeric responses
            value_cols = [col for col in group.columns if col.lower() in ['value', 'response', 'answer', 'item_response']]
            if value_cols:
                unique_values = group[value_cols[0]].dropna().unique()
                numeric_values = [v for v in unique_values if isinstance(v, (int, float)) or 
                                 (isinstance(v, str) and v.replace('.', '', 1).isdigit())]
                
                if numeric_values:
                    for val in numeric_values:
                        try:
                            val_num = int(float(val))
                            question['options'][val_num] = f"Option {val_num}"
                        except (ValueError, TypeError):
                            pass
        
        # Add to questions dictionary
        questions[str(q_id)] = question
    
    return questions

def parse_responses(df: pd.DataFrame, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Parse all responses for each question.
    
    Args:
        df: The survey data DataFrame
        questions: The survey structure dictionary
    
    Returns:
        Updated questions dictionary with responses
    """
    # Find columns that might contain responses
    response_columns = []
    potential_cols = ['value', 'response', 'response_text', 'answer', 'text_response', 'open_response']
    
    for col in df.columns:
        if col in potential_cols or any(term in col.lower() for term in ['value', 'response', 'answer']):
            response_columns.append(col)
    
    if not response_columns:
        print("Warning: No response columns identified.")
        print("Available columns:", df.columns.tolist())
        # Try to use any column that's not obviously metadata
        exclude_terms = ['id', 'question', 'options', 'timestamp', 'date', 'time', 'user', 'participant', 'source']
        response_columns = [col for col in df.columns if not any(term in col.lower() for term in exclude_terms)]
        if response_columns:
            print(f"Using columns as possible response fields: {response_columns}")
    
    # Find columns that might contain question IDs
    question_id_columns = []
    potential_cols = ['question_id', 'id', 'q_id', 'question_identifier', 'item_id', 'item']
    
    for col in df.columns:
        if col in potential_cols or any(term in col.lower() for term in ['question', 'id', 'item']):
            question_id_columns.append(col)
    
    if not question_id_columns:
        print("Warning: No question ID columns identified.")
        print("Available columns:", df.columns.tolist())
        return questions
    
    # Process each row to extract responses
    for _, row in df.iterrows():
        # Try to find question ID in the row
        q_id = None
        for col in question_id_columns:
            if col in row and not pd.isna(row[col]):
                q_id = str(row[col])  # Convert to string for consistency
                break
        
        if not q_id or q_id not in questions:
            continue
        
        # Try to find response value in the row
        response_value = None
        for col in response_columns:
            if col in row and not pd.isna(row[col]):
                response_value = row[col]
                break
        
        if not pd.isna(response_value):
            # Add response to the question
            questions[q_id]['responses'].append(response_value)
    
    return questions

def analyze_multiple_choice(question: Dict[str, Any]) -> Dict[Any, int]:
    """
    Analyze multiple choice question responses.
    
    Args:
        question: Question dictionary with responses
    
    Returns:
        Dictionary with counts for each option
    """
    counts = defaultdict(int)
    valid_option_values = set(question['options'].keys())
    
    # For debugging
    invalid_responses = []
    
    # If we have no valid options defined, try to infer them from responses
    if not valid_option_values:
        print(f"No valid options defined for question {question['id']}. Attempting to infer from responses...")
        all_values = []
        for response in question['responses']:
            if isinstance(response, str) and response.startswith('value:'):
                values = response.replace('value:', '').strip()
                all_values.extend([int(val) for val in re.findall(r'\d+', values)])
            elif isinstance(response, (int, float)):
                all_values.append(int(response))
        
        # Count occurrences of each value
        value_counts = Counter(all_values)
        # Use values that appear multiple times or in a reasonable range (e.g., 1-20)
        for val, count in value_counts.items():
            if count > 1 or (1 <= val <= 20):
                valid_option_values.add(val)
                question['options'][val] = f"Option {val}"
        
        if valid_option_values:
            print(f"Inferred {len(valid_option_values)} valid options for question {question['id']}: {valid_option_values}")
    
    for response in question['responses']:
        # Skip empty responses
        if pd.isna(response) or response == '':
            continue
            
        # Process the response to get potential values
        found_values = []
        
        # Handle responses in format "value: X, Y, Z"
        if isinstance(response, str) and response.startswith('value:'):
            # Extract the value part
            value_part = response.split('|')[0] if '|' in response else response
            values = value_part.replace('value:', '').strip()
            found_values = [int(val) for val in re.findall(r'\d+', values)]
        
        # Handle JSON array format
        elif isinstance(response, str) and response.startswith('[') and response.endswith(']'):
            try:
                import json
                values = json.loads(response)
                if isinstance(values, list):
                    for val in values:
                        if isinstance(val, (int, float)) or (isinstance(val, str) and val.isdigit()):
                            found_values.append(int(float(val)))
            except:
                # If JSON parsing fails, try regex
                found_values = [int(val) for val in re.findall(r'\d+', response)]
        
        # Handle simple numeric responses
        elif isinstance(response, (int, float)) or (isinstance(response, str) and response.replace('.', '', 1).isdigit()):
            try:
                found_values = [int(float(response))]
            except (ValueError, TypeError):
                pass
        
        # Handle string responses that might be option texts
        elif isinstance(response, str):
            # Look for the option by label
            found = False
            for val, label in question['options'].items():
                if response.strip() == label.strip():
                    found_values = [val]
                    found = True
                    break
            
            # If not found by label, check for any numbers in the string
            if not found:
                found_values = [int(val) for val in re.findall(r'\d+', response)]
        
        # Only count values that match valid options
        valid_values = [val for val in found_values if val in valid_option_values]
        invalid_values = [val for val in found_values if val not in valid_option_values]
        
        # Store invalid responses for debugging
        if invalid_values:
            invalid_responses.append((response, invalid_values))
        
        # If there are no valid values but we found invalid ones, report special case
        if not valid_values and invalid_values and '|' in str(response):
            # This might be a case where text follows the value
            print(f"Response with text component for question {question['id']}: {response}")
        
        # Count the valid values
        for val in valid_values:
            counts[val] += 1
        
        # If we couldn't find any valid values but have some potential values, 
        # we might want to add the first one as "Other" or similar
        if not valid_values and found_values and len(found_values) == 1:
            other_key = "Other responses"
            counts[other_key] += 1
    
    # Debug output - shows responses that might be causing issues
    if invalid_responses:
        print(f"Found {len(invalid_responses)} invalid responses for question {question['id']}:")
        for resp, vals in invalid_responses[:5]:  # Show first 5 for brevity
            print(f"  Response '{resp}' yielded invalid values: {vals}")
        
        # Suggest checking the options mapping
        options_str = ", ".join([f"{k}: {v}" for k, v in question['options'].items()])
        print(f"  Valid options are: {options_str}")
    
    # Convert counts to use option text
    labeled_counts = {}
    for val, count in counts.items():
        if isinstance(val, str):  # Already a label (like "Other responses")
            labeled_counts[val] = count
        elif val in question['options']:
            labeled_counts[question['options'][val]] = count
        else:
            # This shouldn't happen given our filtering above, but just in case
            labeled_counts[f"Option {val}"] = count
    
    return labeled_counts if labeled_counts else counts

def analyze_open_ended(question: Dict[str, Any]) -> List[str]:
    """
    Collect open-ended responses.
    
    Args:
        question: Question dictionary with responses
    
    Returns:
        List of text responses
    """
    text_responses = []
    for response in question['responses']:
        if isinstance(response, str) and len(response.strip()) > 0:
            text_responses.append(response.strip())
    
    return text_responses

def plot_multiple_choice(question_id: str, question_text: str, counts: Dict[str, int], 
                       output_dir: str, file_format: str, dpi: int) -> str:
    """
    Create a histogram for multiple choice question responses.
    
    Args:
        question_id: The question ID
        question_text: The question text
        counts: Counts of responses for each option
        output_dir: Directory to save the plot
        file_format: File format for saving the plot
        dpi: DPI for the output figure
    
    Returns:
        Path to the saved figure
    """
    plt.figure(figsize=(12, 7))
    
    # Check if this is likely a numerical question
    is_numerical_question = any(term in question_text.lower() for term in 
                              ['how many', 'number of', 'age', 'years', 'count', 'quantity'])
    
    # Try to extract numeric values from labels when possible
    numeric_values = {}
    for key in counts.keys():
        if isinstance(key, (int, float)):
            numeric_values[key] = float(key)
        elif isinstance(key, str) and key.isdigit():
            numeric_values[key] = float(key)
        elif isinstance(key, str):
            # Look for numbers at the end of strings like "Text: 5"
            match = re.search(r':\s*(\d+)$', key)
            if match:
                numeric_values[key] = float(match.group(1))
            else:
                # Try to find any number in the string
                numbers = re.findall(r'\d+', key)
                if numbers:
                    numeric_values[key] = float(numbers[0])
    
    # Sort the items
    if numeric_values and (is_numerical_question or len(numeric_values) >= len(counts) * 0.5):
        # If we have numeric values and it's likely a numerical question, sort by values
        sorted_items = [(key, counts[key]) for key in counts.keys()]
        sorted_items.sort(key=lambda x: numeric_values.get(x[0], float('inf')))
    else:
        # Otherwise sort by frequency (descending)
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    # Extract labels and values from sorted items
    labels = []
    values = [item[1] for item in sorted_items]
    
    # Process labels - if numerical question, use just numbers as labels
    for item in sorted_items:
        key = item[0]
        if is_numerical_question and key in numeric_values:
            # For numerical questions, use just the number as the label
            labels.append(str(int(numeric_values[key])))
        else:
            labels.append(str(key))
    
    # Create the bar chart
    bars = plt.bar(range(len(values)), values, width=0.7, color='#3498db', edgecolor='#2980b9')
    
    # Add count values on top of each bar
    for i, v in enumerate(values):
        plt.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Set labels and title
    plt.xlabel('Response Options', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f"{question_text}", fontsize=14, wrap=True)
    
    # Set x-axis tick labels with wrapped text
    plt.xticks(range(len(labels)), [textwrap.fill(label, 30) for label in labels], 
              rotation=45, ha='right', fontsize=10)
    
    # Adjust layout and add grid
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    filename = f"{question_id.replace(' ', '_').replace('/', '_')}.{file_format}"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return filepath

def analyze_survey(file_path: str, output_dir: str, file_format: str, dpi: int) -> Dict[str, Any]:
    """
    Main function to analyze a single survey data file.
    
    Args:
        file_path: Path to the survey data CSV file
        output_dir: Directory to save output figures
        file_format: File format for saving figures
        dpi: DPI for output figures
    
    Returns:
        Dictionary with analysis results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV file
    df = pd.read_csv(file_path)
    return analyze_survey_dataframe(df, output_dir, file_format, dpi)

def analyze_survey_dataframe(df: pd.DataFrame, output_dir: str, file_format: str, dpi: int) -> Dict[str, Any]:
    """
    Analyze survey data from a DataFrame.
    
    Args:
        df: The survey data DataFrame
        output_dir: Directory to save output figures
        file_format: File format for saving figures
        dpi: DPI for output figures
    
    Returns:
        Dictionary with analysis results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analyzing DataFrame with {len(df)} rows and {len(df.columns)} columns")
    print("Columns:", df.columns.tolist())
    
    # Check data structure - Specifically looking for item-based columns
    item_columns = [col for col in df.columns if col.startswith('item_')]
    if item_columns:
        print(f"Found {len(item_columns)} item-related columns: {item_columns}")
        
        # If this is a MindLogger format with item_ prefix columns
        if all(col in df.columns for col in ['item_id', 'item_name', 'item_prompt']):
            print("Detected MindLogger format with item_ prefixed columns")
            
            # Map MindLogger columns to our expected columns
            df['question_id'] = df['item_id']
            df['question_text'] = df['item_prompt']
            if 'item_response_options' in df.columns:
                df['options_text'] = df['item_response_options']
            if 'item_response' in df.columns:
                df['value'] = df['item_response']
    
    # Inspect a sample of rows
    inspect_sample = min(5, len(df))
    if inspect_sample > 0:
        print(f"\nInspecting first {inspect_sample} rows to understand data structure:")
        sample_df = df.head(inspect_sample)
        for idx, row in sample_df.iterrows():
            print(f"\nRow {idx}:")
            for col, val in row.items():
                if not pd.isna(val) and col in ['question_id', 'question_text', 'options_text', 'value', 
                                               'item_id', 'item_prompt', 'item_response_options', 'item_response']:
                    print(f"  {col}: {val}")
    
    # Clean up column names - handle different possible formats
    if 'question_id' not in df.columns:
        # Try to identify columns with question IDs, text, and options
        potential_cols = ['id', 'question', 'q_id', 'question_identifier', 'item_id']
        for col in potential_cols:
            if col in df.columns:
                print(f"Using '{col}' as question_id")
                df['question_id'] = df[col]
                break
    
    if 'question_text' not in df.columns:
        potential_cols = ['question', 'text', 'q_text', 'prompt', 'item_prompt', 'item_text', 'item_name']
        for col in potential_cols:
            if col in df.columns:
                print(f"Using '{col}' as question_text")
                df['question_text'] = df[col]
                break
    
    if 'options_text' not in df.columns:
        potential_cols = ['options', 'choices', 'answers', 'response_options', 'item_response_options']
        for col in potential_cols:
            if col in df.columns:
                print(f"Using '{col}' as options_text")
                df['options_text'] = df[col]
                break
    
    if 'value' not in df.columns:
        potential_cols = ['response', 'answer', 'text_response', 'open_response', 'item_response', 'response_text']
        for col in potential_cols:
            if col in df.columns:
                print(f"Using '{col}' as value")
                df['value'] = df[col]
                break
    
    # For MindLogger data, directly process without using extract_survey_structure
    if all(col in df.columns for col in ['question_id', 'question_text']):
        print("Processing using direct method for MindLogger format...")
        questions = {}
        
        # Group by question ID
        for q_id, group in df.groupby('question_id'):
            if pd.isna(q_id):
                continue
                
            q_text = group['question_text'].iloc[0]
            options_text = group['options_text'].iloc[0] if 'options_text' in group.columns else None
            
            question = {
                'id': q_id,
                'text': q_text if not pd.isna(q_text) else f"Question {q_id}",
                'options': {},
                'type': 'multiple_choice' if not pd.isna(options_text) else 'open_ended',
                'responses': []
            }
            
            # Parse options if they exist
            if not pd.isna(options_text):
                question['options'] = extract_option_mapping(options_text)
            
            # Add responses from this group
            if 'value' in group.columns:
                for response in group['value'].dropna():
                    question['responses'].append(response)
            
            # Add to questions dictionary
            questions[str(q_id)] = question
        
        print(f"Directly identified {len(questions)} unique questions")
    else:
        # Use the standard extraction method
        questions = extract_survey_structure(df)
        questions = parse_responses(df, questions)
    
    # Count responses per question for debugging
    for q_id, question in questions.items():
        print(f"Question '{q_id}' has {len(question['responses'])} responses")
    
    # Analyze each question and generate visualizations
    results = {
        'total_questions': len(questions),
        'multiple_choice_questions': 0,
        'open_ended_questions': 0,
        'figures': [],
        'text_responses': {}
    }
    
    for q_id, question in questions.items():
        if question['type'] == 'multiple_choice':
            counts = analyze_multiple_choice(question)
            if counts:
                results['multiple_choice_questions'] += 1
                figure_path = plot_multiple_choice(
                    q_id, question['text'], counts, output_dir, file_format, dpi
                )
                results['figures'].append({
                    'question_id': q_id,
                    'path': figure_path,
                    'counts': counts
                })
        else:
            text_responses = analyze_open_ended(question)
            if text_responses:
                results['open_ended_questions'] += 1
                results['text_responses'][q_id] = {
                    'question_text': question['text'],
                    'responses': text_responses
                }
    
    # Create a summary report using the improved generate_report function
    report_path = generate_report(results, questions, output_dir)
    
    if results['open_ended_questions'] > 0:
        results['text_responses_csv'] = export_text_responses_to_csv(results, output_dir)

    results['report_path'] = report_path
    return results

def process_directory(directory: str, output_dir: str, file_format: str, dpi: int) -> Dict[str, Any]:
    """
    Process all CSV files in the specified directory by concatenating them into a single dataset.
    
    Args:
        directory: Directory containing CSV files
        output_dir: Directory for output
        file_format: File format for figures
        dpi: DPI for output figures
    
    Returns:
        Results dictionary for the combined analysis
    """
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.lower().endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in directory: {directory}")
        return {}
    
    print(f"Found {len(csv_files)} CSV files to combine and analyze")
    
    # Read and concatenate all CSV files
    all_dfs = []
    for file_name in csv_files:
        file_path = os.path.join(directory, file_name)
        print(f"Reading file: {file_name}")
        try:
            df = pd.read_csv(file_path)
            # Add source file information if desired
            df['source_file'] = file_name
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
    
    if not all_dfs:
        print("No valid CSV files could be read")
        return {}
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined dataset has {len(combined_df)} rows")
    
    # Analyze the combined dataset
    results = analyze_survey_dataframe(combined_df, output_dir, file_format, dpi)
    results['source_files'] = csv_files
    results['total_files'] = len(csv_files)
    
    return results
    
def generate_report(results, questions, output_dir):
    """Generate a summary report of the survey analysis."""
    report_path = os.path.join(output_dir, 'survey_report.txt')
    with open(report_path, 'w') as f:
        f.write("Survey Analysis Report\n")
        f.write("====================\n\n")
        f.write(f"Total Questions: {results['total_questions']}\n")
        f.write(f"Multiple Choice Questions: {results['multiple_choice_questions']}\n")
        f.write(f"Open-Ended Questions: {results['open_ended_questions']}\n\n")
        
        f.write("Multiple Choice Questions:\n")
        f.write("--------------------------\n")
        for figure in results['figures']:
            # Get question text instead of ID
            q_text = next((q['text'] for q in questions.values() if q['id'] == figure['question_id']), 
                          f"Question {figure['question_id']}")
            
            f.write(f"\nQuestion: {q_text}\n\n")
            f.write("Responses:\n")
            
            # Check if this is likely a numerical question
            is_numerical_question = any(term in q_text.lower() for term in 
                                      ['how many', 'number of', 'age', 'years', 'count', 'quantity'])
            
            # Try to extract numeric values from labels when possible
            numeric_values = {}
            for key in figure['counts'].keys():
                if isinstance(key, (int, float)):
                    numeric_values[key] = float(key)
                elif isinstance(key, str) and key.isdigit():
                    numeric_values[key] = float(key)
                elif isinstance(key, str):
                    # Look for numbers at the end of strings like "Text: 5"
                    match = re.search(r':\s*(\d+)$', key)
                    if match:
                        numeric_values[key] = float(match.group(1))
                    else:
                        # Try to find any number in the string
                        numbers = re.findall(r'\d+', key)
                        if numbers:
                            numeric_values[key] = float(numbers[0])
            
            # Sort the items using the same logic as in plot_multiple_choice
            if numeric_values and (is_numerical_question or len(numeric_values) >= len(figure['counts']) * 0.5):
                # If we have numeric values and it's likely a numerical question, sort by values
                sorted_items = [(key, figure['counts'][key]) for key in figure['counts'].keys()]
                sorted_items.sort(key=lambda x: numeric_values.get(x[0], float('inf')))
            else:
                # Otherwise sort by frequency (descending)
                sorted_items = sorted(figure['counts'].items(), key=lambda x: x[1], reverse=True)
            
            for option, count in sorted_items:
                f.write(f"  {option}: {count}\n")
        
        f.write("\n\nOpen-Ended Questions:\n")
        f.write("---------------------\n")
        for q_id, data in results['text_responses'].items():
            # Use question text instead of ID
            f.write(f"\nQuestion: {data['question_text']}\n\n")
            f.write("Responses:\n")
            for i, response in enumerate(data['responses'], 1):
                f.write(f"  {i}. {response}\n")
    
    return report_path

def export_text_responses_to_csv(results, output_dir):
    """
    Export all text responses to a CSV file.
    
    Args:
        results: Results dictionary containing text_responses
        output_dir: Directory to save the CSV file
    
    Returns:
        Path to the saved CSV file
    """
    import csv
    
    # Prepare the output file
    csv_path = os.path.join(output_dir, 'text_responses.csv')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        # Create CSV writer
        csv_writer = csv.writer(csvfile)
        
        # Write header row
        csv_writer.writerow(['Question', 'Response'])
        
        # Write each response
        for q_id, data in results['text_responses'].items():
            question_text = data['question_text']
            
            for response in data['responses']:
                if response:  # Only write non-empty responses
                    csv_writer.writerow([question_text, response])
    
    print(f"Text responses exported to: {csv_path}")
    return csv_path

def load_filter_rules(filter_file: str) -> List[Dict[str, Any]]:
    """
    Load user filtering rules from a CSV file.
    
    Args:
        filter_file: Path to the CSV file containing filtering rules
        
    Returns:
        List of dictionaries with filtering rules
    """
    try:
        rules_df = pd.read_csv(filter_file)
        required_columns = ['question_id', 'operation', 'value']
        
        # Check that required columns exist
        missing_columns = [col for col in required_columns if col not in rules_df.columns]
        if missing_columns:
            print(f"Error: Filter rules CSV is missing required columns: {missing_columns}")
            return []
        
        # Parse the rules
        rules = []
        for _, row in rules_df.iterrows():
            rule = {
                'question_id': row['question_id'],
                'operation': row['operation'],
                'value': row['value'],
            }
            
            # Special handling for "in_list" operation which needs a list value
            if rule['operation'] == 'in_list' and isinstance(rule['value'], str):
                try:
                    # Try to parse as JSON list
                    import json
                    rule['value'] = json.loads(rule['value'].replace("'", '"'))
                except:
                    # Fallback to simple string splitting
                    values = rule['value'].strip('[]').split(',')
                    rule['value'] = [v.strip() for v in values]
            
            # Add reason if available
            if 'reason' in row and not pd.isna(row['reason']):
                rule['reason'] = row['reason']
            
            rules.append(rule)
        
        print(f"Loaded {len(rules)} filtering rules from {filter_file}")
        return rules
    
    except Exception as e:
        print(f"Error loading filter rules: {e}")
        return []

# Update main function to use the filter file
def main():
    args = parse_args()
    
    # Create base output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load filter rules if specified
    exclusion_rules = []
    if args.filter:
        if args.filter_file:
            exclusion_rules = load_filter_rules(args.filter_file)
        else:
            # Default rules if no file specified
            exclusion_rules = [
                {
                    'question_id': 'attention_check',
                    'operation': 'not_equals',
                    'value': 4,
                    'reason': 'Failed attention check'
                }
            ]
            print("Using default filter rules. Specify --filter-file for custom rules.")
    
    if args.file:
        # Process a single file
        print(f"Analyzing survey data from file: {args.file}")
        results = analyze_survey(args.file, args.output, args.format, args.dpi)
        
        print("\nAnalysis complete!")
        print(f"Total questions analyzed: {results['total_questions']}")
        print(f"Multiple choice questions: {results['multiple_choice_questions']}")
        print(f"Open-ended questions: {results['open_ended_questions']}")
        print(f"Generated {len(results['figures'])} visualizations")
        print(f"Report saved to: {results['report_path']}")
        print(f"All output saved to directory: {args.output}")
    
    elif args.dir:
        # Process all CSV files in the directory as one combined dataset
        print(f"Analyzing all survey data files in directory: {args.dir} as one combined dataset")
        combined_results = process_directory(args.dir, args.output, args.format, args.dpi)
        
        if combined_results:
            print("\nCombined analysis complete!")
            print(f"Processed {combined_results['total_files']} files into one dataset")
            print(f"Total questions analyzed: {combined_results['total_questions']}")
            print(f"Multiple choice questions: {combined_results['multiple_choice_questions']}")
            print(f"Open-ended questions: {combined_results['open_ended_questions']}")
            print(f"Generated {len(combined_results['figures'])} visualizations")
            print(f"Report saved to: {combined_results['report_path']}")
            print(f"All output saved to directory: {args.output}")
            
            # Add information about source files to the report
            with open(combined_results['report_path'], 'a') as f:
                f.write("\n\nSource Files:\n")
                f.write("-------------\n")
                for idx, file_name in enumerate(combined_results['source_files'], 1):
                    f.write(f"{idx}. {file_name}\n")

if __name__ == "__main__":
    main()