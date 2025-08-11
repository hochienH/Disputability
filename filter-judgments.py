import json
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

def filter_conditions(file_path):
    # setting condtions for filtering, if qualified, return True
    # if file path not contains '行政' return False
    if '行政' not in file_path:
        return False
    # open file_path with json, if the "JFULL" attribute not contains '判決' in first 30 non-space-characters, return False
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        jfull_text = data.get("JFULL", "")
        # Check the first 30 non-space characters
        non_space_text = ''.join(c for c in jfull_text if not c.isspace())
        if '判決' not in non_space_text[:30]:
            return False
    # if all conditions are met, return True
    return True

def process_rar_file(rar_file_info):
    """Process a single RAR file"""
    file_name, file_path = rar_file_info
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract the current .rar file
        os.system(f'unrar x "{file_path}" "{temp_dir}"')
        
        processed_count = 0
        # Filter all the json files in this temporary directory
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.json'):
                    json_file_path = os.path.join(root, file)
                    if filter_conditions(json_file_path):
                        # copy the qualified json file to '../data/filtered_judgments'
                        target_path = os.path.join('../data/filtered_judgments', file)
                        with open(json_file_path, 'r', encoding='utf-8') as src_file:
                            data = json.load(src_file)
                        with open(target_path, 'w', encoding='utf-8') as dest_file:
                            json.dump(data, dest_file, ensure_ascii=False, indent=4)
                        processed_count += 1
        
        print(f"Processed {file_name} - Found {processed_count} qualified files")
        return file_name, processed_count

def filter_judgments():
    # if '../data/filtered_judgments' not exists, create it
    if not os.path.exists('../data/filtered_judgments'):
        os.makedirs('../data/filtered_judgments')
    
    # read opendata directory path from OPENDATA_PATH.txt
    with open('OPENDATA_PATH.txt', 'r', encoding='utf-8') as f:
        opendata_path = f.read().strip()
    if not os.path.exists(opendata_path):
        print(f"Directory {opendata_path} does not exist.")
        return
    
    # Collect all RAR files
    rar_files = []
    for file_name in os.listdir(opendata_path):
        if file_name.endswith('.rar'):
            file_path = os.path.join(opendata_path, file_name)
            rar_files.append((file_name, file_path))
    
    print(f"Found {len(rar_files)} RAR files to process")

    # Process RAR files using 16 threads
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(process_rar_file, rar_files))
    
    # Summary
    total_processed = sum(count for _, count in results)
    print(f"All files processed. Total qualified files: {total_processed}")

if __name__ == "__main__":
    filter_judgments()
    print("Filtering completed. Check '../data/filtered_judgments' for results.")