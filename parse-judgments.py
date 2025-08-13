#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從判決書 JSON 檔案中提取主文部分並轉換為 CSV 格式
使用多線程處理，輸出16個分割檔案
"""

import os
import re
import json
import csv
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

def extract_main_text(jfull_content):
    """
    從 JFULL 內容中提取主文部分（在"主文"和"中華民國...年月日"之間的文字）
    
    Args:
        jfull_content (str): JFULL 屬性的完整內容
        
    Returns:
        str: 提取的主文內容，如果沒有找到則返回空字串
    """
    # 定義要匹配的正則表達式模式
    pattern1 = r'^\s*主\s*文\s*$'  # 匹配 "主文"
    pattern2 = r'^\s*中\s*華\s*民\s*國.*年.*月.*日\s*$'  # 匹配 "中華民國...年...月...日"
    
    # 編譯正則表達式，使用 MULTILINE 模式
    regex1 = re.compile(pattern1, re.MULTILINE)
    regex2 = re.compile(pattern2, re.MULTILINE)
    
    # 找到主文的位置
    main_match = regex1.search(jfull_content)
    if not main_match:
        return ""
    
    # 找到日期的位置
    date_match = regex2.search(jfull_content)
    if not date_match:
        return ""
    
    # 確保主文在日期之前
    if main_match.end() >= date_match.start():
        return ""
    
    # 提取主文部分（從主文結束到日期開始之間的內容）
    main_text = jfull_content[main_match.end():date_match.start()]
    
    return main_text.strip()

def clean_and_split_text(text):
    """
    清理文字並按句號分割成句子
    
    Args:
        text (str): 原始文字
        
    Returns:
        list: 句子列表
    """
    if not text:
        return []
    
    # 移除所有換行符號和多餘空格
    cleaned_text = re.sub(r'\s+', '', text)
    
    # 按句號分割
    sentences = cleaned_text.split('。')
    
    # 過濾掉空句子
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    return sentences

def process_json_files(input_dir, output_prefix, num_threads=16):
    """
    使用多線程處理所有 JSON 檔案並輸出多個 CSV 檔案
    
    Args:
        input_dir (str): 輸入目錄路徑
        output_prefix (str): 輸出 CSV 檔案前綴
        num_threads (int): 線程數量，預設為16
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"錯誤：目錄 {input_path} 不存在")
        return
    
    # 獲取所有 JSON 檔案
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"在 {input_path} 中沒有找到 JSON 檔案")
        return
    
    print(f"找到 {len(json_files)} 個 JSON 檔案")
    print(f"使用 {num_threads} 個線程處理")
    
    # 分割檔案列表
    files_per_thread = math.ceil(len(json_files) / num_threads)
    file_chunks = []
    
    for i in range(num_threads):
        start_idx = i * files_per_thread
        end_idx = min((i + 1) * files_per_thread, len(json_files))
        if start_idx < len(json_files):
            chunk = json_files[start_idx:end_idx]
            file_chunks.append((i, chunk))
    
    print(f"分割為 {len(file_chunks)} 個工作塊")
    
    # 使用線程池處理
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有工作
        future_to_chunk = {
            executor.submit(process_file_chunk, chunk_id, files, output_prefix): chunk_id 
            for chunk_id, files in file_chunks
        }
        
        # 收集結果
        total_processed = 0
        total_sentences = 0
        total_errors = 0
        
        for future in as_completed(future_to_chunk):
            chunk_id = future_to_chunk[future]
            try:
                processed, sentences, errors = future.result()
                total_processed += processed
                total_sentences += sentences
                total_errors += errors
                print(f"線程 {chunk_id} 完成：處理 {processed} 個檔案，{sentences} 個句子")
            except Exception as e:
                print(f"線程 {chunk_id} 發生錯誤: {e}")
                total_errors += 1
    
    print(f"\n🎯 全部處理完成！")
    print(f"總檔案數: {len(json_files)}")
    print(f"成功處理: {total_processed}")
    print(f"錯誤檔案: {total_errors}")
    print(f"總句子數: {total_sentences}")
    print(f"輸出檔案: {output_prefix}_*.csv")

def process_file_chunk(chunk_id, json_files, output_prefix):
    """
    處理一個檔案塊
    
    Args:
        chunk_id (int): 塊編號
        json_files (list): 要處理的 JSON 檔案列表
        output_prefix (str): 輸出檔案前綴
        
    Returns:
        tuple: (處理的檔案數, 句子數, 錯誤數)
    """
    output_file = f"{output_prefix}_{chunk_id:02d}.csv"
    
    processed_files = 0
    total_sentences = 0
    error_files = 0
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 寫入標題行
        writer.writerow(['檔名', '句子編號', '句子內容', 'DISPUTABILITY'])
        
        for i, json_file in enumerate(json_files, 1):
            try:
                # 每處理 100 個檔案顯示進度
                if i % 100 == 0:
                    print(f"線程 {chunk_id}: {i}/{len(json_files)} ({i/len(json_files)*100:.1f}%)")
                
                # 讀取 JSON 檔案
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 檢查是否有 JFULL 屬性
                if 'JFULL' not in data:
                    continue
                
                # 提取主文部分
                main_text = extract_main_text(data['JFULL'])
                
                if not main_text:
                    continue
                
                # 清理並分割文字
                sentences = clean_and_split_text(main_text)
                
                if not sentences:
                    continue
                
                # 獲取 DISPUTABILITY
                disputability = data.get('DISPUTABILITY', '')
                
                # 寫入 CSV
                filename = json_file.stem  # 不含副檔名的檔名
                for sentence_num, sentence in enumerate(sentences, 1):
                    writer.writerow([filename, sentence_num, sentence, disputability])
                
                processed_files += 1
                total_sentences += len(sentences)
                
            except Exception as e:
                error_files += 1
                if error_files <= 3:  # 每個線程只顯示前3個錯誤
                    print(f"線程 {chunk_id} 錯誤：處理檔案 {json_file.name} 時發生錯誤: {e}")
    
    return processed_files, total_sentences, error_files

def main():
    """主函數"""
    print("🔍 判決書主文提取工具 (多線程版本)")
    print("=" * 60)
    
    # 設定輸入和輸出路徑
    input_directory = "../data/filtered_judgments2"
    output_prefix = "judgments_sentences"
    num_threads = 16
    
    print(f"輸入目錄: {input_directory}")
    print(f"輸出檔案前綴: {output_prefix}")
    print(f"線程數量: {num_threads}")
    print(f"將產生: {output_prefix}_00.csv 到 {output_prefix}_{num_threads-1:02d}.csv")
    print("-" * 60)
    
    # 開始處理
    process_json_files(input_directory, output_prefix, num_threads)
    
    print("\n✅ 多線程處理完成！")
    print(f"📁 生成的檔案格式: {output_prefix}_XX.csv (共 {num_threads} 個檔案)")
    print("📝 CSV 欄位: 檔名, 句子編號, 句子內容, DISPUTABILITY")

if __name__ == "__main__":
    main()
