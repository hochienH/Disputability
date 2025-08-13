#!/usr/bin/env python3
"""
多功能 DISPUTABILITY 處理程序
1. 為沒有 DISPUTABILITY 屬性的檔案添加該屬性
2. 重新處理 DISPUTABILITY = 0 的檔案（這些檔案在理論上不應該為 0）
合併自 add-disputability.py 和 reprocess-zero-disputability.py，以後者邏輯為主
"""

import json
import os
import time
import threading
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import sys
import argparse

# Selenium 相關導入
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import unquote

def extract_case_number(link_text):
    """
    從連結文字中提取案件字號
    例如：「新店簡易庭 112 年度 店簡 字第 1141 號判決(113.03.13)」
    提取：「112店簡1141」
    """
    # 使用正則表達式提取年度、字別、案號
    pattern = r'(\d+)\s*年度\s*([^字]+)\s*字第\s*(\d+)\s*號'
    match = re.search(pattern, link_text)
    
    if match:
        year = match.group(1)
        case_type = match.group(2).strip()
        case_number = match.group(3)
        return f"{year}{case_type}{case_number}"
    
    return None

def get_disputability(filename, driver):
    """
    取得指定檔名的爭議案件數量
    
    Args:
        filename (str): 檔案名稱
        driver: 已初始化的 WebDriver
        
    Returns:
        int: 獨一無二的字號數量，最少為 1（包含自己）
    """
    
    try:
        # 建構 URL
        url = f"https://judgment.judicial.gov.tw/FJUD/data.aspx?ty=JD&id={filename}"
        
        # 載入頁面
        driver.get(url)
        
        # 等待一下讓頁面完全載入
        time.sleep(5)  # 增加等待時間
        
        # 嘗試尋找 panel-body 元素
        links = []
        try:
            panel_body = driver.find_element(By.CLASS_NAME, "panel-body")
            links = panel_body.find_elements(By.TAG_NAME, "a")
        except NoSuchElementException:
            # 嘗試直接尋找所有 a 標籤
            all_links = driver.find_elements(By.TAG_NAME, "a")
            
            # 篩選相關連結
            for link in all_links:
                link_text = link.text.strip()
                if '號' in link_text and ('判決' in link_text or '裁定' in link_text):
                    links.append(link)
        
        # 收集所有連結資訊
        unique_case_numbers = set()
        
        # 從檔名中提取當前案件字號用於比較
        current_case_number = None
        if filename:
            # 解析檔名格式：例如 "TPBA,93,簡,420,20050413,1"
            parts = filename.split(',')
            if len(parts) >= 4:
                year = parts[1]
                case_type = parts[2]
                case_number = parts[3]
                current_case_number = f"{year}{case_type}{case_number}"
        
        for link in links:
            try:
                link_text = link.text.strip()
                link_href = link.get_attribute('href')
                
                if link_text and link_href:
                    # 提取案件字號
                    case_number = extract_case_number(link_text)
                    if case_number:
                        unique_case_numbers.add(case_number)
                    
            except Exception as e:
                continue
        
        # 檢查是否包含自己，沒有包含則+1
        result_count = len(unique_case_numbers)
        if current_case_number and current_case_number not in unique_case_numbers:
            result_count += 1
        
        # 確保最少為 1
        return max(result_count, 1)
            
    except Exception as e:
        print(f"處理 {filename} 時發生錯誤: {e}")
        return 1  # 出錯時返回 1 而不是 0

def find_zero_disputability_files(data_dir):
    """找出所有 DISPUTABILITY = 0 的檔案"""
    zero_files = []
    data_path = Path(data_dir)
    
    for json_file in data_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data.get('DISPUTABILITY') == 0:
                zero_files.append(json_file)
                
        except Exception as e:
            print(f"讀取檔案 {json_file.name} 時發生錯誤: {e}")
    
    return zero_files

def find_missing_disputability_files(data_dir):
    """找出所有沒有 DISPUTABILITY 屬性的檔案"""
    missing_files = []
    data_path = Path(data_dir)
    
    for json_file in data_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'DISPUTABILITY' not in data:
                missing_files.append(json_file)
                
        except Exception as e:
            print(f"讀取檔案 {json_file.name} 時發生錯誤: {e}")
    
    return missing_files

class DisputabilityProcessor:
    def __init__(self, data_dir="../data/filtered_judgments2", num_workers=8, mode="reprocess"):
        """
        初始化處理器
        
        Args:
            data_dir (str): 資料目錄路徑
            num_workers (int): 工作線程數量
            mode (str): 處理模式
                - "reprocess": 重新處理 DISPUTABILITY = 0 的檔案
                - "add": 為沒有 DISPUTABILITY 屬性的檔案添加該屬性
                - "all": 處理所有檔案（先添加缺失的，再重新處理為0的）
        """
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.mode = mode
        self.results_queue = queue.Queue()
        self.error_queue = queue.Queue()
    
    def get_target_files(self):
        """根據模式獲取需要處理的檔案"""
        if self.mode == "reprocess":
            return find_zero_disputability_files(self.data_dir)
        elif self.mode == "add":
            return find_missing_disputability_files(self.data_dir)
        elif self.mode == "all":
            # 先處理缺失的，再處理為0的
            missing_files = find_missing_disputability_files(self.data_dir)
            zero_files = find_zero_disputability_files(self.data_dir)
            return missing_files + zero_files
        else:
            raise ValueError(f"不支援的模式: {self.mode}")
    
    def get_all_json_files(self):
        """獲取所有 JSON 檔案的路徑"""
        json_files = list(self.data_dir.glob("*.json"))
        print(f"找到 {len(json_files)} 個 JSON 檔案")
        return json_files
    def __init__(self, data_dir="../data/filtered_judgments2", num_workers=8):
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.results_queue = queue.Queue()
        self.error_queue = queue.Queue()
        
    def extract_filename_from_json(self, json_file_path):
        """從 JSON 檔案路徑提取檔名（不含 .json 副檔名）"""
        return json_file_path.stem
    
    def process_single_file(self, json_file_path, worker_id, driver):
        """處理單個 JSON 檔案"""
        try:
            # 讀取 JSON 檔案
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 記錄原始值
            original_disputability = data.get('DISPUTABILITY', 'N/A')
            
            # 根據模式決定是否處理
            if self.mode == "add" and 'DISPUTABILITY' in data:
                return {
                    'file': json_file_path.name,
                    'status': 'skipped',
                    'original_disputability': original_disputability,
                    'new_disputability': original_disputability,
                    'worker_id': worker_id
                }
            
            # 提取檔名用於查詢
            filename = self.extract_filename_from_json(json_file_path)
            
            # 獲取爭議度
            new_disputability = get_disputability(filename, driver)
            
            # 更新 DISPUTABILITY 屬性
            data['DISPUTABILITY'] = new_disputability
            
            # 寫回檔案
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            return {
                'file': json_file_path.name,
                'status': 'success',
                'original_disputability': original_disputability,
                'new_disputability': new_disputability,
                'worker_id': worker_id
            }
            
        except Exception as e:
            return {
                'file': json_file_path.name,
                'status': 'error',
                'error': str(e),
                'worker_id': worker_id
            }
    
    def worker_process(self, file_chunk, worker_id):
        """工作線程處理函數"""
        print(f"Worker {worker_id} 開始處理 {len(file_chunk)} 個檔案 (模式: {self.mode})")
        
        # 為每個工作線程創建一個瀏覽器實例
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # 啟用無頭模式，背景操作
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        driver = None
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            
            processed_count = 0
            skipped_count = 0
            
            for i, json_file in enumerate(file_chunk):
                try:
                    result = self.process_single_file(json_file, worker_id, driver)
                    self.results_queue.put(result)
                    
                    if result['status'] == 'success':
                        processed_count += 1
                        if self.mode == "reprocess":
                            print(f"Worker {worker_id}: {json_file.name} - {result['original_disputability']} → {result['new_disputability']}")
                    elif result['status'] == 'skipped':
                        skipped_count += 1
                    
                    # 每處理指定數量的檔案報告一次進度
                    report_interval = 5 if self.mode == "reprocess" else 10
                    if (i + 1) % report_interval == 0:
                        print(f"Worker {worker_id}: 已處理 {i + 1}/{len(file_chunk)} 個檔案 (處理: {processed_count}, 跳過: {skipped_count})")
                        
                except Exception as e:
                    error_info = {
                        'file': json_file.name,
                        'worker_id': worker_id,
                        'error': str(e)
                    }
                    self.error_queue.put(error_info)
            
            print(f"Worker {worker_id} 完成所有檔案處理 (總處理: {processed_count}, 總跳過: {skipped_count})")
            
        finally:
            if driver:
                driver.quit()
    
    def split_files_into_chunks(self, files, num_chunks):
        """將檔案列表分割成指定數量的塊"""
        chunk_size = len(files) // num_chunks
        remainder = len(files) % num_chunks
        
        chunks = []
        start_idx = 0
        
        for i in range(num_chunks):
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_chunk_size
            
            if start_idx < len(files):
                chunks.append(files[start_idx:end_idx])
            else:
                chunks.append([])
                
            start_idx = end_idx
            
        return chunks
    
    def run(self):
        """主要執行函數"""
        start_time = time.time()
        
        # 根據模式獲取需要處理的檔案
        print(f"模式: {self.mode}")
        if self.mode == "reprocess":
            print("正在尋找 DISPUTABILITY = 0 的檔案...")
            target_files = find_zero_disputability_files(self.data_dir)
            action_description = "重新處理"
        elif self.mode == "add":
            print("正在尋找沒有 DISPUTABILITY 屬性的檔案...")
            target_files = find_missing_disputability_files(self.data_dir)
            action_description = "添加 DISPUTABILITY 屬性"
        elif self.mode == "all":
            print("正在尋找需要處理的檔案...")
            missing_files = find_missing_disputability_files(self.data_dir)
            zero_files = find_zero_disputability_files(self.data_dir)
            target_files = missing_files + zero_files
            action_description = "處理"
            print(f"找到 {len(missing_files)} 個缺失 DISPUTABILITY 屬性的檔案")
            print(f"找到 {len(zero_files)} 個 DISPUTABILITY = 0 的檔案")
        
        if not target_files:
            print(f"沒有找到需要{action_description}的檔案")
            return
        
        print(f"找到 {len(target_files)} 個檔案需要{action_description}")
        
        # 將檔案分割成塊
        file_chunks = self.split_files_into_chunks(target_files, self.num_workers)
        
        print(f"將 {len(target_files)} 個檔案分成 {self.num_workers} 個工作塊")
        for i, chunk in enumerate(file_chunks):
            print(f"  Worker {i}: {len(chunk)} 個檔案")
        
        # 使用線程池執行
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有任務
            futures = []
            for worker_id, file_chunk in enumerate(file_chunks):
                if file_chunk:  # 只處理非空的塊
                    future = executor.submit(self.worker_process, file_chunk, worker_id)
                    futures.append(future)
            
            # 等待所有任務完成
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Worker 執行過程中出現錯誤: {e}")
        
        # 收集結果
        results = []
        errors = []
        
        while not self.results_queue.empty():
            results.append(self.results_queue.get())
        
        while not self.error_queue.empty():
            errors.append(self.error_queue.get())
        
        # 統計結果
        success_count = len([r for r in results if r['status'] == 'success'])
        skipped_count = len([r for r in results if r['status'] == 'skipped'])
        error_count = len(errors) + len([r for r in results if r['status'] == 'error'])
        
        # 統計爭議度分布
        disputability_distribution = {}
        for result in results:
            if result['status'] == 'success':
                new_value = result['new_disputability']
                disputability_distribution[new_value] = disputability_distribution.get(new_value, 0) + 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n=== {action_description}結果統計 ===")
        print(f"需要{action_description}的檔案數: {len(target_files)}")
        print(f"成功{action_description}: {success_count}")
        print(f"跳過: {skipped_count}")
        print(f"錯誤: {error_count}")
        print(f"總處理時間: {total_time:.2f} 秒")
        print(f"平均每個檔案: {total_time/len(target_files):.3f} 秒")
        
        if disputability_distribution:
            print(f"\n=== 爭議度分布 ===")
            for disputability, count in sorted(disputability_distribution.items()):
                percentage = (count / success_count) * 100 if success_count > 0 else 0
                print(f"  {disputability}: {count} 檔案 ({percentage:.2f}%)")
        
        # 顯示錯誤詳情
        if errors:
            print(f"\n=== 錯誤詳情 ===")
            for error in errors[:10]:  # 只顯示前10個錯誤
                print(f"檔案: {error['file']}, Worker: {error['worker_id']}, 錯誤: {error['error']}")
            if len(errors) > 10:
                print(f"... 還有 {len(errors) - 10} 個錯誤")
        
        # 保存處理報告
        report = {
            'mode': self.mode,
            'total_target_files': len(target_files),
            'success_count': success_count,
            'skipped_count': skipped_count,
            'error_count': error_count,
            'total_time': total_time,
            'avg_time_per_file': total_time/len(target_files),
            'disputability_distribution': disputability_distribution,
            'results': results,
            'errors': errors
        }
        
        mode_name = self.mode.replace("/", "_")
        report_file = f"disputability_{mode_name}_report_{int(time.time())}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n處理報告已保存至: {report_file}")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='DISPUTABILITY 處理程序')
    parser.add_argument('--mode', 
                       choices=['reprocess', 'add', 'all'], 
                       default='reprocess',
                       help='處理模式：reprocess=重新處理為0的檔案, add=為缺失的檔案添加屬性, all=處理所有需要的檔案')
    parser.add_argument('--workers', 
                       type=int, 
                       default=8,
                       help='工作線程數量 (預設: 8)')
    parser.add_argument('--data-dir', 
                       default="../data/filtered_judgments2",
                       help='資料目錄路徑 (預設: ../data/filtered_judgments2)')
    
    args = parser.parse_args()
    
    print(f"開始處理檔案...")
    print(f"模式: {args.mode}")
    print(f"工作線程數: {args.workers}")
    print(f"資料目錄: {args.data_dir}")
    
    if args.mode == "reprocess":
        print("重新處理 DISPUTABILITY = 0 的檔案")
    elif args.mode == "add":
        print("為沒有 DISPUTABILITY 屬性的檔案添加該屬性")
    elif args.mode == "all":
        print("處理所有需要的檔案")
    
    processor = DisputabilityProcessor(
        data_dir=args.data_dir,
        num_workers=args.workers,
        mode=args.mode
    )
    processor.run()

if __name__ == "__main__":
    main()
