#!/usr/bin/env python3
"""
篩選 ../data/filtered_judgments 中 JTITLE 包含「稅」字的 .json 檔案
將符合條件的檔案移動到 ../data/filtered_judgments2
"""

import json
import os
import shutil
from pathlib import Path

def filter_tax_related_files():
    """篩選並移動包含「稅」字的 JTITLE 檔案"""
    
    # 設定來源和目標目錄路徑
    source_dir = Path("../data/filtered_judgments")
    target_dir = Path("../data/filtered_judgments2")
    
    # 檢查來源目錄是否存在
    if not source_dir.exists():
        print(f"錯誤：來源目錄 {source_dir} 不存在")
        return
    
    # 創建目標目錄（如果不存在）
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"目標目錄：{target_dir.absolute()}")
    
    print(f"正在掃描目錄：{source_dir.absolute()}")
    
    # 統計處理的檔案數量
    total_files = 0
    processed_files = 0
    moved_files = 0
    error_files = 0
    
    # 遍歷目錄中的所有 JSON 檔案
    for json_file in source_dir.glob("*.json"):
        total_files += 1
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 檢查是否有 JTITLE 欄位
                if 'JTITLE' in data:
                    JTITLE_value = data['JTITLE']
                    processed_files += 1
                    
                    # 檢查 JTITLE 是否包含「稅」字
                    if '稅' in JTITLE_value:
                        # 移動檔案到目標目錄
                        target_file = target_dir / json_file.name
                        shutil.move(str(json_file), str(target_file))
                        moved_files += 1
                        print(f"移動檔案：{json_file.name} (JTITLE: {JTITLE_value})")
                else:
                    print(f"警告：檔案 {json_file.name} 沒有 JTITLE 欄位")
                    
        except json.JSONDecodeError as e:
            print(f"錯誤：無法解析 JSON 檔案 {json_file.name}: {e}")
            error_files += 1
        except Exception as e:
            print(f"錯誤：處理檔案 {json_file.name} 時發生錯誤: {e}")
            error_files += 1
        
        # 每處理 1000 個檔案顯示進度
        if total_files % 1000 == 0:
            print(f"已處理 {total_files} 個檔案...")
    
    # 輸出統計結果
    print(f"\n篩選完成！")
    print(f"總共檔案數：{total_files}")
    print(f"成功處理：{processed_files}")
    print(f"移動檔案數：{moved_files}")
    print(f"錯誤檔案：{error_files}")
    print(f"剩餘檔案數：{len(list(source_dir.glob('*.json')))}")
    
    return {
        'total_files': total_files,
        'processed_files': processed_files,
        'moved_files': moved_files,
        'error_files': error_files
    }

if __name__ == "__main__":
    filter_tax_related_files()
