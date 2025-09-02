#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
將判決書句子轉換為 BERT 嵌入
讀取 judgments_sentences_{00-15}.csv 並生成對應的 sentences_embeddings_{00-15}.csv
"""

import os
import csv
import pandas as pd
import torch
from transformers import (
    BertTokenizerFast,
    AutoModel,
)
from pathlib import Path
import argparse
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 定義資料夾路徑
DATA_DIR = "/Users/hochienhuang/JRAR/projects/Disputability/data/Dataset"

class SentenceEmbeddingProcessor:
    def __init__(self, max_length=512):
        """
        初始化處理器
        
        Args:
            max_length (int): BERT 輸入的最大長度
        """
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 為每個線程創建獨立的 tokenizer 和 model
        self._local = threading.local()
        
        print("🔄 正在載入 BERT 中文模型和分詞器...")
        # 主線程中預載入一次
        self.tokenizer = BertTokenizerFast.from_pretrained('ckiplab/bert-base-chinese')
        self.model = AutoModel.from_pretrained('ckiplab/bert-base-chinese')
        self.model.to(self.device)
        self.model.eval()  # 設置為評估模式
        
        # 獲取模型的隱藏層維度
        self.hidden_size = self.model.config.hidden_size
        
        print(f"✅ 模型載入完成！")
        print(f"📐 設備: {self.device}")
        print(f"📏 嵌入維度: {self.hidden_size}")
    
    def get_tokenizer(self):
        """獲取線程本地的 tokenizer"""
        if not hasattr(self._local, 'tokenizer'):
            self._local.tokenizer = BertTokenizerFast.from_pretrained('ckiplab/bert-base-chinese')
        return self._local.tokenizer
    
    def get_model(self):
        """獲取線程本地的 model"""
        if not hasattr(self._local, 'model'):
            self._local.model = AutoModel.from_pretrained('ckiplab/bert-base-chinese')
            self._local.model.to(self.device)
            self._local.model.eval()
        return self._local.model
    
    def process_sentence(self, sentence):
        """
        將句子轉換為固定長度的 input_ids 和 attention_mask
        
        Args:
            sentence (str): 輸入句子
            
        Returns:
            tuple: (input_ids_list, attention_mask_list)
        """
        # 使用線程本地的 tokenizer
        tokenizer = self.get_tokenizer()
        
        # 使用 BERT 分詞器進行編碼
        encoded = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # 轉換為 Python list (方便存入 CSV)
        input_ids = encoded['input_ids'][0].tolist()
        attention_mask = encoded['attention_mask'][0].tolist()
        
        return input_ids, attention_mask
    
    def check_progress(self, output_file):
        """
        檢查已處理的進度
        
        Args:
            output_file (str): 輸出檔案路徑
            
        Returns:
            int: 已處理的行數
        """
        if not os.path.exists(output_file):
            return 0
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                # 計算行數（減去標題行）
                line_count = sum(1 for line in f) - 1
                return max(0, line_count)
        except Exception as e:
            print(f"⚠️ 檢查進度時發生錯誤: {e}")
            return 0
    
    def process_file(self, input_file, output_file, resume=True):
        """
        處理單個 CSV 檔案
        
        Args:
            input_file (str): 輸入檔案路徑
            output_file (str): 輸出檔案路徑
            resume (bool): 是否從上次中斷處繼續
        """
        if not os.path.exists(input_file):
            print(f"❌ 輸入檔案不存在: {input_file}")
            return
        
        # 檢查進度
        start_row = 0
        if resume:
            start_row = self.check_progress(output_file)
            if start_row > 0:
                print(f"📍 從第 {start_row + 1} 行繼續處理 {input_file}")
        
        # 讀取輸入檔案
        print(f"📖 正在讀取 {input_file}...")
        try:
            df = pd.read_csv(input_file)
            total_rows = len(df)
            print(f"   總共 {total_rows:,} 行數據")
            
            if start_row >= total_rows:
                print(f"✅ {input_file} 已完全處理完成")
                return
                
        except Exception as e:
            print(f"❌ 讀取檔案錯誤: {e}")
            return
        
        # 檢查必要欄位
        required_columns = ['檔名', '句子編號', '句子內容', 'DISPUTABILITY']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ 缺少必要欄位: {missing_columns}")
            return
        
        # 準備輸出檔案
        file_mode = 'a' if start_row > 0 else 'w'
        write_header = start_row == 0
        
        print(f"💾 開始處理並寫入 {output_file}...")
        
        try:
            with open(output_file, file_mode, newline='', encoding='utf-8') as csvfile:
                # 定義 CSV 欄位
                fieldnames = [
                    '檔名', '句子編號', 'DISPUTABILITY', 
                    'input_ids', 'attention_mask'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # 寫入標題行
                if write_header:
                    writer.writeheader()
                
                # 處理數據
                processed_count = 0
                error_count = 0
                
                # 使用 tqdm 顯示進度條
                for idx in tqdm(range(start_row, total_rows), 
                              desc=f"處理 {os.path.basename(input_file)}", 
                              initial=start_row, 
                              total=total_rows):
                    
                    try:
                        row = df.iloc[idx]
                        sentence = str(row['句子內容'])
                        
                        # 跳過空句子
                        if not sentence or sentence.strip() == '':
                            continue
                        
                        # 生成嵌入
                        input_ids, attention_mask = self.process_sentence(sentence)
                        
                        # 寫入結果
                        writer.writerow({
                            '檔名': row['檔名'],
                            '句子編號': row['句子編號'],
                            'DISPUTABILITY': row['DISPUTABILITY'],
                            'input_ids': str(input_ids),  # 轉換為字串存儲
                            'attention_mask': str(attention_mask)
                        })
                        
                        processed_count += 1
                        
                        # 每處理 1000 行強制刷新
                        if processed_count % 1000 == 0:
                            csvfile.flush()
                            
                    except Exception as e:
                        error_count += 1
                        if error_count <= 5:  # 只顯示前5個錯誤
                            print(f"⚠️ 處理第 {idx + 1} 行時發生錯誤: {e}")
                
                print(f"✅ 完成處理 {input_file}")
                print(f"   成功處理: {processed_count:,} 行")
                print(f"   錯誤數量: {error_count}")
                
        except Exception as e:
            print(f"❌ 寫入檔案錯誤: {e}")
            
    def process_file_wrapper(self, file_id, resume=True):
        """
        檔案處理包裝函數，用於多執行緒
        
        Args:
            file_id (str): 檔案 ID (00-15)
            resume (bool): 是否從上次中斷處繼續
            
        Returns:
            dict: 處理結果
        """
        input_file = os.path.join(DATA_DIR, f"judgments_sentences_{file_id}.csv")
        output_file = os.path.join(DATA_DIR, f"sentences_embeddings_{file_id}.csv")
        
        start_time = time.time()
        
        try:
            if not os.path.exists(input_file):
                return {
                    'file_id': file_id,
                    'status': 'skipped',
                    'message': f'檔案不存在: {input_file}',
                    'duration': 0
                }
            
            # 執行處理
            self.process_file(input_file, output_file, resume)
            
            duration = time.time() - start_time
            return {
                'file_id': file_id,
                'status': 'success',
                'message': f'處理完成',
                'duration': duration
            }
            
        except Exception as e:
            duration = time.time() - start_time
            return {
                'file_id': file_id,
                'status': 'error',
                'message': f'錯誤: {str(e)}',
                'duration': duration
            }

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='將判決書句子轉換為 BERT 嵌入')
    parser.add_argument('--file_id', type=str, help='處理特定檔案 ID (00-15)')
    parser.add_argument('--max_length', type=int, default=512, help='BERT 輸入最大長度')
    parser.add_argument('--no_resume', action='store_true', help='不從中斷處繼續，重新開始')
    parser.add_argument('--threads', type=int, default=4, help='執行緒數量 (預設: 4)')
    
    args = parser.parse_args()
    
    print("🚀 多執行緒句子嵌入處理器")
    print("=" * 60)
    
    # 初始化處理器
    processor = SentenceEmbeddingProcessor(max_length=args.max_length)
    
    # 確定要處理的檔案
    if args.file_id:
        # 處理單個檔案
        file_ids = [args.file_id]
    else:
        # 處理所有檔案 (00-15)
        file_ids = [f"{i:02d}" for i in range(16)]
    
    print(f"📂 準備處理檔案: {file_ids}")
    print(f"🔧 設定:")
    print(f"   最大長度: {args.max_length}")
    print(f"   執行緒數量: {args.threads}")
    print(f"   繼續模式: {'否' if args.no_resume else '是'}")
    print("-" * 60)
    
    # 處理檔案
    total_files = len(file_ids)
    start_time = time.time()
    
    if args.file_id:
        # 單檔案處理（不使用多執行緒）
        print(f"\n📄 處理單個檔案: {args.file_id}")
        
        input_file = os.path.join(DATA_DIR, f"judgments_sentences_{args.file_id}.csv")
        output_file = os.path.join(DATA_DIR, f"sentences_embeddings_{args.file_id}.csv")
        
        if not os.path.exists(input_file):
            print(f"❌ 檔案不存在: {input_file}")
            return
        
        processor.process_file(
            input_file=input_file,
            output_file=output_file,
            resume=not args.no_resume
        )
    else:
        # 多檔案多執行緒處理
        print(f"\n🧵 使用 {args.threads} 個執行緒同時處理 {total_files} 個檔案")
        
        # 建立執行緒池
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            # 提交所有任務
            future_to_file = {
                executor.submit(processor.process_file_wrapper, file_id, not args.no_resume): file_id
                for file_id in file_ids
                if os.path.exists(os.path.join(DATA_DIR, f"judgments_sentences_{file_id}.csv"))
            }
            
            # 處理完成的任務
            completed = 0
            results = []
            
            for future in as_completed(future_to_file):
                file_id = future_to_file[future]
                completed += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    status_icon = "✅" if result['status'] == 'success' else "⚠️" if result['status'] == 'skipped' else "❌"
                    duration_str = f"{result['duration']:.1f}s" if result['duration'] > 0 else "0s"
                    
                    print(f"{status_icon} [{completed:2d}/{len(future_to_file):2d}] {file_id}: {result['message']} ({duration_str})")
                    
                except Exception as exc:
                    print(f"❌ [{completed:2d}/{len(future_to_file):2d}] {file_id}: 執行時發生錯誤: {exc}")
                    results.append({
                        'file_id': file_id,
                        'status': 'error',
                        'message': f'執行錯誤: {str(exc)}',
                        'duration': 0
                    })
        
        # 顯示總結
        total_duration = time.time() - start_time
        success_count = sum(1 for r in results if r['status'] == 'success')
        error_count = sum(1 for r in results if r['status'] == 'error')
        skipped_count = sum(1 for r in results if r['status'] == 'skipped')
        
        print("\n" + "=" * 60)
        print(f"📊 處理總結:")
        print(f"   ✅ 成功: {success_count} 個檔案")
        print(f"   ❌ 錯誤: {error_count} 個檔案")
        print(f"   ⚠️ 跳過: {skipped_count} 個檔案")
        print(f"   ⏱️ 總耗時: {total_duration:.1f} 秒")
        print(f"   🚀 平均每檔案: {total_duration/len(results):.1f} 秒" if results else "")
    
    print(f"\n🎯 全部處理完成！")
    print(f"📁 輸出檔案: {DATA_DIR}/sentences_embeddings_*.csv")

if __name__ == "__main__":
    main()
