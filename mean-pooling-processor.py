#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡化版本：直接從 input_ids 計算 mean pooling（基於 token embeddings）
讀取 sentences_embeddings_{00-15}.csv 並生成對應的 pooled_embeddings_{00-15}.csv
"""

import os
import csv
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizerFast, AutoModel
from pathlib import Path
import argparse
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import ast
import warnings
warnings.filterwarnings('ignore')

# 定義資料夾路徑
DATA_DIR = "/Users/hochienhuang/JRAR/projects/Disputability/data/Dataset"

class SimpleMeanPoolingProcessor:
    def __init__(self):
        """
        初始化簡化 Mean Pooling 處理器
        使用 BERT embeddings 進行 mean pooling
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("🔄 正在載入 BERT 中文模型...")
        # 載入 tokenizer 和 model
        self.tokenizer = BertTokenizerFast.from_pretrained('ckiplab/bert-base-chinese')
        self.model = AutoModel.from_pretrained('ckiplab/bert-base-chinese')
        self.model.to(self.device)
        self.model.eval()
        
        # 獲取 embedding layer
        self.embedding_layer = self.model.embeddings.word_embeddings
        
        # 獲取模型的隱藏層維度
        self.hidden_size = self.model.config.hidden_size
        self.max_length = 512  # BERT 最大序列長度
        
        print(f"✅ 模型載入完成！")
        print(f"📐 設備: {self.device}")
        print(f"📏 嵌入維度: {self.hidden_size}")
    
    def simple_mean_pooling(self, input_ids, attention_mask):
        """
        使用 token embeddings 進行簡單的 mean pooling
        
        Args:
            input_ids (list): BERT token IDs
            attention_mask (list): 注意力遮罩
            
        Returns:
            np.ndarray: Mean pooled 嵌入向量 (維度: hidden_size)
        """
        try:
            # 轉換為 tensor
            input_ids_tensor = torch.tensor(input_ids, device=self.device)
            attention_mask_tensor = torch.tensor(attention_mask, device=self.device)
            
            # 獲取 token embeddings
            with torch.no_grad():
                token_embeddings = self.embedding_layer(input_ids_tensor)  # [seq_len, hidden_size]
                
                # 應用 attention mask
                masked_embeddings = token_embeddings * attention_mask_tensor.unsqueeze(-1)  # [seq_len, hidden_size]
                
                # 計算有效 token 數量
                valid_token_count = attention_mask_tensor.sum()
                
                if valid_token_count > 0:
                    # Mean pooling
                    mean_pooled = masked_embeddings.sum(dim=0) / valid_token_count  # [hidden_size]
                else:
                    # 如果沒有有效 token，返回零向量
                    mean_pooled = torch.zeros(self.hidden_size, device=self.device)
                
                # 轉換為 numpy array
                embedding = mean_pooled.cpu().numpy()
                
                return embedding
                
        except Exception as e:
            print(f"⚠️ Mean pooling 處理錯誤: {e}")
            # 返回零向量作為備用
            return np.zeros(self.hidden_size, dtype=np.float32)
    
    def batch_simple_mean_pooling(self, input_ids_batch, attention_mask_batch):
        """
        批量處理 token embeddings mean pooling
        
        Args:
            input_ids_batch (list): 批量 BERT token IDs
            attention_mask_batch (list): 批量注意力遮罩
            
        Returns:
            np.ndarray: 批量 Mean pooled 嵌入向量
        """
        try:
            embeddings = []
            for input_ids, attention_mask in zip(input_ids_batch, attention_mask_batch):
                embedding = self.simple_mean_pooling(input_ids, attention_mask)
                embeddings.append(embedding)
            
            return np.array(embeddings)
            
        except Exception as e:
            print(f"⚠️ 批量 Mean pooling 處理錯誤: {e}")
            batch_size = len(input_ids_batch)
            return np.zeros((batch_size, self.hidden_size), dtype=np.float32)
    
    def parse_list_string(self, list_str):
        """
        解析字串形式的 list
        
        Args:
            list_str (str): 字串形式的 list, 例如 "[1, 2, 3]"
            
        Returns:
            list: 解析後的 list
        """
        try:
            return ast.literal_eval(list_str)
        except Exception as e:
            print(f"⚠️ 解析 list 字串錯誤: {e}")
            return []
    
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
    
    def process_file(self, input_file, output_file, batch_size=100, resume=True):
        """
        處理單個 CSV 檔案，進行簡化 mean pooling
        
        Args:
            input_file (str): 輸入檔案路徑
            output_file (str): 輸出檔案路徑
            batch_size (int): 批量大小
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
        required_columns = ['檔名', '句子編號', 'DISPUTABILITY', 'input_ids', 'attention_mask']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ 缺少必要欄位: {missing_columns}")
            return
        
        # 準備輸出檔案
        file_mode = 'a' if start_row > 0 else 'w'
        write_header = start_row == 0
        
        print(f"💾 開始批量處理並寫入 {output_file}...")
        
        try:
            with open(output_file, file_mode, newline='', encoding='utf-8') as csvfile:
                # 定義 CSV 欄位 - 使用單一 embedding 欄位
                fieldnames = [
                    '檔名', '句子編號', 'DISPUTABILITY', 'embedding'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # 寫入標題行
                if write_header:
                    writer.writeheader()
                
                # 處理數據
                processed_count = 0
                error_count = 0
                
                # 計算批量數量
                remaining_rows = total_rows - start_row
                num_batches = (remaining_rows + batch_size - 1) // batch_size
                
                # 使用 tqdm 顯示進度條
                with tqdm(total=remaining_rows, 
                         desc=f"簡化處理 {os.path.basename(input_file)}", 
                         initial=0) as pbar:
                    
                    for batch_idx in range(num_batches):
                        batch_start = start_row + batch_idx * batch_size
                        batch_end = min(batch_start + batch_size, total_rows)
                        
                        # 準備批量數據
                        batch_input_ids = []
                        batch_attention_masks = []
                        batch_metadata = []
                        
                        valid_indices = []
                        
                        for idx in range(batch_start, batch_end):
                            try:
                                row = df.iloc[idx]
                                
                                # 解析 input_ids 和 attention_mask
                                input_ids = self.parse_list_string(row['input_ids'])
                                attention_mask = self.parse_list_string(row['attention_mask'])
                                
                                # 檢查長度
                                if len(input_ids) != self.max_length or len(attention_mask) != self.max_length:
                                    error_count += 1
                                    if error_count <= 5:
                                        print(f"⚠️ 第 {idx + 1} 行長度不符: input_ids={len(input_ids)}, attention_mask={len(attention_mask)}")
                                    continue
                                
                                batch_input_ids.append(input_ids)
                                batch_attention_masks.append(attention_mask)
                                batch_metadata.append({
                                    '檔名': row['檔名'],
                                    '句子編號': row['句子編號'],
                                    'DISPUTABILITY': row['DISPUTABILITY']
                                })
                                valid_indices.append(idx)
                                
                            except Exception as e:
                                error_count += 1
                                if error_count <= 5:
                                    print(f"⚠️ 處理第 {idx + 1} 行時發生錯誤: {e}")
                        
                        # 如果批量中有有效數據，進行處理
                        if batch_input_ids:
                            try:
                                # 進行批量簡化 mean pooling
                                embeddings = self.batch_simple_mean_pooling(batch_input_ids, batch_attention_masks)
                                
                                # 寫入結果
                                for i, (embedding, metadata) in enumerate(zip(embeddings, batch_metadata)):
                                    output_row = metadata.copy()
                                    
                                    # 將嵌入向量轉換為 list 字串格式
                                    embedding_list = embedding.tolist()
                                    output_row['embedding'] = str(embedding_list)
                                    
                                    writer.writerow(output_row)
                                    processed_count += 1
                                
                                # 強制刷新
                                csvfile.flush()
                                
                            except Exception as e:
                                error_count += len(batch_input_ids)
                                print(f"⚠️ 批量處理錯誤: {e}")
                        
                        # 更新進度條
                        pbar.update(batch_end - batch_start)
                
                print(f"✅ 完成處理 {input_file}")
                print(f"   成功處理: {processed_count:,} 行")
                print(f"   錯誤數量: {error_count}")
                
        except Exception as e:
            print(f"❌ 寫入檔案錯誤: {e}")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='對 BERT input_ids 進行簡化 mean pooling')
    parser.add_argument('--file_id', type=str, help='處理特定檔案 ID (00-15)')
    parser.add_argument('--batch_size', type=int, default=100, help='批量大小 (預設: 100)')
    parser.add_argument('--no_resume', action='store_true', help='不從中斷處繼續，重新開始')
    
    args = parser.parse_args()
    
    print("🚀 簡化 Mean Pooling 處理器")
    print("=" * 60)
    
    # 初始化處理器
    processor = SimpleMeanPoolingProcessor()
    
    # 確定要處理的檔案
    if args.file_id:
        # 處理單個檔案
        file_ids = [args.file_id]
    else:
        # 處理所有檔案 (00-15)
        file_ids = [f"{i:02d}" for i in range(16)]
    
    print(f"📂 準備處理檔案: {file_ids}")
    print(f"🔧 設定:")
    print(f"   批量大小: {args.batch_size}")
    print(f"   繼續模式: {'否' if args.no_resume else '是'}")
    print(f"   輸出維度: {processor.hidden_size}")
    print("-" * 60)
    
    # 處理檔案
    start_time = time.time()
    
    for file_id in file_ids:
        print(f"\n📄 處理檔案: {file_id}")
        
        input_file = os.path.join(DATA_DIR, f"sentences_embeddings_{file_id}.csv")
        output_file = os.path.join(DATA_DIR, f"simple_pooled_embeddings_{file_id}.csv")
        
        if not os.path.exists(input_file):
            print(f"⚠️ 檔案不存在，跳過: {input_file}")
            continue
        
        processor.process_file(
            input_file=input_file,
            output_file=output_file,
            batch_size=args.batch_size,
            resume=not args.no_resume
        )
    
    total_duration = time.time() - start_time
    
    print("\n" + "=" * 60)
    print(f"🎯 全部處理完成！")
    print(f"📁 輸出檔案: {DATA_DIR}/simple_pooled_embeddings_*.csv")
    print(f"📏 每個句子的嵌入維度: {processor.hidden_size}")
    print(f"⏱️ 總耗時: {total_duration:.1f} 秒")

if __name__ == "__main__":
    main()
