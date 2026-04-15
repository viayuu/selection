#!/usr/bin/env python3
"""
直接查看指定 pickle (.pkl) 文件内容的脚本
"""

import pickle
import pprint

# ====== 在这里直接写入你的 pkl 文件路径 ======
PKL_FILE_PATH = "easynco_v3_bridge/exports/nss_like_export_final_full/OVRPLtest/raw_label.pkl"   # 修改为你的实际路径
# ============================================

def main():
    print(f"正在读取文件: {PKL_FILE_PATH}")
    print("⚠️  警告: pickle 文件可能执行恶意代码，请确保来源可信！\n")

    try:
        with open(PKL_FILE_PATH, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"错误: 文件不存在 - {PKL_FILE_PATH}")
        return
    except Exception as e:
        print(f"加载失败: {e}")
        return

    print("文件内容如下：\n")
    pprint.pprint(data)

if __name__ == "__main__":
    main()