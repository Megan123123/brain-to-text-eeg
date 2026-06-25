# Brain-to-Text EEG Neural Decoding

課程：資料探勘（1141 Data Mining）

## 專案簡介

本專案基於 [Brain-to-Text 競賽](https://eval.ai/web/challenges/challenge-page/2099/overview) 資料集，目標是將 256 通道皮質神經訊號解碼為文字，協助 ALS 患者透過神經義肢進行溝通。

## 資料集

- **來源**：Card et al. (2024) - *An accurate and rapidly calibrating speech neuroprosthesis*
- **受試者**：45 歲 ALS 患者
- **設備**：256 個皮質內電極（左腹側中央前回）
- **資料量**：11,000+ 次試驗，收集期間長達 20 個月
- **特徵**：Threshold crossings + spike band power（共 512 個特徵）

## 主要效能指標（原始研究）

| 指標 | 數值 |
|------|------|
| 第一天準確率（50詞彙） | 99.6% |
| 第二天準確率（125,000詞彙） | 90.2% |
| 長期持續準確率 | 97.5%（8.4 個月） |
| 解碼速度 | ~32 字/分鐘 |

## 專案結構

```
.
├── brain_to_text_analysis.py   # 主要分析腳本
├── midterm/                    # 期中報告相關
│   ├── brain_to_text_analysis.py
│   ├── competition_summary.md
│   ├── progress_report_analysis.py
│   ├── progress_report_guide.md
│   ├── requirements.txt
│   └── ...
├── Lab1/                       # Lab 作業
│   └── DM2025Labs/
├── *.png                       # 分析結果視覺化圖表
└── 2508.00758v1.pdf            # 相關論文
```

## 視覺化分析

- 神經訊號時間序列 (`neural_signals_timeseries.png`)
- 頻率分析 (`frequency_analysis.png`, `frequency_analysis_psd.png`)
- 電極相關性 (`electrode_correlations.png`, `channel_correlations.png`)
- 訊號分布與正規化效果 (`normalization_effects.png`)
- 濾波效果比較 (`filtering_effects.png`)

## 技術方法

- **前處理**：訊號濾波、Z-score 正規化、特徵工程
- **建模**：RNN / LSTM / Transformer 架構
- **評估指標**：Word Error Rate (WER)、Character Error Rate (CER)

## 環境設置

```bash
pip install -r midterm/requirements.txt
```
