# 📘 第 5 週｜顧客關係管理（CRM）與市場區隔實務

> 對應教科書：CH2《機器學習》2-5（分群）節、CH4《人工智慧與商業應用》4-5（個案分析—客戶分群）節

---

## 🎯 本週學習目標

上完這個單元，各位將能夠：

1. 說出 K-means 分群演算法的運作邏輯，並能手算一個簡化的分群範例。
2. 說出 RFM 分析的三個核心指標（Recency、Frequency、Monetary）分別衡量什麼，以及為什麼顧客留存比開發新客更划算。
3. 把 RFM 數值轉換成等級分數，並依十種顧客屬性標籤，判斷一位顧客屬於哪個客群。
4. 針對不同客群，設計對應的差異化行銷方案。

---

## Part 1｜主題課程：K-means 分群與 RFM 顧客價值分析

### 一、K-means 分群演算法

K-means 是最常見的非監督式學習方法，依據資料之間的相似性，把資料自動分成 K 個群組，使同一群內的資料點彼此相似度高、不同群之間差異大。使用者只需要事先指定要分成幾群（K 值），演算法就會反覆執行兩個步驟，直到分群結果穩定：

1. **資料指派**：計算每筆資料到目前各群中心的距離，指派給距離最近的群。
2. **更新群中心**：根據每一群目前的成員，重新計算群內所有點的平均座標，把群中心移動到這個新位置。

這兩個步驟不斷交替，直到群中心不再明顯移動、成員不再換組為止。

**手算範例（教科書原例）**：假設平面上有六個點 A(0,0)、B(1,1)、C(2,1)、D(2,4)、E(3,3)、F(3,5)，要分成 K=2 群：

| 回合 | 群中心 | 分群結果 |
|---|---|---|
| 初始化 | 隨機選 B(1,1)、C(2,1) 作為初始中心 | — |
| 第一次指派 | — | 離 B 近：{A, B, D} → 第一群；離 C 近：{C, E, F} → 第二群 |
| 更新中心 | 第一群新中心 (1, 1.67)；第二群新中心 (2.67, 3) | — |
| 第二次指派 | — | C 發現離第一群中心更近，跳槽過去；D 發現離第二群中心更近，跳槽過去 |
| 收斂 | 再次計算新中心，確認沒有人再跳槽 | 分群完成 |

**K-means 的優點與限制**：概念直觀、實作容易、計算效率高，適合大量資料的初步探索；但分群結果對 K 值的選擇很敏感，初始中心的位置不同也可能導致不同結果，且當資料呈現非球狀或群組大小差異懸殊時，K-means 往往無法反映真實的群集結構。實務上通常需要多次嘗試不同的 K 值，並搭配其他方法交叉驗證。

### 二、RFM 分析：從交易紀錄看見顧客價值

在數據量龐大的情況下，企業往往難以判斷該從何處著手分析顧客行為，RFM 分析正是一項實用且具策略價值的工具。研究指出，**企業若能將顧客留存率提升 5%，利潤可能增加 25% 至 95%**；而向既有顧客銷售的成功機率約 60%–70%，吸引新顧客購買的成功率卻通常只有 5%–20%。這說明維繫既有顧客關係，遠比單純開發新客戶更具成本效益，也是 RFM 分析在行銷領域被廣泛使用的原因。

RFM 由三個核心指標組成：

| 指標 | 定義 | 意義 |
|---|---|---|
| **Recency（最近購買時間）** | 顧客距離最近一次購買，已經過了多久 | 數值越小代表顧客越活躍，數值越大則可能已逐漸流失 |
| **Frequency（購買頻率）** | 顧客在特定期間內的購買次數 | 頻率越高，通常代表與品牌關係越穩定，屬於忠誠顧客 |
| **Monetary（消費金額）** | 顧客在特定期間內的總消費金額 | 金額越高，代表對企業營收的貢獻越大 |

### 三、RFM 分數與十大客群標籤

要讓不同顧客之間的行為特徵容易比較，實務上會把 R、F、M 三個數值型指標，各自轉換成 1 到 5 分的等級分數（例如把所有顧客的 Recency 依五等分切分，最近購買的一組給 5 分，最久沒消費的一組給 1 分；Frequency、Monetary 同理，數值越高分數越高）。把 Recency 分數與 Frequency 分數組合起來，就能得到一組兩位數的代碼（例如「55」代表近期活躍且購買頻率也高），依照這組代碼，可以把顧客劃分成十種策略意義明確的客群：

| 客群 | R、F 特徵 | 說明 | 建議行銷方案 |
|---|---|---|---|
| **冠軍** | R 高、F 高 | 最近才購買，且購買頻率也高，是企業最有價值的頂級顧客 | VIP 專屬禮遇、優先體驗新品、口碑推薦邀請 |
| **忠誠客戶** | R 中高、F 高 | 購買頻率高且仍持續消費，是穩定的核心客群 | 會員升級、積分獎勵、忠誠回饋計畫 |
| **潛在忠誠者** | R 高、F 中 | 近期活躍且已有一定購買次數，若妥善經營可轉為忠誠客戶 | 引導式行銷，鼓勵累積購買次數、加入會員 |
| **有前途** | R 高、F 低 | 最近有消費，但購買次數不多，具有成長潛力 | 強化第二次購買誘因，如限時折扣或組合優惠 |
| **新客戶** | R 高、F 極低 | 近期首次購買，仍處於培養階段 | 新手引導、首購禮、品牌故事溝通 |
| **需要關注** | R 中、F 中 | 屬於中等活躍顧客，若未適時經營，可能逐漸流失 | 主動關懷、個人化推薦喚醒興趣 |
| **即將睡眠** | R 中低、F 低 | 購買頻率不高，且最近消費時間開始拉長 | 提醒式行銷、優惠券刺激回購 |
| **風險中** | R 低、F 高 | 過去曾多次購買，但近期未再消費，具有流失風險 | 個人化召回訊息、專屬折扣積極挽回 |
| **不能失去** | R 極低、F 高 | 高頻消費但近期未購買，屬於高價值但流失風險高的關鍵客戶 | 客戶經理主動介入、高關懷優先處理 |
| **冬眠中** | R 低、F 低 | 購買時間久遠且購買次數少，屬於低活躍且低價值顧客 | 低成本再行銷測試，長期無回應則可考慮排除 |

管理上的關鍵是：**十個客群不需要用同一套行銷手段對待**——對「冠軍」與「忠誠客戶」應該投入資源做關係深化，對「風險中」與「不能失去」則要優先搶救，對「冬眠中」的顧客則可能只值得用最低成本的方式測試，把更多預算留給前面幾類客群。

---

## Part 2｜Vibe Coding 實作演練：市場區隔與 RFM 客群標籤

### 範例 ①　用 K-means 做市場區隔

**題目定義**：用真實的零售顧客資料，依「年齡、年收入、消費分數」三個特徵，把顧客分成 3 個市場區隔，並解讀每一群的輪廓特徵。

**資料來源（真實公開資料，可直接下載）：**
<https://raw.githubusercontent.com/gakudo-ai/open-datasets/refs/heads/main/Mall_Customers.csv>

**給 AI 的提示詞：**
```
請用 Python 幫我做以下事情：
1. 讀取這個網址的 CSV 資料：
   https://raw.githubusercontent.com/gakudo-ai/open-datasets/refs/heads/main/Mall_Customers.csv
2. 取出 Age、Annual Income (k$)、Spending Score (1-100) 三個欄位作為分群特徵
3. 用 K-means 演算法，將顧客分成 3 群（K=3）
4. 印出每一群的顧客數量，以及每一群在三個特徵上的平均值（也就是各群的「輪廓」）
5. 畫一張以 Annual Income 為 X 軸、Spending Score 為 Y 軸的散佈圖，
   用不同顏色標示三個群組，並用黑色叉號標出各群中心
請給我可以在 Google Colab 直接執行的完整程式碼。
```

**預期產出的程式碼骨架：**
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

url = "https://raw.githubusercontent.com/gakudo-ai/open-datasets/refs/heads/main/Mall_Customers.csv"
df = pd.read_csv(url)

features = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(features)

print(df["Cluster"].value_counts())
print(df.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean())

plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"], c=df["Cluster"], cmap="viridis")
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 1], centers[:, 2], c="black", marker="x", s=200)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("顧客市場區隔（K=3）")
plt.show()
```

**延伸練習**：把 K 改成 5，重新畫一次散佈圖，觀察分群結果變得更細緻還是更破碎？試著替每一群取一個像「理性消費族」「高價值年輕客群」這樣的名字。

### 範例 ②　完整 RFM 分析流程，自動貼上十大客群標籤

**題目定義**：用真實的電商交易紀錄，完整計算每位顧客的 R、F、M 指標，轉換成分數，並依照 Part 1 的規則自動貼上十大客群標籤，統計各客群的人數與消費占比。

**資料來源（真實公開資料，UK 線上零售商 2010–2011 年交易紀錄，這正是 RFM 分析最早被提出的案例資料集）：**
```
!pip install ucimlrepo --quiet
```

**給 AI 的提示詞：**
```
請用 Python 幫我做以下事情：
1. 用 pip 安裝 ucimlrepo 套件，並用 fetch_ucirepo(id=352) 讀取
   「Online Retail」線上零售交易資料集
2. 資料前處理：移除缺失 CustomerID 的資料列、移除 Quantity 或
   UnitPrice 為負數的退貨紀錄，新增一個 TotalPrice = Quantity × UnitPrice 欄位
3. 設定分析基準日為資料集中最後一筆交易日期的隔一天
4. 依 CustomerID 分組，計算每位顧客的：
   - Recency：距離最後一次購買的天數
   - Frequency：購買次數（不重複的 InvoiceNo 數量）
   - Monetary：TotalPrice 的加總
5. 把 R、F、M 三個指標分別依五等分轉換成 1-5 分（Recency 分數方向要反過來，
   越接近的給分越高）
6. 組合 Recency 分數與 Frequency 分數，依照以下規則自動貼上客群標籤：
   55、54、45 → 冠軍；44、43、34、35 → 忠誠客戶；53、52 → 潛在忠誠者；
   51 → 新客戶；41、42、31、32、33 → 有前途；24、23 → 需要關注；
   22、21 → 即將睡眠；15、14、13 → 風險中；25 → 不能失去；11、12 → 冬眠中
   （如果組合沒對應到，標記為「其他」）
7. 統計每個客群的顧客人數，以及各客群 Monetary 總和占全體的百分比
請給我可以在 Google Colab 直接執行的完整程式碼。
```

**預期產出的程式碼骨架：**
```python
!pip install ucimlrepo --quiet
import pandas as pd
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=352)
df = pd.concat([dataset.data.features, dataset.data.targets], axis=1) if dataset.data.targets is not None else dataset.data.features.copy()

df = df.dropna(subset=["CustomerID"])
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

today_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (today_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum"),
)

rfm["R_score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["RFM_CODE"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str)

segment_map = {
    r"5[4-5]|45": "冠軍",
    r"[3-4][3-5]": "忠誠客戶",
    r"5[2-3]": "潛在忠誠者",
    r"51": "新客戶",
    r"[3-4][1-2]": "有前途",
    r"[2-3][2-4]": "需要關注",
    r"2[1-2]": "即將睡眠",
    r"1[3-5]": "風險中",
    r"25": "不能失去",
    r"1[1-2]": "冬眠中",
}
rfm["Segment"] = rfm["RFM_CODE"].replace(segment_map, regex=True)

summary = rfm.groupby("Segment").agg(
    人數=("Monetary", "count"),
    消費總額=("Monetary", "sum"),
)
summary["消費占比"] = (summary["消費總額"] / summary["消費總額"].sum() * 100).round(1)
print(summary.sort_values("消費總額", ascending=False))
```

**延伸練習**：找出「不能失去」與「風險中」這兩群顧客各有幾人、貢獻了多少消費金額，思考如果流失這群人，對整體營收的衝擊有多大，藉此判斷該投入多少行銷預算去挽回。

**管理意涵**：這個練習把「顧客關係管理」從一句口號，變成一份具體的、可以排優先順序的名單——哪些客群該花錢挽回、哪些客群該用最低成本測試、哪些客群其實可以放棄，都能用數字支持決策，而不是憑印象。

---

## Part 3｜RFM 行銷方案擬定工作坊

**目標**：針對 Part 2 範例②產出的十大客群（或用自己企業的顧客資料），挑選其中 2–3 個客群，用 Excel 或試算表整理出具體的行銷方案。

**操作步驟：**
1. 從十大客群中，挑出對你的企業最關鍵的 2–3 個客群（建議至少包含一個「高價值」與一個「高風險」客群）。
2. 用 Excel／Google 試算表，列出這幾個客群的人數、消費金額占比。
3. 針對每個客群，設計具體的行銷方案：用什麼管道（簡訊／email／APP 推播）、提供什麼誘因（折扣／贈品／專屬服務）、預期成效如何衡量。

```
【RFM 行銷方案擬定表】
客群名稱：___________
人數／消費占比：___________
行銷目標：□提升回購　□深化黏著度　□防止流失　□挽回沉睡客戶
行銷管道：___________
具體誘因／方案內容：___________
預期成效衡量方式：___________
```

---

## Part 4｜論文導讀方向

本週對應課程規劃中的「論文研討 Round 1（梯次 A）：AI 於精準行銷與顧客價值管理」，示範論文聚焦 RFM 與機器學習結合的顧客分群方法：

| 論文標題 | 期刊／年份 | 網址 |
|---|---|---|
| A Mathematical Model for Customer Segmentation Leveraging Deep Learning, Explainable AI, and RFM Analysis in Targeted Marketing | *Mathematics* (MDPI), 2023, 11(18), 3930 | <https://www.mdpi.com/2227-7390/11/18/3930> |
| An Automated Machine Learning Framework for Interpretable Customer Segmentation in Financial Services | *Journal of Risk and Financial Management* (MDPI), 2025, 13(4), 243 | <https://www.mdpi.com/2227-7072/13/4/243> |

導讀示範重點：這兩篇論文都是在 RFM 的基礎上，進一步結合機器學習或深度學習技術。閱讀時可以留意——作者為什麼認為傳統 RFM 不夠用？他們新增的技術解決了什麼限制？如果你是行銷主管，會不會直接採用這麼複雜的模型，還是傳統 RFM 分群已經夠用？這個「多複雜的模型才划算」的判斷，正是經理人在導入 AI 時最需要的商業判斷力。

---

## ✅ 課後練習

1. 完成「RFM 行銷方案擬定表」，作為期中報告「AI 導入方案設計」段落的補充素材（若你的期中報告主題是顧客關係管理相關）。
2. 把 Part 2 的兩個 Vibe Coding 範例，自己在 Colab 跑過一次，並記錄下你資料集中「冠軍」與「冬眠中」客群的人數與消費占比差距。
3. 準備本週或後續幾週要上台的論文閱覽簡報。
