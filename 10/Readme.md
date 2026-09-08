# 📘 第 10 週｜自然語言處理與市場輿情洞察

> 對應教科書：CH5《自然語言處理》5-1～5-4 節

---

## 🎯 本週學習目標

上完這個單元，各位將能夠：

1. 說出自然語言理解（NLU）與自然語言生成（NLG）的差異，並各舉一個商業應用例子。
2. 說出自然語言處理的基本流程（資料蒐集 → 斷詞 → 關鍵詞擷取 → 向量化 → 模型訓練）。
3. 親手跑一次真實的顧客評論情感分析，理解「正面／負面」判讀背後的技術邏輯。
4. 針對品牌或產品的社群輿情，規劃一套監控與危機應對的初步流程。

---

## Part 1｜主題課程：自然語言處理的技術與應用

### 一、NLU 與 NLG：理解與生成

自然語言處理（NLP）的核心可分為兩項技術：

| 技術 | 定義 | 商業應用 |
|---|---|---|
| **自然語言理解（NLU）** | 讓電腦正確理解語言內容，包含文法分析、語意詮釋與語境判讀 | 情緒與意圖分析、網路輿情分析、顧客評論分析 |
| **自然語言生成（NLG）** | 讓電腦主動「說話」或「寫作」，根據理解的內容自動生成語法正確、語意連貫的輸出 | 問答系統（單輪／多輪）、自動摘要生成 |

**NLU 應用範例**：企業可分析消費者評論與回饋內容，了解顧客對產品或服務的滿意程度；政府機構則可分析民眾在社群平台上的意見，掌握政策風向，這類應用也常被稱為「網路輿情分析」。

**NLG 應用範例**：單輪問答系統適合處理「訂單何時到貨」這類明確、一次性的查詢；多輪問答系統則能記住先前對話內容，逐步縮小選項範圍，例如金融理財規劃中先了解財務狀況，再詢問投資偏好，最終提出個人化建議。自動摘要生成則能從大量文字資料中萃取重點，廣泛應用於新聞閱讀、研究文獻整理與商業報告分析。

### 二、自然語言處理的基本流程

一套完整的 NLP 分析流程，通常包含以下步驟：

1. **資料蒐集**：透過網路爬蟲擷取特定網站或平台的文字資料，作為原始語料
2. **斷詞與斷句**：中文不像英文可用空格分詞，因此斷詞格外重要。台灣常見的斷詞工具為中研院開發的 CKIP，能將句子切分為詞彙單位，並加上詞性與語意標註（例如把「台北」標註為行政區）
3. **關鍵詞擷取與語意分析**：常用 TF-IDF（詞頻—逆向文件頻率）評估詞語重要性——若某詞在特定文章中頻繁出現，但在其他文章中相對少見，就可能是該文章的關鍵詞
4. **數值化表示**：把詞語轉換為詞向量（Word Embeddings）或句向量，讓電腦能進行數學運算
5. **模型訓練與應用**：用向量化後的資料訓練分類、生成或其他任務模型

### 三、情感分析：從評論到商業洞察

情感分析是 NLU 最常見的商業應用之一。系統透過辨識語句中的關鍵詞彙與語氣（如「好棒」「失望」「不推薦」），推斷使用者的情緒傾向。這類技術可應用於：

- **顧客服務**：分析問卷或開放式回饋，評估顧客滿意度
- **產品評論**：辨識「劇情緊湊」「值得再看一次」等關鍵詞彙，統計整體情緒傾向
- **品牌輿情監控**：即時掌握社群媒體上對品牌或產品的討論風向，作為公關危機應對的依據

---

## Part 2｜Vibe Coding 實作演練：餐廳評論情感分析

### 範例①　完整跑一次情感分析流程

**題目定義**：用真實的餐廳顧客評論資料，訓練一個分類模型，自動判斷評論是正面還是負面，並找出「正面評論」與「負面評論」中最常出現的關鍵字。

**資料來源（真實公開資料，1000 則標註過的餐廳評論，可直接下載）：**
<https://raw.githubusercontent.com/aadimangla/Restaurant-Reviews-Sentiment-Analysis/master/Dataset/Restaurant_Reviews.tsv>

**給 AI 的提示詞：**
```
請用 Python 幫我做以下事情：
1. 讀取這個網址的 TSV 資料（用 tab 分隔）：
   https://raw.githubusercontent.com/aadimangla/Restaurant-Reviews-Sentiment-Analysis/master/Dataset/Restaurant_Reviews.tsv
   欄位包含 Review（評論文字）與 Liked（1=正面，0=負面）
2. 用 TF-IDF 把評論文字轉換成數值特徵
3. 切分訓練集與測試集，訓練一個羅吉斯迴歸分類模型
4. 印出模型的準確率與混淆矩陣
5. 分別列出「正面評論」與「負面評論」中，TF-IDF 分數最高的前 10 個關鍵字
請給我可以在 Google Colab 直接執行的完整程式碼。
```

**預期產出的程式碼骨架：**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

url = "https://raw.githubusercontent.com/aadimangla/Restaurant-Reviews-Sentiment-Analysis/master/Dataset/Restaurant_Reviews.tsv"
df = pd.read_csv(url, delimiter="\t", quoting=3)

X_train, X_test, y_train, y_test = train_test_split(
    df["Review"], df["Liked"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

print("準確率：", accuracy_score(y_test, y_pred))
print("混淆矩陣：\n", confusion_matrix(y_test, y_pred))

feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]
top_positive = sorted(zip(coefs, feature_names), reverse=True)[:10]
top_negative = sorted(zip(coefs, feature_names))[:10]
print("最能代表正面評論的關鍵字：", [w for _, w in top_positive])
print("最能代表負面評論的關鍵字：", [w for _, w in top_negative])
```

**延伸練習**：把 `max_features` 從 1000 調高到 3000，觀察準確率與關鍵字列表有什麼變化；並思考——如果這是你自己餐廳的評論資料，看到「slow」「rude」這類詞出現在負面關鍵字榜首，你會優先調整哪個環節？

### 範例②　用生成式 AI 快速摘要一批評論的痛點

**題目定義**：不用寫程式，直接把一批評論貼給 LLM，練習用提示詞快速萃取「顧客痛點」與「品牌優勢」，體驗 NLG 自動摘要的實際操作方式。

**給 AI 的提示詞：**
```
以下是 10 則顧客對我們餐廳的評論（請貼上你自己蒐集或虛構的評論文字）。
請幫我：
1. 統計大致的正負面評論比例
2. 歸納出出現頻率最高的 3 個顧客痛點
3. 歸納出出現頻率最高的 3 個品牌優勢
4. 針對痛點，各提出一個具體的改善建議
請用表格呈現。
```

**管理意涵**：範例①示範的是「規則化、可重複執行」的情感分析（適合每天自動處理大量評論）；範例②示範的則是「彈性、需要人工貼上文字」的生成式 AI 摘要（適合偶爾針對一批評論做深入質化分析）。兩種方式在實務上經常互補使用。

---

## Part 3｜品牌輿情監控與危機應對規劃工作坊

**目標**：針對自己企業或熟悉的品牌，規劃一套社群輿情監控與危機應對的初步流程。

**操作步驟：**
1. 列出目前最主要的顧客意見來源管道（Google 評論、社群留言、客服信件等）。
2. 討論：如果情感分析系統偵測到負面評論比例突然上升，應該由哪個部門、在多長時間內做出回應？
3. 設計一套簡單的分級機制（例如：負面評論中若出現「食安」「安全」等關鍵字，須立即通報主管）。

```
【品牌輿情監控規劃表】
主要意見來源管道：___________
負面評論異常上升時的應變流程：___________
高風險關鍵字與通報機制：___________
```

---

## Part 4｜論文導讀方向與 Round 2 選題準備

本週不安排正式的論文導讀場次，但請開始準備 Round 2（生成式 AI／LLM／RAG／Agent 主題）的候選論文，並與教師進行選題諮詢。可參考的搜尋方向：GenAI 對組織生產力的影響、LLM 於專業領域的決策支援、企業 RAG 知識庫構建、AI Agent 於供應鏈的應用——這幾個主題會分別對應第 11 至 14 週的內容。

---

## ✅ 課後練習

1. 完成「品牌輿情監控規劃表」。
2. 把 Part 2 的兩個範例都跑過一次，比較規則化情感分析與生成式 AI 摘要的差異。
3. 鎖定 1–2 篇 Round 2 候選論文，準備下週的選題諮詢。
