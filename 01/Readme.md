# 📘 第 1 週｜人工智慧的發展、素養與產業應用

> 對應教科書：CH1《人工智慧之發展及演進》1-1、1-2、1-3 節（1-4 風險與挑戰留待第 2 週）

---

## 🎯 本週學習目標

1. 說出人工智慧從符號推理、專家系統、機器學習╱深度學習到生成式 AI 的四個發展階段，以及每個階段的關鍵歷史事件。
2. 用「知道 → 瞭解 → 熟悉 → 內化 → 影響」五層次，檢視自己目前的 AI 素養落在哪個階段。
3. 親手（或跟著操作）跑過一次「規則型 AI → 機器學習 → 深度學習 → 生成式 AI」四個階段的實際範例，理解每個階段技術上真正在做什麼。
4. 拆解一篇真實期刊論文的研究方法與資料處理流程，並用 Vibe Coding 簡易重現其中的核心分析邏輯。

---

## Part 1｜主題課程：AI 四階段發展史與素養思維

### 1-1｜人工智慧發展的四個階段

| 階段 | 年代 | 關鍵事件 | 參考條目 | 對經營管理的意義 |
|---|---|---|---|---|
| **① 早期探索與符號 AI** | 1950s～1970s 中期 | 1950 圖靈測試；1956 達特茅斯會議正式提出「AI」一詞；邏輯理論家、一般問題解決器、1966 ELIZA；1973 Lighthill 報告導致第一次 AI 寒冬 | [Turing test](https://en.wikipedia.org/wiki/Turing_test)、[Dartmouth workshop](https://en.wikipedia.org/wiki/Dartmouth_workshop)、[ELIZA](https://en.wikipedia.org/wiki/ELIZA)、[Lighthill report](https://en.wikipedia.org/wiki/Lighthill_report) | 智慧被定義為「可形式化的邏輯規則」——這也是後來專家系統與現今 if-then 商業流程的思想起點 |
| **② 專家系統時期** | 1970s 末～1980s 中期 | 史丹佛 MYCIN（醫療診斷）、DEC 的 XCON（訂單組裝自動化）、1982 日本第五代電腦計畫；1987 LISP 機器市場崩潰引發第二次 AI 寒冬 | [Expert system](https://en.wikipedia.org/wiki/Expert_system)、[MYCIN](https://en.wikipedia.org/wiki/Mycin)、[XCON](https://en.wikipedia.org/wiki/Xcon) | 第一次證明 AI 可以「取代特定領域的專業判斷」，但也暴露知識工程瓶頸——規則難維護、無法自我學習 |
| **③ 機器學習與深度學習時期** | 1990s 中～2010s 末 | 決策樹、SVM、貝氏網路成熟；1997 Deep Blue 擊敗 Kasparov；2012 AlexNet 在 ImageNet 競賽大幅超越傳統方法，正式宣告深度學習時代 | [Machine learning](https://en.wikipedia.org/wiki/Machine_learning)、[Deep Blue](https://en.wikipedia.org/wiki/Deep_Blue_(chess_computer))、[ImageNet](https://en.wikipedia.org/wiki/ImageNet)、[AlexNet](https://en.wikipedia.org/wiki/AlexNet) | 決策邏輯從「人寫規則」變成「機器從資料中學規則」——這正是本課程後續機器學習單元的核心 |
| **④ 生成式 AI 與大型語言模型時期** | 2020～現今 | 2018 Transformer 架構成熟；2022 對話式系統普及，AI 走入大眾生活；RAG、AI Agent、多模態模型相繼出現 | [Generative artificial intelligence](https://en.wikipedia.org/wiki/Generative_artificial_intelligence)、[Large language model](https://en.wikipedia.org/wiki/Large_language_model)、[Retrieval-augmented generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) | AI 從「分析輔助」進化到「主動生成、甚至自主執行任務」——這是本課程後段生成式 AI 單元的核心主題 |

**思考題（每階段任選一題）：**
- 如果你的公司還停留在「規則型 AI」，哪個部門的 SOP 最容易被寫成 if-then 規則？
- MYCIN 用幾百條規則做醫療診斷，你公司有沒有類似「靠老師傅經驗判斷」但其實可以規則化的流程？
- Deep Blue 靠搜尋演算法打敗西洋棋世界冠軍，你的產業裡有沒有類似「規則明確、但複雜度極高」的決策情境？
- 生成式 AI 現在最直接衝擊你公司哪一種「知識型工作」（企劃、法務、客服、教育訓練）？

### 補充｜電腦視覺技術的三個轉折（教科書 1-1-2）

| 轉折 | 代表模型 | 能力 |
|---|---|---|
| 物件偵測 | [YOLO](https://en.wikipedia.org/wiki/You_Only_Look_Once)（2016） | 找出物體位置與類別 |
| 整體語意建模 | [Vision Transformer](https://en.wikipedia.org/wiki/Vision_transformer)（2020） | 理解整張圖片的結構關係 |
| 多模態理解與生成 | [CLIP](https://en.wikipedia.org/wiki/CLIP_(machine_learning))（2021）、BLIP（2022）、[Segment Anything Model](https://en.wikipedia.org/wiki/Segment_Anything)（2023） | 圖文對齊、生成敘述、像素級分割 |

這條演進路徑，會在後續「電腦視覺與智慧零售╱服務體驗」單元進一步展開，這裡先建立整體印象即可。

### 1-2｜AI 素養與思維

**AI 學習的五個層次**：知道 → 瞭解 → 熟悉 → 內化 → 影響。

**判別力工具｜加減乘除思維**：

| 招式 | 做法 |
|---|---|
| 加法 | 多輪追問、修正提示詞，探索更全面的觀點 |
| 減法 | 降低對 AI 輸出的無條件信任，主動查證 |
| 乘法 | 請 AI 提供多元立場、不同分析角度 |
| 除法 | 必要時放下 AI，回到原始資料與自身判斷 |

---

## Part 2｜Vibe Coding 實作演練：AI 四階段技術體驗

### 什麼是 Vibe Coding？

「Vibe Coding」指的是：**你不需要會寫程式，只要能把問題用自然語言講清楚，AI 就能幫你寫出可執行的程式碼**。這門課刻意避開底層程式撰寫，但「動手跑一次」跟「只聽概念」的理解深度完全不同——尤其之後導讀期刊論文時，看得懂論文裡「這個模型做了什麼」會需要一點親手跑過的經驗。

以下四個範例，每一個都包含：
- **題目定義**：這個範例具體要解決什麼問題
- **給 AI 的提示詞**：可以直接複製貼上到 ChatGPT / Claude / Gemini 或 Colab 內建的 AI 助手
- **預期產出的程式碼**：可以直接貼到 Google Colab 執行
- **延伸練習**：自己動手改一個小地方，加深理解

這份教材可以是課堂上即時操作的示範，也可以是各位下課後自己打開一份 Colab 筆記本，照著步驟跑一遍的自學材料——兩種方式效果相同。

準備工作：到 [colab.research.google.com](https://colab.research.google.com/) 新增一份筆記本即可，不需要安裝任何軟體。

---

### 範例 ①　規則型 AI：訂單狀態查詢客服機器人

**題目定義**：模擬第一階段「符號 AI」的邏輯——用明確的 if-then 規則，讓程式根據訂單編號回覆對應的客服訊息，沒有任何「學習」成分。

**給 AI 的提示詞：**
```
請幫我用 Python 寫一個「訂單狀態查詢」的規則型客服機器人：
1. 用一個字典儲存訂單編號與對應狀態（已下單/備貨中/已出貨/已送達）
2. 使用者輸入訂單編號後，機器人依狀態回覆對應的客服訊息
3. 如果輸入的編號不存在，要回覆「查無此訂單，請確認編號」
請直接給我可以在 Google Colab 執行的完整程式碼。
```

**預期產出的程式碼（可直接執行）：**
```python
orders = {
    "A001": "已出貨",
    "A002": "備貨中",
    "A003": "已送達",
}

def check_order(order_id):
    if order_id not in orders:
        return "查無此訂單，請確認編號"
    status = orders[order_id]
    if status == "已下單":
        return "您的訂單已成立，我們會盡快為您備貨"
    elif status == "備貨中":
        return "您的訂單正在備貨中，預計 1-2 個工作天出貨"
    elif status == "已出貨":
        return "您的訂單已出貨，請留意物流簡訊通知"
    elif status == "已送達":
        return "您的訂單已送達，感謝您的購買"

print(check_order("A001"))
print(check_order("A999"))
```

**延伸練習**：請 AI 幫你新增一種「退貨中」的狀態，並設計對應的客服回覆訊息。

**管理意涵**：這種邏輯完全「靠人寫規則」，優點是可解釋、可控；缺點是規則一多就難維護——這正是專家系統時期最終遇到的瓶頸。

---

### 範例 ②　機器學習：信用卡違約風險分類

**題目定義**：用真實的信用卡客戶資料，訓練一個機器學習模型，預測客戶「下個月是否會違約」，並體驗「決策門檻」如何影響誤判的方向。這個範例的主題，跟本週 Part 4 要精讀的期刊論文完全對應——讀完論文後，可以回頭比較「論文的做法」跟「這裡的簡化做法」差在哪裡。

**資料來源（真實公開資料集，可直接下載）**：
UCI Machine Learning Repository — *Default of Credit Card Clients*（台灣信用卡客戶資料，3 萬筆）
<https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>

**給 AI 的提示詞：**
```
請用 Python 幫我做以下事情：
1. 用 pip 安裝 ucimlrepo 套件，並用 fetch_ucirepo(id=350) 讀取
   「Default of Credit Card Clients」資料集
2. 用 scikit-learn 的羅吉斯迴歸（Logistic Regression）做二元分類，
   預測客戶下個月是否會違約
3. 切分訓練集與測試集，印出準確率（accuracy）與混淆矩陣（confusion matrix）
4. 接著把預測機率的判斷門檻從預設的 0.5 改成 0.3，
   重新印出混淆矩陣，並告訴我 precision 和 recall 各自怎麼變化
請給我可以在 Google Colab 直接執行的完整程式碼。
```

**預期產出的程式碼骨架（實際內容 AI 會依提示詞完整補齊）：**
```python
!pip install ucimlrepo --quiet

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dataset = fetch_ucirepo(id=350)
X = dataset.data.features
y = dataset.data.targets.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 預設門檻 0.5
y_pred = model.predict(X_test)
print("準確率：", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# 調整門檻為 0.3
y_prob = model.predict_proba(X_test)[:, 1]
y_pred_030 = (y_prob >= 0.3).astype(int)
print("門檻 0.3 的混淆矩陣：")
print(confusion_matrix(y_test, y_pred_030))
print(classification_report(y_test, y_pred_030))
```

**延伸練習**：把門檻改成 0.7，觀察 precision／recall 又會怎麼變化；並想一想——如果你是風控主管，把門檻調低（更容易判斷為「會違約」）的成本是什麼？調高的成本又是什麼？

**管理意涵**：門檻調整＝「誤攔好客戶（false positive）」vs.「放過壞客戶（false negative）」的成本權衡，這正是後續機器學習單元要深入談的「決策門檻」議題，也直接對應信用卡詐騙偵測、不良品檢測等真實產業情境。

---

### 範例 ③　深度學習：智慧檢驗影像判讀

**題目定義**：用一個現成的預訓練深度學習影像模型，判讀一張產品照片，模擬「智慧檢驗」情境——不需要自己訓練模型，直接呼叫別人訓練好的模型。

**給 AI 的提示詞：**
```
請用 Python 的 transformers 套件，幫我做以下事情：
1. 載入一個預訓練的圖片分類模型（例如 google/vit-base-patch16-224）
2. 讓我可以上傳一張圖片（用 Google Colab 的檔案上傳功能）
3. 印出模型預測的前 3 個類別名稱與對應的信心分數（百分比）
請給我可以在 Google Colab 直接執行的完整程式碼。
```

**預期產出的程式碼骨架：**
```python
!pip install transformers torch --quiet

from transformers import pipeline
from google.colab import files

classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

uploaded = files.upload()
image_path = list(uploaded.keys())[0]

results = classifier(image_path, top_k=3)
for r in results:
    print(f"{r['label']}：{r['score']*100:.1f}%")
```

**延伸練習**：換一張不同的照片測試看看，觀察模型在哪些情境下判斷準確、哪些情境下容易誤判（例如角度特殊、光線不足的照片）。

**管理意涵**：這示範的正是「機器視覺全檢」的技術基礎。可以延伸討論：傳統抽檢 vs. AI 全檢的投資報酬率（ROI）該怎麼試算？誤判成本 vs. 人力抽檢成本如何比較？

---

### 範例 ④　生成式 AI：提示工程初體驗

**題目定義**：用同一個大型語言模型，透過「修改提示詞」讓輸出品質產生明顯差異，體驗提示工程（prompt engineering）的基本邏輯。

**第一版提示詞（刻意寫得籠統）：**
```
幫我寫一封信用卡逾期提醒簡訊
```

**優化後的提示詞（明確定義語氣、長度、必要資訊）：**
```
請幫我寫一則不超過 70 字的簡訊，提醒客戶信用卡帳單已逾期 3 天。
語氣需維持品牌友善，但要清楚告知逾期可能產生的利息與信用影響，
並附上一個聯繫客服的方式（客服專線：0800-123-456）。
```

**操作方式**：把兩個版本分別貼到 ChatGPT / Claude / Gemini，比較兩次輸出的差異——內容具體度、語氣掌握度、是否包含必要資訊。

**如果想在 Colab 中用 API 呼叫（進階選用）：**
```python
!pip install anthropic --quiet
import anthropic

client = anthropic.Anthropic(api_key="你的API金鑰")

prompt = """請幫我寫一則不超過 70 字的簡訊，提醒客戶信用卡帳單已逾期 3 天。
語氣需維持品牌友善，但要清楚告知逾期可能產生的利息與信用影響，
並附上一個聯繫客服的方式（客服專線：0800-123-456）。"""

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=200,
    messages=[{"role": "user", "content": prompt}]
)
print(response.content[0].text)
```

**延伸練習**：再修改一次提示詞，要求「語氣更嚴肅、強調法律追訴可能性」，觀察輸出如何隨提示詞調整而改變。

**管理意涵**：同一個模型，提示詞的精準度會直接決定產出品質——這是生成式 AI 落地應用的第一道關卡，也是後續提示工程單元要系統化學習的內容。

---

## Part 3｜AI 應用機會探索工作坊

**為什麼不用「自己企業」的真實資料？** 多數同學目前不容易取得自己企業的真實內部資料（涉及機密、跨部門申請流程），與其卡在資料取得，不如先用幾個**產業情境範本**練習分析框架，之後若能取得自己企業的真實資訊，隨時可以替換進來——分析工具是一樣的，換的只是輸入的情境。

**目標**：練習用「五構面數位成熟度」框架，分析一個產業情境的痛點與 AI 導入機會，同時也是在為後續選擇論文閱覽方向、探索研究興趣領域做暖身。

**操作步驟：**
1. 從下方四個情境中，選一個你比較有興趣、或跟你工作背景相關的（如果你已經很清楚自己企業的公開資訊或產業報告，也可以直接替換成你自己熟悉的產業情境）。
2. 用五構面自評表（策略、資料、流程、技術、人才），針對這個情境打分數，找出最弱的兩項。
3. 具體化痛點：哪個環節、造成什麼損失。
4. 畫出 AI 機會地圖（效益 × 難度 2×2 矩陣），標出優先題目。
5. 想一想：這個情境和哪個產業別的期刊論文最有可能相關？（呼應下方示範論文庫）

**四個產業情境範本（可任選其一）：**

> **情境 A｜製造業**：中小型金屬零件代工廠，員工 120 人，主要客戶為汽車零組件廠。目前品管仍以人工抽檢為主，客訴多集中在「批次瑕疵未被及時發現」；業務接單仍靠 Excel＋電話追蹤，常發生交期延誤。

> **情境 B｜零售／電商**：中型連鎖服飾零售商，會員數穩定成長，但整體回購率停滯。行銷預算分散在各社群通路，缺乏系統化的顧客分群工具，行銷人員只能憑經驗判斷該對誰發送哪種優惠。

> **情境 C｜金融服務**：消費性信貸業務單位，核貸審查仍高度仰賴人工判斷收入證明與聯徵資料，撥款速度慢，不同審查人員的判斷標準不一致，偶有爭議案件。

> **情境 D｜連鎖餐飲服務業**：顧客評論分散在 Google 評論、外送平台、社群留言等多個管道，各分店經理難以即時掌握口碑趨勢與客訴熱點，往往等到客訴擴散才注意到問題。

**五構面自評表**（1-5 分）：策略、資料、流程、技術、人才。

```
【AI 應用機會探索表】
選定情境：□A 製造業　□B 零售/電商　□C 金融服務　□D 連鎖餐飲服務　□其他（自訂）：___________

一、五構面自評（1-5分）：策略__ 資料__ 流程__ 技術__ 人才__
二、最弱兩項：1.______ 2.______
三、痛點描述：
  - 痛點1：
  - 痛點2：
  - 痛點3：
四、AI機會地圖：
  [效益高/難度低]優先執行：
  [效益高/難度高]中期規劃：
  [效益低/難度低]小規模試點：
  [效益低/難度高]暫緩：
五、這個情境跟下方哪一篇（或哪一類）示範論文最相關？為什麼？
```

---

## Part 4｜論文導讀方向

本課程每一堂課固定包含三個部分：**主題課程**、**Vibe Coding 實作**、**論文導讀**。論文導讀不是只列書單，而是希望帶各位實際拆解一篇論文的研究方法與模型設計，必要時搭配 Vibe Coding 簡易重現其中的分析邏輯，讓「讀論文」跟「動手做」互相印證。

### 教師示範精讀：一篇論文的研究方法怎麼讀

以下用示範論文庫中的信用風險論文，示範一份完整的論文精讀該抓哪些重點：

> **Credit Risk Prediction Using Machine Learning and Deep Learning: A Study on Credit Card Customers**（*Risks*, 2024, 12(11), 174）<https://www.mdpi.com/2227-9091/12/11/174>

**① 研究問題與資料**：這篇論文想解決「怎麼把信用卡客戶分成『好客戶』與『壞客戶』」的問題。資料來自 Kaggle 上的真實銀行資料（已去識別化），拆成兩個檔案：`application_record.csv`（申請人基本資料：性別、是否有房有車、收入、學歷、職業等）與 `credit_record.csv`（每月還款狀態紀錄），兩者用客戶 ID 串接。

**② 目標變數怎麼定義（這是全篇最關鍵的設計）**：論文先觀察每個帳戶開卡後 12 個月內的還款狀態，把「逾期 60 天以上（狀態碼 2、3、4、5）」定義為「壞客戶（1）」，其餘定義為「好客戶（0）」。這個「先觀察一段固定期間，再回頭定義標籤」的做法，叫做「績效觀察窗（performance window）」，是信用風險建模的標準做法——值得注意的是，論文作者是先畫出「壞帳率隨帳齡變化」的趨勢圖，發現大約 12 個月後壞帳率趨於穩定，才決定用 12 個月當作觀察窗，而不是隨意假設。

**③ 資料前處理**：處理離群值（用 IQR 方法）、缺失值（例如「職業類型」欄位缺失 32%）；最關鍵的是**類別極度不平衡**——好客戶占 98.7%，壞客戶僅 1.3%，論文使用 **SMOTE 過採樣**，合成少數類別（壞客戶）的樣本，讓訓練資料的兩類比例接近 1:1。

**④ 模型選擇與比較**：論文一次訓練了六種模型（隨機森林、羅吉斯迴歸、類神經網路、AdaBoost、XGBoost、LightGBM），用準確率、精確率、召回率、F1、ROC-AUC、MCC 六種指標全面比較，而不是只看單一指標。最終 **XGBoost 表現最好，準確率達 99.4%**。

**⑤ 找出關鍵特徵**：論文用 XGBoost 的特徵重要性分析，指出**年齡、收入、工作年資、家庭人數**是預測違約最關鍵的四個變數——這正是論文對「管理啟示」最直接的貢獻：银行審查信用卡申請時，這幾個欄位值得優先關注。

**這樣讀論文的好處**：把一篇論文拆解成「問題定義 → 目標變數怎麼設計 → 資料前處理 → 模型比較 → 關鍵發現」五個固定步驟，之後看任何一篇實證論文，都可以用同樣的順序去抓重點，這也是《課程作業與報告格式規範》A 節「論文閱覽簡報」要求各位練習的能力。

### Vibe Coding 簡易重現：用同樣的邏輯跑一次

**題目定義**：論文使用的原始資料（`application_record.csv` / `credit_record.csv`）需要 Kaggle 帳號才能下載，且欄位需要額外拼接與整理，較不適合課堂即時操作。這裡改用 Part 2 範例②已經用過的 UCI 信用卡資料集，練習論文中同樣的「不平衡資料處理」與「多模型比較」邏輯，體會研究方法本身可以套用到不同資料集上。

**給 AI 的提示詞：**
```
接續 Part 2 範例②讀取的 UCI 信用卡違約資料集，請幫我做以下事情：
1. 印出目標變數（是否違約）的兩個類別各自的筆數與比例，
   確認資料是否不平衡
2. 用 imbalanced-learn 套件的 SMOTE，對訓練集做過採樣，
   讓兩個類別的比例接近 1:1
3. 分別訓練三個模型：羅吉斯迴歸、隨機森林、XGBoost
4. 針對這三個模型，統一印出準確率、精確率、召回率、F1 分數，
   整理成一個表格方便比較
5. 針對表現最好的模型，印出前 5 個最重要的特徵
請給我可以在 Google Colab 直接執行的完整程式碼。
```

**預期產出的程式碼骨架：**
```python
!pip install imbalanced-learn xgboost --quiet

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print(y.value_counts(normalize=True))

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

models = {
    "羅吉斯迴歸": LogisticRegression(max_iter=1000),
    "隨機森林": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
}

results = []
for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    results.append({
        "模型": name,
        "準確率": accuracy_score(y_test, y_pred),
        "精確率": precision_score(y_test, y_pred),
        "召回率": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
    })

print(pd.DataFrame(results))

best_model = models["XGBoost"]
importances = pd.Series(best_model.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False).head(5))
```

**延伸練習**：比較 SMOTE 前後，模型在「少數類別（違約客戶）」的召回率有沒有明顯提升；並把重要特徵排名跟論文的發現（年齡、收入、工作年資、家庭人數）比對，看看是不是類似的欄位也排在前面。

**管理意涵**：這個練習說明——即使換一份資料集，論文裡「先看資料平不平衡、決定要不要過採樣、比較多個模型、找出關鍵特徵」這套研究方法本身是可以遷移複製的。之後各位在做論文閱覽簡報時，也可以試著用自己找的論文，設計類似的簡易重現。

### 📚 示範論文庫（涵蓋不同產業｜開放取用可直接下載）

| 產業別 | 論文標題 | 期刊／年份 | 網址 |
|---|---|---|---|
| 綜合／數位轉型 | Artificial Intelligence and Business Strategy towards Digital Transformation: A Research Agenda | *Sustainability*, 2021, 13(4), 2025 | <https://www.mdpi.com/2071-1050/13/4/2025> |
| 綜合／數位轉型 | AI-Powered Innovation in Digital Transformation: Key Pillars and Industry Impact | *Sustainability*, 2024, 16(5), 1790 | <https://www.mdpi.com/2071-1050/16/5/1790> |
| 製造業／預測性維護 | Hybrid Deep Learning for Predictive Maintenance in Industrial Machinery Using LSTM and MLP Models | *Machines*, 2026, 14(2), 191 | <https://www.mdpi.com/2075-1702/14/2/191> |
| 金融業／信用風險 | Credit Risk Prediction Using Machine Learning and Deep Learning: A Study on Credit Card Customers（本週精讀範例） | *Risks*, 2024, 12(11), 174 | <https://www.mdpi.com/2227-9091/12/11/174> |
| 零售／服務業／顧客關係 | Machine Learning Based Customer Churn Prediction in Home Appliance Rental Business | *Journal of Big Data*, 2023, 10:41 | <https://journalofbigdata.springeropen.com/articles/10.1186/s40537-023-00721-8> |

---

## ✅ 課後練習

1. 完成「AI 應用機會探索表」，並想一想這個情境跟哪一類論文主題比較接近——這會是你後續選擇論文閱覽簡報、探索研究興趣領域的起點。
2. 把 Part 2 的四個 Vibe Coding 範例，自己在 Colab 跑過一次（不需要全部成功，卡住的地方正是下次上課可以討論的問題）。
3. 把 Part 4 的簡易重現範例也跑過一次，並找時間把信用風險那篇論文完整讀一遍，練習用「問題定義 → 目標變數 → 資料前處理 → 模型比較 → 關鍵發現」五步驟寫下你自己的筆記。
