# 第 11 週：混合式 AI 決策架構——結構方程模型結合類神經網路（SEM-ANN）

> 課程模組：第三模組｜資料驅動 AI：預測、分群與可解釋性模型（第 8–11 週）
> 本週定位：作為第三模組之收官週，本週整合本課程橫跨兩大方法論典範之學習成果——以第 1、3 週所學之 PLS-SEM（知識論基礎為理論驗證、假設檢定）作為第一階段，篩選出具統計顯著性之前因變數；再以第 8–10 週所學之機器學習脈絡下的類神經網路（ANN，知識論基礎為預測準確度最大化）作為第二階段，捕捉變數間傳統線性模型難以偵測之深層非線性與補償性關聯，是全課程「決策智能與混合式 AI」核心定位最具體的方法論實踐。

> 教材版本：v1.0｜適用對象：在職專班研究方法與 AI 應用課程｜先修基礎：第 3 週（PLS-SEM）、第 8–10 週（機器學習方法）
> 使用工具：Google Colab（Python 3）｜主要套件：`plspm`（PLS-SEM，沿用第 3 週設定）、`scikit-learn`（`MLPRegressor`、`KFold`、`permutation_importance`）、`pandas`、`matplotlib`（本週所有程式碼已實際測試驗證可正常執行）

---

## 目錄

1. [學習目標](#學習目標)
2. [本週知識地圖](#本週知識地圖)
3. [理論基礎篇](#理論基礎篇)
   1. [3.1 混合式研究架構的方法論動機](#31-混合式研究架構的方法論動機)
   2. [3.2 兩階段混合研究法：SEM 作為變數篩選機制](#32-兩階段混合研究法sem-作為變數篩選機制)
   3. [3.3 類神經網路基礎：多層感知器結構與正向傳播](#33-類神經網路基礎多層感知器結構與正向傳播)
   4. [3.4 反向傳播與模型訓練](#34-反向傳播與模型訓練)
   5. [3.5 k 折交叉驗證](#35-k-折交叉驗證)
   6. [3.6 敏感度分析與正規化相對重要性](#36-敏感度分析與正規化相對重要性)
   7. [3.7 SEM 線性效果與 ANN 非線性重要度的比較邏輯](#37-sem-線性效果與-ann-非線性重要度的比較邏輯)
   8. [3.8 論文中 SEM-ANN 章節的標準寫法架構](#38-論文中-sem-ann-章節的標準寫法架構)
4. [研究設計實例：行動支付使用者滿意度與忠誠度——PLS-SEM 驗證與 ANN 非線性權重補強](#研究設計實例行動支付使用者滿意度與忠誠度pls-sem-驗證與-ann-非線性權重補強)
5. [Colab 實作環境建置](#colab-實作環境建置)
6. [Colab 實作：Step by Step 完整程式碼](#colab-實作step-by-step-完整程式碼)
7. [Vibe Coding 提示詞（Prompt）實作範例集](#vibe-coding-提示詞prompt實作範例集)
8. [結果呈現與分析：碩士論文寫法示例](#結果呈現與分析碩士論文寫法示例)
9. [常見統計誤區與 Q&A](#常見統計誤區與-qa)
10. [延伸研究方向：智慧觀光平台體驗滿意度前因分析——SEM-ANN 雙階段混合模型](#延伸研究方向智慧觀光平台體驗滿意度前因分析sem-ann-雙階段混合模型)
11. [課後作業與練習](#課後作業與練習)
12. [參考文獻與延伸閱讀（已查核連結）](#參考文獻與延伸閱讀已查核連結)
13. [附錄](#附錄)
14. [第三模組總結：資料驅動型 AI 方法家族回顧](#第三模組總結資料驅動型-ai-方法家族回顧)
15. [下週預告](#下週預告)

---

## 學習目標

完成本週課程後，學生應能夠：

1. 說明混合式研究架構（Two-Stage Hybrid Approach）整合 SEM 與 ANN 之方法論動機，理解其如何截長補短兩種方法論典範。
2. 說明多層感知器（MLP）之結構組成——輸入層、隱藏層、輸出層，以及正向傳播之運算邏輯。
3. 說明反向傳播演算法與梯度下降法如何驅動類神經網路之參數學習。
4. 使用 Python（`scikit-learn`）建構具備 k 折交叉驗證之 MLP 迴歸模型。
5. 計算並解讀敏感度分析與正規化相對重要性（Normalized Importance），辨識類神經網路視角下的關鍵預測因子。
6. 比較 SEM 線性路徑係數（總效果）與 ANN 非線性重要度排名之異同，理解兩者差異之方法論意涵。
7. 依照論文「研究結果與討論」章節寫法，將 SEM-ANN 兩階段分析結果轉譯為雙軸雷達圖與具學術規範的文字敘述。
8. 回顧並整合第 8–11 週資料驅動型 AI 方法家族之完整脈絡，為期末專題之方法論選擇建立完整工具箱。

---

## 本週知識地圖

| 構面 | 內容 | 對應方法 | 對應 Python 套件 |
|---|---|---|---|
| 第一階段：假設檢定 | 檢驗理論路徑關係是否具統計顯著性 | PLS-SEM（沿用第 3 週） | `plspm` |
| 變數篩選 | 萃取通過顯著性檢定之前因變數 | Bootstrapping 顯著性判讀 | — |
| 第二階段：非線性補強 | 以顯著前因變數為輸入，預測依變數 | 多層感知器（MLP） | `sklearn.neural_network.MLPRegressor` |
| 模型驗證 | 確保預測效能不受單一資料切分影響 | k 折交叉驗證 | `sklearn.model_selection.KFold` |
| 重要度分析 | 衡量各輸入變數對預測結果之邊際貢獻 | 敏感度分析、置換重要性 | `sklearn.inspection.permutation_importance` |
| 結果比較 | 對照線性總效果與非線性重要度排名 | 正規化相對重要性、雙軸雷達圖 | `matplotlib` |

---

## 理論基礎篇

### 3.1 混合式研究架構的方法論動機

本課程第一模組（第 1–3 週）介紹之 PLS-SEM，其知識論基礎是**理論驗證取向**——研究者依循既有理論，事先假設構念間之因果路徑關係，再以資料檢驗這些假設路徑是否達統計顯著；本課程第三模組（第 8–10 週）介紹之機器學習方法，其知識論基礎則是**預測準確度取向**——不預設任何理論路徑限制，讓演算法自主從資料中學習最能準確預測目標變數的規則，即使這些規則涉及複雜的非線性或變數間交互補償關係。

**兩種取向各自之限制**：PLS-SEM 假設構念間關係為線性可加（linear and additive），若真實世界中變數間存在顯著的非線性效果或補償性關聯（compensatory relationship，即某一前因變數表現不佳時，可由另一前因變數之優異表現部分抵銷其對依變數之負面影響），PLS-SEM 之線性路徑係數估計可能無法完整捕捉此類複雜關係；反之，類神經網路雖能捕捉高度非線性關係、預測準確度通常更高，卻是典型的「黑盒子」模型（見第 9、10 週已討論之可解釋性議題），難以直接回答「這條因果路徑是否具有統計顯著性」此類理論驗證問題，也無法提供 PLS-SEM 完整之測量模型品質評鑑（收斂效度、區別效度等）。

**混合式研究架構（Two-Stage Hybrid Approach）**正是為了截長補短此二者而發展（見本週參考文獻中 Leong, Hew, Lee, & Ooi, 2015 之奠基性研究）：**第一階段以 PLS-SEM 檢驗理論假設路徑之顯著性，確保研究建立在嚴謹的理論驗證基礎上；第二階段僅萃取通過顯著性檢定之前因變數，作為類神經網路之輸入層，讓 ANN 專注於捕捉這些「已確認具理論意義」之變數與依變數之間可能存在的深層非線性關聯**，此一設計巧妙地讓兩種方法論在研究流程中各司其職、優勢互補，而非簡單並列比較。

### 3.2 兩階段混合研究法：SEM 作為變數篩選機制

延續 3.1 節之邏輯，SEM-ANN 混合架構之具體操作流程如下：

**第一階段（SEM）**：依循第 3 週所學之完整 PLS-SEM 分析流程（測量模型評鑑 → 結構模型路徑係數 → Bootstrapping 顯著性檢定），求得各前因構念對依變數之直接與間接路徑係數，並判斷其統計顯著性。

**變數篩選準則**：僅有在 SEM 階段通過顯著性檢定（ $p &lt; .05$ ，或 95% 信賴區間不含 0）之前因構念，才被視為具備理論與實證雙重支持之「候選預測變數」，納入第二階段 ANN 模型；未通過顯著性檢定之構念，則予以排除，不納入 ANN 輸入層。此一篩選機制之方法論價值在於：**避免 ANN 模型將未經理論驗證支持之雜訊變數也一併納入預測，確保混合模型之預測基礎仍根植於嚴謹之理論檢定，而非單純的資料探勘（data dredging）**。

**第二階段（ANN）**：以第一階段萃取之顯著前因變數（之構念分數）作為輸入層，依變數作為輸出層，訓練一個多層感知器模型，並透過敏感度分析求得各輸入變數之正規化相對重要性，藉此檢視 ANN 視角下的變數重要度排名，是否與 SEM 階段之線性路徑係數排名一致；若排名出現顯著落差，則暗示變數與依變數之間，可能存在傳統線性 SEM 路徑係數估計未能完整捕捉之非線性或補償性關聯，此一發現本身即具有重要的理論延伸價值。

### 3.3 類神經網路基礎：多層感知器結構與正向傳播

**多層感知器（Multilayer Perceptron, MLP）**是類神經網路家族中最基礎、也是 SEM-ANN 混合研究文獻中最廣泛採用之網路架構（見本週參考文獻），其結構包含三種層級：

- **輸入層（Input Layer）**：對應本週研究情境中，SEM 階段篩選出之各顯著前因構念分數，每一個輸入神經元對應一個前因變數。
- **隱藏層（Hidden Layer）**：一層或多層由人工神經元組成之中介層，透過非線性**激活函數（activation function，如 ReLU、Sigmoid）**，賦予網路捕捉非線性關係之能力——若無非線性激活函數，無論疊加多少層，整個網路在數學上仍等價於一個單純的線性模型，這正是類神經網路能捕捉非線性關係之數學關鍵。
- **輸出層（Output Layer）**：對應本週研究情境中之依變數（如忠誠度）預測值。

**正向傳播（Forward Propagation）**：資料從輸入層開始，逐層向前傳遞至輸出層之運算過程。對隱藏層中第 $j$ 個神經元：

$$
h_j = \sigma\left(\sum_{i=1}^{p} w_{ij} x_i + b_j\right)
$$

其中 $x_i$ 為輸入變數、 $w_{ij}$ 為連接輸入神經元 $i$ 與隱藏神經元 $j$ 之權重、 $b_j$ 為偏誤項（bias）、 $\sigma(\cdot)$ 為非線性激活函數。此一運算在每一層之間重複進行，直至輸出層產生最終預測值 $\hat{y}$ 。

### 3.4 反向傳播與模型訓練

類神經網路之參數（各層連接權重 $w$ 與偏誤項 $b$ ）學習，透過**反向傳播（Backpropagation）演算法**搭配**梯度下降法（Gradient Descent）**進行：

1. 以當前參數執行正向傳播，計算預測值 $\hat{y}$ 與實際值 $y$ 之間的**損失函數（loss function）**，迴歸問題常用均方誤差（Mean Squared Error, MSE）：

$$
L = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

2. 運用微積分之鏈鎖法則（chain rule），從輸出層反向逐層計算損失函數對每一個權重參數之梯度（偏導數），此即「反向傳播」名稱之由來。
3. 依梯度方向，以**學習率（learning rate）**控制之步伐大小，逐步調整所有權重參數，使損失函數值降低。
4. 重複步驟 1–3（每一輪完整的正向與反向傳播稱為一次疊代），直至損失函數收斂或達到預設之最大疊代次數。

`scikit-learn` 之 `MLPRegressor` 已將此完整訓練流程封裝為簡潔的 `.fit()` 方法，研究者僅需設定隱藏層結構、激活函數、求解器（solver）等超參數，無需手動實作反向傳播之數學運算細節，這也是本週 Colab 實作能以相對簡潔的程式碼完成 ANN 模型訓練之原因。

### 3.5 k 折交叉驗證

單一次的訓練測試集切分，其模型效能評估結果，可能因切分方式之隨機性而產生偏誤（尤其在本週研究情境常見之中小樣本規模下更為明顯）。**k 折交叉驗證（k-fold Cross-Validation）**是更穩健的模型效能評估方式，完整流程為：

1. 將全部樣本隨機均分為 $k$ 個大小相近的子集（本週提示詞實作要求 $k=10$ ，即十折交叉驗證）。
2. 依序以其中 $k-1$ 個子集作為訓練資料，訓練一個 ANN 模型，並以剩餘的 1 個子集作為測試資料，計算模型效能（如 $R^2$ ）。
3. 重複步驟 2 共 $k$ 次，確保每一個子集皆曾經作為測試集使用過一次。
4. 將 $k$ 次測試效能取平均數（與標準差），作為模型效能之最終穩健估計，相較單一次切分之評估結果，更能反映模型在不同資料子集上的穩定性與泛化能力。

### 3.6 敏感度分析與正規化相對重要性

**敏感度分析（Sensitivity Analysis）**之目的，是衡量 ANN 模型中，每一個輸入變數對輸出預測值之邊際影響程度，是 ANN 模型可解釋性之核心分析工具（呼應第 9、10 週已介紹之可解釋性 AI 議題）。本週 Colab 實作採用**置換重要性（Permutation Importance）**方法——此為現代機器學習可解釋性文獻中廣泛採用、且統計性質嚴謹之敏感度分析技術，其核心邏輯為：對已訓練完成之模型，逐一將某個輸入變數之數值隨機打亂（切斷其與依變數之真實對應關係），觀察模型預測效能（如 $R^2$ ）下降之幅度；下降幅度越大，代表該變數對模型預測越重要。

**正規化相對重要性（Normalized Importance, NI）**：為了讓不同變數之重要度數值具有直觀可比較性，SEM-ANN 混合研究文獻之標準做法（見本週參考文獻），是將每個變數之敏感度重要性數值，除以所有變數中最大之重要性數值，轉換為百分比形式：

$$
NI_j = \frac{Importance_j}{\max_i(Importance_i)} \times 100\%
$$

重要性最高之變數，其 $NI = 100\%$ ，其餘變數依相對比例呈現，此一正規化處理，正是本週提示詞實作要求計算之「正規化相對重要性（Normalized Importance %）」，也是本週理論篇 3.7 節與研究結果呈現中，用以對照 SEM 線性效果與 ANN 非線性重要度排名之標準化基礎。

### 3.7 SEM 線性效果與 ANN 非線性重要度的比較邏輯

完成兩階段分析後，SEM-ANN 混合研究之核心產出，是**對照 SEM 之線性總效果（direct + indirect effect，見第 3 週理論篇 3.6 節）排序與 ANN 之正規化相對重要性排序**。兩種排序結果之關係，可能出現以下型態：

| 對照結果 | 方法論意涵 |
|---|---|
| 兩者排序高度一致 | 該前因變數與依變數之關係，主要呈現線性可加特性，SEM 之估計已能良好捕捉此關係 |
| ANN 重要度顯著高於 SEM 相對排序 | 該變數可能與其他變數存在未被 SEM 線性模型捕捉之非線性交互作用或補償性關聯 |
| ANN 重要度顯著低於 SEM 相對排序 | 該變數在 SEM 中呈現顯著線性效果，但在納入其他變數之非線性模型情境下，其邊際預測貢獻反而被其他變數之非線性效果所稀釋 |

此一比較邏輯之學術價值，並非為了判定「哪一種方法比較正確」（兩者本就基於不同之統計假設與目的），而是**透過兩種方法論視角之對照，更完整地理解變數間關係之複雜性，為後續理論精進（如是否應在模型中納入交互作用項、是否應考慮非線性效果之理論解釋）提供實證線索**，這正是混合式研究方法相較單一方法論取徑，更具方法論嚴謹度與理論延伸價值之處。

### 3.8 論文中 SEM-ANN 章節的標準寫法架構

1. **第一階段 SEM 分析結果**：測量模型品質、結構路徑係數與顯著性檢定結果（可直接沿用第 3 週之報告架構）。
2. **顯著前因變數篩選說明**：明確列出通過顯著性檢定、納入第二階段之變數。
3. **第二階段 ANN 模型設定**：輸入層變數、隱藏層結構、激活函數、求解器等超參數說明。
4. **k 折交叉驗證效能報告**：各折 $R^2$ 值、平均值與標準差。
5. **敏感度分析與正規化相對重要性表**：各輸入變數之重要性數值與百分比排序。
6. **SEM 線性效果 vs. ANN 非線性重要度對照表與雙軸雷達圖**。
7. **兩階段結果對照之理論意涵討論**。

---

## 研究設計實例：行動支付使用者滿意度與忠誠度——PLS-SEM 驗證與 ANN 非線性權重補強

### 4.1 研究背景與動機

行動支付服務之使用者忠誠度，受知覺有用性、知覺易用性與信任等多重前因變數共同影響，然而既有台灣本土研究（見本週參考文獻）多僅採用單一 PLS-SEM 方法檢定線性路徑關係，較少進一步探討這些前因變數之間，是否存在傳統線性模型難以偵測之非線性或補償性關聯——例如，當使用者對行動支付平台之信任程度極高時，即使其知覺易用性表現普通，是否仍能維持高忠誠度（即信任對易用性產生某種補償效果）？此類問題，正是本週 SEM-ANN 混合架構所欲探討之核心研究問題。

### 4.2 研究目的

1. 以 PLS-SEM 檢驗知覺有用性、知覺易用性、信任、滿意度對行動支付忠誠度之路徑關係與顯著性。
2. 萃取 SEM 階段通過顯著性檢定之前因構念，作為類神經網路之輸入層。
3. 訓練具備 10 折交叉驗證之 MLP 模型，預測使用者忠誠度。
4. 計算敏感度分析與正規化相對重要性，並與 SEM 線性總效果進行對照分析。

### 4.3 研究構念架構

| 構念代碼 | 構念名稱 | 操作型定義 |
|---|---|---|
| PU | 知覺有用性 | 使用者認為行動支付服務能提升其交易效率與便利性之程度 |
| PEOU | 知覺易用性 | 使用者認為操作行動支付服務所需付出之認知努力程度 |
| TR | 信任 | 使用者對行動支付平台之交易安全性與服務可靠性之信任程度 |
| SAT | 滿意度 | 使用者對整體行動支付使用經驗之情感性評價 |
| LOY | 忠誠度 | 使用者持續使用並推薦該行動支付服務之行為意圖 |

### 4.4 研究假設與路徑架構

- H1：知覺有用性正向影響滿意度。
- H2：知覺易用性正向影響滿意度。
- H3：信任正向影響滿意度。
- H4：信任正向影響忠誠度。
- H5：滿意度正向影響忠誠度。

### 4.5 研究對象與資料來源

建議以有實際行動支付使用經驗（近 3 個月內至少 1 次使用紀錄）之消費者為研究對象，依循第 3 週理論篇 4.5 節之 PLS-SEM 樣本數規劃原則，樣本數建議至少 250–300 份以上，本課程範例採用模擬資料，樣本數 $n=350$ 。

### 4.6 資料分析流程規劃

```
【第一階段：SEM】
問卷資料（PU、PEOU、TR、SAT、LOY 各構念題項）
        │
        ▼
PLS-SEM 測量模型與結構模型分析（沿用第 3 週完整流程）
        │
        ▼
Bootstrapping 顯著性檢定，判斷各路徑是否顯著
        │
        ▼
萃取通過顯著性檢定之構念，取得其構念分數
        │
        ▼
【第二階段：ANN】
以顯著構念分數作為 MLP 輸入層，忠誠度為輸出層
        │
        ▼
10 折交叉驗證訓練 MLP 模型，評估預測效能
        │
        ▼
置換重要性敏感度分析 → 正規化相對重要性
        │
        ▼
【整合對照】
SEM 線性總效果排序 vs. ANN 非線性重要度排序
        │
        ▼
繪製雙軸雷達圖 + 撰寫理論意涵討論
```

---

## Colab 實作環境建置

```python
# ============================================================
# Cell 0：Colab 環境建置與套件安裝
# ------------------------------------------------------------
# 重要相容性說明：第一階段 PLS-SEM 沿用第 3 週所述之 plspm
# 套件相容性設定，須鎖定 pandas < 2.2 版本。執行完本 Cell
# 後，若 Colab 提示需要重新啟動執行階段，請務必重新啟動後
# 再繼續執行後續 Cell。
# ============================================================

!pip install "pandas<2.2" --quiet
!pip install plspm --quiet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(f"目前 pandas 版本：{pd.__version__}（應為 2.1.x）")

import plspm.config as spc
from plspm.plspm import Plspm
from plspm.scheme import Scheme
from plspm.mode import Mode

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

# ------------------------------------------------------------
# 設定中文字型（沿用第 1–10 週相同設定邏輯）
# ------------------------------------------------------------
!wget -q https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf -O /content/NotoSansTC.otf
from matplotlib import font_manager
font_manager.fontManager.addfont('/content/NotoSansTC.otf')
plt.rcParams['font.family'] = 'Noto Sans CJK TC'
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

print("環境建置完成，本週分兩階段：plspm（PLS-SEM）+ scikit-learn（ANN）。")
```

---

## Colab 實作：Step by Step 完整程式碼

### Step 1：模擬行動支付使用者問卷資料

```python
# ============================================================
# Cell 1：模擬行動支付滿意度與忠誠度研究之問卷資料
# ------------------------------------------------------------
# 教學目的：依 4.4 節研究假設路徑，模擬具已知因果結構之資料，
# 驗證兩階段分析流程能否正確還原此結構。正式研究請將本 Cell
# 替換為讀取真實問卷檔案的程式碼。
# ============================================================

np.random.seed(42)
n_samples = 350

PU = np.random.normal(0, 1, n_samples)
PEOU = np.random.normal(0, 1, n_samples)
TR = np.random.normal(0, 1, n_samples)
SAT = 0.35 * PU + 0.25 * PEOU + 0.30 * TR + np.random.normal(0, 0.6, n_samples)
LOY = 0.20 * TR + 0.55 * SAT + np.random.normal(0, 0.55, n_samples)

def generate_items(latent, loading, n_items, noise, prefix):
    items = {}
    for i in range(n_items):
        raw = loading * latent + np.random.normal(0, noise, len(latent))
        items[f'{prefix}{i}'] = raw
    return items

data = pd.DataFrame({
    **generate_items(PU, 0.80, 4, 0.5, 'pu'),
    **generate_items(PEOU, 0.78, 4, 0.5, 'peou'),
    **generate_items(TR, 0.82, 4, 0.5, 'tr'),
    **generate_items(SAT, 0.80, 3, 0.5, 'sat'),
    **generate_items(LOY, 0.83, 3, 0.5, 'loy'),
})

print(f"模擬問卷資料維度：{data.shape[0]} 位受訪者 × {data.shape[1]} 個題項")
data.head()
```

### Step 2：第一階段——PLS-SEM 模型設定與求解

```python
# ============================================================
# Cell 2：PLS-SEM 結構模型設定與求解（沿用第 3 週完整方法）
# ============================================================

structure = spc.Structure()
structure.add_path(["PU"], ["SAT"])
structure.add_path(["PEOU"], ["SAT"])
structure.add_path(["TR"], ["SAT", "LOY"])
structure.add_path(["SAT"], ["LOY"])

config = spc.Config(structure.path(), scaled=True)
config.add_lv_with_columns_named("PU", Mode.A, data, "pu")
config.add_lv_with_columns_named("PEOU", Mode.A, data, "peou")
config.add_lv_with_columns_named("TR", Mode.A, data, "tr")
config.add_lv_with_columns_named("SAT", Mode.A, data, "sat")
config.add_lv_with_columns_named("LOY", Mode.A, data, "loy")

sem_model = Plspm(data, config, Scheme.PATH, bootstrap=True, bootstrap_iterations=500)

print("=== 表 1：PLS-SEM 路徑係數矩陣 ===")
display(sem_model.path_coefficients().round(3))

print("\n=== 表 2：結構模型解釋力（R²）===")
display(sem_model.inner_summary()[['type', 'r_squared']].round(3))
```

### Step 3：Bootstrapping 顯著性檢定與變數篩選

```python
# ============================================================
# Cell 3：路徑顯著性檢定，篩選顯著前因變數
# ============================================================

bootstrap_paths = sem_model.bootstrap().paths()
bootstrap_paths['Significant'] = np.where(
    (bootstrap_paths['perc.025'] > 0) | (bootstrap_paths['perc.975'] < 0),
    '是（p < .05）', '否'
)

print("=== 表 3：路徑係數 Bootstrapping 顯著性檢定結果 ===")
display(bootstrap_paths.round(3))

hypothesis_map = {
    'PU -> SAT': 'H1', 'PEOU -> SAT': 'H2', 'TR -> SAT': 'H3',
    'TR -> LOY': 'H4', 'SAT -> LOY': 'H5'
}
print("\n=== 研究假設檢定結果 ===")
for path, hyp in hypothesis_map.items():
    row = bootstrap_paths.loc[path]
    result = '成立' if row['Significant'].startswith('是') else '不成立'
    print(f"{hyp}（{path}）：β = {row['original']:.3f}，{row['Significant']} → 假設{result}")

# 萃取通過顯著性檢定之總效果（direct + indirect），作為第二階段 ANN 輸入變數之依據
effects_table = sem_model.effects()
loy_effects = effects_table[effects_table['to'] == 'LOY'][['from', 'direct', 'indirect', 'total']]
print("\n=== 表 4：各構念對忠誠度（LOY）之總效果 ===")
display(loy_effects.round(3))

print("\n本研究之 PU、PEOU、TR、SAT 四項構念，其對 LOY 之路徑（直接或間接）")
print("皆通過顯著性檢定，故全數納入第二階段 ANN 模型作為輸入變數。")
```

### Step 4：萃取構念分數，準備 ANN 輸入資料

```python
# ============================================================
# Cell 4：萃取 PLS-SEM 構念分數，作為 ANN 輸入資料
# ------------------------------------------------------------
# 提示詞實作對照（前置步驟）：
# 「提取第 3 週之顯著前因變數作為類神經網路輸入層。」
# ============================================================

construct_scores = sem_model.scores()
print("=== 表 5：PLS-SEM 構念分數（前 5 筆）===")
display(construct_scores.head())

ann_input_vars = ['PU', 'PEOU', 'TR', 'SAT']   # 第一階段通過顯著性檢定之前因構念
ann_output_var = 'LOY'

X_raw = construct_scores[ann_input_vars].values
y_raw = construct_scores[ann_output_var].values

# 正規化至 0-1 區間，利於類神經網路訓練之數值穩定性
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

print(f"\nANN 輸入層維度：{X_scaled.shape[0]} 筆樣本 × {X_scaled.shape[1]} 個輸入變數")
print(f"輸入變數：{ann_input_vars}")
print(f"輸出變數：{ann_output_var}")
```

### Step 5：第二階段——10 折交叉驗證訓練 MLP 模型

```python
# ============================================================
# Cell 5：具備 10 折交叉驗證之 MLP 模型訓練
# ------------------------------------------------------------
# 提示詞實作對照：
# 「建立具備 10-fold 交叉驗證的 MLP 模型預測忠誠度。」
# ============================================================

kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_r2_scores = []
fold_rmse_scores = []

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_raw[train_idx], y_raw[test_idx]

    mlp = MLPRegressor(
        hidden_layer_sizes=(5,),   # 單一隱藏層，5 個神經元（因輸入變數僅 4 個，維持精簡架構）
        activation='relu',          # ReLU 激活函數，賦予網路捕捉非線性關係之能力
        solver='lbfgs',             # 適合中小樣本規模之最佳化求解器
        max_iter=2000,
        random_state=42
    )
    mlp.fit(X_train, y_train)
    pred = mlp.predict(X_test)

    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(np.mean((y_test - pred) ** 2))
    fold_r2_scores.append(r2)
    fold_rmse_scores.append(rmse)
    print(f"第 {fold_idx:>2} 折：R² = {r2:.4f}，RMSE = {rmse:.4f}")

cv_results = pd.DataFrame({
    'Fold': range(1, 11), 'R2': fold_r2_scores, 'RMSE': fold_rmse_scores
})
print("\n=== 表 6：10 折交叉驗證結果摘要 ===")
display(cv_results.round(4))
print(f"\n平均 R² = {np.mean(fold_r2_scores):.4f}（標準差 = {np.std(fold_r2_scores):.4f}）")
print(f"平均 RMSE = {np.mean(fold_rmse_scores):.4f}")
```

### Step 6：訓練最終模型並執行敏感度分析

```python
# ============================================================
# Cell 6：最終模型訓練與敏感度分析
# ------------------------------------------------------------
# 提示詞實作對照：
# 「計算各變數的靈敏度與正規化相對重要性（Normalized
#   Importance %）。」
# ============================================================

final_mlp = MLPRegressor(
    hidden_layer_sizes=(5,), activation='relu', solver='lbfgs',
    max_iter=2000, random_state=42
)
final_mlp.fit(X_scaled, y_raw)
final_r2 = r2_score(y_raw, final_mlp.predict(X_scaled))
print(f"最終模型（全樣本配適）R² = {final_r2:.4f}")

# 置換重要性敏感度分析（見理論篇 3.6 節）
perm_result = permutation_importance(
    final_mlp, X_scaled, y_raw, n_repeats=30, random_state=42
)
raw_importance = pd.Series(perm_result.importances_mean, index=ann_input_vars)

# 正規化相對重要性：除以最大值，轉換為百分比
normalized_importance = (raw_importance / raw_importance.max()) * 100

ann_importance_table = pd.DataFrame({
    'Raw_Importance': raw_importance,
    'Normalized_Importance(%)': normalized_importance
}).sort_values('Normalized_Importance(%)', ascending=False)

print("\n=== 表 7：ANN 敏感度分析與正規化相對重要性 ===")
display(ann_importance_table.round(2))
```

### Step 7：SEM 與 ANN 結果對照表

```python
# ============================================================
# Cell 7：SEM 線性總效果 vs. ANN 非線性重要度對照表
# ------------------------------------------------------------
# 提示詞實作對照：
# 「SEM 線性係數 vs. ANN 非線性重要度排名對照表。」
# ============================================================

sem_total_effects = loy_effects.set_index('from')['total']
sem_normalized = (sem_total_effects / sem_total_effects.max()) * 100

comparison_table = pd.DataFrame({
    'SEM_Total_Effect': sem_total_effects,
    'SEM_Normalized(%)': sem_normalized,
    'ANN_Normalized_Importance(%)': normalized_importance
}).loc[ann_input_vars].sort_values('SEM_Normalized(%)', ascending=False)

comparison_table['Rank_Difference'] = (
    comparison_table['SEM_Normalized(%)'].rank(ascending=False)
    - comparison_table['ANN_Normalized_Importance(%)'].rank(ascending=False)
)

print("=== 表 8：SEM 線性總效果 vs. ANN 正規化重要度對照表 ===")
display(comparison_table.round(2))
```

### Step 8：雙軸雷達圖視覺化

```python
# ============================================================
# Cell 8：SEM vs. ANN 雙軸雷達圖
# ============================================================

categories = ann_input_vars
n_cat = len(categories)
angles = np.linspace(0, 2 * np.pi, n_cat, endpoint=False).tolist()
angles_closed = angles + [angles[0]]

sem_vals = comparison_table.loc[categories, 'SEM_Normalized(%)'].tolist()
sem_vals_closed = sem_vals + [sem_vals[0]]
ann_vals = comparison_table.loc[categories, 'ANN_Normalized_Importance(%)'].tolist()
ann_vals_closed = ann_vals + [ann_vals[0]]

fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
ax.plot(angles_closed, sem_vals_closed, 'o-', linewidth=2.5, label='SEM 線性總效果（正規化）', color='#2980b9')
ax.fill(angles_closed, sem_vals_closed, alpha=0.15, color='#2980b9')
ax.plot(angles_closed, ann_vals_closed, 's--', linewidth=2.5, label='ANN 正規化重要度', color='#e74c3c')
ax.fill(angles_closed, ann_vals_closed, alpha=0.15, color='#e74c3c')

ax.set_xticks(angles)
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylim(0, 105)
ax.set_title('圖 1：SEM 線性總效果 vs. ANN 非線性重要度 雙軸雷達圖', fontsize=13, pad=25)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
plt.tight_layout()
plt.savefig('sem_ann_radar.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Step 9：匯出所有分析結果

```python
# ============================================================
# Cell 9：匯出完整分析結果至 Excel
# ============================================================

with pd.ExcelWriter('SEM_ANN分析結果_Week11.xlsx') as writer:
    sem_model.path_coefficients().round(3).to_excel(writer, sheet_name='SEM路徑係數')
    bootstrap_paths.round(3).to_excel(writer, sheet_name='SEM顯著性檢定')
    loy_effects.round(3).to_excel(writer, sheet_name='SEM總效果', index=False)
    cv_results.round(4).to_excel(writer, sheet_name='ANN十折交叉驗證', index=False)
    ann_importance_table.round(2).to_excel(writer, sheet_name='ANN正規化重要度')
    comparison_table.round(2).to_excel(writer, sheet_name='SEM_ANN對照表')

print("所有統計結果已匯出至 SEM_ANN分析結果_Week11.xlsx，可於 Colab 左側檔案面板下載。")
print(f"\nSEM 階段：LOY 之 R² = {sem_model.inner_summary().loc['LOY','r_squared']:.3f}")
print(f"ANN 階段：10 折平均 R² = {np.mean(fold_r2_scores):.3f}")
print(f"ANN 最重要前因變數：{ann_importance_table.index[0]}")
```

---

## Vibe Coding 提示詞（Prompt）實作範例集

**範例 1：SEM 顯著變數篩選**

> 我已完成 PLS-SEM 分析，請幫我從 Bootstrapping 顯著性檢定結果中，篩選出所有對忠誠度（LOY）具有顯著直接或間接總效果的前因構念，並列出這些構念的名稱，作為後續類神經網路模型的輸入變數清單。

**範例 2：10 折交叉驗證 MLP**

> 請使用 scikit-learn 的 MLPRegressor，以剛才篩選出的顯著前因構念分數作為輸入（記得先做 Min-Max 正規化），忠誠度作為輸出，建立一個具備 10 折交叉驗證的類神經網路迴歸模型，並回報每一折與平均的 R² 及 RMSE。

**範例 3：敏感度分析與正規化重要性**

> 請對訓練好的 MLP 模型執行置換重要性分析，計算每個輸入變數的原始重要性數值，並將其除以最大值轉換為百分比形式的正規化相對重要性，輸出成一張由高到低排序的表格。

**範例 4：SEM 與 ANN 對照**

> 請將 SEM 階段各構念對忠誠度的總效果（正規化為百分比）與 ANN 階段的正規化相對重要性整理成同一張對照表，並計算兩種排序方法的名次差異，幫我判斷哪些構念在兩種方法下的重要度排名出現明顯落差。

**範例 5：雙軸雷達圖**

> 請將 SEM 正規化總效果與 ANN 正規化重要度，繪製在同一張雷達圖上，兩組數據請用不同顏色與不同線型（實線 vs 虛線）區分並加上圖例，方便直接比較兩種方法對同一組前因變數的重要度判讀是否一致。

**範例 6：結果段落初稿撰寫**

> 根據以下統計結果（SEM：SAT→LOY β=.485為最強路徑，TR對LOY總效果.381次之；ANN：SAT正規化重要度100%，TR僅32.1%，PU從SEM排名第2掉到ANN排名第4；10折平均R²=.295），請以碩士論文研究結果章節的學術寫作語氣，撰寫一段約 300 字的中文分析段落，說明兩階段結果對照的理論意涵。

---

## 結果呈現與分析：碩士論文寫法示例

以下段落數值取自本週 Colab 範例程式碼之實際執行結果，供學生對照模仿寫作邏輯（實際數值請以自己資料之 Colab 輸出為準）。

> **4.1 第一階段：PLS-SEM 路徑分析結果**
>
> 本研究第一階段以 PLS-SEM 檢驗研究假設，經 500 次 Bootstrapping 重抽樣檢定，結果顯示：知覺有用性對滿意度具顯著正向影響（β = .385，t = 9.45），支持 H1；知覺易用性對滿意度具顯著正向影響（β = .259，t = 6.98），支持 H2；信任對滿意度（β = .327，t = 7.48）與忠誠度（β = .222，t = 5.05）均具顯著正向影響，支持 H3、H4；滿意度對忠誠度具顯著正向影響（β = .485，t = 9.47），支持 H5。模型對滿意度與忠誠度之解釋力分別為 R² = .325 與 R² = .359。由於四項前因構念（知覺有用性、知覺易用性、信任、滿意度）對忠誠度之路徑（直接或間接）均通過顯著性檢定，故全數納入第二階段類神經網路模型作為輸入變數。
>
> **4.2 第二階段：ANN 模型預測效能**
>
> 以第一階段篩選之四項顯著構念分數作為輸入層，忠誠度作為輸出層，訓練具備 10 折交叉驗證之多層感知器模型（單一隱藏層，5 個神經元，ReLU 激活函數），10 折測試集之平均 R² 為 .295（標準差 = .142），顯示模型在不同資料子集上之預測效能存在一定波動，此一現象反映了中小樣本規模下類神經網路模型預測穩定度之常見限制，亦提醒研究者在報告 ANN 效能時，應同時呈現各折結果之離散程度，而非僅報告單一平均值。
>
> **4.3 SEM 與 ANN 結果對照：非線性關聯之發現**
>
> 將 SEM 階段各構念對忠誠度之正規化總效果，與 ANN 階段之正規化相對重要性對照，結果顯示：滿意度在兩種方法下均為最重要之前因變數（正規化重要度皆為 100%），顯示滿意度對忠誠度之影響關係，主要呈現穩定之線性可加特性；然而，信任之 SEM 正規化總效果達 78.6%（總效果 .381，次於滿意度），但其 ANN 正規化重要度僅為 32.1%，排名落差達兩位；知覺有用性之落差更為明顯，SEM 正規化總效果為 38.6%，ANN 正規化重要度卻僅有 2.3%，降至四項變數中最低。此一發現顯示，信任與知覺有用性對忠誠度之預測貢獻，在納入其他變數之非線性模型情境下，其邊際貢獻相對被滿意度之強力線性效果所稀釋，暗示滿意度可能在忠誠度形成過程中，扮演比 SEM 線性路徑係數所呈現更具支配性之角色，此一發現為後續研究是否應進一步檢驗滿意度之中介或調節效果，提供了具體的實證線索。

**APA 格式三線表範例：SEM 與 ANN 對照表**

| 構念 | SEM 總效果 | SEM 正規化（%） | ANN 正規化重要度（%） | 排名差異 |
|---|---|---|---|---|
| 滿意度（SAT） | .485 | 100.0 | 100.0 | 0 |
| 信任（TR） | .381 | 78.6 | 32.1 | -1 |
| 知覺有用性（PU） | .187 | 38.6 | 2.3 | -2 |
| 知覺易用性（PEOU） | .125 | 25.8 | 8.7 | +1 |

*註：以上數值為本週 Colab 範例實際執行結果，實際研究請以自己資料之輸出為準。ANN 10 折平均 R² = .295（SD = .142）。*

---

## 常見統計誤區與 Q&A

**Q1：SEM 階段的 R²（.359）跟 ANN 階段的平均 R²（.295），為什麼 ANN 反而比較低？不是說類神經網路預測力通常比較強嗎？**
這是一個很好的觀察，但不能簡單類比兩個 R² 數值。SEM 之 R² 是在**全部樣本**上配適得出的模型解釋力；ANN 之 10 折交叉驗證 R²，則是在**每次皆為模型訓練時完全未見過的測試子集**上計算而得，兩者之評估基礎不同（樣本內配適 vs. 樣本外預測），本質上並不適合直接比較數值大小。此外，本週範例之樣本數（ $n=350$ ）與輸入變數數（4 個）相對精簡，ANN 模型之隱藏層神經元數量也刻意設定得較為保守（僅 5 個），這是為了在有限樣本下降低過度配適風險所做的取捨，若樣本規模更大、模型架構更複雜（如增加隱藏層層數），ANN 之樣本外預測力有可能進一步提升，但也需要更謹慎地控制過度配適風險。文獻上（見本週參考文獻）確實有研究報告 ANN 之 $R^2$ 顯著高於 PLS-SEM，但也有研究呈現較為接近或甚至較低之結果，此一差異高度取決於資料本身之線性／非線性程度與樣本規模，不宜一概而論。

**Q2：為什麼要先做 SEM 篩選顯著變數，再做 ANN？可以省略 SEM 階段，直接把所有可能的前因變數都丟進 ANN 訓練嗎？**
技術上可行，但這樣做會喪失混合式研究架構的核心方法論價值。如理論篇 3.2 節所述，SEM 階段之顯著性篩選機制，確保納入 ANN 的每一個輸入變數，都已通過嚴謹的理論假設檢定，使整個研究建立在「理論驗證」與「預測補強」雙重基礎之上；若省略此一篩選步驟、直接將所有變數（包含未經理論支持者）都納入 ANN，雖然可能在單純預測準確度上略有提升，但整個研究會退化為純粹的資料探勘，喪失了 SEM-ANN 混合方法論相較單純機器學習方法，在理論貢獻與學術嚴謹度上的核心優勢，這也是本週提示詞實作特別強調「提取第 3 週之顯著前因變數」此一步驟的方法論用意。

**Q3：本週的隱藏層只設定 5 個神經元、1 層，會不會太簡單了？可以增加更多層、更多神經元嗎？**
可以，但應謹慎權衡。理論上，增加隱藏層層數與神經元數量，能讓類神經網路捕捉更複雜之非線性關係，但也會大幅增加模型參數量，在本週範例這種中小樣本規模（ $n=350$ ，僅 4 個輸入變數）下，過於複雜的網路結構容易導致過度配適（模型在訓練資料上表現優異，但泛化至新資料的能力反而下降），這也是本週範例刻意採用精簡架構（單層、5 個神經元）之設計考量。SEM-ANN 混合研究文獻中，多數研究採用之隱藏層神經元數量，通常介於輸入變數數的 1 至 2 倍之間，作為兼顧模型表達能力與過度配適風險之經驗法則，實務上仍建議透過交叉驗證比較不同網路架構之效能，而非單純採用越複雜越好之直覺判斷。

**Q4：置換重要性（Permutation Importance）和第 9、10 週學過的特徵重要性、SHAP 值，都是在講「哪個變數重要」，這些方法之間有什麼關係？**
這是本課程第三模組（第 8–11 週）一以貫之的核心方法論主題——「模型可解釋性」，置換重要性、隨機森林原生特徵重要性（第 9 週）與 SHAP 值（第 10 週），都屬於廣義的特徵重要性衡量方法，但各自之統計原理與適用範圍不同：隨機森林原生重要性衡量特徵在樹分割過程中降低不純度之平均貢獻，僅適用於樹狀模型；SHAP 值基於 Shapley 值理論，具備嚴謹的可加性數學保證，可適用於任意模型（搭配對應之 Explainer）；本週採用之置換重要性，則是一種更為通用、不依賴模型內部結構、僅需觀察「打亂某變數後預測效能下降多少」之黑盒子診斷方法，適用於包含類神經網路在內之任意模型類型，這正是本週選用置換重要性作為 ANN 敏感度分析工具之原因——類神經網路不像決策樹具有可直接拆解的樹狀分割結構，難以直接套用第 9、10 週介紹之方法，置換重要性提供了一個模型無關（model-agnostic）的替代方案。

**Q5：如果 SEM 的顯著性檢定結果顯示某個前因變數「不顯著」，是不是就完全不用管這個變數了？**
不一定，這取決於研究目的與後續分析設計。若研究目的單純是建構最終之混合預測模型，則遵循本週理論篇 3.2 節之標準做法，將不顯著變數排除於 ANN 輸入層之外是合理的；但若研究者對該不顯著變數之理論角色仍有進一步探究興趣（例如懷疑其效果可能透過與其他變數之非線性交互作用才會顯現，而非單純的直接線性效果），仍可考慮將其保留於 ANN 模型中進行探索性分析，並在論文中明確說明此為超出標準兩階段流程之額外探索性嘗試，而非直接遵循「不顯著即排除」之機械式規則，畢竟 SEM 之顯著性檢定是在線性可加之假設下進行，本就有可能低估具有非線性效果之變數的真實重要性，這也是本週 4.3 節「TR 與 PU 在兩種方法下排名落差」發現所啟發之進一步思考方向。

---

## 延伸研究方向：智慧觀光平台體驗滿意度前因分析——SEM-ANN 雙階段混合模型

### 6.1 研究背景與理論基礎

智慧觀光平台（結合行動應用程式、擴增實境導覽、個人化推薦系統等智慧科技之觀光服務平台）之遊客體驗滿意度，同樣可能受多重前因變數之非線性與補償性關聯影響——例如當平台之個人化推薦精準度極高時，即使介面操作稍嫌複雜，遊客滿意度是否仍能維持在高水準（即推薦精準度對操作複雜度之補償效果）？此一研究問題情境，與本週研究範例之行動支付忠誠度模型高度類似。本週參考文獻中之運動觀光社群媒體重遊意願 SEM-ANN 研究，已示範將此混合方法應用於觀光體驗情境，可作為本延伸研究方向之直接方法論參照。

### 6.2 建議研究設計

1. **理論框架**：延續本週兩階段 SEM-ANN 混合架構，第一階段以 PLS-SEM 檢驗智慧觀光平台體驗前因變數對滿意度（及後續行為意圖）之路徑關係。
2. **候選前因構念（範例）**：
   - 個人化推薦精準度
   - 擴增實境導覽沉浸感
   - 平台操作便利性
   - 即時資訊更新可靠度
   - 社群互動功能豐富度
3. **依變數**：遊客體驗滿意度，可進一步延伸至重遊意願或口碑推薦意圖。
4. **研究對象**：建議以實際使用過特定智慧觀光平台（如國家風景區官方 App、智慧博物館導覽系統）之遊客為研究對象。
5. **分析流程**：完全比照本週 Colab 實作流程（PLS-SEM 顯著性篩選 → MLP 十折交叉驗證 → 置換重要性敏感度分析 → SEM/ANN 雙軸雷達圖對照），僅需替換構念定義與問卷資料，即可直接複用本週所有程式碼架構。
6. **管理實務意涵**：此類研究可協助智慧觀光平台營運單位，辨識出哪些體驗構面之改善投資，在納入非線性補償效果考量後，實際上對滿意度提升之邊際效益可能被低估或高估，較單純依賴 SEM 線性路徑係數之投資優先順序判斷更具決策參考價值。

### 6.3 給學生的思考練習

請思考：本週研究範例與延伸研究方向，皆屬於「消費者／使用者滿意度與忠誠度」之研究主題類型。若你的期末專題研究主題，並非典型的滿意度忠誠度模型（例如是第 9 週之財務危機預測、或第 10 週之房地產估價），你認為 SEM-ANN 混合架構是否仍然適用？在什麼樣的研究情境下，兩階段混合方法的方法論價值最能充分發揮？在什麼樣的情境下，可能不必要或不適合採用此一混合架構？

---

## 課後作業與練習

**練習一：非線性效果強度敏感度分析**
請修改 Step 1 中資料生成邏輯，額外加入一個 TR 與 SAT 之交互作用項（例如在 LOY 之生成公式中加入 `0.3 * TR * SAT` 之非線性項），重新執行完整兩階段分析流程，觀察 SEM 與 ANN 兩種方法之重要度排名落差是否較原範例更為明顯，並說明你觀察到的規律。

**練習二：真實問卷資料實作**
請延續第 3 週已蒐集或設計之 PLS-SEM 問卷資料架構（至少包含 3 個前因構念與 1 個依變數），實際發放蒐集後，套用本週完整兩階段 Colab 程式碼執行 SEM-ANN 混合分析。請繳交：(1) 原始 CSV 檔、(2) 執行後的 Colab Notebook（.ipynb）、(3) 一頁 A4 的結果摘要（比照本週「結果呈現與分析」段落之寫法）。

**練習三：隱藏層架構敏感度分析**
請將 Step 5、6 中 MLP 之隱藏層設定，分別調整為 `(3,)`、`(8,)`、`(5,5)`（兩層）三種不同架構，重新執行 10 折交叉驗證，比較不同架構下平均 R² 與標準差之變化，並討論模型複雜度與過度配適風險之權衡。

**練習四：文獻延伸閱讀報告**
請從本週「參考文獻與延伸閱讀」清單中，任選一篇 SEM-ANN 混合方法應用之期刊論文，撰寫一頁重點摘要，內容須包含：(1) 該研究之 SEM 階段構念與路徑假設、(2) 通過顯著性檢定並納入 ANN 之變數、(3) 報告之 ANN 正規化相對重要性排序、(4) SEM 與 ANN 排序是否一致，該研究如何討論此一發現之理論意涵。

**練習五：跨模型可解釋性方法比較**
請結合第 9 週（隨機森林特徵重要性）、第 10 週（SHAP 值）與本週（置換重要性）三種方法，對同一組資料（如本週模擬資料，將依變數改為二元分類問題）分別計算特徵重要性，比較三種方法之排序結果是否一致，並綜合說明三種方法各自之理論基礎與適用情境差異。

---

## 參考文獻與延伸閱讀（已查核連結）

1. Leong, L.-Y., Hew, T.-S., Lee, V.-H., & Ooi, K.-B. (2015). An SEM-artificial-neural-network analysis of the relationships between SERVPERF, customer satisfaction and loyalty among low-cost and full-service airline. *Expert Systems with Applications*, 42(19), 6620-6634.
   https://www.academia.edu/83773314/Predicting_the_determinants_of_mobile_payment_acceptance_A_hybrid_SEM_neural_network_approach
   （SEM-ANN 混合研究方法之奠基性經典論文，提出兩階段混合研究架構之核心邏輯，為本週理論篇 3.1、3.2 節之直接方法論依據。）

2. How to generate loyalty in mobile payment services? An integrative dual SEM-ANN analysis.
   https://cronfa.swan.ac.uk/Record/cronfa62978
   （直接以行動支付忠誠度為研究主題之 SEM-ANN 混合方法期刊論文，與本週研究設計範例主題完全對應，可作為文獻回顧核心參照。）

3. Deciphering Loyalty in P2P M-Payment Platforms: A Combined SEM and Artificial Neural Network Investigation.
   https://doi.org/10.2139/ssrn.4786584
   （行動支付忠誠度之 SEM-ANN 混合研究，並納入企業社會責任與品牌識別等延伸構念，可作為研究架構延伸之參考範本。）

4. 影響行動支付使用意圖之研究。臺灣博碩士論文知識加值系統。
   https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi/login?o=dnclcdr&s=id%3D%22106CSU00399010%22.&searchmode=basic
   （以 UTAUT2 理論為基礎、SmartPLS 進行分析之台灣本土行動支付使用意圖研究，可作為本週研究設計範例之構念設計與台灣本土情境參照。）

5. The impact of social media on university students' revisit intention in sports tourism: A hybrid method based on SEM and ANN.
   https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12040268/
   （以 SEM-ANN 混合方法探討運動觀光重遊意願之期刊論文，並完整報告 ANN 正規化重要度排序，為本週延伸研究方向「智慧觀光平台」之直接方法論參照。）

6. Predicting the determinants of mobile payment acceptance: A hybrid SEM-neural network approach.
   https://www.researchgate.net/publication/322150284_Predicting_the_determinants_of_mobile_payment_acceptance_A_hybrid_SEM-neural_network_approach
   （行動支付採用決定因素之 SEM-ANN 混合研究，示範敏感度分析與模型效能比較之完整報告方式。）

**方法論經典文獻（建議延伸閱讀，非本次線上搜尋來源，圖書館或資料庫可查閱）**：

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.
- Chong, A. Y.-L. (2013). Predicting m-commerce adoption determinants: A neural network approach. *Expert Systems with Applications*, 40(2), 523-530.
- Hair, J. F., Sarstedt, M., Ringle, C. M., & Gudergan, S. P. (2018). *Advanced Issues in Partial Least Squares Structural Equation Modeling*. SAGE Publications.
- Sarstedt, M., & Mooi, E. (2019). Cluster analysis. In *A Concise Guide to Market Research* (pp. 301-354). Springer.

---

## 附錄

### 附錄 A：PLS-SEM（第 3 週）與 ANN（本週）方法對照表

| 比較構面 | PLS-SEM（第 3 週） | ANN（本週） |
|---|---|---|
| 知識論基礎 | 理論驗證、假設檢定 | 預測準確度最大化 |
| 關係假設 | 線性可加 | 可捕捉非線性、補償性關聯 |
| 統計顯著性檢定 | 有（Bootstrapping） | 無直接對應機制 |
| 可解釋性 | 高（完整路徑係數與測量模型） | 中（僅能透過敏感度分析等事後方法） |
| 適用樣本規模 | 中小樣本即可（10 倍法則） | 樣本規模越大越能發揮優勢 |
| 混合架構中之角色 | 第一階段：變數篩選與理論驗證 | 第二階段：非線性關係補強 |

### 附錄 B：術語中英對照表

| 中文術語 | 英文術語 | 縮寫 |
|---|---|---|
| 兩階段混合研究法 | Two-Stage Hybrid Approach | — |
| 多層感知器 | Multilayer Perceptron | MLP |
| 正向傳播 | Forward Propagation | — |
| 反向傳播 | Backpropagation | — |
| 梯度下降法 | Gradient Descent | — |
| 激活函數 | Activation Function | — |
| 隱藏層 | Hidden Layer | — |
| k 折交叉驗證 | k-fold Cross-Validation | — |
| 敏感度分析 | Sensitivity Analysis | — |
| 置換重要性 | Permutation Importance | — |
| 正規化相對重要性 | Normalized Importance | NI |
| 補償性關聯 | Compensatory Relationship | — |

### 附錄 C：常見程式錯誤排解（Debugging Tips）

| 錯誤現象 | 常見原因 | 排解建議 |
|---|---|---|
| `MLPRegressor` 訓練時出現不收斂警告 | `max_iter` 設定過低，或學習率設定不當 | 提高 `max_iter`（如本週範例之 2000），或改用 `solver='adam'` 搭配 `learning_rate_init` 調整 |
| 10 折交叉驗證各折 R² 差異極大 | 樣本數過少，導致每折測試集樣本數不足，估計不穩定 | 檢查樣本數是否足夠（建議至少 200–300 筆），或改用重複 k 折交叉驗證取得更穩定估計 |
| `permutation_importance` 計算結果每次執行不同 | 未設定 `random_state` 參數 | 確認 `permutation_importance()` 呼叫時已指定固定隨機種子 |
| SEM 構念分數與 ANN 輸入資料維度不吻合 | `construct_scores` 欄位順序與 `ann_input_vars` 清單不一致 | 使用 `construct_scores[ann_input_vars]` 明確指定欄位順序後再轉換為 `.values` |
| 雙軸雷達圖兩條線幾乎重疊、難以比較 | SEM 與 ANN 正規化計算基準不一致（如其中一者忘記乘以 100） | 確認兩組資料皆已正確轉換為 0–100 之百分比尺度 |

### 附錄 D：繳交前自我檢核清單

- [ ] 已完整報告第一階段 PLS-SEM 之路徑係數與顯著性檢定結果
- [ ] 已明確列出通過顯著性檢定、納入 ANN 之前因變數
- [ ] 已說明 ANN 模型之超參數設定（隱藏層結構、激活函數、求解器）
- [ ] 已報告 10 折交叉驗證之各折與平均效能指標
- [ ] 已計算並報告敏感度分析與正規化相對重要性
- [ ] 已將 SEM 線性總效果與 ANN 正規化重要度整理為對照表
- [ ] 已繪製雙軸雷達圖呈現兩種方法之重要度排序對照
- [ ] 已針對排名落差較大之變數，提出具體之理論意涵討論
- [ ] 所有統計結果之文字敘述與表格數值一致，無謄寫錯誤
- [ ] 已在研究限制中說明 ANN 模型可解釋性之侷限性

### 附錄 E：研究倫理提醒

本週研究設計涉及消費者行動支付使用資料（或延伸研究方向中之遊客體驗資料），除延續前十週已說明之知情同意、匿名性等基本倫理原則外，特別提醒：混合式研究方法因涉及兩階段分析，資料蒐集階段應確保樣本規模足以支撐兩個階段（尤其是對樣本規模較敏感之 ANN 階段）之分析需求，避免因中途發現樣本不足而需要重新蒐集資料所造成之受訪者重複參與負擔；此外，SEM-ANN 混合分析結果若涉及對特定消費者群體之差異化行銷或服務策略建議，應同樣留意第 8 週附錄已提及之演算法應用公平性議題，避免資料驅動之決策建議對特定族群造成非預期之不利影響。

---

## 第三模組總結：資料驅動型 AI 方法家族回顧

隨著本週課程結束，本課程第三模組（第 8–11 週）之資料驅動型 AI 方法已完整介紹完畢。以下總結表回顧整個模組之方法論演進脈絡：

| 週次 | 方法 | 學習性質 | 核心產出 | 可解釋性工具 |
|---|---|---|---|---|
| 第 8 週 | K-Means / PCA | 非監督式學習 | 客戶／個案分群結構 | 群集輪廓表 |
| 第 9 週 | 決策樹 / 隨機森林 | 監督式學習（分類） | 風險預警分類模型 | 特徵重要性（Gini/Entropy） |
| 第 10 週 | XGBoost / LightGBM | 監督式學習（迴歸／分類） | 高精度預測模型 | SHAP（Shapley 值） |
| 第 11 週 | SEM-ANN | 混合式（理論驗證＋預測補強） | 兩階段驗證與非線性補強模型 | 置換重要性、正規化相對重要性 |

**第三模組與第一、二模組之整合回顧**：本課程至此已完整涵蓋三大方法論典範——第一模組（第 1–3 週）之資料驅動型統計推論（EFA、迴歸、PLS-SEM）、第二模組（第 4–7 週）之知識驅動型決策科學（AHP、DEMATEL、DANP-VIKOR、BWM/IFS）、第三模組（第 8–11 週）之資料驅動型機器學習（分群、集成學習、梯度提升、混合式 AI）。本週之 SEM-ANN 架構，正是首次將第一模組與第三模組之方法論加以正式整合的示範，學生應能體會到：**優秀的量化研究者，不應將自己侷限於單一方法論典範，而應依研究問題之本質，靈活調度、甚至創造性地整合不同方法論工具，這正是本課程「決策智能與混合式 AI」核心定位之終極體現**。

---

## 下週預告

第 12 週將進入第四模組「深度學習與生成式 AI 決策智能」，主題為「時序深度學習模型：長短期記憶網路（LSTM）趨勢模擬」。學生將從本週之淺層類神經網路（MLP），進一步學習能處理序列與時間相依資料之深度學習架構——循環神經網路（RNN）與其改良版本長短期記憶網路（LSTM），理解其門控機制（遺忘門、輸入門、輸出門）如何解決傳統 RNN 之長期依賴問題。研究範例將以「半導體產業鏈關鍵原物料價格震盪與需求預測模型」為主題，並延伸至「綠能電廠發電量預測與儲能調度排程策略模擬」之期末專題發想方向，正式開啟本課程最後一個模組——深度學習與生成式 AI 之學習旅程。
