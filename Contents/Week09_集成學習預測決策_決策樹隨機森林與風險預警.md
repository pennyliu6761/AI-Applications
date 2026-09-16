# 第 9 週：集成學習預測決策——決策樹、隨機森林與風險預警

> 課程模組：第三模組｜資料驅動 AI：預測、分群與可解釋性模型（第 8–11 週）
> 本週定位：承接第 8 週非監督式分群之基礎，正式進入監督式學習（Supervised Learning）情境——當資料具有明確的標籤（如企業是否發生財務危機）時，如何運用決策樹之資訊獲利與基尼不純度分割準則，建構具可解釋性的分類模型，並透過隨機森林之集成學習（Bagging）架構大幅提升預測穩定度與準確度，同時學習處理實務資料中極為常見的類別不平衡問題，是企業風險預警系統建構之核心方法論基礎。

> 教材版本：v1.0｜適用對象：在職專班研究方法與 AI 應用課程｜先修基礎：第 8 週（非監督式學習：K-Means/PCA）
> 使用工具：Google Colab（Python 3）｜主要套件：`scikit-learn`（`DecisionTreeClassifier`、`RandomForestClassifier`、分類評估指標）、`imbalanced-learn`（`SMOTE`）、`pandas`、`matplotlib`、`seaborn`（本週所有程式碼已實際測試驗證可正常執行）

---

## 目錄

1. [學習目標](#學習目標)
2. [本週知識地圖](#本週知識地圖)
3. [理論基礎篇](#理論基礎篇)
   1. [3.1 從非監督式到監督式學習：分群到分類的橋接](#31-從非監督式到監督式學習分群到分類的橋接)
   2. [3.2 決策樹演算法原理：遞迴分割與停止準則](#32-決策樹演算法原理遞迴分割與停止準則)
   3. [3.3 資訊獲利與資訊增益](#33-資訊獲利與資訊增益)
   4. [3.4 基尼不純度](#34-基尼不純度)
   5. [3.5 集成學習與 Bagging 架構：隨機森林原理](#35-集成學習與-bagging-架構隨機森林原理)
   6. [3.6 類別不平衡問題與處理策略](#36-類別不平衡問題與處理策略)
   7. [3.7 分類模型評估指標](#37-分類模型評估指標)
   8. [3.8 特徵重要性判讀](#38-特徵重要性判讀)
   9. [3.9 論文中分類預測模型章節的標準寫法架構](#39-論文中分類預測模型章節的標準寫法架構)
4. [研究設計實例：企業財務危機早期預警機制與違約風險預測模型](#研究設計實例企業財務危機早期預警機制與違約風險預測模型)
5. [Colab 實作環境建置](#colab-實作環境建置)
6. [Colab 實作：Step by Step 完整程式碼](#colab-實作step-by-step-完整程式碼)
7. [Vibe Coding 提示詞（Prompt）實作範例集](#vibe-coding-提示詞prompt實作範例集)
8. [結果呈現與分析：碩士論文寫法示例](#結果呈現與分析碩士論文寫法示例)
9. [常見統計誤區與 Q&A](#常見統計誤區與-qa)
10. [延伸研究方向：科技業關鍵技術人才非預期離職傾向之預警與預測研究](#延伸研究方向科技業關鍵技術人才非預期離職傾向之預警與預測研究)
11. [課後作業與練習](#課後作業與練習)
12. [參考文獻與延伸閱讀（已查核連結）](#參考文獻與延伸閱讀已查核連結)
13. [附錄](#附錄)
14. [下週預告](#下週預告)

---

## 學習目標

完成本週課程後，學生應能夠：

1. 說明監督式學習與第 8 週非監督式學習之本質差異，理解「標籤資料」在分類預測模型中的角色。
2. 說明決策樹演算法之遞迴分割原理，並計算資訊獲利（Entropy-based Information Gain）與基尼不純度（Gini Impurity）。
3. 說明集成學習（Ensemble Learning）與 Bagging 架構之核心邏輯，理解隨機森林如何透過多棵決策樹之投票降低過度配適風險。
4. 辨識類別不平衡問題，並運用 SMOTE 合成少數類別過採樣與類別權重調整（Class Weighting）兩種策略因應。
5. 使用 Python（`scikit-learn`、`imbalanced-learn`）建構完整的分類模型訓練管道。
6. 計算並判讀混淆矩陣、Precision、Recall、F1 分數與 AUC-ROC 曲線，全面評估分類模型效能。
7. 萃取並繪製特徵重要性排序長條圖，辨識風險預警模型中的關鍵驅動因子。
8. 依照論文「研究結果與討論」章節寫法，將分類預測模型結果轉譯為混淆矩陣熱圖、ROC 曲線與具學術規範的文字敘述。

---

## 本週知識地圖

| 構面 | 內容 | 對應方法 | 對應 Python 套件 |
|---|---|---|---|
| 問題性質 | 具備明確標籤之分類預測問題 | 監督式學習 | — |
| 單一分類器 | 遞迴分割特徵空間，建立可解釋規則 | 決策樹（分割準則：Entropy／Gini） | `sklearn.tree.DecisionTreeClassifier` |
| 集成分類器 | 多棵決策樹投票，降低過度配適 | 隨機森林（Bagging） | `sklearn.ensemble.RandomForestClassifier` |
| 資料不平衡處理 | 少數類別（如危機公司）樣本過少 | SMOTE、類別權重調整 | `imblearn.over_sampling.SMOTE` |
| 模型評估 | 全面評估分類效能，兼顧不同錯誤型態 | 混淆矩陣、Precision/Recall/F1、AUC-ROC | `sklearn.metrics` |
| 結果解釋 | 辨識關鍵風險驅動因子 | 特徵重要性 | `RandomForestClassifier.feature_importances_` |

---

## 理論基礎篇

### 3.1 從非監督式到監督式學習：分群到分類的橋接

第 8 週介紹之 K-Means 分群，資料中並不存在任何「正確答案」，演算法完全依樣本特徵相似性自主浮現分群結構；本週起進入之**監督式學習（Supervised Learning）**，資料中每一筆樣本皆附帶一個明確的「標籤（label）」——例如本週研究範例中，每一家企業樣本都有一個「是否於次年發生財務危機」之標籤（是／否），模型訓練之目標，是學習一組能從企業特徵（財務比率）準確預測此標籤的規則。

這正是監督式與非監督式學習在方法論本質上的關鍵差異：非監督式學習沒有「正確答案」可供模型學習比對，其評估標準（如第 8 週之輪廓係數）著重於分群結構本身的統計特性；監督式學習則存在客觀的「正確答案」，其模型評估標準（本週將介紹之 Precision、Recall、F1、AUC）著重於模型預測結果與真實標籤之間的一致程度。本週研究範例「企業財務危機早期預警機制」，正是監督式學習在風險管理領域最典型的應用情境——透過歷史上已知結果的企業樣本（哪些公司確實發生過財務危機），訓練模型學習危機企業與健全企業之特徵差異規律，進而對尚未發生危機的現有企業進行風險預測。

### 3.2 決策樹演算法原理：遞迴分割與停止準則

決策樹（Decision Tree）是一種以樹狀結構呈現決策規則的監督式學習演算法，其核心邏輯為**遞迴二元分割（Recursive Binary Splitting）**：

1. 從根節點（包含全部訓練樣本）開始，逐一檢視每個特徵之每個可能分割點，計算「若以此特徵、此分割點將樣本切分為兩群，能讓子節點之類別純度（class purity）提升多少」。
2. 選擇能讓純度提升幅度最大之特徵與分割點，作為當前節點之分割規則，將樣本分為左右兩個子節點。
3. 對每個子節點重複步驟 1–2，持續遞迴分割，直至達到**停止準則（stopping criteria）**，如：節點內樣本數低於最小門檻、樹的深度達到預設上限（`max_depth`）、或節點內樣本已完全純化（單一類別）。

決策樹之最大優勢在於**高度可解釋性**——訓練完成後之樹狀結構，可直接轉譯為一系列「若…則…」的決策規則（例如「若利息保障倍數 &lt; 2.0 且負債比率 &gt; 0.6，則預測為財務危機」），管理者無需具備統計背景即可直接理解模型之決策邏輯，這也是決策樹相較於許多「黑盒子」機器學習模型，在風險管理、信用評等等需要高度可課責性（accountability）之應用領域，至今仍被廣泛採用之核心原因。然而，單一決策樹也存在容易**過度配適（overfitting）**訓練資料、預測穩定度較低之限制，這正是本週理論篇 3.5 節將介紹之隨機森林集成學習技術所欲解決的核心問題。

### 3.3 資訊獲利與資訊增益

決策樹分割規則之選擇，關鍵在於如何量化「類別純度提升程度」，最經典的兩種量化指標為資訊獲利（源自 Quinlan 提出之 ID3、C4.5 演算法）與基尼不純度（源自 Breiman 等人提出之 CART 演算法，見本週理論篇 3.4 節）。

**熵（Entropy）**：源自資訊理論，衡量一個節點內類別分布的「混亂程度」或「不確定性」：

$$
Entropy(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)
$$

其中 $p_i$ 為節點 $S$ 中屬於第 $i$ 類的樣本比例， $c$ 為類別數。當節點內樣本完全屬於同一類別時（純度最高）， $Entropy = 0$ ；當節點內各類別樣本數完全平均分布時（純度最低、不確定性最大）， $Entropy$ 達最大值（二元分類情境下為 $\log_2 2 = 1$ ）。

**資訊增益（Information Gain）**：衡量以特定特徵 $A$ 分割節點 $S$ 後，熵下降（不確定性減少）的幅度：

$$
IG(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Entropy(S_v)
$$

決策樹演算法在每個節點，選擇能使資訊增益最大化之特徵作為分割依據，此即為 `scikit-learn` 中 `criterion='entropy'` 參數所對應之分割準則，也是本週提示詞實作中決策樹分割邏輯之數學基礎。

### 3.4 基尼不純度

**基尼不純度（Gini Impurity）**由 Breiman、Friedman、Olshen 與 Stone 於 1984 年提出之 CART（Classification and Regression Trees）演算法採用，是另一種衡量節點類別純度的指標，計算上較熵運算更為簡潔（不涉及對數運算）：

$$
Gini(S) = 1 - \sum_{i=1}^{c} p_i^2
$$

與熵相同， $Gini(S) = 0$ 代表節點完全純化，數值越大代表節點內類別混合程度越高（二元分類情境下之最大值為 0.5）。`scikit-learn` 之 `DecisionTreeClassifier` 預設採用 `criterion='gini'`。

**Entropy 與 Gini 之選擇實務**：兩種指標在絕大多數實務情境下會得出高度相似（雖非完全相同）之分割結果與最終模型效能，本週 Colab 實作將以相同資料集實際比較兩者之分類結果差異，讓學生具體觀察此一現象；文獻上普遍認為兩者效能差異通常不大，Gini 因運算效率較高（無對數運算），在大型資料集或需要頻繁重新訓練模型之情境下略具優勢，Entropy 則因源自資訊理論、在學術論文中之理論闡述較具說服力，兩者之選擇更多是實務工程考量而非統計顯著性之差異。

### 3.5 集成學習與 Bagging 架構：隨機森林原理

**集成學習（Ensemble Learning）**之核心哲學是「三個臭皮匠，勝過一個諸葛亮」——與其仰賴單一模型之預測，不如訓練多個模型，並整合（如多數決投票）其預測結果，通常能得到比任一單一模型更穩定、更準確的預測效能。

**隨機森林（Random Forest）**由 Breiman（2001，發表於 *Machine Learning*，見本週參考文獻）提出，是決策樹之集成學習延伸，核心採用 **Bagging（Bootstrap Aggregating）架構**，並額外引入特徵隨機性，完整流程如下：

1. **拔靴重抽樣（Bootstrap Sampling）**：從原始訓練樣本中，採取後放回抽樣方式，重複抽取與原始樣本數相同之樣本，建構多組（通常數百組）拔靴樣本。
2. **特徵隨機選取（Random Feature Selection）**：在建構每一棵決策樹的每一個分割節點時，並非考慮全部特徵，而是隨機選取一部分特徵子集（`scikit-learn` 中對應 `max_features` 參數），僅從此子集中選擇最佳分割特徵。
3. **獨立建樹**：以每一組拔靴樣本、搭配上述特徵隨機性，各自獨立訓練一棵（通常不修剪、允許充分生長的）決策樹。
4. **多數決整合（Aggregation）**：對新樣本進行預測時，讓森林中每一棵樹各自投票，以多數決（分類問題）或平均值（迴歸問題）作為最終預測結果。

**隨機森林為何能降低過度配適風險**：單一決策樹容易過度配適訓練資料中的雜訊，導致對新資料的預測不穩定；隨機森林透過「拔靴重抽樣」與「特徵隨機選取」雙重隨機性機制，確保森林中各棵樹彼此之間的預測誤差具有低相關性（每棵樹「犯錯」的方式不盡相同），根據統計學中變異數縮減之基本原理，將多個低相關性、各自帶有隨機誤差之預測結果加以平均或投票整合，能有效抵銷個別樹之隨機誤差，使整體模型之預測效能與穩定度顯著優於任一單一決策樹，這正是集成學習相較單一模型之核心價值所在。

### 3.6 類別不平衡問題與處理策略

實務資料中，如本週研究範例之「財務危機」標籤，發生危機之公司樣本數通常遠少於正常經營之公司樣本（**類別不平衡，Class Imbalance**），若直接以此不平衡資料訓練分類模型，模型容易傾向「全部預測為多數類別（正常）」以追求表面上的高整體準確率，卻完全喪失辨識少數類別（危機公司）的實質預測能力，這在風險預警應用情境中是極為嚴重的問題——畢竟預警系統存在的核心目的，正是要準確辨識出少數的高風險個案。

**處理策略一：SMOTE（Synthetic Minority Over-sampling Technique）**：由 Chawla、Bowyer、Hall 與 Kegelmeyer（2002，發表於 *Journal of Artificial Intelligence Research*，見本週參考文獻）提出，其核心邏輯並非單純複製少數類別樣本（該做法容易導致模型對重複樣本過度配適），而是在少數類別樣本的特徵空間中，透過 $k$ 近鄰（k-Nearest Neighbors）插值方式，**合成產生新的、介於現有少數類別樣本之間的人工樣本**，藉此在不喪失原始資料資訊、且避免單純重複複製之副作用的前提下，達到平衡訓練資料類別比例之目的，這正是本週提示詞實作中 `imblearn` 套件所實作之核心演算法。

**處理策略二：類別權重調整（Class Weighting）**：不改變訓練資料本身之樣本組成，而是在模型訓練之損失函數中，賦予少數類別更高的錯誤懲罰權重（`scikit-learn` 中對應 `class_weight='balanced'` 參數，會自動依類別樣本數之反比設定權重），使模型在學習過程中，對少數類別之誤判施以更大的懲罰，進而促使模型更加重視少數類別之辨識準確度。

**兩種策略之選擇**：SMOTE 透過實際增加少數類別之「有效樣本數」發揮作用，通常需要搭配訓練／測試集切分時特別注意（**必須先切分訓練集與測試集，再僅對訓練集執行 SMOTE**，避免合成樣本的資訊滲漏至測試集造成評估結果過度樂觀，此為初學者極常犯的方法論錯誤，詳見本週 Q&A）；類別權重調整則不涉及資料合成，實作相對單純，兩者亦可合併使用。本週 Colab 實作將完整示範兩種策略之正確實作方式與結果比較。

### 3.7 分類模型評估指標

**混淆矩陣（Confusion Matrix）**：以 $2 \times 2$ 矩陣（二元分類情境）呈現模型預測結果與真實標籤之交叉分布：

| | 預測為正常 | 預測為危機 |
|---|---|---|
| **實際為正常** | 真陰性 TN | 偽陽性 FP |
| **實際為危機** | 偽陰性 FN | 真陽性 TP |

**衍生評估指標**：

$$
Precision = \frac{TP}{TP + FP} \qquad \text{（預測為危機的樣本中，真正是危機的比例——「精確度」）}
$$

$$
Recall = \frac{TP}{TP + FN} \qquad \text{（實際為危機的樣本中，被成功預測出的比例——「召回率／敏感度」）}
$$

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall} \qquad \text{（Precision 與 Recall 之調和平均數）}
$$

**指標選擇之實務意涵**：在財務危機預警情境中，Recall（是否成功抓出所有真正的危機公司，避免漏判）通常比 Precision（預測為危機的公司中有多少真的是危機，避免誤判正常公司）更受重視，因為「漏判危機公司」（偽陰性，FN）之代價（如放款損失、投資損失）通常遠高於「誤判正常公司為危機」（偽陽性，FP）之代價（如過度保守的信用額度），這也是為何本週理論篇 3.6 節強調處理類別不平衡問題之重要性——若模型因類別不平衡而傾向不辨識危機類別，將直接導致 Recall 偏低，使預警系統喪失實務價值。

**AUC-ROC（Area Under the Receiver Operating Characteristic Curve）**：ROC 曲線以「偽陽性率（FPR = FP/(FP+TN)）」為橫軸、「真陽性率（TPR，即 Recall）」為縱軸，呈現模型在不同分類門檻值（threshold）下之權衡表現；AUC 為此曲線下方所圍面積，介於 0.5（等同隨機猜測）到 1（完美分類）之間，是不受特定門檻值選擇影響、能綜合評估模型整體辨識能力之常用指標，也是本週提示詞實作要求計算之核心績效指標。

### 3.8 特徵重要性判讀

隨機森林之另一項重要產出，是**特徵重要性（Feature Importance）**——衡量每個特徵在整個森林的分割決策過程中，對降低不純度（Entropy 或 Gini）之平均貢獻程度。`scikit-learn` 中之 `feature_importances_` 屬性，計算原理為：加總該特徵在森林中所有節點分割時，所貢獻之不純度下降量（並依該節點涵蓋之樣本比例加權），再除以森林中的樹木總數進行正規化，所有特徵之重要性總和為 1。特徵重要性排序長條圖，是風險預警模型研究中極具管理溝通價值的產出——它直接回答了「究竟是哪些財務指標，最能預示企業即將陷入財務危機」此一實務關鍵問題，這正是本週提示詞實作「輸出特徵重要性排序長條圖」之核心目的。

### 3.9 論文中分類預測模型章節的標準寫法架構

1. **資料集描述**：樣本來源、樣本數、類別分布（含不平衡比例）。
2. **類別不平衡處理方法說明**：SMOTE 或類別權重調整，並明確說明訓練／測試集切分與 SMOTE 執行之先後順序。
3. **模型設定與訓練過程**：決策樹與隨機森林之超參數設定（如 `max_depth`、`n_estimators`）。
4. **混淆矩陣**：以熱圖呈現，並說明各象限（TP/TN/FP/FN）之實務意涵。
5. **分類評估指標表**：Precision、Recall、F1（含各類別與總體之 Macro/Weighted 平均）。
6. **ROC 曲線與 AUC 值**。
7. **特徵重要性排序長條圖**：並討論關鍵風險因子之管理意涵。
8. **模型比較與選擇依據討論**（如決策樹 vs. 隨機森林、Entropy vs. Gini、有無 SMOTE 之比較）。

---

## 研究設計實例：企業財務危機早期預警機制與違約風險預測模型

### 4.1 研究背景與動機

企業財務危機之早期預警，對金融機構授信決策、投資人風險評估與主管機關監理，皆具有重要實務價值。既有台灣本土研究（見本週參考文獻）多採用羅吉斯迴歸等傳統統計方法建構預警模型，近年機器學習方法（尤其隨機森林）因能捕捉財務指標間之非線性交互關係、且預測穩定度較高，逐漸成為財務危機預測研究之主流方法選擇。本研究以決策樹與隨機森林，結合 SMOTE 類別不平衡處理技術，建構企業財務危機早期預警模型。

### 4.2 研究目的

1. 蒐集企業財務比率資料，建構財務危機（違約）預測資料集。
2. 比較決策樹（Entropy 與 Gini 準則）與隨機森林之分類預測效能。
3. 運用 SMOTE 處理財務危機樣本相對稀少之類別不平衡問題。
4. 計算並比較模型之混淆矩陣、Precision/Recall/F1 與 AUC-ROC，評估模型實務可用性。
5. 萃取特徵重要性，辨識企業財務危機之關鍵預警指標。

### 4.3 預測特徵架構

| 特徵代碼 | 特徵名稱 | 操作型定義 |
|---|---|---|
| F1 | 流動比率 | 流動資產／流動負債，衡量短期償債能力 |
| F2 | 負債比率 | 總負債／總資產，衡量財務槓桿程度 |
| F3 | 資產報酬率（ROA） | 稅後淨利／總資產，衡量整體獲利能力 |
| F4 | 利息保障倍數 | 息前稅前淨利／利息費用，衡量償付利息之能力 |
| F5 | 營收成長率 | 本期營收相對前期之成長百分比 |
| F6 | 保留盈餘比率 | 保留盈餘／總資產，衡量企業累積獲利之財務韌性 |

### 4.4 研究對象與資料來源

建議以台灣經濟新報（TEJ）資料庫或公開資訊觀測站之上市櫃公司財務資料為研究對象，依循本週參考文獻中台灣本土研究常見之做法，以「危機公司（如全額交割股、下市櫃、重整）」與「正常經營配對公司（同產業、同期間、相似資產規模）」建構研究樣本；本課程範例為教學示範，採用模擬資料，模擬情境中財務危機發生率約 12%，符合實務上財務危機事件相對稀少之類別不平衡特性。

### 4.5 資料分析流程規劃

```
蒐集企業財務比率資料與危機標籤
        │
        ▼
資料清理（極端值檢查、遺漏值處理）
        │
        ▼
切分訓練集與測試集（分層抽樣，維持類別比例一致）
        │
        ▼
【僅對訓練集】以 SMOTE 進行類別平衡處理
        │
        ▼
訓練決策樹模型（比較 Entropy 與 Gini 準則）
        │
        ▼
訓練隨機森林模型（Bagging 集成架構）
        │
        ▼
於測試集（原始未經 SMOTE 處理之真實類別分布）評估模型
        │
        ▼
計算混淆矩陣、Precision/Recall/F1、AUC-ROC
        │
        ▼
繪製 ROC 曲線 + 混淆矩陣熱圖
        │
        ▼
萃取特徵重要性 + 繪製排序長條圖
        │
        ▼
撰寫研究結果與風險管理意涵討論
```

---

## Colab 實作環境建置

```python
# ============================================================
# Cell 0：Colab 環境建置與套件安裝
# ------------------------------------------------------------
# 說明：imbalanced-learn（imblearn）套件提供 SMOTE 等類別
# 不平衡處理技術，Colab 環境通常未預先安裝，需手動安裝。
# ============================================================

!pip install imbalanced-learn --quiet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_auc_score, roc_curve, accuracy_score)
from imblearn.over_sampling import SMOTE

# ------------------------------------------------------------
# 設定中文字型（沿用第 1–8 週相同設定邏輯）
# ------------------------------------------------------------
!wget -q https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf -O /content/NotoSansTC.otf
from matplotlib import font_manager
font_manager.fontManager.addfont('/content/NotoSansTC.otf')
plt.rcParams['font.family'] = 'Noto Sans CJK TC'
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')

print("環境建置完成，本週使用 scikit-learn 與 imbalanced-learn 建立分類預測管道。")
```

---

## Colab 實作：Step by Step 完整程式碼

### Step 1：模擬企業財務危機預測資料集

```python
# ============================================================
# Cell 1：模擬具有類別不平衡特性之企業財務危機資料集
# ------------------------------------------------------------
# 教學目的：模擬財務危機發生率約 12% 之不平衡資料集，並刻意
# 加入合理雜訊使危機／正常公司之特徵分布存在部分重疊，更
# 貼近真實財務資料之預測難度（避免模型輕易達到 100% 準確率
# 這種不切實際的教學示範）。正式研究請將本 Cell 替換為讀取
# TEJ 或公開資訊觀測站財務資料之程式碼。
# ============================================================

np.random.seed(42)
n = 800

# 財務危機標籤：約 12% 為危機公司，符合實務不平衡特性
distress = np.random.binomial(1, 0.12, n)

def make_feature(base_normal, base_distress, noise, is_distress):
    """
    依危機／正常標籤，分別以不同平均數產生特徵值，並加入
    共同的隨機雜訊，使兩類別之特徵分布存在部分重疊。
    """
    return np.where(
        is_distress == 1,
        np.random.normal(base_distress, noise, n),
        np.random.normal(base_normal, noise, n)
    )

df = pd.DataFrame({
    'current_ratio': np.clip(make_feature(1.8, 1.3, 0.6, distress), 0, None),
    'debt_ratio': np.clip(make_feature(0.45, 0.65, 0.20, distress), 0, 2),
    'roa': make_feature(0.05, -0.02, 0.08, distress),
    'interest_coverage': np.clip(make_feature(7.0, 2.0, 5.0, distress), -5, 25),
    'revenue_growth': make_feature(0.06, -0.08, 0.18, distress),
    'retained_earnings_ratio': make_feature(0.20, -0.02, 0.22, distress),
    'distress': distress
})

print("=== 表 1：類別分布（0=正常經營，1=財務危機）===")
display(df['distress'].value_counts().to_frame('樣本數'))
print(f"\n財務危機發生率：{df['distress'].mean()*100:.1f}%")

print("\n=== 表 2：各類別特徵平均值對照 ===")
display(df.groupby('distress').mean().round(3))
```

### Step 2：訓練測試集切分（分層抽樣）

```python
# ============================================================
# Cell 2：訓練測試集切分
# ------------------------------------------------------------
# 統計原理：採用分層抽樣（stratify=y），確保訓練集與測試集
# 之財務危機比例與原始資料集一致，避免因隨機切分導致某一
# 子集之類別比例產生偏誤。
# ============================================================

features = ['current_ratio', 'debt_ratio', 'roa', 'interest_coverage',
            'revenue_growth', 'retained_earnings_ratio']
X = df[features]
y = df['distress']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"訓練集樣本數：{len(X_train)}，危機比例：{y_train.mean()*100:.2f}%")
print(f"測試集樣本數：{len(X_test)}，危機比例：{y_test.mean()*100:.2f}%")
print("\n重要提醒：測試集將全程維持原始真實類別比例，")
print("後續 SMOTE 僅套用於訓練集，確保模型評估結果反映真實世界表現。")
```

### Step 3：決策樹訓練（比較 Entropy 與 Gini 準則）

```python
# ============================================================
# Cell 3：決策樹模型訓練與 Entropy／Gini 準則比較
# ============================================================

dt_results = {}
for criterion in ['entropy', 'gini']:
    dt = DecisionTreeClassifier(
        criterion=criterion, max_depth=4,
        class_weight='balanced',   # 以類別權重調整因應不平衡，暫不使用 SMOTE
        random_state=42
    )
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    dt_results[criterion] = {'model': dt, 'accuracy': acc, 'report': report}

    print(f"=== 決策樹（{criterion} 準則）分類報告 ===")
    print(classification_report(y_test, y_pred, digits=3,
                                 target_names=['正常經營', '財務危機']))

print("兩種分割準則之整體準確率比較：")
for criterion, result in dt_results.items():
    print(f"  {criterion}: accuracy = {result['accuracy']:.4f}")
```

### Step 4：SMOTE 類別平衡處理

```python
# ============================================================
# Cell 4：SMOTE 合成少數類別過採樣
# ------------------------------------------------------------
# 提示詞實作對照（前置步驟）：
# 「使用 scikit-learn 與 imblearn 建立分類管道」
# ------------------------------------------------------------
# 關鍵方法論提醒：SMOTE 僅套用於訓練集（X_train, y_train），
# 測試集必須維持原始真實類別分布，此為避免資料滲漏
# （data leakage）之標準做法（詳見理論篇 3.6 節與本週 Q&A）。
# ============================================================

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("SMOTE 處理前訓練集類別分布：")
print(y_train.value_counts().to_dict())
print("\nSMOTE 處理後訓練集類別分布：")
print(y_train_smote.value_counts().to_dict())
print(f"\n訓練樣本數由 {len(X_train)} 筆增加至 {len(X_train_smote)} 筆")
```

### Step 5：隨機森林訓練（SMOTE 平衡後）

```python
# ============================================================
# Cell 5：隨機森林分類器訓練
# ------------------------------------------------------------
# 提示詞實作對照：
# 「訓練隨機森林分類器，計算混淆矩陣、Precision/Recall/F1
#   與 AUC-ROC。」
# ============================================================

rf = RandomForestClassifier(
    n_estimators=300,     # 森林中決策樹之數量
    max_depth=6,           # 限制樹深，避免過度配適
    random_state=42
)
rf.fit(X_train_smote, y_train_smote)

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]   # 危機類別之預測機率，用於 ROC/AUC 計算

print("=== 隨機森林（SMOTE 平衡後訓練）分類報告 ===")
print(classification_report(y_test, y_pred_rf, digits=3,
                             target_names=['正常經營', '財務危機']))

rf_auc = roc_auc_score(y_test, y_proba_rf)
print(f"AUC-ROC = {rf_auc:.4f}")
```

### Step 6：混淆矩陣熱圖

```python
# ============================================================
# Cell 6：混淆矩陣視覺化
# ============================================================

cm = confusion_matrix(y_test, y_pred_rf)
cm_df = pd.DataFrame(cm, index=['實際：正常', '實際：危機'],
                      columns=['預測：正常', '預測：危機'])

print("=== 表 3：混淆矩陣 ===")
display(cm_df)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True,
            annot_kws={'fontsize': 14})
plt.title('圖 1：隨機森林混淆矩陣熱圖', fontsize=13)
plt.ylabel('實際類別')
plt.xlabel('預測類別')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"\n真陰性 TN={tn}，偽陽性 FP={fp}，偽陰性 FN={fn}，真陽性 TP={tp}")
print(f"Precision（危機類別）= {tp/(tp+fp):.3f}")
print(f"Recall（危機類別）= {tp/(tp+fn):.3f}")
```

### Step 7：ROC 曲線繪製

```python
# ============================================================
# Cell 7：ROC 曲線繪製
# ============================================================

fpr, tpr, thresholds = roc_curve(y_test, y_proba_rf)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, linewidth=2.5, color='#2980b9',
         label=f'隨機森林（AUC = {rf_auc:.3f}）')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='隨機猜測基準線（AUC = 0.5）')
plt.xlabel('偽陽性率（False Positive Rate）')
plt.ylabel('真陽性率（True Positive Rate / Recall）')
plt.title('圖 2：ROC 曲線', fontsize=13)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Step 8：特徵重要性排序長條圖

```python
# ============================================================
# Cell 8：特徵重要性萃取與視覺化
# ------------------------------------------------------------
# 提示詞實作對照：
# 「輸出特徵重要性（Feature Importance）排序長條圖。」
# ============================================================

feature_names_zh = {
    'current_ratio': '流動比率', 'debt_ratio': '負債比率', 'roa': '資產報酬率',
    'interest_coverage': '利息保障倍數', 'revenue_growth': '營收成長率',
    'retained_earnings_ratio': '保留盈餘比率'
}

feature_importance = pd.Series(rf.feature_importances_, index=features)
feature_importance = feature_importance.sort_values(ascending=True)
feature_importance.index = [feature_names_zh[f] for f in feature_importance.index]

print("=== 表 4：特徵重要性排序 ===")
display(feature_importance.sort_values(ascending=False).round(4))

plt.figure(figsize=(8, 5))
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(feature_importance)))
bars = plt.barh(feature_importance.index, feature_importance.values, color=colors, edgecolor='#2c3e50')
for bar, val in zip(bars, feature_importance.values):
    plt.text(val + 0.003, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=10)
plt.xlabel('特徵重要性')
plt.title('圖 3：隨機森林特徵重要性排序長條圖', fontsize=13)
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n關鍵風險預警指標：{feature_importance.index[-1]}（重要性 = {feature_importance.values[-1]:.3f}）")
```

### Step 9：決策樹 vs. 隨機森林效能比較

```python
# ============================================================
# Cell 9：模型效能綜合比較
# ============================================================

comparison_records = []
for criterion, result in dt_results.items():
    r = result['report']
    comparison_records.append({
        'Model': f'決策樹（{criterion}）',
        'Accuracy': result['accuracy'],
        'Precision(危機)': r['1']['precision'],
        'Recall(危機)': r['1']['recall'],
        'F1(危機)': r['1']['f1-score'],
    })

rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
comparison_records.append({
    'Model': '隨機森林（SMOTE）',
    'Accuracy': accuracy_score(y_test, y_pred_rf),
    'Precision(危機)': rf_report['1']['precision'],
    'Recall(危機)': rf_report['1']['recall'],
    'F1(危機)': rf_report['1']['f1-score'],
})

comparison_df = pd.DataFrame(comparison_records)
print("=== 表 5：模型效能綜合比較表 ===")
display(comparison_df.round(3))
```

### Step 10：匯出所有分析結果

```python
# ============================================================
# Cell 10：匯出完整分析結果至 Excel
# ============================================================

with pd.ExcelWriter('財務危機預警模型分析結果_Week09.xlsx') as writer:
    df.groupby('distress').mean().round(3).to_excel(writer, sheet_name='類別特徵對照')
    comparison_df.round(3).to_excel(writer, sheet_name='模型效能比較', index=False)
    cm_df.to_excel(writer, sheet_name='混淆矩陣')
    feature_importance.sort_values(ascending=False).round(4).to_frame('重要性').to_excel(writer, sheet_name='特徵重要性')

print("所有統計結果已匯出至 財務危機預警模型分析結果_Week09.xlsx，可於 Colab 左側檔案面板下載。")
print(f"\n最終模型（隨機森林 + SMOTE）：AUC = {rf_auc:.4f}")
print(f"關鍵風險預警指標：{feature_importance.index[-1]}")
```

---

## Vibe Coding 提示詞（Prompt）實作範例集

**範例 1：類別不平衡診斷**

> 我的財務危機預測資料集中，危機類別僅占約 12%。請幫我計算類別分布，並提醒我在切分訓練測試集時應使用分層抽樣（stratify），同時說明如果不處理這種不平衡問題，模型可能會出現什麼樣的預測偏誤。

**範例 2：SMOTE 正確實作**

> 請使用 imblearn 的 SMOTE 對我的訓練集進行類別平衡處理，但請務必確認只對訓練集執行 SMOTE，測試集必須保持原始真實類別分布，並解釋為什麼這個順序很重要。

**範例 3：決策樹與隨機森林比較**

> 請分別以 entropy 與 gini 兩種分割準則訓練決策樹，並訓練一個隨機森林模型（使用 SMOTE 平衡後的訓練集），比較三個模型在測試集上對「財務危機」類別的 Precision、Recall、F1 分數，整理成一張比較表。

**範例 4：混淆矩陣與 ROC 曲線**

> 請畫出隨機森林模型的混淆矩陣熱圖（使用 seaborn，數值以整數顯示），並繪製 ROC 曲線，圖上請標示 AUC 數值，同時加上一條對角線代表隨機猜測的基準線。

**範例 5：特徵重要性**

> 請從訓練好的隨機森林模型中萃取特徵重要性，由小到大排序後繪製水平長條圖，並在每個長條後方標示重要性數值，圖表中文標籤請對應到易懂的財務比率名稱。

**範例 6：結果段落初稿撰寫**

> 根據以下統計結果（決策樹 entropy 準則：危機類別 Precision=.531、Recall=.548；隨機森林+SMOTE：危機類別 Precision=.618、Recall=.677、AUC=.912；混淆矩陣 TN=196,FP=13,FN=10,TP=21；最重要特徵為利息保障倍數，重要性.235），請以碩士論文研究結果章節的學術寫作語氣，撰寫一段約 300 字的中文分析段落。

---

## 結果呈現與分析：碩士論文寫法示例

以下段落數值取自本週 Colab 範例程式碼之實際執行結果，供學生對照模仿寫作邏輯（實際數值請以自己資料之 Colab 輸出為準）。

> **4.1 決策樹分割準則比較**
>
> 本研究首先以 Entropy 與 Gini 兩種分割準則分別訓練決策樹模型（皆搭配類別權重調整以因應資料不平衡）。結果顯示，Entropy 準則之整體準確率為 87.9%，對財務危機類別之 Precision 為 .531、Recall 為 .548、F1 為 .540；Gini 準則之整體準確率為 82.1%，危機類別之 Precision 為 .400、Recall 為 .774、F1 為 .527。兩種準則呈現出不同的錯誤型態權衡——Gini 準則之 Recall 較高（較能抓出真正的危機公司），但 Precision 明顯偏低（誤判正常公司為危機之比例較高）；此一發現顯示分割準則之選擇，在類別不平衡且樣本重疊程度較高之資料情境下，確實可能對模型之錯誤型態分布產生實質影響，而非僅是理論上的細微差異。
>
> **4.2 隨機森林與 SMOTE 平衡處理成效**
>
> 進一步以 SMOTE 平衡訓練集後訓練隨機森林模型，測試集（維持原始真實類別分布）評估結果顯示，整體準確率提升至 90.4%，財務危機類別之 Precision 為 .618、Recall 為 .677、F1 為 .646，均優於單一決策樹模型之表現，AUC-ROC 達 .912，顯示模型具備良好之整體辨識能力。混淆矩陣顯示，測試集 31 家實際發生財務危機之公司中，模型成功辨識出 21 家（真陽性），僅有 10 家未被成功辨識（偽陰性）；209 家正常經營公司中，196 家被正確分類，13 家遭誤判為危機（偽陽性）。此一結果印證了本週理論篇 3.5 節所述之集成學習優勢：透過 Bagging 架構整合多棵決策樹之預測，相較單一決策樹展現出更穩定、更優異之分類效能。
>
> **4.3 關鍵風險預警特徵**
>
> 特徵重要性分析結果顯示，六項財務比率指標之相對重要性由高至低依序為：利息保障倍數（.235）、保留盈餘比率（.177）、負債比率（.173）、營收成長率（.151）、資產報酬率（.140）、流動比率（.124）。利息保障倍數作為最關鍵之預警指標，其管理意涵為：企業支付利息費用之能力，是預示財務危機最敏感之先行指標，此一發現與既有財務危機預測文獻（見本週參考文獻）中普遍強調償債能力指標之核心地位相互呼應，可作為金融機構授信風險評估與主管機關監理預警系統，優先關注之財務指標依據。

**APA 格式三線表範例：模型效能綜合比較表**

| 模型 | 整體準確率 | Precision（危機） | Recall（危機） | F1（危機） | AUC |
|---|---|---|---|---|---|
| 決策樹（Entropy） | .879 | .531 | .548 | .540 | — |
| 決策樹（Gini） | .821 | .400 | .774 | .527 | — |
| 隨機森林（SMOTE） | .904 | .618 | .677 | .646 | .912 |

*註：以上數值為本週 Colab 範例實際執行結果，實際研究請以自己資料之輸出為準。*

---

## 常見統計誤區與 Q&A

**Q1：為什麼一定要先切分訓練測試集、再對訓練集執行 SMOTE？如果先對全部資料做 SMOTE 再切分，會有什麼問題？**
這是類別不平衡處理中最常見、也最嚴重的方法論錯誤，稱為**資料滲漏（Data Leakage）**。若先對全部資料執行 SMOTE 再切分訓練測試集，測試集中可能包含由「訓練集樣本」插值合成而來的人工樣本（或其近鄰資訊已被模型間接學習），導致模型在測試集上的評估效能被人為高估，無法反映模型面對真正全新、未曾見過之資料時的實際預測能力。正確順序永遠是：先切分訓練測試集，SMOTE 僅套用於訓練集，測試集全程維持原始真實資料與類別分布，這也是本週 Colab 實作 Step 2 特別強調此一順序的原因。

**Q2：我的整體準確率（accuracy）高達 90% 以上，這樣模型是不是已經很好了？**
在類別不平衡情境下，整體準確率是極具誤導性的指標，務必格外謹慎。若財務危機發生率僅 12%，一個「無論如何都預測為正常經營」的無意義模型，僅憑此策略就能達到 88% 的整體準確率，卻完全沒有任何實質的危機辨識能力（危機類別之 Recall 為 0）。這正是本週理論篇 3.7 節強調應同時檢視 Precision、Recall、F1（尤其是少數類別的這些指標）與 AUC-ROC，而非僅憑整體準確率判斷模型優劣之核心原因，論文寫作時應避免僅呈現整體準確率、迴避少數類別表現不佳之情形。

**Q3：隨機森林的 `n_estimators`（樹的數量）和 `max_depth`（樹的深度）這些超參數，該怎麼決定最適合的數值？**
本週課程示範之超參數設定（`n_estimators=300`、`max_depth=6`）為教學示範用之合理預設值，正式研究中應透過**超參數調校（Hyperparameter Tuning）**系統化決定，常見做法為網格搜尋（Grid Search）或隨機搜尋（Random Search）搭配交叉驗證（Cross-Validation），在多組候選超參數組合中，選擇能使驗證集（而非測試集）效能最佳之組合，測試集應保留至最終模型評估階段才使用，避免因反覆依測試集結果調整超參數而產生另一種形式的資料滲漏。此為進階實作技巧，有興趣之學生可查閱 `scikit-learn` 之 `GridSearchCV`、`RandomizedSearchCV` 相關文件。

**Q4：決策樹的可解釋性這麼好，為什麼還需要用隨機森林？隨機森林不就變成看不懂的黑盒子了嗎？**
這是機器學習模型選擇中「可解釋性」與「預測效能」之間的經典權衡（trade-off）。單一決策樹確實可以完整視覺化為一組「若…則…」規則，管理者可逐條檢視；隨機森林由數百棵樹組成，無法逐一檢視每棵樹的規則，就此意義而言確實喪失了單一決策樹的完整可解釋性。但隨機森林仍保留了**特徵重要性**此一「全域可解釋性（global interpretability）」工具，能回答「哪些特徵整體而言最重要」，只是無法像單一決策樹一樣回答「這一筆特定樣本，究竟是依循哪一條具體規則被分類」此類「個別可解釋性（local interpretability）」問題。若研究情境對個別樣本層次的可解釋性有更高要求（如需要向個別客戶說明其信用評等被拒之具體原因），第 10 週將介紹之 SHAP 值方法，正是為了在維持隨機森林等集成模型高預測效能的同時，補足此一個別可解釋性缺口而發展的技術，敬請期待。

**Q5：我可以同時使用 SMOTE 和 class_weight='balanced' 兩種類別不平衡處理策略嗎？**
技術上可以同時使用，但實務上通常不建議疊加使用兩種強度相近的不平衡處理策略，因為這可能導致模型對少數類別的重視程度「過度矯正」，反而使 Precision 大幅下降（大量正常公司被誤判為危機）。實務建議是先單獨嘗試其中一種策略（如本週範例僅對隨機森林使用 SMOTE、決策樹僅使用 class_weight），觀察評估指標表現，若效果不理想，再考慮謹慎地嘗試組合使用，並透過驗證集效能謹慎評估是否過度矯正，而非直接同時套用兩種策略作為預設做法。

---

## 延伸研究方向：科技業關鍵技術人才非預期離職傾向之預警與預測研究

### 6.1 研究背景與理論基礎

科技業關鍵技術人才之非預期離職，對企業研發能量與專案延續性造成重大衝擊，及早辨識具高離職風險之關鍵人才並啟動留任機制，是人力資源管理領域日益重視之資料驅動應用情境。本週參考文獻中之台灣科技大學 AIdea 平台員工離職預測競賽資料集，已提供年齡層、績效、最高學歷、出差數、請假數等多項可能影響離職傾向之特徵變數，可作為本延伸研究方向之直接資料來源與方法論參照，此一應用情境與本週研究範例「企業財務危機預警」在方法論結構上高度類似——同樣是監督式二元分類問題（是否離職／是否發生危機）、同樣面臨類別不平衡問題（離職員工通常為少數）、同樣重視 Recall 指標（避免漏判高風險關鍵人才）。

### 6.2 建議研究設計

1. **理論框架**：延續本週決策樹與隨機森林分類架構，並可進一步整合人力資源管理理論（如工作滿意度、組織承諾相關構念）作為特徵設計依據。
2. **候選預測特徵（範例）**：
   - 年資與職涯發展停滯指標（如晉升間隔時間）
   - 近期績效考核趨勢（是否連續下滑）
   - 薪酬相對市場水準之競爭力
   - 出差頻率與工作負荷指標
   - 請假／加班模式異常變化
   - 主管關係與團隊氛圍調查評分（如有相關人資調查資料）
3. **研究對象**：建議以科技業研發部門關鍵技術人才（如資深工程師、專案技術主管）之歷史人資資料為研究對象，並以「是否於次一年度內非預期離職」作為預測標籤。
4. **分析流程**：完全比照本週 Colab 實作流程（資料切分 → SMOTE 平衡 → 決策樹與隨機森林訓練 → 混淆矩陣／ROC／特徵重要性），僅需替換特徵定義與資料來源，即可直接複用本週所有程式碼架構。
5. **管理實務意涵**：此類研究可協助人力資源部門，將原本仰賴主管主觀經驗判斷的人才留任決策，轉化為具備資料實證基礎的系統化預警機制，及早對高風險關鍵人才啟動個別化留任對話或職涯發展規劃，惟研究者應特別留意此類模型應用於實際人事決策時所涉及之倫理議題（見本週附錄之研究倫理提醒）。

### 6.3 給學生的思考練習

請思考：企業財務危機預警模型之「偽陽性」（誤判正常公司為危機），主要代價是授信或投資決策上的過度保守；但員工離職預警模型之「偽陽性」（誤判穩定員工為高離職風險），若主管因此對該員工採取不必要的特別關注或留任介入，可能產生什麼樣的員工感受與職場關係層面的負面影響？這樣的差異，是否會改變你在此延伸研究中對 Precision 與 Recall 相對重視程度的設計思考（例如是否仍應像本週財務危機範例一樣，優先重視 Recall）？

---

## 課後作業與練習

**練習一：類別不平衡程度敏感度分析**
請修改 Step 1 中 `np.random.binomial(1, 0.12, n)` 之危機發生率參數，分別調整為 5%（更不平衡）與 30%（較平衡），重新執行完整流程，比較不同不平衡程度下，SMOTE 處理前後之模型效能差異是否有所不同，並說明你觀察到的規律。

**練習二：真實資料集實作**
請自行尋找一份公開的二元分類資料集（例如 Kaggle 之信用卡違約、客戶流失等資料集，或本週參考文獻中提及之 AIdea 平台員工離職預測資料集），套用本週完整 Colab 程式碼執行決策樹與隨機森林分類分析。請繳交：(1) 資料來源說明、(2) 執行後的 Colab Notebook（.ipynb）、(3) 一頁 A4 的結果摘要（比照本週「結果呈現與分析」段落之寫法）。

**練習三：決策樹視覺化與規則解讀**
請使用 `sklearn.tree.plot_tree()` 函式，將 Step 3 訓練完成之決策樹（建議設定 `max_depth=3` 以利閱讀）完整視覺化繪製出來，並任選一條從根節點到葉節點的完整路徑，以文字說明這條路徑代表的具體決策規則及其財務意涵。

**練習四：文獻延伸閱讀報告**
請從本週「參考文獻與延伸閱讀」清單中，任選一篇財務危機預測或機器學習分類相關之期刊論文或台灣碩士論文，撰寫一頁重點摘要，內容須包含：(1) 該研究之預測特徵與資料規模、(2) 是否處理類別不平衡問題及採用方法、(3) 報告之模型效能指標、(4) 該研究提出之關鍵風險因子為何。

**練習五：超參數調校延伸**
請使用 `sklearn.model_selection.GridSearchCV`，針對隨機森林之 `n_estimators`（如 100/200/300/500）與 `max_depth`（如 4/6/8/10）進行網格搜尋，以交叉驗證之 F1 分數（`scoring='f1'`）作為調校依據，找出本週資料集之最適超參數組合，並比較調校後與本週範例預設參數之效能差異。

---

## 參考文獻與延伸閱讀（已查核連結）

1. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
   https://link.springer.com/article/10.1023/A:1010933404324
   （隨機森林之原始創始論文，完整推導 Bagging 集成架構與特徵隨機選取之理論基礎，為本週理論篇 3.5 節之直接方法論依據。）

2. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357.
   https://www.researchgate.net/publication/220543125_SMOTE_Synthetic_Minority_Over-sampling_Technique
   （SMOTE 之原始創始論文，為本週理論篇 3.6 節之直接方法論依據。）

3. 使用機器學習演算法加入市場變數來預測財務危機。Airiti Library 華藝線上圖書館。
   https://www.airitilibrary.com/Article/Detail/U0001-0628230526563044
   （比較邏吉斯迴歸、支援向量機、隨機森林、K-近鄰演算法於財務危機預測之台灣期刊論文，發現隨機森林預測能力最穩定準確，與本週研究設計範例方法論選擇相互呼應。）

4. 企業財務危機預測之研究-以台灣上市櫃公司為例。臺灣博碩士論文知識加值系統。
   https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi?o=dnclcdr&s=id=%22098KUAS8320033%22.&searchmode=basic
   （台灣本土財務危機預測碩士論文，採配對樣本設計、財務與公司治理變數並重，可作為研究設計參考範本。）

5. 財務危機預測方法與比較。Airiti Library 華藝線上圖書館。
   https://www.airitilibrary.com/Article/Detail/U0001-0154240612422011
   （比較羅吉斯迴歸、隨機森林、支援向量機、神經網路等多種方法於財務危機預測之台灣期刊論文，並指出經濟成長率與公司治理指標為關鍵影響因子。）

6. 企業違約機率預測－使用羅吉斯迴歸模型。臺灣博碩士論文知識加值系統。
   https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi?o=dnclcdr&s=id=%22095KUAS0213021%22.&searchmode=basic
   （以 TEJ 資料庫建構企業違約機率模型之台灣碩士論文，並以 ROC 曲線與 K-S 檢定驗證模型效力，可作為傳統統計方法與本週機器學習方法之比較基準。）

7. AIdea 人工智慧共創平台：員工離職預測。致理科技大學／國立臺灣科技大學。
   https://www.aidea-web.tw/topic/6fcb6b35-f85e-4444-a342-63b1f38cea08
   （台灣本土人才離職預測資料集與競賽平台，為本週延伸研究方向「科技業關鍵技術人才離職預警」之直接資料來源參照。）

**方法論經典文獻（建議延伸閱讀，非本次線上搜尋來源，圖書館或資料庫可查閱）**：

- Quinlan, J. R. (1986). Induction of decision trees. *Machine Learning*, 1(1), 81-106.
- Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.
- Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and Regression Trees*. Wadsworth.
- Shannon, C. E. (1948). A mathematical theory of communication. *The Bell System Technical Journal*, 27(3), 379-423.
- Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *The Journal of Finance*, 23(4), 589-609.

---

## 附錄

### 附錄 A：決策樹與隨機森林方法對照表

| 比較構面 | 決策樹 | 隨機森林 |
|---|---|---|
| 模型結構 | 單一樹狀結構 | 多棵樹之集成（通常數百棵） |
| 可解釋性 | 高（可完整視覺化規則） | 中（僅提供全域特徵重要性，非個別規則） |
| 過度配適風險 | 較高（尤其樹深較深時） | 較低（Bagging 與特徵隨機性降低變異） |
| 訓練速度 | 快 | 較慢（需訓練多棵樹，惟可平行化） |
| 適用情境 | 需要高度可解釋規則之情境 | 追求預測穩定度與準確度之情境 |

### 附錄 B：Entropy 與 Gini 分割準則對照表

| 比較構面 | Entropy（資訊獲利） | Gini（基尼不純度） |
|---|---|---|
| 起源 | 資訊理論（Shannon, 1948）；ID3/C4.5（Quinlan） | CART（Breiman et al., 1984） |
| 計算公式 | $-\sum p_i \log_2(p_i)$ | $1-\sum p_i^2$ |
| 運算複雜度 | 較高（涉及對數運算） | 較低 |
| 數值範圍（二元分類） | 0 至 1 | 0 至 0.5 |
| 實務差異 | 通常與 Gini 結果相近，惟極端不平衡資料下可能產生不同分割結果 | 同左 |

### 附錄 C：術語中英對照表

| 中文術語 | 英文術語 | 縮寫 |
|---|---|---|
| 監督式學習 | Supervised Learning | — |
| 決策樹 | Decision Tree | — |
| 資訊獲利／資訊增益 | Information Gain | IG |
| 基尼不純度 | Gini Impurity | — |
| 集成學習 | Ensemble Learning | — |
| 拔靴集成 | Bootstrap Aggregating | Bagging |
| 隨機森林 | Random Forest | RF |
| 類別不平衡 | Class Imbalance | — |
| 合成少數類別過採樣技術 | Synthetic Minority Over-sampling Technique | SMOTE |
| 類別權重調整 | Class Weighting | — |
| 混淆矩陣 | Confusion Matrix | — |
| 精確度 | Precision | — |
| 召回率 | Recall | — |
| 曲線下面積 | Area Under the Curve | AUC |
| 特徵重要性 | Feature Importance | — |
| 資料滲漏 | Data Leakage | — |

### 附錄 D：常見程式錯誤排解（Debugging Tips）

| 錯誤現象 | 常見原因 | 排解建議 |
|---|---|---|
| 模型測試集效能異常優異（如準確率接近 100%） | 資料滲漏：SMOTE 於切分前執行，或特徵中意外包含與標籤高度相關之洩漏變數 | 確認 SMOTE 僅套用於訓練集，並檢查特徵是否包含不應存在之未來資訊 |
| 少數類別 Recall 為 0（模型完全無法辨識危機類別） | 未處理類別不平衡問題，或 `class_weight` 參數設定錯誤 | 確認已執行 SMOTE 或設定 `class_weight='balanced'` |
| `SMOTE` 執行時拋出 `ValueError`（近鄰數量不足） | 少數類別樣本數過少，小於 SMOTE 預設之 `k_neighbors=5` | 調整 `SMOTE(k_neighbors=較小數值)`，或檢查資料是否有足夠之少數類別樣本 |
| `feature_importances_` 加總不等於 1 | 誤將未訓練完成之模型物件用於萃取特徵重要性 | 確認 `rf.fit()` 已成功執行後才呼叫 `rf.feature_importances_` |
| ROC 曲線呈現不合理的鋸齒狀 | 測試集樣本數過少，導致機率門檻值變化時 TPR/FPR 跳動劇烈 | 檢查測試集樣本數是否足夠（建議至少 100 筆以上），或考慮以交叉驗證取得更穩定之 ROC 曲線估計 |

### 附錄 E：繳交前自我檢核清單

- [ ] 已報告類別分布與不平衡程度
- [ ] 已說明類別不平衡處理方法（SMOTE 或類別權重），並確認正確之訓練測試切分順序
- [ ] 已比較 Entropy 與 Gini 兩種分割準則（如適用）
- [ ] 已報告決策樹與隨機森林之效能比較
- [ ] 已報告混淆矩陣，並繪製熱圖
- [ ] 已報告 Precision、Recall、F1（含少數類別之個別指標，而非僅整體準確率）
- [ ] 已繪製 ROC 曲線並報告 AUC 值
- [ ] 已萃取並繪製特徵重要性排序長條圖
- [ ] 已針對關鍵風險特徵提出具體管理實務意涵討論
- [ ] 所有統計結果之文字敘述與表格數值一致，無謄寫錯誤

### 附錄 F：研究倫理提醒

本週研究設計涉及使用企業財務資料（或延伸研究方向中之員工人事資料）建構風險預警與預測模型，除延續前八週已說明之知情同意、匿名性等基本倫理原則外，特別提醒：若研究成果涉及對特定企業信用評等或個別員工離職風險之預測應用，應審慎考量模型預測錯誤（尤其偽陽性）可能對被預測對象造成之實質影響（如企業被誤判為高風險而遭限縮授信、員工被誤判為高離職風險而遭受不必要之特別關注或職涯發展限制），機器學習模型之預測結果應作為輔助管理決策之參考依據，而非取代人為專業判斷之唯一標準，此為資料驅動型人事與信用決策應用中日益受到重視之演算法問責（algorithmic accountability）倫理議題。

---

## 下週預告

第 10 週將延續本週監督式學習之脈絡，進入「高效梯度提升機（XGBoost/LightGBM）與可解釋性 AI（XAI/SHAP）」。學生將學習比本週隨機森林（平行集成之 Bagging 架構）更進階之循序集成學習技術——梯度提升機（Boosting），透過逐步修正前一棵樹之殘差誤差，通常能取得比隨機森林更高之預測準確度；同時將學習可解釋性 AI 核心理論——夏普利值（Shapley Values，源自合作賽局理論），如何拆解機器學習模型之「黑盒子」決策邏輯，提供個別樣本層次之可解釋性（正好補足本週 Q&A 中所指出隨機森林在個別樣本解釋力上的侷限）。研究範例將以「房地產實價登錄自動化估價模型與特徵非線性權重透明化研究」為主題，並延伸至「金融科技授信決策中運用 XAI 消除演算法偏見與合規性稽核」之期末專題發想方向。
