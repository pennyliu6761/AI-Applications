# 第 10 週：高效梯度提升機（XGBoost/LightGBM）與可解釋性 AI（XAI/SHAP）

> 課程模組：第三模組｜資料驅動 AI：預測、分群與可解釋性模型（第 8–11 週）
> 本週定位：承接第 9 週隨機森林之 Bagging 集成架構，介紹另一種更為強大、應用更廣泛之集成學習範式——梯度提升機（Gradient Boosting Machine）之循序學習原理，並以業界標準工具 XGBoost 與 LightGBM 實作高精度迴歸預測模型；同時正式回應第 9 週 Q&A 所遺留之「黑盒子模型可解釋性」議題，引入源自合作賽局理論之 Shapley 值與其現代機器學習實作 SHAP，拆解梯度提升模型之預測邏輯，兼顧預測準確度與決策透明度。

> 教材版本：v1.0｜適用對象：在職專班研究方法與 AI 應用課程｜先修基礎：第 9 週（決策樹、隨機森林與風險預警）
> 使用工具：Google Colab（Python 3）｜主要套件：`xgboost`、`lightgbm`、`shap`、`scikit-learn`、`pandas`、`matplotlib`（本週所有程式碼已實際測試驗證可正常執行）

---

## 目錄

1. [學習目標](#學習目標)
2. [本週知識地圖](#本週知識地圖)
3. [理論基礎篇](#理論基礎篇)
   1. [3.1 從 Bagging 到 Boosting：集成學習的第二種架構](#31-從-bagging-到-boosting集成學習的第二種架構)
   2. [3.2 梯度提升機原理：殘差學習與加法模型](#32-梯度提升機原理殘差學習與加法模型)
   3. [3.3 XGBoost：正則化與二階泰勒展開優化](#33-xgboost正則化與二階泰勒展開優化)
   4. [3.4 LightGBM：直方圖演算法與葉優先生長策略](#34-lightgbm直方圖演算法與葉優先生長策略)
   5. [3.5 可解釋性 AI（XAI）的必要性](#35-可解釋性-aixai的必要性)
   6. [3.6 Shapley 值理論基礎：合作賽局理論](#36-shapley-值理論基礎合作賽局理論)
   7. [3.7 SHAP：統一可加性特徵歸因框架](#37-shap統一可加性特徵歸因框架)
   8. [3.8 SHAP 視覺化：Summary／Beeswarm 圖與 Waterfall 圖](#38-shap-視覺化summarybeeswarm-圖與-waterfall-圖)
   9. [3.9 論文中 XGBoost／SHAP 章節的標準寫法架構](#39-論文中-xgboostshap-章節的標準寫法架構)
4. [研究設計實例：房地產實價登錄自動化估價模型與特徵非線性權重透明化研究](#研究設計實例房地產實價登錄自動化估價模型與特徵非線性權重透明化研究)
5. [Colab 實作環境建置](#colab-實作環境建置)
6. [Colab 實作：Step by Step 完整程式碼](#colab-實作step-by-step-完整程式碼)
7. [Vibe Coding 提示詞（Prompt）實作範例集](#vibe-coding-提示詞prompt實作範例集)
8. [結果呈現與分析：碩士論文寫法示例](#結果呈現與分析碩士論文寫法示例)
9. [常見統計誤區與 Q&A](#常見統計誤區與-qa)
10. [延伸研究方向：金融科技授信決策中運用 XAI 消除演算法偏見與合規性稽核](#延伸研究方向金融科技授信決策中運用-xai-消除演算法偏見與合規性稽核)
11. [課後作業與練習](#課後作業與練習)
12. [參考文獻與延伸閱讀（已查核連結）](#參考文獻與延伸閱讀已查核連結)
13. [附錄](#附錄)
14. [下週預告](#下週預告)

---

## 學習目標

完成本週課程後，學生應能夠：

1. 說明 Boosting 與第 9 週 Bagging 兩種集成學習架構之本質差異，理解梯度提升機之循序殘差學習原理。
2. 說明 XGBoost 相較傳統梯度提升機之關鍵改良——正則化項與二階泰勒展開近似。
3. 說明 LightGBM 之直方圖演算法與葉優先生長策略，理解其相較 XGBoost 之效率優勢。
4. 使用 Python（`xgboost`、`lightgbm`）建構高精度之迴歸預測模型，並以 R²、MAE、RMSE、MAPE 評估模型效能。
5. 說明可解釋性 AI（XAI）之必要性，理解「預測準確度」與「模型可解釋性」之間的權衡關係。
6. 說明 Shapley 值源自合作賽局理論之數學原理，並理解 SHAP 如何將此理論應用於機器學習模型解釋。
7. 使用 `shap` 套件計算 SHAP 值，繪製全域 Summary Plot（Beeswarm）與局部特定樣本之 Waterfall Plot。
8. 依照論文「研究結果與討論」章節寫法，將梯度提升模型預測結果與 SHAP 解釋結果轉譯為具學術規範的文字敘述與圖表。

---

## 本週知識地圖

| 構面 | 內容 | 對應方法 | 對應 Python 套件 |
|---|---|---|---|
| 集成學習架構 | 循序修正前一輪模型之殘差誤差 | 梯度提升（Gradient Boosting） | — |
| 高效實作一 | 正則化、二階近似、稀疏感知 | XGBoost | `xgboost.XGBRegressor` |
| 高效實作二 | 直方圖分箱、葉優先生長 | LightGBM | `lightgbm.LGBMRegressor` |
| 迴歸模型評估 | 評估連續數值預測之準確度 | R²、MAE、RMSE、MAPE | `sklearn.metrics` |
| 可解釋性理論基礎 | 公平分配各特徵對預測之貢獻 | Shapley 值（合作賽局理論） | — |
| 可解釋性實作 | 計算並視覺化特徵貢獻 | SHAP（TreeExplainer） | `shap` |
| 全域解釋 | 呈現整體資料集之特徵影響型態 | Summary／Beeswarm Plot | `shap.summary_plot` |
| 局部解釋 | 呈現單一樣本之預測推論過程 | Waterfall Plot | `shap.plots.waterfall` |

---

## 理論基礎篇

### 3.1 從 Bagging 到 Boosting：集成學習的第二種架構

第 9 週介紹之隨機森林，屬於**平行集成（Parallel Ensemble）**架構——森林中每一棵決策樹皆獨立、同步訓練（各自基於不同的拔靴樣本與特徵子集），彼此之間互不影響，最終以多數決或平均整合各樹之預測結果。本週介紹之**梯度提升機（Gradient Boosting Machine, GBM）**，則屬於**循序集成（Sequential Ensemble）**架構，又稱 **Boosting**：模型並非同時獨立訓練多棵樹，而是「一棵接著一棵」依序訓練，且**每一棵新樹的訓練目標，是專門修正前面所有樹加總起來之預測結果與真實答案之間的殘差誤差**。

這正是 Bagging 與 Boosting 兩種集成學習架構之根本差異：Bagging 透過「多個獨立、弱相關的預測器互相平均」降低變異數（variance），適合處理容易過度配適、預測不穩定的高變異模型（如未修剪的深度決策樹）；Boosting 則透過「每一步都針對性地修正前一步尚未解決的錯誤」逐步降低偏誤（bias），通常能達到比 Bagging 更高的預測準確度，但也因為模型訓練過程高度依序、彼此緊密關聯，若不加以適當正則化控制，更容易過度配適訓練資料，這正是本週理論篇 3.3 節將介紹 XGBoost 之正則化機制的重要動機。

### 3.2 梯度提升機原理：殘差學習與加法模型

梯度提升機由 Friedman（2001，見本週參考文獻）系統化提出，其核心邏輯可表述為一個**加法模型（Additive Model）**：

$$
F_M(x) = \sum_{m=1}^{M} \gamma_m h_m(x)
$$

其中 $h_m(x)$ 為第 $m$ 棵決策樹（通常為淺層之弱學習器，weak learner）， $\gamma_m$ 為該棵樹之權重貢獻， $M$ 為總樹數。模型訓練採**逐步（stagewise）加入新樹**的方式進行：

1. 初始化模型 $F_0(x)$ 為一個簡單的常數預測（如訓練資料目標值之平均數）。
2. 對第 $m$ 輪疊代，計算目前模型 $F_{m-1}(x)$ 在每筆訓練樣本上之**負梯度（negative gradient）**——在均方誤差損失函數下，此負梯度恰等於當前之殘差 $y_i - F_{m-1}(x_i)$ 。
3. 訓練一棵新的決策樹 $h_m(x)$ ，目標是盡可能準確地擬合這些殘差（即學習「前面的模型還沒有學到、還錯在哪裡」）。
4. 將此新樹以適當的學習率（learning rate） $\eta$ 加入模型： $F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$ 。
5. 重複步驟 2–4 直至達到預設之樹木數量或其他停止準則。

此處之**學習率 $\eta$ **（`scikit-learn` 系列套件中對應 `learning_rate` 參數）扮演關鍵角色：較小之學習率使每一棵樹的貢獻更為保守、需要更多棵樹才能達到相同配適程度，但通常能換取更好的泛化能力（降低過度配適風險），這是梯度提升機超參數調校中最重要的權衡考量之一。

### 3.3 XGBoost：正則化與二階泰勒展開優化

**XGBoost（eXtreme Gradient Boosting）**由 Chen 與 Guestrin（2016，發表於 *KDD*，見本週參考文獻）提出，是梯度提升機之工程與統計雙重優化實作，相較傳統 GBM 之關鍵改良包括：

**正則化目標函數（Regularized Objective）**：XGBoost 在損失函數中額外加入樹複雜度之懲罰項，直接將模型複雜度控制內建於最佳化目標本身，而非僅依賴事後之樹深度限制：

$$
Obj = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{m=1}^{M} \Omega(h_m)
$$

$$
\Omega(h) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2
$$

其中 $T$ 為樹的葉節點數、 $w_j$ 為各葉節點之預測值、 $\gamma$ 與 $\lambda$ 為正則化強度超參數。此設計使 XGBoost 天生具備比傳統 GBM 更強之抗過度配適能力，這也是 XGBoost 在資料科學競賽與實務應用中被廣泛採用之核心原因之一。

**二階泰勒展開近似（Second-Order Taylor Approximation）**：XGBoost 在求解每一輪新樹之最佳分割時，同時運用損失函數之一階導數（梯度）與二階導數（海森矩陣），相較傳統 GBM 僅使用一階梯度資訊，能更精確地逼近最優解、加速收斂。

**稀疏感知與加權分位數略圖**：XGBoost 針對真實世界資料中常見之缺失值與稀疏特徵，設計了專屬之處理機制，能自動學習缺失值應歸屬左子節點或右子節點之最佳預設方向，無需事先進行繁瑣的缺失值填補。

### 3.4 LightGBM：直方圖演算法與葉優先生長策略

**LightGBM（Light Gradient Boosting Machine）**由 Ke 等人（2017，發表於 *NeurIPS*，見本週參考文獻，微軟研究院開發）提出，設計目標是在大型資料集情境下，進一步提升梯度提升機之訓練效率：

**直方圖演算法（Histogram-based Algorithm）**：將連續特徵值離散化分箱（binning）為有限數量之直方圖區間，尋找最佳分割點時僅需在有限的直方圖區間中搜尋，而非如傳統做法般需檢視每一個獨特特徵值，大幅降低運算複雜度與記憶體用量。

**葉優先生長策略（Leaf-wise Growth Strategy）**：傳統決策樹（包含 XGBoost 預設）多採用**逐層生長（level-wise growth）**，即同一深度的所有節點會被同時展開；LightGBM 則採用**葉優先生長**，每一輪僅選擇當前所有葉節點中，能讓損失函數下降最多的那一個節點進行分割。此策略通常能以較少的樹木數量達到較低的訓練誤差，但也因為容易產生較不平衡、較深的樹結構，在小資料集情境下有較高之過度配適風險，實務上須格外留意 `max_depth` 或 `num_leaves` 等參數之設定。

**XGBoost 與 LightGBM 之選擇實務**：兩者在絕大多數表格型資料（tabular data）預測任務中，效能表現通常相當接近（本週 Colab 實作將以相同資料集實際比較兩者之預測效能），LightGBM 因訓練速度更快，在特徵數與樣本數皆龐大之情境下優勢較為明顯；XGBoost 則因發展較早、社群生態與相關工具（包含本週稍後介紹之 SHAP 套件）之整合支援更為成熟，仍是目前學術研究與實務應用中最廣泛採用之梯度提升機實作。

### 3.5 可解釋性 AI（XAI）的必要性

如第 9 週 Q&A 所述，隨機森林與梯度提升機等集成學習模型，雖然預測準確度通常優於單一決策樹，卻也犧牲了「個別樣本層次」之完整可解釋性——面對「這一筆特定的房屋估價，模型究竟是依據哪些特徵、各自貢獻了多少，才得出這個預測總價」此類問題，僅有全域特徵重要性（如第 9 週介紹之 `feature_importances_`）並不足以回答。

**可解釋性 AI（Explainable AI, XAI）**正是為了解決此一「準確度與透明度」權衡困境而發展的方法論領域，其實務價值至少展現於三個層面：(1) **模型除錯與信任建立**：協助研究者與決策者確認模型是否依循合理的邏輯做出預測，而非依賴資料中的虛假相關性；(2) **法規遵循與可課責性**：金融信用評等、醫療診斷輔助等高風險應用領域，監理法規（如歐盟一般資料保護規則 GDPR 賦予之「解釋權」）日益要求演算法決策具備可解釋性；(3) **管理決策支援**：如本週研究範例之房地產估價情境，「哪些特徵、以何種方向與幅度影響了這筆房產的估值」本身就是極具實務價值的資訊，而非僅止於得出一個預測數字。

### 3.6 Shapley 值理論基礎：合作賽局理論

**Shapley 值（Shapley Value）**由諾貝爾經濟學獎得主 Lloyd Shapley 於 1953 年在合作賽局理論（Cooperative Game Theory）脈絡下提出（見本週參考文獻），原始問題情境為：一群參與者合作完成一項任務並共同創造出某個總價值，應該如何「公平地」將這個總價值分配給每一位參與者，才能反映出每個人對團隊貢獻的真實大小？

Shapley 值之數學定義，是計算每個參與者 $i$ 在**所有可能的參與順序排列**中，其加入合作聯盟後所帶來之**邊際貢獻（marginal contribution）**的平均值：

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} \left[ v(S \cup \{i\}) - v(S) \right]
$$

其中 $N$ 為全體參與者集合、 $S$ 為不包含參與者 $i$ 的任意子集合、 $v(S)$ 為聯盟 $S$ 所能創造之價值。此公式看似複雜，其直覺意涵其實相當清晰：**公平地考量參與者 $i$ 在「所有可能加入團隊的先後順序」下，平均而言能為團隊帶來多少額外貢獻**，Shapley（1953）證明此一分配方式，是唯一同時滿足效率性（總分配等於總價值）、對稱性（貢獻相同者分配相同）、虛擬性（無貢獻者分配為零）與可加性（多個賽局之分配可直接加總）四項公理的分配方案。

**Shapley 值如何應用於機器學習模型解釋**：將此一經濟學理論框架「翻譯」至機器學習情境——把「合作完成任務的參與者」對應為「模型的各個特徵」，把「團隊共同創造的總價值」對應為「模型對某一筆樣本的預測值（相對於基準期望值之差異）」，即可計算出「每一個特徵，對這一筆特定樣本之預測結果，貢獻了多少」，這正是 SHAP 方法之理論根基。

### 3.7 SHAP：統一可加性特徵歸因框架

**SHAP（SHapley Additive exPlanations）**由 Lundberg 與 Lee（2017，發表於 *NeurIPS*，見本週參考文獻）提出，將 Shapley 值理論系統化應用於機器學習模型解釋，並證明其與另外多種既有解釋方法（如 LIME）在數學形式上具有統一性，是目前學術界與業界最廣泛採用之模型解釋框架。SHAP 之核心性質為**局部準確性（local accuracy）**：對任一樣本 $x$ ，其預測值可精確分解為基準期望值加上所有特徵之 SHAP 值加總：

$$
f(x) = \phi_0 + \sum_{j=1}^{p} \phi_j
$$

其中 $\phi_0$ 為模型在整個訓練資料集上之**期望預測值（base value，即模型對「不知道任何特徵資訊」時的平均預測）**， $\phi_j$ 為第 $j$ 個特徵之 SHAP 值，代表該特徵將預測值從基準期望值「推動」了多少（正值代表向上推動、負值代表向下推動）。此一可加性分解特性，正是本週提示詞實作「解析單一特徵對預測目標的正負推力」之數學基礎，也是 SHAP 相較其他解釋方法（許多方法僅提供相對排序、缺乏此種嚴謹可加性保證）更具理論說服力之處。

**TreeSHAP**：針對決策樹集成模型（如本週使用之 XGBoost、LightGBM），Lundberg 等人（2020，見本週參考文獻）進一步提出 TreeSHAP 演算法，能利用樹狀結構之特性，以多項式時間（而非原始 Shapley 值定義所需的指數時間）精確計算 SHAP 值，這正是本週 Colab 實作採用 `shap.TreeExplainer` 之技術基礎，使 SHAP 值計算在實務規模的資料集上具備可行的運算效率。

### 3.8 SHAP 視覺化：Summary／Beeswarm 圖與 Waterfall 圖

**Summary Plot（又稱 Beeswarm Plot）**：呈現**全域（global）**解釋觀點，將測試集中每一筆樣本、每一個特徵之 SHAP 值，同時繪製於一張圖上——縱軸依特徵重要性排序（通常取該特徵 SHAP 值絕對值之平均數）、橫軸為 SHAP 值大小（代表對預測值之推力方向與幅度），並以顏色深淺編碼該特徵之原始數值高低（如紅色代表特徵值較高、藍色代表較低）。此圖形不僅能呈現「哪些特徵整體而言最重要」，更能同時呈現「該特徵數值較高或較低時，通常會將預測值往哪個方向推動」，資訊密度遠高於單純的特徵重要性長條圖，是本週提示詞實作要求繪製之核心全域解釋圖表。

**Waterfall Plot（瀑布圖）**：呈現**局部（local）**解釋觀點，針對**單一特定樣本**，以瀑布圖形式，由下而上（或由左而右）逐一呈現：從基準期望值 $\phi_0$ 出發，每一個特徵之 SHAP 值如何一步步將預測結果「推動」至該樣本最終的預測值，正貢獻（推高預測值之特徵）與負貢獻（拉低預測值之特徵）以不同顏色區分，是本週提示詞實作要求繪製、用於「解析單一特徵對預測目標的正負推力」之核心局部解釋圖表，也是實務上回答「這一筆特定樣本，模型為什麼會做出這個預測」此類問題最直觀有力的視覺化工具。

### 3.9 論文中 XGBoost／SHAP 章節的標準寫法架構

1. **模型設定與超參數說明**：樹木數量、學習率、樹深等關鍵超參數設定。
2. **模型效能評估**：R²、MAE、RMSE、MAPE（迴歸問題）或第 9 週介紹之分類評估指標（分類問題）。
3. **模型效能比較**：如 XGBoost 與 LightGBM 之比較，或與第 9 週隨機森林之比較。
4. **SHAP 全域解釋**：Summary/Beeswarm Plot，並討論關鍵特徵之整體影響方向。
5. **SHAP 局部解釋**：選取具代表性之個案（如預測誤差較大或較小之樣本、具管理意義之極端案例），繪製 Waterfall Plot 並逐一說明特徵貢獻。
6. **管理實務意涵討論**：結合全域與局部解釋結果，提出具體之決策建議或制度設計意涵。

---

## 研究設計實例：房地產實價登錄自動化估價模型與特徵非線性權重透明化研究

### 4.1 研究背景與動機

台灣自 2012 年施行不動產實價登錄制度後（見本週參考文獻中之內政部地政司官方資料），累積之交易資料為房地產自動化估價模型（Automated Valuation Model, AVM）之發展提供了堅實的資料基礎。既有台灣本土研究（見本週參考文獻）已嘗試以類神經網路、堆疊泛化等機器學習方法建構房價預測模型，然而此類模型多聚焦於預測準確度之提升，較少系統化呈現「各項房屋特徵如何以非線性方式影響估價結果」此一對買賣雙方與仲介從業人員同樣重要的透明化資訊。本研究以 XGBoost 建構房價預測模型，並運用 SHAP 值拆解模型之非線性特徵權重，兼顧估價準確度與決策透明度。

### 4.2 研究目的

1. 蒐集房地產實價登錄交易資料，建構房屋特徵與總價之預測資料集。
2. 訓練 XGBoost 與 LightGBM 迴歸模型，比較兩者之估價準確度。
3. 運用 SHAP 值進行全域特徵影響分析，辨識影響房價的關鍵非線性因子。
4. 針對特定房屋估價案例，以 SHAP Waterfall Plot 呈現個別估價結果之推論過程，提升估價透明度。

### 4.3 預測特徵架構

| 特徵代碼 | 特徵名稱 | 操作型定義 |
|---|---|---|
| F1 | 屋齡 | 建物完工至交易當下之年數 |
| F2 | 建坪 | 建物登記總面積（坪） |
| F3 | 所在樓層 | 交易標的所在樓層數 |
| F4 | 總樓層數 | 該建物之總樓層數 |
| F5 | 捷運站距離 | 交易標的至最近捷運站之直線距離（公尺） |
| F6 | 房間數 | 交易標的之房間數量 |
| F7 | 是否含車位 | 交易是否包含停車位（0/1） |

### 4.4 研究對象與資料來源

建議以內政部不動產交易實價查詢服務網（見本週參考文獻）之特定行政區、特定期間集合式住宅交易資料為研究對象，樣本數建議至少 500–1000 筆以上，以確保梯度提升機模型有足夠訓練資料學習特徵間之非線性交互關係。本課程範例為教學示範，採用模擬資料，模擬邏輯中刻意加入捷運站距離對房價之非線性（指數遞減）影響關係，用以示範梯度提升機相較線性迴歸模型，能更準確捕捉此類非線性特徵效果之方法論優勢。

### 4.5 資料分析流程規劃

```
蒐集實價登錄房屋特徵與總價資料
        │
        ▼
資料清理（極端值檢查、遺漏值處理）
        │
        ▼
切分訓練集與測試集
        │
        ▼
訓練 XGBoost 迴歸模型
        │
        ▼
訓練 LightGBM 迴歸模型（效能對照）
        │
        ▼
計算 R²、MAE、RMSE、MAPE，比較兩模型效能
        │
        ▼
選定最終模型，以 SHAP TreeExplainer 計算 SHAP 值
        │
        ▼
繪製 Summary/Beeswarm Plot（全域解釋）
        │
        ▼
選取代表性個案，繪製 Waterfall Plot（局部解釋）
        │
        ▼
撰寫研究結果與估價透明化管理意涵討論
```

---

## Colab 實作環境建置

```python
# ============================================================
# Cell 0：Colab 環境建置與套件安裝
# ------------------------------------------------------------
# 說明：xgboost 與 shap 套件 Colab 環境通常已預先安裝，
# lightgbm 亦多已預裝，惟仍建議執行安裝指令確保版本相容性。
# ============================================================

!pip install xgboost lightgbm shap --quiet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ------------------------------------------------------------
# 設定中文字型（沿用第 1–9 週相同設定邏輯）
# ------------------------------------------------------------
!wget -q https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf -O /content/NotoSansTC.otf
from matplotlib import font_manager
font_manager.fontManager.addfont('/content/NotoSansTC.otf')
plt.rcParams['font.family'] = 'Noto Sans CJK TC'
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')

print("環境建置完成，本週使用 xgboost、lightgbm 與 shap 建構估價模型與解釋分析。")
```

---

## Colab 實作：Step by Step 完整程式碼

### Step 1：模擬房地產實價登錄交易資料

```python
# ============================================================
# Cell 1：模擬具有非線性特徵效果之房地產交易資料
# ------------------------------------------------------------
# 教學目的：刻意在資料生成階段，讓「捷運站距離」以指數遞減
# 方式非線性影響單價（距離越近、溢價效果越強，但邊際效果
# 隨距離增加而快速遞減），驗證梯度提升機模型能否正確捕捉
# 此類線性迴歸模型難以直接處理的非線性關係。正式研究請將
# 本 Cell 替換為讀取內政部實價登錄資料之程式碼。
# ============================================================

np.random.seed(42)
n = 1000

building_age = np.clip(np.random.exponential(15, n), 0, 50)          # 屋齡
floor_area = np.clip(np.random.normal(35, 12, n), 10, 100)            # 建坪
floor_level = np.random.randint(1, 25, n)                             # 所在樓層
total_floors = floor_level + np.random.randint(0, 15, n)              # 總樓層數
dist_mrt = np.clip(np.random.exponential(800, n), 50, 5000)           # 捷運站距離（公尺）
rooms = np.random.choice([1, 2, 3, 4, 5], n, p=[0.10, 0.25, 0.35, 0.22, 0.08])  # 房間數
has_parking = np.random.binomial(1, 0.55, n)                          # 是否含車位

# 每坪單價模型：屋齡負向線性影響；捷運站距離非線性（指數遞減）正向影響；
# 樓層與車位正向影響；並加入隨機雜訊模擬其他未觀察因素
price_per_ping = (
    45
    - 0.25 * building_age
    + 8 * np.exp(-dist_mrt / 1000)   # 非線性核心：距離越近，溢價效果越強
    + 0.3 * floor_level
    + 3 * has_parking
    + np.random.normal(0, 4, n)
)
price_per_ping = np.clip(price_per_ping, 10, None)
total_price = price_per_ping * floor_area   # 總價（萬元）= 每坪單價 × 建坪

df = pd.DataFrame({
    'building_age': building_age, 'floor_area': floor_area,
    'floor_level': floor_level, 'total_floors': total_floors,
    'dist_mrt': dist_mrt, 'rooms': rooms, 'has_parking': has_parking,
    'total_price': total_price
})

print(f"模擬交易資料維度：{df.shape[0]} 筆交易 × {df.shape[1]} 個變數")
print("\n=== 表 1：資料描述性統計 ===")
display(df.describe().round(2))
```

### Step 2：訓練測試集切分

```python
# ============================================================
# Cell 2：訓練測試集切分
# ============================================================

features = ['building_age', 'floor_area', 'floor_level', 'total_floors',
            'dist_mrt', 'rooms', 'has_parking']
X = df[features]
y = df['total_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"訓練集樣本數：{len(X_train)}，測試集樣本數：{len(X_test)}")
```

### Step 3：XGBoost 迴歸模型訓練

```python
# ============================================================
# Cell 3：XGBoost 迴歸模型訓練與評估
# ------------------------------------------------------------
# 提示詞實作對照（前半段）：
# 「使用 XGBoost / LightGBM 訓練迴歸預測模型。」
# ============================================================

xgb_model = xgb.XGBRegressor(
    n_estimators=300,       # 樹木總數
    max_depth=4,             # 每棵樹之最大深度（控制模型複雜度）
    learning_rate=0.05,      # 學習率（見理論篇 3.2 節）
    subsample=0.8,           # 每棵樹訓練時之樣本抽樣比例（降低過度配適）
    colsample_bytree=0.8,    # 每棵樹訓練時之特徵抽樣比例
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

def evaluate_regression(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"=== {model_name} 效能評估 ===")
    print(f"R² = {r2:.4f}")
    print(f"MAE（平均絕對誤差）= {mae:.2f} 萬元")
    print(f"RMSE（均方根誤差）= {rmse:.2f} 萬元")
    print(f"MAPE（平均絕對百分比誤差）= {mape:.2f}%")
    return {'model': model_name, 'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

xgb_metrics = evaluate_regression(y_test, xgb_pred, 'XGBoost')
```

### Step 4：LightGBM 迴歸模型訓練與效能比較

```python
# ============================================================
# Cell 4：LightGBM 迴歸模型訓練與效能比較
# ============================================================

lgb_model = lgb.LGBMRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)

lgb_metrics = evaluate_regression(y_test, lgb_pred, 'LightGBM')

comparison_df = pd.DataFrame([xgb_metrics, lgb_metrics])
print("\n=== 表 2：XGBoost 與 LightGBM 效能比較表 ===")
display(comparison_df.round(4))
```

### Step 5：SHAP 值計算

```python
# ============================================================
# Cell 5：SHAP 值計算（以最終選定之 XGBoost 模型為例）
# ------------------------------------------------------------
# 提示詞實作對照（後半段）：
# 「導入 shap 套件計算 SHAP 值。」
# ------------------------------------------------------------
# 說明：TreeExplainer 專為樹狀集成模型設計，利用理論篇 3.7
# 節介紹之 TreeSHAP 演算法，能以多項式時間精確計算 SHAP 值。
# ============================================================

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_test)

print(f"SHAP 值矩陣維度：{shap_values.values.shape}（測試樣本數 × 特徵數）")
print(f"基準期望值（base value）= {shap_values.base_values[0]:.2f} 萬元")
print("（此為模型在完全不知道任何特徵資訊時，對房價的平均預測值）")

# 驗證局部準確性：基準值 + 該樣本所有 SHAP 值加總，應等於模型預測值
sample_idx = 0
manual_prediction = shap_values.base_values[sample_idx] + shap_values.values[sample_idx].sum()
model_prediction = xgb_model.predict(X_test.iloc[[sample_idx]])[0]
print(f"\n驗證局部準確性（樣本 {sample_idx}）：")
print(f"基準值 + SHAP 值加總 = {manual_prediction:.2f}")
print(f"模型直接預測值 = {model_prediction:.2f}")
print(f"（兩者應幾乎完全相等，此即為理論篇 3.7 節之局部準確性數學性質）")
```

### Step 6：SHAP Summary Plot（全域解釋）

```python
# ============================================================
# Cell 6：SHAP Summary Plot（Beeswarm 圖）
# ------------------------------------------------------------
# 提示詞實作對照：
# 「繪製全域 Summary Plot（Beeswarm）。」
# ============================================================

feature_names_zh = {
    'building_age': '屋齡', 'floor_area': '建坪', 'floor_level': '所在樓層',
    'total_floors': '總樓層數', 'dist_mrt': '捷運站距離',
    'rooms': '房間數', 'has_parking': '是否含車位'
}
X_test_zh = X_test.rename(columns=feature_names_zh)

plt.figure(figsize=(9, 6))
shap.summary_plot(shap_values.values, X_test_zh, show=False)
plt.title('圖 1：SHAP Summary Plot（Beeswarm）— 全域特徵影響分析', fontsize=13, pad=15)
plt.tight_layout()
plt.savefig('shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.show()

# 各特徵之平均絕對 SHAP 值（另一種全域重要性排序方式，與原生 feature_importances_ 對照）
mean_abs_shap = pd.Series(
    np.abs(shap_values.values).mean(axis=0), index=features
).sort_values(ascending=False)
mean_abs_shap.index = [feature_names_zh[f] for f in mean_abs_shap.index]

print("=== 表 3：各特徵平均絕對 SHAP 值排序 ===")
display(mean_abs_shap.round(2))
```

### Step 7：SHAP Waterfall Plot（局部解釋）

```python
# ============================================================
# Cell 7：SHAP Waterfall Plot（單一樣本局部解釋）
# ------------------------------------------------------------
# 提示詞實作對照：
# 「局部特定樣本的 Waterfall Plot，解析單一特徵對預測目標的
#   正負推力。」
# ============================================================

# 選取一筆代表性樣本進行局部解釋（此處選取測試集第一筆樣本）
sample_idx = 0
sample_features = X_test.iloc[sample_idx]

print(f"=== 樣本 {sample_idx} 之原始特徵值 ===")
for f in features:
    print(f"  {feature_names_zh[f]}：{sample_features[f]:.2f}")
print(f"\n實際成交總價：{y_test.iloc[sample_idx]:.2f} 萬元")
print(f"模型預測總價：{xgb_model.predict(X_test.iloc[[sample_idx]])[0]:.2f} 萬元")

shap_values_zh = shap.Explanation(
    values=shap_values.values[sample_idx],
    base_values=shap_values.base_values[sample_idx],
    data=X_test_zh.iloc[sample_idx].values,
    feature_names=X_test_zh.columns.tolist()
)

plt.figure(figsize=(9, 6))
shap.plots.waterfall(shap_values_zh, show=False)
plt.title(f'圖 2：樣本 {sample_idx} SHAP Waterfall Plot（局部解釋）', fontsize=12, pad=15)
plt.tight_layout()
plt.savefig('shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Step 8：模型原生特徵重要性與 SHAP 排序對照

```python
# ============================================================
# Cell 8：模型原生特徵重要性 vs. SHAP 重要性對照
# ============================================================

native_importance = pd.Series(xgb_model.feature_importances_, index=features)
native_importance.index = [feature_names_zh[f] for f in native_importance.index]
native_importance = native_importance.sort_values(ascending=False)

importance_comparison = pd.DataFrame({
    'XGBoost原生重要性(Gain)': native_importance,
    'SHAP平均絕對值': mean_abs_shap
}).sort_values('XGBoost原生重要性(Gain)', ascending=False)

print("=== 表 4：模型原生特徵重要性與 SHAP 排序對照表 ===")
display(importance_comparison.round(4))

print("\n觀察重點：兩種排序方式在次要特徵之相對順序上可能略有差異，")
print("這是因為兩者衡量的統計量不同（見本週 Q&A 詳細說明），")
print("惟最重要與最不重要之特徵，通常兩種方法會得出一致結論。")
```

### Step 9：匯出所有分析結果

```python
# ============================================================
# Cell 9：匯出完整分析結果至 Excel
# ============================================================

with pd.ExcelWriter('房價估價模型分析結果_Week10.xlsx') as writer:
    comparison_df.round(4).to_excel(writer, sheet_name='模型效能比較', index=False)
    importance_comparison.round(4).to_excel(writer, sheet_name='特徵重要性對照')

print("所有統計結果已匯出至 房價估價模型分析結果_Week10.xlsx，可於 Colab 左側檔案面板下載。")
print(f"\n最終模型：XGBoost，R² = {xgb_metrics['R2']:.4f}，MAPE = {xgb_metrics['MAPE']:.2f}%")
print(f"最關鍵估價特徵（SHAP）：{mean_abs_shap.index[0]}")
```

---

## Vibe Coding 提示詞（Prompt）實作範例集

**範例 1：XGBoost 與 LightGBM 效能比較**

> 請分別訓練 XGBoost 與 LightGBM 迴歸模型，使用相同的超參數設定（樹數 300、學習率 0.05、樹深 4），並計算兩者在測試集上的 R²、MAE、RMSE、MAPE，整理成一張比較表，告訴我哪個模型表現較好。

**範例 2：SHAP 值計算與局部準確性驗證**

> 請使用 shap 套件的 TreeExplainer 對我的 XGBoost 模型計算 SHAP 值，並針對任一測試樣本，驗證「基準期望值加上該樣本所有特徵的 SHAP 值加總」是否確實等於模型的實際預測值，用這個驗證來解釋 SHAP 局部準確性這個性質的意義。

**範例 3：SHAP Beeswarm 圖**

> 請繪製 SHAP Summary Plot（Beeswarm 圖），特徵名稱請使用中文標籤，並告訴我哪個特徵整體而言對房價影響最大，以及該特徵數值較高時通常會把預測價格往上還是往下推。

**範例 4：SHAP Waterfall 圖**

> 請選擇測試集中的第一筆樣本，繪製 SHAP Waterfall Plot，圖上請顯示從基準期望值到最終預測值的完整推論過程，並用文字說明哪些特徵是這筆樣本估價偏高或偏低的主要原因。

**範例 5：特徵重要性方法比較**

> 請將 XGBoost 原生的特徵重要性（基於 Gain）與 SHAP 平均絕對值兩種排序方式整理成對照表，如果兩者排序不完全一致，請解釋可能的原因。

**範例 6：結果段落初稿撰寫**

> 根據以下統計結果（XGBoost：R²=.935，MAPE=7.02%；LightGBM：R²=.934；SHAP 顯示建坪、屋齡、捷運站距離為前三大影響特徵；某樣本基準值1868.88萬元，建坪貢獻+519.55萬元，捷運站距離貢獻-70.02萬元，最終預測2554.92萬元），請以碩士論文研究結果章節的學術寫作語氣，撰寫一段約 300 字的中文分析段落。

---

## 結果呈現與分析：碩士論文寫法示例

以下段落數值取自本週 Colab 範例程式碼之實際執行結果，供學生對照模仿寫作邏輯（實際數值請以自己資料之 Colab 輸出為準）。

> **4.1 模型效能評估與比較**
>
> 本研究以 XGBoost 與 LightGBM 分別訓練房價預測迴歸模型，結果顯示：XGBoost 之測試集 R² 為 .935，平均絕對誤差（MAE）為 118.18 萬元，均方根誤差（RMSE）為 149.88 萬元，平均絕對百分比誤差（MAPE）為 7.02%；LightGBM 之測試集 R² 為 .934，MAE 為 118.37 萬元，RMSE 為 150.55 萬元，兩模型之預測效能高度相近，差異在小數點後第三位，此一發現與理論篇 3.4 節所述——兩者在多數表格型資料預測任務中效能表現相當——完全一致。本研究採 XGBoost 作為最終分析模型，主要考量其與 SHAP 套件之整合支援較為成熟。
>
> **4.2 SHAP 全域解釋分析**
>
> SHAP Summary Plot 分析結果顯示，七項房屋特徵依平均絕對 SHAP 值排序，建坪為影響房價最關鍵之特徵，其次依序為屋齡、捷運站距離、所在樓層、是否含車位、總樓層數、房間數。值得注意的是，SHAP 排序與 XGBoost 原生特徵重要性（基於 Gain 之排序）在次要特徵之相對順序上略有差異——原生重要性顯示「是否含車位」排序高於「捷運站距離」，SHAP 分析則呈現相反排序，此一發現顯示兩種特徵重要性衡量方式（一者衡量特徵在樹分割過程中降低誤差之平均貢獻，另一者衡量特徵對個別樣本預測值之平均邊際貢獻）確實可能得出不完全一致之次要特徵排序，論文中應同時報告兩者並說明其方法論差異。Beeswarm 圖進一步顯示，捷運站距離之 SHAP 值與其原始數值呈現非線性負向關係——距離越近（顏色越藍，數值越低），SHAP 值越偏向正值（推升房價），且此一推升效果在距離極近時特別強烈，印證了模型確實成功捕捉到理論篇資料生成邏輯中刻意加入之非線性指數遞減關係。
>
> **4.3 SHAP 局部解釋：個案估價透明化**
>
> 以測試集中一筆代表性樣本為例，該樣本實際成交總價為 2,531.16 萬元，模型預測為 2,554.92 萬元，預測誤差相對較小。SHAP Waterfall Plot 顯示，模型之基準期望值為 1,868.88 萬元，其中建坪為最主要之正向貢獻特徵（+519.55 萬元），其次為所在樓層（+112.69 萬元）與屋齡（+68.43 萬元，該樣本屋齡相對年輕，故為正向貢獻）；捷運站距離則為主要負向貢獻特徵（-70.02 萬元，顯示該樣本距離捷運站相對較遠）。此一逐項拆解之估價推論過程，相較僅呈現一個預測總價之傳統估價模型，能為房屋買賣雙方與仲介從業人員提供更具體、更可追溯之估價依據說明，此為本研究運用 XAI 技術提升房地產自動化估價透明度之核心貢獻。

**APA 格式三線表範例：模型效能比較表**

| 模型 | R² | MAE（萬元） | RMSE（萬元） | MAPE |
|---|---|---|---|---|
| XGBoost | .935 | 118.18 | 149.88 | 7.02% |
| LightGBM | .934 | 118.37 | 150.55 | — |

*註：以上數值為本週 Colab 範例實際執行結果，實際研究請以自己資料之輸出為準。*

---

## 常見統計誤區與 Q&A

**Q1：SHAP 值和第 9 週學過的特徵重要性（`feature_importances_`），不是都在講「哪個特徵比較重要」嗎？為什麼結果會不完全一樣？**
這是本週最核心的方法論釐清重點，本週 Step 8 已實際示範此一現象。第 9 週介紹之特徵重要性（基於 Gini 或 Gain），衡量的是「該特徵在整個森林／樹群的所有分割節點中，平均而言降低了多少不純度或誤差」，是一個**全域、聚合層次**的統計量，且與訓練過程本身直接掛鉤；SHAP 值則是基於 Shapley 值理論，衡量「該特徵對『每一筆個別樣本』之預測值的邊際貢獻」，先計算出個別樣本層次的貢獻，再取平均絕對值作為全域重要性，是一個**由局部推導至全域**的統計量，兩者之數學定義本質不同，因此在次要特徵的相對排序上，出現不完全一致的結果是正常現象，並非模型或程式碼有誤，論文中若同時呈現兩種排序方式，應明確說明此一方法論差異，而非迴避不一致之處。

**Q2：Waterfall Plot 裡面的「基準期望值」（base value）到底代表什麼？為什麼每個樣本的基準值都一樣？**
基準期望值 $\phi_0$ ，代表模型在「完全不知道任何特徵資訊」的情況下，對目標變數（本週範例為房價）所能給出的最佳猜測，數學上等於模型在整個訓練資料集上之平均預測值，這正是為什麼在同一個訓練好的模型下，所有測試樣本之 Waterfall Plot 起點（基準值）都相同的原因——它是模型的一個全域屬性，而非因樣本而異的數值。Waterfall Plot 的核心價值，正是呈現「這筆樣本的各項特徵，如何將預測結果從這個『毫無資訊』的起點，一步步推動至該樣本最終、具體的預測值」。

**Q3：訓練梯度提升機模型時，`learning_rate`（學習率）和 `n_estimators`（樹木數量）這兩個超參數，感覺是互相牽制的，該怎麼決定？**
確實如此，這是梯度提升機超參數調校中最經典的權衡關係。較小的學習率，代表每一棵新樹對整體模型的貢獻更保守、更謹慎，通常能得到較好的泛化能力，但也代表需要更多棵樹（更大的 `n_estimators`）才能讓模型充分收斂至理想的配適程度；反之，較大的學習率能讓模型較快收斂，但每一步的修正較為激進，容易錯過最優解或導致過度配適。實務上常見做法是先設定一個相對較小的學習率（如本週範例之 0.05，甚至更小如 0.01），再透過交叉驗證搜尋能讓驗證集效能最佳之樹木數量，而非隨意設定兩者數值後直接訓練。

**Q4：SHAP 值計算聽起來牽涉到「所有可能的特徵排列組合」，這樣運算量不會很大嗎？為什麼本週的 Colab 實作跑起來速度還算可以接受？**
理論篇 3.6 節之 Shapley 值原始定義，確實需要考慮 $2^p$ 種可能的特徵子集組合（ $p$ 為特徵數），對特徵數較多的模型而言，計算量會呈指數級成長，難以直接應用於實務規模的資料集。這正是理論篇 3.7 節特別介紹 TreeSHAP 演算法的原因——Lundberg 等人（2020）證明，針對決策樹集成模型（如本週使用之 XGBoost），可以利用樹狀結構本身的特性（每個樣本在每棵樹中只會經過一條特定路徑），設計出時間複雜度僅與樹的深度和節點數呈多項式關係（而非指數關係）的精確計算演算法，這正是本週 Colab 實作能在合理時間內完成 SHAP 值計算的技術基礎，也是為什麼本週特別選用 `shap.TreeExplainer`（而非適用於任意模型、但運算效率較低的通用 `shap.KernelExplainer`）之原因。

**Q5：如果我的模型不是決策樹集成模型（例如是類神經網路），還可以用 SHAP 解釋嗎？**
可以，SHAP 套件除了本週使用之 `TreeExplainer`（專為樹狀模型優化），也提供其他 Explainer 類型，如適用於類神經網路等深度學習模型之 `DeepExplainer`，以及不依賴模型內部結構、可套用於任意「黑盒子」模型（包含類神經網路、支援向量機甚至非機器學習之複雜規則系統）之通用型 `KernelExplainer`（原理上以隨機抽樣近似計算 Shapley 值，運算效率低於 TreeSHAP，但適用範圍最廣）。這也是 SHAP 相較許多僅適用於特定模型類型之解釋方法，在理論篇 3.7 節所強調「統一框架」之實務展現——研究者可依所採用之模型類型，選擇對應之 Explainer，而不需要為每種模型另外學習一套解釋工具。

---

## 延伸研究方向：金融科技授信決策中運用 XAI 消除演算法偏見與合規性稽核

### 6.1 研究背景與理論基礎

金融科技（Fintech）業者與數位銀行日益採用機器學習模型（如本週介紹之 XGBoost）進行信用評分與授信決策，然而此類「黑盒子」模型若缺乏適當之可解釋性機制，可能面臨兩大挑戰：其一，模型可能在訓練過程中無意間學習到與敏感屬性（如性別、年齡、地區）高度相關之代理變數，產生演算法偏見（algorithmic bias），對特定族群造成系統性不利之授信待遇；其二，金融監理法規日益要求授信決策具備可追溯、可解釋之依據，缺乏透明度之模型難以通過合規性稽核。本週參考文獻中之信用卡違約預測研究，即以公開之台灣信用卡帳戶資料（UCI 信用卡違約資料集，包含 2005 年台灣 30,000 筆信用卡帳戶資料），示範以 XGBoost 結合 SHAP 進行信用風險建模與解釋穩定性評估，可作為本延伸研究方向之直接方法論參照與可取用之公開資料來源。

### 6.2 建議研究設計

1. **理論框架**：延續本週 XGBoost + SHAP 分析架構，並額外納入演算法公平性（algorithmic fairness）評估指標，檢驗模型預測結果是否對不同人口統計子群體存在系統性差異待遇。
2. **候選預測特徵（範例）**：
   - 信用歷史相關特徵（過去繳款紀錄、信用額度使用率）
   - 帳單與繳款金額趨勢特徵
   - 人口統計特徵（須特別留意是否直接或間接納入受法規保護之敏感屬性）
3. **研究對象**：可採用本週參考文獻中提及之 UCI 公開信用卡違約資料集（台灣資料），或與金融機構合作取得去識別化之真實授信資料。
4. **分析流程**：完全比照本週 Colab 實作流程（XGBoost 模型訓練 → SHAP 全域與局部解釋），並額外增加：(a) 依人口統計子群體分別計算 SHAP 值分布，檢驗是否存在系統性差異；(b) 檢驗 SHAP 解釋結果在模型重新訓練（如更換隨機種子）後之穩定性（見本週參考文獻信用卡違約研究之核心關切議題）。
5. **管理與監理實務意涵**：此類研究可協助金融機構建立具備 SHAP 解釋依據之授信決策稽核軌跡，滿足金融監理機關對演算法可課責性之要求，同時透過系統化之公平性檢驗，及早辨識並修正模型中潛藏之演算法偏見，降低監理裁罰與商譽風險。

### 6.3 給學生的思考練習

請思考：若本延伸研究之 SHAP 分析發現，「申請人所在地區郵遞區號」此一特徵，對模型預測結果具有顯著之 SHAP 貢獻，但地區郵遞區號本身可能與種族或社經地位等敏感屬性存在高度相關性（即所謂「代理變數」問題），你認為研究者應該如何處理此一發現？直接將該特徵從模型中移除，是否就能完全解決演算法偏見問題？為什麼？

---

## 課後作業與練習

**練習一：非線性效果強度敏感度分析**
請修改 Step 1 中捷運站距離對單價之非線性影響係數（原設定為 `8 * np.exp(-dist_mrt / 1000)`），將係數 8 調整為 3（減弱非線性效果）與 15（增強非線性效果），重新執行完整流程，觀察 SHAP Beeswarm 圖中捷運站距離特徵之視覺型態變化，並說明你觀察到的規律。

**練習二：真實資料集實作**
請自行至內政部不動產交易實價查詢服務網或政府資料開放平台，下載特定行政區之實價登錄交易資料，套用本週完整 Colab 程式碼執行 XGBoost 估價模型與 SHAP 解釋分析。請繳交：(1) 資料來源與篩選條件說明、(2) 執行後的 Colab Notebook（.ipynb）、(3) 一頁 A4 的結果摘要（比照本週「結果呈現與分析」段落之寫法）。

**練習三：Waterfall Plot 案例選取延伸**
請分別選取測試集中「模型預測誤差最大」與「模型預測誤差最小」之兩筆樣本，各自繪製 Waterfall Plot，比較兩筆樣本之 SHAP 貢獻結構有何不同，並嘗試推論模型在何種特徵組合情境下，預測準確度可能較差。

**練習四：文獻延伸閱讀報告**
請從本週「參考文獻與延伸閱讀」清單中，任選一篇 XGBoost、LightGBM 或 SHAP 相關之期刊論文或台灣碩士論文，撰寫一頁重點摘要，內容須包含：(1) 該研究之預測任務與特徵架構、(2) 報告之模型效能指標、(3) 是否運用 XAI 方法進行解釋分析及其發現、(4) 該研究提出之實務應用建議為何。

**練習五：DeepExplainer 或 KernelExplainer 延伸應用**
請查閱 `shap` 套件官方文件中關於 `KernelExplainer` 之說明，嘗試將本週 XGBoost 模型改用 `KernelExplainer`（而非 `TreeExplainer`）計算 SHAP 值，比較兩者計算所需時間與最終 SHAP 值結果之差異，並說明為何 `TreeExplainer` 在樹狀模型情境下是更有效率的選擇。

---

## 參考文獻與延伸閱讀（已查核連結）

1. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)*, 785-794.
   https://arxiv.org/abs/1603.02754
   （XGBoost 之原始創始論文，完整推導正則化目標函數與二階泰勒展開近似之數學基礎，為本週理論篇 3.3 節之直接方法論依據。）

2. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
   https://arxiv.org/pdf/1810.08744
   （LightGBM 之原始創始論文出處對照，說明直方圖演算法與葉優先生長策略，為本週理論篇 3.4 節之直接方法論依據。）

3. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
   https://arxiv.org/abs/1705.07874
   （SHAP 方法之原始創始論文，完整推導統一可加性特徵歸因框架，為本週理論篇 3.7 節之直接方法論依據。）

4. Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56-67.
   https://arxiv.org/pdf/2402.04211
   （TreeSHAP 演算法之延伸論文出處對照，說明如何以多項式時間精確計算樹狀模型之 SHAP 值，為本週理論篇 3.7 節與 Q&A 之重要延伸依據。）

5. 以類神經網路方法建構房價估價模型-以高雄市實價登錄資料為例。國立高雄科技大學。
   https://dba.nkust.edu.tw/uploads/asset/data/623426362e6356240dbf2eaf/28.pdf
   （台灣本土以機器學習方法（類神經網路）結合實價登錄資料建構房價估價模型之研究，並納入捷運交通便捷性與總體經濟變數，與本週研究設計範例之特徵架構高度相關。）

6. 應用實價登錄建立以聚類方法之堆疊泛化房價預測模型-以桃園市區分建物房價資料為例。國立政治大學學術集成。
   https://ah.nccu.edu.tw/item?item_id=158566
   （台灣本土以堆疊泛化集成學習方法建構房價預測模型之研究，可作為本週梯度提升機方法之比較延伸閱讀。）

7. 中華民國內政部地政司全球資訊網：不動產交易實價登錄。
   https://www.land.moi.gov.tw/chhtml/property
   （台灣實價登錄制度之官方資料來源，為本週研究設計範例真實資料蒐集之直接依據。）

8. 信用风险管理中的 SHAP 稳定性：信用卡违约模型案例研究。alphaXiv。
   https://www.alphaxiv.org/zh/abs/2508.01851
   （以台灣 UCI 信用卡違約公開資料集（30,000 筆帳戶，2005 年），結合 XGBoost 與 SHAP 進行信用風險建模與解釋穩定性評估之研究，為本週延伸研究方向「金融科技授信決策 XAI」之直接方法論參照與可取用之公開資料來源。）

**方法論經典文獻（建議延伸閱讀，非本次線上搜尋來源，圖書館或資料庫可查閱）**：

- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189-1232.
- Shapley, L. S. (1953). A value for n-person games. In H. W. Kuhn & A. W. Tucker (Eds.), *Contributions to the Theory of Games II* (pp. 307-317). Princeton University Press.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135-1144.
- Molnar, C. (2022). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable* (2nd ed.).

---

## 附錄

### 附錄 A：Bagging（第 9 週）與 Boosting（本週）方法對照表

| 比較構面 | Bagging（隨機森林，第 9 週） | Boosting（梯度提升機，本週） |
|---|---|---|
| 訓練方式 | 平行、獨立訓練多棵樹 | 循序、依前一輪殘差訓練新樹 |
| 核心目標 | 降低變異數（variance），穩定預測 | 降低偏誤（bias），提升準確度 |
| 過度配適風險 | 較低（多樹平均自然抵銷雜訊） | 較高（需靠正則化、學習率控制） |
| 典型代表 | Random Forest | XGBoost、LightGBM、GBM |
| 訓練速度 | 可平行化，較快 | 循序依賴，較慢（惟 LightGBM 已大幅優化） |

### 附錄 B：XGBoost 與 LightGBM 方法對照表

| 比較構面 | XGBoost | LightGBM |
|---|---|---|
| 起源 | Chen & Guestrin (2016) | Ke et al. (2017)，微軟研究院 |
| 樹生長策略 | 逐層生長（level-wise，預設） | 葉優先生長（leaf-wise） |
| 分割搜尋 | 精確貪婪演算法／近似分位數 | 直方圖分箱演算法 |
| 大資料集效率 | 良好 | 更優（訓練速度通常更快） |
| 社群生態與工具整合 | 最成熟（含 SHAP 深度整合） | 快速成長中 |

### 附錄 C：術語中英對照表

| 中文術語 | 英文術語 | 縮寫 |
|---|---|---|
| 梯度提升機 | Gradient Boosting Machine | GBM |
| 加法模型 | Additive Model | — |
| 學習率 | Learning Rate | — |
| 正則化 | Regularization | — |
| 可解釋性人工智慧 | Explainable AI | XAI |
| 合作賽局理論 | Cooperative Game Theory | — |
| Shapley 值 | Shapley Value | — |
| 統一可加性特徵歸因 | SHapley Additive exPlanations | SHAP |
| 基準期望值 | Base Value | — |
| 局部準確性 | Local Accuracy | — |
| 全域解釋 | Global Interpretation | — |
| 局部解釋 | Local Interpretation | — |
| 演算法偏見 | Algorithmic Bias | — |
| 演算法問責 | Algorithmic Accountability | — |

### 附錄 D：常見程式錯誤排解（Debugging Tips）

| 錯誤現象 | 常見原因 | 排解建議 |
|---|---|---|
| `shap.TreeExplainer` 初始化時拋出錯誤 | 傳入尚未訓練完成之模型物件 | 確認 `xgb_model.fit()` 已成功執行後才建立 Explainer |
| Waterfall Plot 繪製時特徵名稱顯示為英文而非中文 | 未使用 `shap.Explanation` 物件重新指定 `feature_names` | 依 Step 7 做法，重新建構帶有中文特徵名稱之 `Explanation` 物件 |
| SHAP 值加總與模型預測值不完全相等（有微小誤差） | 屬浮點數運算之正常捨入誤差 | 誤差通常在小數點後多位，屬正常現象，非程式錯誤 |
| `summary_plot` 繪製之圖形中文顯示為方框 | matplotlib 中文字型設定未套用至 shap 繪圖函式 | 確認 Cell 0 之字型設定已正確執行，且在呼叫 `shap.summary_plot()` 前已設定完成 |
| LightGBM 訓練時出現大量警告訊息 | 未設定 `verbose=-1` 參數 | 於 `LGBMRegressor()` 初始化時加入 `verbose=-1` 抑制冗餘訊息輸出 |

### 附錄 E：繳交前自我檢核清單

- [ ] 已說明模型超參數設定（樹數、學習率、樹深等）
- [ ] 已報告 R²、MAE、RMSE、MAPE 等迴歸評估指標
- [ ] 已比較 XGBoost 與 LightGBM（或與第 9 週隨機森林）之效能差異
- [ ] 已使用 `shap.TreeExplainer` 計算 SHAP 值，並驗證局部準確性
- [ ] 已繪製 SHAP Summary/Beeswarm Plot 並討論全域特徵影響型態
- [ ] 已選取具代表性樣本繪製 Waterfall Plot 並逐項解釋特徵貢獻
- [ ] 已比較模型原生特徵重要性與 SHAP 排序，並說明兩者差異之方法論原因
- [ ] 已針對關鍵特徵之非線性影響型態提出具體管理實務意涵
- [ ] 所有統計結果之文字敘述與表格數值一致，無謄寫錯誤
- [ ] 已在研究限制中說明模型可解釋性工具（SHAP）本身之侷限性（如僅反映相關性而非因果關係）

### 附錄 F：研究倫理提醒

本週研究設計涉及使用房地產實價登錄資料（或延伸研究方向中之信用卡違約資料）建構自動化估價或授信決策模型，除延續前九週已說明之知情同意、匿名性等基本倫理原則外，特別提醒：實價登錄資料雖為政府公開資料，仍應留意資料中是否包含足以識別特定交易當事人之細節資訊（如過於精確之地址），使用時應依循資料開放使用規範；若研究成果涉及自動化估價或授信模型之實務部署，應特別留意 SHAP 等 XAI 工具雖能提升模型透明度，但其解釋結果反映的是統計相關性而非因果關係，不應被過度解讀為「證明」某特徵「導致」了特定估價或授信結果，此一區辨對於避免模型解釋結果被誤用於歧視性決策辯護，具有重要之研究倫理意涵。

---

## 下週預告

第 11 週將進入第三模組之收官週「混合式 AI 決策架構：結構方程模型結合類神經網路（SEM-ANN）」，學生將學習如何整合第一模組（第 1–3 週）之 PLS-SEM 線性因果路徑分析，與本模組（第 8–10 週）所學之機器學習技術，建構兩階段混合研究架構——以 SEM 檢驗線性因果假設與指標顯著性，以類神經網路（ANN）補強捕捉變數間之深層非線性與補償性關聯，兼取兩種方法論典範之長處。研究範例將以「行動支付使用者滿意度與忠誠度：PLS-SEM 驗證與類神經網路非線性權重補強」為主題，並延伸至「智慧觀光平台體驗滿意度前因分析：SEM-ANN 雙階段混合模型」之期末專題發想方向，同時作為第三模組（第 8–11 週）資料驅動型 AI 方法之總結收官。
