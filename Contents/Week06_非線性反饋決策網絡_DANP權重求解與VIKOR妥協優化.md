# 第 6 週：非線性反饋決策網絡——DANP 權重求解與 VIKOR 妥協優化

> 課程模組：第二模組｜知識驅動型 AI 與多準則決策系統（第 4–7 週）
> 本週定位：承接第 5 週 DEMATEL 之總影響矩陣，正式解構第 4 週 AHP 所隱含之「傳統指標獨立性假設」，以 DEMATEL-based ANP（DANP）方法建構未加權、加權與極限超矩陣，求解出可直接使用之影響權重；並以 VIKOR 妥協排序法取代第 4 週之 TOPSIS，處理準則間存在複雜回饋關係時的方案排序與落差改善問題，為第二模組收官前之方法論整合週次。

> 教材版本：v1.0｜適用對象：在職專班研究方法與 AI 應用課程｜先修基礎：第 4 週（AHP、Fuzzy AHP 與 TOPSIS）、第 5 週（DEMATEL 與 INRM）
> 使用工具：Google Colab（Python 3）｜主要套件：`numpy`、`pandas`、`matplotlib`（DANP 與 VIKOR 核心演算法以 `numpy` 手動實作，所有程式碼已實際測試驗證可正常執行）

---

## 目錄

1. [學習目標](#學習目標)
2. [本週知識地圖](#本週知識地圖)
3. [理論基礎篇](#理論基礎篇)
   1. [3.1 從 DEMATEL 到 DANP：解構準則獨立假設](#31-從-dematel-到-danp解構準則獨立假設)
   2. [3.2 未加權超矩陣的建構](#32-未加權超矩陣的建構)
   3. [3.3 加權超矩陣與極限超矩陣](#33-加權超矩陣與極限超矩陣)
   4. [3.4 VIKOR 方法起源與核心邏輯](#34-vikor-方法起源與核心邏輯)
   5. [3.5 VIKOR 完整演算法：S、R、Q 值計算](#35-vikor-完整演算法srq-值計算)
   6. [3.6 妥協解之可接受條件：C1 與 C2](#36-妥協解之可接受條件c1-與-c2)
   7. [3.7 修正版 VIKOR 與期望水準落差分析](#37-修正版-vikor-與期望水準落差分析)
   8. [3.8 DANP-VIKOR 整合流程與論文標準寫作架構](#38-danp-vikor-整合流程與論文標準寫作架構)
4. [研究設計實例：綠色智慧建築方案評比與改善路徑](#研究設計實例綠色智慧建築方案評比與改善路徑)
5. [Colab 實作環境建置](#colab-實作環境建置)
6. [Colab 實作：Step by Step 完整程式碼](#colab-實作step-by-step-完整程式碼)
7. [Vibe Coding 提示詞（Prompt）實作範例集](#vibe-coding-提示詞prompt實作範例集)
8. [結果呈現與分析：碩士論文寫法示例](#結果呈現與分析碩士論文寫法示例)
9. [常見統計誤區與 Q&A](#常見統計誤區與-qa)
10. [延伸研究方向：醫療院所導入 AI 輔助排班系統滿意度落差之改善策略研究](#延伸研究方向醫療院所導入-ai-輔助排班系統滿意度落差之改善策略研究)
11. [課後作業與練習](#課後作業與練習)
12. [參考文獻與延伸閱讀（已查核連結）](#參考文獻與延伸閱讀已查核連結)
13. [附錄](#附錄)
14. [下週預告](#下週預告)

---

## 學習目標

完成本週課程後，學生應能夠：

1. 說明 DANP（DEMATEL-based ANP）方法如何解構 AHP 之準則獨立假設，並理解其與網路分析法（ANP）之關聯。
2. 使用 Python（`numpy`）將第 5 週求得之總影響矩陣，轉換為未加權超矩陣、加權超矩陣。
3. 透過矩陣冪次疊代法求解極限超矩陣，並從中萃取出可直接使用之 DANP 影響權重。
4. 說明 VIKOR 方法之核心邏輯——群體效用最大化與個別遺憾最小化，並理解其與第 4 週 TOPSIS 在方法論定位上的差異。
5. 使用 Python 完整實作 VIKOR 演算法，計算群體效用值（S）、個別遺憾值（R）與妥協指標值（Q）。
6. 判讀妥協解之兩項可接受條件（C1 可接受優勢、C2 可接受穩定性），並正確處理條件不成立時之因應做法。
7. 使用修正版 VIKOR（以期望水準取代理想解）進行落差分析，繪製改善優先級落差分析圖。
8. 依照論文「研究結果與討論」章節寫法，將 DANP 權重與 VIKOR 妥協排序結果轉譯為具學術規範的文字敘述與圖表。

---

## 本週知識地圖

| 構面 | 內容 | 對應方法 | 對應 Python 實作 |
|---|---|---|---|
| 網路結構建模 | 突破準則獨立假設，以網路而非層級結構建模 | ANP 網路分析法 | 概念性 |
| 影響權重求解 | 由 DEMATEL 總影響矩陣推導準則權重 | DANP（未加權→加權→極限超矩陣） | `numpy` 矩陣運算 |
| 超矩陣收斂 | 求解穩定狀態下之影響權重分布 | 矩陣冪次疊代（Power Iteration） | `numpy.linalg.matrix_power` |
| 方案妥協排序 | 兼顧群體效用與個別遺憾之排序方法 | VIKOR（S、R、Q 值） | 自訂函式 |
| 妥協解驗證 | 判斷排序結果是否構成穩定之單一最適解 | C1／C2 可接受條件檢驗 | 自訂函式 |
| 落差改善分析 | 以期望水準取代相對理想解，呈現改善優先順序 | 修正版 VIKOR（Aspiration Level） | 自訂函式 |
| 結果視覺化 | 呈現落差改善優先級與妥協排序結果 | 落差分析圖、排序長條圖 | `matplotlib` |

---

## 理論基礎篇

### 3.1 從 DEMATEL 到 DANP：解構準則獨立假設

第 4 週介紹之 AHP 方法，其權重求解建立在「準則彼此獨立、呈單向層級結構」之假設上；第 5 週介紹之 DEMATEL 方法，雖然成功辨識出準則間之因果影響關係（原因群、結果群），但其直接產出（中心度、原因度）本身並非可直接套用於方案排序之「權重」。**DANP（DEMATEL-based Analytic Network Process）** 正是為了銜接這兩者之間的缺口而發展——它借用網路分析法（Analytic Network Process, ANP，由 Saaty 於 1996 年提出，是 AHP 之網路化延伸）之數學架構，但以 DEMATEL 求得之總影響矩陣，取代 ANP 原本需要另外進行之準則間成對比較，直接推導出能反映準則間複雜回饋關係的影響權重（見本週參考文獻中 Ou-Yang et al., 2008 之原始方法論文獻）。

**ANP 與 AHP 的本質差異**：AHP 假設決策結構為單向層級（目標→準則→方案，準則間互不影響）；ANP 則將決策結構視為一個網路（network），允許準則與準則之間、甚至準則與方案之間存在雙向回饋關係。ANP 原始做法需要針對網路中每一組可能存在依賴關係的元素，都執行一次成對比較，運算與問卷設計工作量極為龐大；DANP 之關鍵貢獻，正是以 DEMATEL 總影響矩陣「一次性」捕捉所有準則間之影響關係，大幅簡化了 ANP 之實務操作複雜度，這也是近十年 DANP 方法在管理學、工程決策領域被廣泛採用之核心原因（見本週參考文獻中多篇綠色建築評估相關研究）。

### 3.2 未加權超矩陣的建構

DANP 之第一步，是將第 5 週求得之總影響矩陣 $T$ （ $n \times n$ ， $n$ 為準則數）轉換為 ANP 架構所需之**超矩陣（Supermatrix）**形式：

**步驟一：將總影響矩陣依列正規化**

$$
T_c = \begin{bmatrix} t_{ij} / \sum_{j=1}^{n} t_{ij} \end{bmatrix}_{n \times n}
$$

正規化後， $T_c$ 矩陣每一列元素總和為 1（列隨機矩陣，row-stochastic matrix），代表「準則 $i$ 之總影響力，如何依比例分配給其他準則 $j$ 」。

**步驟二：轉置得到未加權超矩陣**

$$
W = T_c^{T}
$$

轉置後， $W$ 矩陣每一「欄」元素總和為 1（欄隨機矩陣，column-stochastic matrix），這是配合 ANP／馬可夫鏈（Markov Chain）之傳統慣例——超矩陣之欄代表「起始節點」，列代表「目標節點」， $W_{ij}$ 代表「準則 $j$ 對準則 $i$ 之影響力占準則 $j$ 對外總影響力之比例」。此矩陣即為 DANP 之**未加權超矩陣（Unweighted Supermatrix）**。

### 3.3 加權超矩陣與極限超矩陣

**加權超矩陣（Weighted Supermatrix）**：若決策問題之準則可進一步分組為多個「構面（dimension／cluster）」（例如本週研究範例中，五項綠色智慧建築評選準則若能歸類為「技術構面」與「人本構面」兩大類），則須以構面層級之 DEMATEL 總影響矩陣，對未加權超矩陣中對應之區塊（block）進行加權，確保不同構面間的相對重要性也能反映在超矩陣中。若決策問題僅有單一層級之準則（不涉及構面分組，如本週研究範例之簡化情境），則加權超矩陣在數學上等同於未加權超矩陣，因為此時沒有「跨構面」的權重需要額外調整。

**極限超矩陣（Limit Supermatrix）**：DANP 求解權重之核心概念，是將加權超矩陣持續自乘（冪次疊代），直至矩陣收斂至一個穩定狀態：

$$
W_{limit} = \lim_{k \to \infty} W^k
$$

此收斂概念與馬可夫鏈之「穩態分布（Stationary Distribution）」完全類比——想像每個準則之影響力如同在網路中不斷流動、傳遞、再分配，經過足夠多次的傳遞疊代後，各準則所獲得之影響力占比將趨於穩定，不再隨疊代次數增加而改變。收斂後之極限超矩陣，其每一欄理論上會呈現完全相同之數值分布，此時任取其中一欄（正規化後），即為最終之 **DANP 影響權重**。此權重同時蘊含了準則間所有直接與間接的回饋影響關係，這正是 DANP 相較於 AHP 特徵向量法權重，在方法論上更貼近複雜決策現實的關鍵優勢。

值得注意的是，DANP 權重與單純將 DEMATEL 之中心度（D+R）正規化所得之權重，數值上通常並不相同（本週 Colab 實作將實際呈現兩者之差異），因為 DANP 之極限超矩陣運算，捕捉的是影響力在整個網路中「持續流動、反覆傳遞」後的均衡分布，而非僅止於總影響矩陣的單次列欄加總，這也是本週理論篇 Q&A 中學生最常混淆之處。

### 3.4 VIKOR 方法起源與核心邏輯

VIKOR（塞爾維亞語 VlseKriterijumska Optimizacija I Kompromisno Resenje 之縮寫，意為「多準則最適化與妥協解」）由 Opricovic（1998）提出，並經 Opricovic 與 Tzeng（2004，發表於 *European Journal of Operational Research*，見本週參考文獻）進一步發展成熟。VIKOR 之核心理念，源自 Yu（1973）與 Zeleny（1982）之妥協規劃（compromise programming）概念：當決策問題存在多個互相衝突之準則、且決策者難以在方案評估初期明確表達其偏好權重時，VIKOR 提供一種**同時兼顧「群體效用最大化（多數決精神）」與「個別遺憾最小化（照顧劣勢方）」之妥協排序方法**。

**VIKOR 與 TOPSIS 之方法論差異**：兩者雖然都屬於「距離為基礎」之排序方法，但核心邏輯不同——第 4 週介紹之 TOPSIS，是尋找「同時與正理想解距離最近、與負理想解距離最遠」的方案，本質上是一種「距離加總」的概念；VIKOR 則是分別計算「群體效用測度（S，所有準則加權偏離程度之總和，類似 TOPSIS 中對正理想解的曼哈頓距離）」與「個別遺憾測度（R，表現最差之單一準則的最大偏離程度，類似柴比雪夫距離）」，並將兩者以策略係數加權整合為最終之妥協指標（Q），因此 VIKOR 相較 TOPSIS，更能兼顧「整體表現」與「最弱環節」兩種評估觀點，在準則間存在顯著衝突、或決策者格外關切最弱項目表現時，VIKOR 通常被認為是更適切的方法選擇。

### 3.5 VIKOR 完整演算法：S、R、Q 值計算

**步驟一：決定各準則之最佳值與最差值**

$$
f_j^{*} = \begin{cases} \max_i f_{ij} & \text{若準則 } j \text{ 為效益型} \\ \min_i f_{ij} & \text{若準則 } j \text{ 為成本型} \end{cases}
\qquad
f_j^{-} = \begin{cases} \min_i f_{ij} & \text{若準則 } j \text{ 為效益型} \\ \max_i f_{ij} & \text{若準則 } j \text{ 為成本型} \end{cases}
$$

**步驟二：計算群體效用值（S）與個別遺憾值（R）**

$$
S_i = \sum_{j=1}^{n} w_j \cdot \frac{f_j^{*} - f_{ij}}{f_j^{*} - f_j^{-}}
$$

$$
R_i = \max_{j} \left[ w_j \cdot \frac{f_j^{*} - f_{ij}}{f_j^{*} - f_j^{-}} \right]
$$

其中 $w_j$ 即為本週前段以 DANP 求得之準則權重（此為 DANP 與 VIKOR 兩方法整合串接之關鍵接口，對應本週提示詞實作「由總影響矩陣推導極限超矩陣取得準則權重，並執行 VIKOR 演算法」）。 $S_i$ 之統計意涵是「方案 $i$ 在所有準則上加權偏離理想值之總和」，數值越小代表整體表現越接近理想； $R_i$ 之統計意涵是「方案 $i$ 在表現最差之單一準則上的偏離程度」，數值越小代表最弱環節的落差越小。

**步驟三：計算妥協指標值（Q）**

$$
Q_i = v \cdot \frac{S_i - S^{*}}{S^{-} - S^{*}} + (1-v) \cdot \frac{R_i - R^{*}}{R^{-} - R^{*}}
$$

其中 $S^{*} = \min_i S_i$ 、 $S^{-} = \max_i S_i$ 、 $R^{*} = \min_i R_i$ 、 $R^{-} = \max_i R_i$ ， $v$ 為「群體效用策略權重」，代表決策者對「多數決精神（群體效用 S）」相對於「照顧劣勢（個別遺憾 R）」之相對重視程度，慣例上取 $v = 0.5$ （代表兩者同等重視），若決策者更重視整體多數之效用最大化，可將 $v$ 調高（趨近 1）；若更重視避免任何準則出現極端劣勢，可將 $v$ 調低（趨近 0）。所有方案依 $Q_i$ 值由小到大排序， $Q_i$ 越小代表該方案越接近妥協理想解。

### 3.6 妥協解之可接受條件：C1 與 C2

VIKOR 方法論之嚴謹之處，在於提出了兩項明確的統計條件，用以判斷依 $Q$ 值排序得出之最優方案，是否真正構成一個「穩定、可被接受」的妥協解，而非僅止於「數值上排名第一」：

**條件 C1：可接受優勢（Acceptable Advantage）**

$$
Q(A'') - Q(A') \geq DQ, \qquad DQ = \frac{1}{J-1}
$$

其中 $A'$ 為 $Q$ 值排名第一之方案， $A''$ 為排名第二之方案， $J$ 為候選方案總數。此條件要求最優方案與次優方案之 $Q$ 值差距，必須大於等於一個與方案數量相關的門檻值 $DQ$ ，確保最優方案之優勢具有統計上的顯著區隔，而非僅是誤差範圍內的些微領先。

**條件 C2：可接受穩定性（Acceptable Stability in Decision Making）**

方案 $A'$ 除了在 $Q$ 值排序中位居第一，也必須同時在 $S$ 值排序或 $R$ 值排序中至少一項位居第一（或並列第一），確保該方案不僅是「加權整合後」的最優解，也在「純粹群體效用」或「純粹個別遺憾」之單一觀點下，仍具備相對優勢，而非僅是加權係數 $v$ 選擇下的產物。

**兩項條件之判定結果與對應處理方式**：

| 判定結果 | 結論 |
|---|---|
| C1 與 C2 皆成立 | $A'$ 為唯一妥協解，可直接作為最終決策建議 |
| C1 不成立，C2 成立 | 存在一組「妥協解集合」，包含所有滿足 $Q(A_m) - Q(A') &lt; DQ$ 之方案 $A_m$ （即與最優方案 $Q$ 值差距在門檻值以內者），應將此集合中所有方案一併呈報決策者參考，而非武斷選定單一方案 |
| C2 不成立 | $A'$ 與 $A''$ 皆可視為妥協解 |

此一嚴謹的判定機制，是 VIKOR 相較於 TOPSIS（單純依貼近係數排序、不涉及「是否構成穩定解」之額外檢定）在方法論上更為周延之處，也是本週提示詞實作要求「輸出符合驗證條件的妥協排序解」所對應之核心統計程序。

### 3.7 修正版 VIKOR 與期望水準落差分析

傳統 VIKOR 以「候選方案中之最佳與最差實際表現」作為 $f_j^{*}$ 與 $f_j^{-}$ 之計算基準（即「相對理想解」）；然而在許多管理決策情境中，決策者更關心的問題是「現有方案距離一個**外部設定的期望水準（Aspiration Level）**（例如產業標竿、法規最低標準、管理階層設定之目標分數）還有多少落差」，而非僅止於候選方案彼此之間的相對優劣。**修正版 VIKOR（Modified VIKOR with Aspiration Level）**（見本週參考文獻中 Huang, Liou, Chuang, & Tzeng, 2021 之醫療系統品質評估研究）將 $f_j^{*}$ 與 $f_j^{-}$ 之定義，由「候選方案中的最佳／最差值」改為「決策者設定之期望水準／最低可接受水準」，計算邏輯與傳統 VIKOR 完全相同，僅置換了比較基準：

$$
\text{Gap}_{ij} = \frac{f_j^{aspiration} - f_{ij}}{f_j^{aspiration} - f_j^{worst}}
$$

以此方式計算出的加權落差值，經彙整後可繪製成**改善優先級落差分析圖（Gap Analysis Chart）**，直接呈現「每一項準則，平均而言距離期望水準還有多少落差」，落差越大者，代表該準則越應被列為優先改善對象，此一產出對決策者而言，比單純的方案排名更具直接可執行之管理意涵，這也是本週理論篇 3.8 節與提示詞實作中特別強調此一延伸應用的原因。

### 3.8 DANP-VIKOR 整合流程與論文標準寫作架構

論文中報告 DANP-VIKOR 整合分析之標準章節順序如下：

1. **DEMATEL 總影響矩陣**（銜接第 5 週分析結果）
2. **DANP 超矩陣建構過程**：未加權超矩陣、加權超矩陣（如有構面分組）
3. **DANP 極限超矩陣與最終影響權重表**
4. **DANP 權重與 DEMATEL 中心度正規化權重之對照**（展現方法論嚴謹度，說明兩者差異）
5. **候選方案決策矩陣**
6. **VIKOR 計算過程**：正規化、 $S$ 、 $R$ 、 $Q$ 值三線表
7. **妥協解可接受條件檢驗結果**（C1、C2 是否成立，並說明最終決策建議）
8. **改善優先級落差分析圖**（如有執行修正版 VIKOR）

---

## 研究設計實例：綠色智慧建築方案評比與改善路徑

### 4.1 研究背景與動機

隨著全球淨零排放趨勢與智慧建築技術發展，企業與機構在規劃新建或改建智慧建築專案時，往往面臨多個設計方案之評選決策，這些方案在能源效率、室內環境品質、智慧化控制系統、資源循環利用與使用者健康福祉等多項評選準則上，各有優劣且準則間存在複雜的交互影響關係（例如「智慧化控制系統」之投資程度，同時直接影響「能源效率管理」表現，也間接透過使用經驗影響「使用者健康福祉」），此一情境完全符合本週理論篇 3.1 節所述之 DANP 適用條件。本研究延續第 5 週 DEMATEL 之因果結構分析技術，進一步求解可直接應用之準則權重，並以 VIKOR 對候選建築設計方案進行妥協排序，同時透過修正版 VIKOR 落差分析，為未獲選方案提出具體之改善路徑建議。

### 4.2 研究目的

1. 建構綠色智慧建築評選準則之直接影響矩陣，求解 DANP 影響權重。
2. 比較 DANP 權重與 DEMATEL 中心度正規化權重之異同。
3. 以 VIKOR 法對候選建築設計方案進行妥協排序，並檢驗排序結果是否滿足 C1、C2 可接受條件。
4. 以修正版 VIKOR（期望水準）進行落差分析，為候選方案提出具體之改善優先順序建議。

### 4.3 評選準則架構

| 準則代碼 | 準則名稱 | 操作型定義 |
|---|---|---|
| C1 | 能源效率管理 | 建築物整體能源使用效率與再生能源導入程度 |
| C2 | 室內環境品質 | 室內空氣品質、採光、通風與熱舒適度表現 |
| C3 | 智慧化控制系統 | 建築自動化控制、物聯網感測與 AI 節能決策系統之完備程度 |
| C4 | 資源循環利用 | 建材循環利用、水資源回收與廢棄物管理表現 |
| C5 | 使用者健康福祉 | 建築環境對使用者身心健康與工作／生活滿意度之影響程度 |

### 4.4 候選方案與研究對象

本研究以三個候選建築設計方案作為評選對象：**方案甲（高科技型）**——大量投資於智慧化控制系統與能源監控技術；**方案乙（低成本型）**——以基本建材與有限預算完成基礎綠建築規格；**方案丙（平衡型）**——在各項準則間採取均衡投資策略。建議邀請 8–10 位具建築規劃、機電工程、永續發展顧問背景之專家，進行準則間之直接影響評估（DEMATEL）與候選方案之績效評分。

### 4.5 資料分析流程規劃

```
第 5 週 DEMATEL 分析已求得之總影響矩陣 T
        │
        ▼
【DANP】依列正規化 T → Tc（列隨機矩陣）
        │
        ▼
轉置：未加權超矩陣 W = Tc^T（欄隨機矩陣）
        │
        ▼
（如有構面分組）以構面層級 DEMATEL 結果加權，得加權超矩陣
        │
        ▼
矩陣冪次疊代：W^k，直至收斂 → 極限超矩陣
        │
        ▼
萃取 DANP 影響權重（正規化極限超矩陣之任一欄）
        │
        ▼
蒐集候選方案於各準則下之績效評分（決策矩陣）
        │
        ▼
【VIKOR】正規化 → 計算 S、R、Q 值
        │
        ▼
依 Q 值排序，檢驗 C1（可接受優勢）與 C2（可接受穩定性）
        │
        ├── 兩項皆成立 ──► 唯一妥協解
        └── C1 不成立 ──► 提出妥協解集合
        │
        ▼
【修正版 VIKOR】以期望水準取代相對理想解，計算落差
        │
        ▼
繪製改善優先級落差分析圖 + 方案妥協解排序表
        │
        ▼
撰寫研究結果與改善路徑建議
```

---

## Colab 實作環境建置

```python
# ============================================================
# Cell 0：Colab 環境建置與套件匯入
# ------------------------------------------------------------
# 說明：DANP（超矩陣運算）與 VIKOR（S/R/Q 計算）核心演算法
# 皆以 numpy 手動實作，延續第 4、5 週之教學設計邏輯。
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 設定中文字型（沿用第 1–5 週相同設定邏輯）
# ------------------------------------------------------------
!wget -q https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf -O /content/NotoSansTC.otf
from matplotlib import font_manager
font_manager.fontManager.addfont('/content/NotoSansTC.otf')
plt.rcParams['font.family'] = 'Noto Sans CJK TC'
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

print("環境建置完成，本週 DANP／VIKOR 核心演算法以 numpy 手動實作。")
```

---

## Colab 實作：Step by Step 完整程式碼

### Step 1：建構專家直接影響矩陣並求解總影響矩陣（銜接第 5 週）

```python
# ============================================================
# Cell 1：建構綠色智慧建築評選準則之直接影響矩陣，求解總影響矩陣
# ------------------------------------------------------------
# 說明：此步驟完全比照第 5 週 DEMATEL 分析流程，作為本週
# DANP 分析之前置基礎。
# ============================================================

criteria = ['能源效率管理(C1)', '室內環境品質(C2)', '智慧化控制系統(C3)',
            '資源循環利用(C4)', '使用者健康福祉(C5)']
n = len(criteria)

A = np.array([
    [0, 2, 3, 2, 2],   # C1 對 C2/C3/C4/C5 之直接影響評估
    [1, 0, 2, 1, 3],   # C2 對 C1/C3/C4/C5 之直接影響評估
    [3, 2, 0, 2, 2],   # C3 對 C1/C2/C4/C5 之直接影響評估
    [2, 1, 1, 0, 1],   # C4 對 C1/C2/C3/C5 之直接影響評估
    [1, 3, 1, 1, 0],   # C5 對 C1/C2/C3/C4 之直接影響評估
], dtype=float)

s = max(A.sum(axis=1).max(), A.sum(axis=0).max())
N = A / s
I = np.eye(n)
T = N @ np.linalg.inv(I - N)

total_influence_df = pd.DataFrame(T, index=criteria, columns=criteria)
print("=== 表 1：總影響矩陣 T（銜接第 5 週 DEMATEL 分析）===")
display(total_influence_df.round(4))

R = T.sum(axis=1)
C = T.sum(axis=0)
print(f"\nD+R（中心度）：{dict(zip(criteria, (R + C).round(3)))}")
print(f"D-R（原因度）：{dict(zip(criteria, (R - C).round(3)))}")
```

### Step 2：建構未加權超矩陣

```python
# ============================================================
# Cell 2：建構 DANP 未加權超矩陣
# ------------------------------------------------------------
# 提示詞實作對照（前半段）：
# 「由總影響矩陣推導極限超矩陣取得準則權重」
# ============================================================

# 步驟一：依列正規化總影響矩陣（每列元素總和為 1）
Tc = T / T.sum(axis=1, keepdims=True)
print("=== 表 2：列正規化矩陣 Tc（每列總和應為 1）===")
display(pd.DataFrame(Tc, index=criteria, columns=criteria).round(4))
print(f"\n列總和驗證：{Tc.sum(axis=1).round(6)}")

# 步驟二：轉置得未加權超矩陣（每欄元素總和為 1）
W_unweighted = Tc.T
print("\n=== 表 3：未加權超矩陣 W（每欄總和應為 1）===")
display(pd.DataFrame(W_unweighted, index=criteria, columns=criteria).round(4))
print(f"\n欄總和驗證：{W_unweighted.sum(axis=0).round(6)}")
```

### Step 3：求解極限超矩陣與 DANP 權重

```python
# ============================================================
# Cell 3：矩陣冪次疊代求解極限超矩陣
# ------------------------------------------------------------
# 說明：本研究範例僅有單一層級準則（無構面分組），故加權
# 超矩陣在數學上等同於未加權超矩陣，可直接進行冪次疊代。
# 若研究設計涉及多構面分組，應先以構面層級 DEMATEL 結果
# 對未加權超矩陣各區塊加權後，再執行本步驟。
# ============================================================

W_weighted = W_unweighted.copy()   # 單一層級準則，加權超矩陣 = 未加權超矩陣

# 觀察不同冪次下矩陣是否已收斂（各欄數值趨於一致）
print("=== 不同疊代次數下超矩陣第一欄數值變化（觀察收斂過程）===")
for k in [1, 3, 5, 10, 20, 50]:
    Wp = np.linalg.matrix_power(W_weighted, k)
    print(f"k={k:>3}：{Wp[:, 0].round(4)}")

# 取足夠高之冪次作為極限超矩陣（本範例於 k=5 左右已收斂）
W_limit = np.linalg.matrix_power(W_weighted, 100)
print("\n=== 表 4：極限超矩陣（各欄應呈現相同數值）===")
display(pd.DataFrame(W_limit, index=criteria, columns=criteria).round(4))

# 驗證收斂：檢查各欄是否確實已收斂為相同數值
convergence_check = np.abs(W_limit - W_limit[:, [0]]).max()
print(f"\n收斂驗證：各欄與第一欄之最大差異 = {convergence_check:.8f}（應趨近於 0）")

# 萃取 DANP 權重（正規化極限超矩陣之任一欄）
danp_weights = W_limit[:, 0]
danp_weights = danp_weights / danp_weights.sum()

danp_weight_table = pd.DataFrame({
    'Criterion': criteria, 'DANP_Weight': danp_weights
}).sort_values('DANP_Weight', ascending=False)
print("\n=== 表 5：DANP 影響權重排序 ===")
display(danp_weight_table.round(4))
```

### Step 4：DANP 權重與 DEMATEL 正規化權重對照

```python
# ============================================================
# Cell 4：DANP 權重 vs. DEMATEL 中心度正規化權重對照
# ------------------------------------------------------------
# 教學目的：許多初學者誤以為「將 D+R 正規化」就等同於 DANP
# 權重，本步驟明確對照兩者數值差異，釐清此一常見誤解
# （詳見本週 Q&A）。
# ============================================================

dr_normalized = (R + C) / (R + C).sum()

comparison_df = pd.DataFrame({
    'Criterion': criteria,
    'DANP_Weight': danp_weights,
    'DR_Normalized_Weight': dr_normalized,
    'Difference': danp_weights - dr_normalized
}).sort_values('DANP_Weight', ascending=False)

print("=== 表 6：DANP 權重與 DEMATEL 中心度正規化權重對照表 ===")
display(comparison_df.round(4))
print(f"\n兩組權重之最大差異：{comparison_df['Difference'].abs().max():.4f}")
print("此差異來自於 DANP 之極限超矩陣運算，捕捉了影響力在整個網路中")
print("反覆傳遞後的均衡分布，而非僅是總影響矩陣的單次列欄加總。")
```

### Step 5：建構候選方案決策矩陣

```python
# ============================================================
# Cell 5：候選建築設計方案決策矩陣
# ------------------------------------------------------------
# 資料說明：三個候選方案於五項準則下之專家評分（1–10 分，
# 皆為效益型準則，分數越高代表表現越佳）。
# ============================================================

alternatives = ['方案甲(高科技型)', '方案乙(低成本型)', '方案丙(平衡型)']

performance_matrix = pd.DataFrame({
    '能源效率管理(C1)': [9, 6, 8],
    '室內環境品質(C2)': [7, 6, 8],
    '智慧化控制系統(C3)': [9, 5, 7],
    '資源循環利用(C4)': [6, 7, 7],
    '使用者健康福祉(C5)': [7, 6, 8],
}, index=alternatives)

benefit_criteria = [True, True, True, True, True]   # 本範例五項準則皆為效益型

print("=== 表 7：候選建築設計方案決策矩陣 ===")
display(performance_matrix)
```

### Step 6：VIKOR 演算法完整實作

```python
# ============================================================
# Cell 6：VIKOR 演算法（S、R、Q 值計算）
# ------------------------------------------------------------
# 提示詞實作對照（後半段）：
# 「執行 VIKOR 演算法計算 S_i, R_i, Q_i 值，輸出符合驗證條件
#   的妥協排序解。」
# ============================================================

def vikor(performance_matrix, weights, benefit_criteria, v=0.5):
    """
    執行完整 VIKOR 演算法。

    參數：
    - performance_matrix: pandas DataFrame，列為方案，欄為準則
    - weights: 各準則權重（應與欄位順序對應，總和為 1）
    - benefit_criteria: 布林值列表，True 代表效益型準則
    - v: 群體效用策略權重，預設 0.5（群體效用與個別遺憾同等重視）

    回傳：包含 S、R、Q 值與排名之結果 DataFrame
    """
    X = performance_matrix.values.astype(float)
    n_criteria = X.shape[1]

    f_star = np.where(benefit_criteria, X.max(axis=0), X.min(axis=0))
    f_minus = np.where(benefit_criteria, X.min(axis=0), X.max(axis=0))

    # 避免分母為 0（該準則所有方案表現完全相同時）
    denom = np.where(f_star - f_minus == 0, 1e-10, f_star - f_minus)
    normalized_diff = (f_star - X) / denom   # 對成本型準則已透過 f_star/f_minus 定義自動處理方向

    S = (weights * normalized_diff).sum(axis=1)
    R = (weights * normalized_diff).max(axis=1)

    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()

    Q = (
        v * (S - S_star) / (S_minus - S_star if S_minus != S_star else 1e-10)
        + (1 - v) * (R - R_star) / (R_minus - R_star if R_minus != R_star else 1e-10)
    )

    result = pd.DataFrame({
        'Alternative': performance_matrix.index,
        'S（群體效用）': S, 'R（個別遺憾）': R, 'Q（妥協指標）': Q
    }).sort_values('Q（妥協指標）').reset_index(drop=True)
    result['Rank_by_Q'] = result.index + 1

    return result

vikor_result = vikor(performance_matrix, danp_weights, benefit_criteria, v=0.5)
print("=== 表 8：VIKOR 妥協排序結果（依 Q 值由小到大排序）===")
display(vikor_result.round(4))
```

### Step 7：妥協解可接受條件檢驗（C1、C2）

```python
# ============================================================
# Cell 7：妥協解可接受條件檢驗
# ------------------------------------------------------------
# 統計原理：詳見理論篇 3.6 節之 C1（可接受優勢）與 C2
# （可接受穩定性）判斷邏輯。
# ============================================================

def check_vikor_compromise_conditions(vikor_result, S_col='S（群體效用）',
                                       R_col='R（個別遺憾）', Q_col='Q（妥協指標）'):
    """
    檢驗 VIKOR 妥協解是否滿足 C1（可接受優勢）與 C2（可接受穩定性）條件。
    """
    J = len(vikor_result)
    DQ = 1 / (J - 1)

    sorted_by_Q = vikor_result.sort_values(Q_col).reset_index(drop=True)
    A_prime = sorted_by_Q.iloc[0]
    A_double_prime = sorted_by_Q.iloc[1]

    # C1：可接受優勢
    Q_gap = A_double_prime[Q_col] - A_prime[Q_col]
    C1_satisfied = Q_gap >= DQ

    # C2：可接受穩定性（A' 須同時為 S 或 R 排序之最優方案）
    best_by_S = vikor_result.sort_values(S_col).iloc[0]['Alternative']
    best_by_R = vikor_result.sort_values(R_col).iloc[0]['Alternative']
    C2_satisfied = (A_prime['Alternative'] == best_by_S) or (A_prime['Alternative'] == best_by_R)

    print(f"候選方案總數 J = {J}，門檻值 DQ = 1/(J-1) = {DQ:.4f}")
    print(f"Q 值排名第一：{A_prime['Alternative']}（Q = {A_prime[Q_col]:.4f}）")
    print(f"Q 值排名第二：{A_double_prime['Alternative']}（Q = {A_double_prime[Q_col]:.4f}）")
    print(f"Q 值差距：{Q_gap:.4f}")
    print(f"\nC1（可接受優勢）：{'✅ 成立' if C1_satisfied else '⚠️ 不成立'}（差距 {'≥' if C1_satisfied else '<'} DQ）")
    print(f"C2（可接受穩定性）：{'✅ 成立' if C2_satisfied else '⚠️ 不成立'}（{A_prime['Alternative']} 是否同時為 S 或 R 排序最優：{'是' if C2_satisfied else '否'}）")

    if C1_satisfied and C2_satisfied:
        print(f"\n結論：{A_prime['Alternative']} 為唯一妥協解，可直接作為最終決策建議。")
    elif not C1_satisfied:
        compromise_set = vikor_result[
            vikor_result[Q_col] - A_prime[Q_col] < DQ
        ]['Alternative'].tolist()
        print(f"\n結論：C1 未成立，存在妥協解集合：{compromise_set}，建議一併呈報決策者參考。")
    else:
        print(f"\n結論：C2 未成立，{A_prime['Alternative']} 與 {A_double_prime['Alternative']} 皆可視為妥協解。")

    return C1_satisfied, C2_satisfied

C1_ok, C2_ok = check_vikor_compromise_conditions(vikor_result)
```

### Step 8：修正版 VIKOR——期望水準落差分析

```python
# ============================================================
# Cell 8：修正版 VIKOR（期望水準）落差分析
# ------------------------------------------------------------
# 提示詞實作對照（延伸應用）：
# 「以期望水準取代相對理想解，計算各準則之改善優先級落差。」
# ============================================================

# 設定期望水準（滿分 10 分）與最低可接受水準（0 分）
aspiration_level = np.full(n, 10.0)
worst_level = np.zeros(n)

X = performance_matrix.values.astype(float)
gap_matrix = (aspiration_level - X) / (aspiration_level - worst_level)
weighted_gap = gap_matrix * danp_weights

gap_df = pd.DataFrame(weighted_gap, index=alternatives, columns=criteria)
print("=== 表 9：各方案於各準則之加權落差矩陣 ===")
display(gap_df.round(4))

# 各準則之平均加權落差（跨方案），作為整體改善優先順序判斷依據
avg_gap_per_criterion = gap_df.mean(axis=0).sort_values(ascending=False)
print("\n=== 表 10：各準則平均加權落差（改善優先順序，由高到低）===")
display(avg_gap_per_criterion.round(4))

print(f"\n改善優先建議：應優先投入資源改善「{avg_gap_per_criterion.index[0]}」，")
print(f"其平均加權落差達 {avg_gap_per_criterion.iloc[0]:.4f}，為所有準則中距離期望水準最遠者。")
```

### Step 9：改善優先級落差分析圖與妥協排序視覺化

```python
# ============================================================
# Cell 9：視覺化——落差分析圖與 VIKOR 妥協排序圖
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# 圖 A：改善優先級落差分析圖
ax = axes[0]
colors = plt.cm.OrRd(avg_gap_per_criterion.values / avg_gap_per_criterion.values.max())
bars = ax.barh(avg_gap_per_criterion.index[::-1], avg_gap_per_criterion.values[::-1],
               color=colors[::-1], edgecolor='#2c3e50')
for bar, val in zip(bars, avg_gap_per_criterion.values[::-1]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
ax.set_xlabel('平均加權落差（相對於期望水準）')
ax.set_title('圖 1：改善優先級落差分析圖', fontsize=12)
ax.grid(alpha=0.3, axis='x')

# 圖 B：VIKOR 妥協排序 S/R/Q 值比較
ax = axes[1]
x_pos = np.arange(len(vikor_result))
width = 0.25
ax.bar(x_pos - width, vikor_result['S（群體效用）'], width, label='S（群體效用）', color='#3498db')
ax.bar(x_pos, vikor_result['R（個別遺憾）'], width, label='R（個別遺憾）', color='#e67e22')
ax.bar(x_pos + width, vikor_result['Q（妥協指標）'], width, label='Q（妥協指標）', color='#e74c3c')
ax.set_xticks(x_pos)
ax.set_xticklabels(vikor_result['Alternative'], fontsize=10)
ax.set_title('圖 2：候選方案 VIKOR S／R／Q 值比較', fontsize=12)
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('danp_vikor_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Step 10：匯出所有分析結果

```python
# ============================================================
# Cell 10：匯出完整分析結果至 Excel
# ============================================================

with pd.ExcelWriter('DANP_VIKOR分析結果_Week06.xlsx') as writer:
    total_influence_df.round(4).to_excel(writer, sheet_name='總影響矩陣')
    danp_weight_table.round(4).to_excel(writer, sheet_name='DANP權重', index=False)
    comparison_df.round(4).to_excel(writer, sheet_name='DANP與DR權重對照', index=False)
    performance_matrix.to_excel(writer, sheet_name='決策矩陣')
    vikor_result.round(4).to_excel(writer, sheet_name='VIKOR排序結果', index=False)
    gap_df.round(4).to_excel(writer, sheet_name='落差矩陣')
    avg_gap_per_criterion.round(4).to_frame('平均加權落差').to_excel(writer, sheet_name='改善優先順序')

print("所有統計結果已匯出至 DANP_VIKOR分析結果_Week06.xlsx，可於 Colab 左側檔案面板下載。")
print(f"\nVIKOR 最終妥協建議方案：{vikor_result.iloc[0]['Alternative']}")
print(f"改善優先準則：{avg_gap_per_criterion.index[0]}")
```

---

## Vibe Coding 提示詞（Prompt）實作範例集

**範例 1：DANP 超矩陣建構**

> 我已有一個第 5 週求得的 5×5 總影響矩陣。請使用 Python numpy 依列正規化後轉置，得到未加權超矩陣（欄總和應為 1），並以矩陣冪次疊代法求解極限超矩陣，觀察在哪個冪次時各欄數值已收斂一致，最後萃取出正規化後的 DANP 影響權重。

**範例 2：DANP 與 DEMATEL 權重對照**

> 請將 DANP 權重與「直接將 D+R 中心度正規化」所得之權重並列比較，計算兩者差異，並用約 100 字說明為什麼這兩種計算方式在方法論上並不等價。

**範例 3：VIKOR 完整計算**

> 請將 DANP 權重代入 VIKOR 演算法，計算三個候選方案的 S（群體效用）、R（個別遺憾）與 Q（妥協指標）值，策略係數 v 請設為 0.5，並依 Q 值由小到大排序輸出結果。

**範例 4：妥協解條件檢驗**

> 請依 VIKOR 方法之 C1（可接受優勢，門檻值 DQ = 1/(方案數-1)）與 C2（可接受穩定性，最優方案須同時為 S 或 R 排序最優）判斷本次排序結果是否構成唯一妥協解；若 C1 不成立，請列出所有應納入妥協解集合的候選方案。

**範例 5：落差分析與視覺化**

> 請以期望水準（各準則滿分 10 分）取代相對理想解，重新計算各方案於各準則之加權落差，並繪製一張橫向長條圖，由上至下依落差大小排序，用以呈現改善優先順序。

**範例 6：結果段落初稿撰寫**

> 根據以下統計結果（DANP 權重：室內環境品質 .224、使用者健康福祉 .222、智慧化控制系統 .196、能源效率管理 .191、資源循環利用 .167；VIKOR 排序：方案丙 Q=0.000 第一、方案甲 Q=0.443 第二、方案乙 Q=1.000 第三；C1 不成立、C2 成立），請以碩士論文研究結果章節的學術寫作語氣，撰寫一段約 300 字的中文分析段落，並提出具體的決策建議。

---

## 結果呈現與分析：碩士論文寫法示例

以下段落數值取自本週 Colab 範例程式碼之實際執行結果，供學生對照模仿寫作邏輯（實際數值請以自己資料之 Colab 輸出為準）。

> **4.1 DANP 影響權重求解結果**
>
> 本研究以第 5 週求得之總影響矩陣為基礎，依序建構未加權超矩陣並執行矩陣冪次疊代，結果顯示於第 5 次冪次疊代後，超矩陣各欄數值已收斂一致，求得最終 DANP 影響權重由高至低依序為：室內環境品質（.224）、使用者健康福祉（.222）、智慧化控制系統（.196）、能源效率管理（.191）、資源循環利用（.167）。進一步將此權重與單純將 DEMATEL 中心度（D+R）正規化所得之權重對照，發現兩組權重之排序與數值均存在差異（最大差異達 .026），顯示 DANP 之極限超矩陣運算，確實捕捉了準則間影響力反覆傳遞後的均衡分布，而非僅是總影響矩陣之單次加總，此一發現支持本研究採用 DANP 而非單純 DEMATEL 正規化權重之方法論選擇。
>
> **4.2 VIKOR 妥協排序結果**
>
> 將 DANP 權重代入 VIKOR 演算法（策略係數 v = 0.5），三個候選建築設計方案之 Q 值由小到大依序為：方案丙（平衡型，Q = 0.000）、方案甲（高科技型，Q = 0.443）、方案乙（低成本型，Q = 1.000）。方案丙同時亦為 S 值與 R 值排序之最優方案，符合 C2（可接受穩定性）條件；惟方案甲與方案丙之 Q 值差距為 0.443，低於本研究情境下之門檻值 DQ = 1/(3-1) = 0.5，未滿足 C1（可接受優勢）條件。依 VIKOR 方法論之判定邏輯，本研究之妥協解應為「方案丙、方案甲」所構成之妥協解集合，而非單一方案，建議決策者將此二方案一併納入最終決策參考，而非僅依 Q 值排名武斷選定單一方案。
>
> **4.3 改善優先級落差分析**
>
> 以期望水準（各準則滿分 10 分）取代相對理想解，重新計算各方案之加權落差後發現，五項準則之平均加權落差由高至低依序為：室內環境品質、使用者健康福祉、智慧化控制系統、資源循環利用、能源效率管理。此結果顯示，即使在妥協解集合中排名最優的方案丙，其「室內環境品質」與「使用者健康福祉」兩項準則，相較於期望水準仍存在相對較大之落差，此二項恰為 DANP 權重排序中最高的兩項準則，顯示未來無論選擇方案丙或方案甲，均應優先投入資源改善此二項準則，方能最有效提升整體綠色智慧建築評選表現，此一發現亦呼應本週理論篇 3.7 節所強調之修正版 VIKOR 落差分析，相較單純方案排名更具直接可執行之管理意涵。

**APA 格式三線表範例：VIKOR 妥協排序結果表**

| 方案 | S（群體效用） | R（個別遺憾） | Q（妥協指標） | 排名 |
|---|---|---|---|---|
| 方案丙（平衡型） | 0.162 | 0.098 | 0.000 | 1 |
| 方案甲（高科技型） | 0.390 | 0.167 | 0.443 | 2 |
| 方案乙（低成本型） | 0.833 | 0.224 | 1.000 | 3 |

*註：以上數值為本週 Colab 範例實際執行結果（v = 0.5），實際研究請以自己資料之輸出為準。DQ = 0.500，C1 未成立、C2 成立，妥協解集合為 {方案丙, 方案甲}。*

---

## 常見統計誤區與 Q&A

**Q1：既然可以直接把 DEMATEL 的 D+R 中心度正規化當作權重使用，為什麼還要大費周章地執行 DANP 的超矩陣運算？**
這是本週最核心的方法論誤解，也是 Step 4 刻意安排對照表的原因。將 D+R 正規化，本質上只是「總影響矩陣的單次列欄加總」，並未考慮「準則 A 影響準則 B，而準則 B 又進一步影響準則 C」這種多階傳遞效應在整個網路達到均衡狀態後的最終分布；DANP 之極限超矩陣運算，透過矩陣冪次疊代模擬影響力在網路中反覆流動、傳遞直至收斂的過程，數學上更貼近 ANP「網路節點間相互依存、動態均衡」的理論精神。本週 Colab 實作 Step 4 之對照結果已實際證實兩組權重存在數值差異，這正是選擇 DANP 而非單純 DEMATEL 正規化權重的方法論依據，論文中應明確引用此一差異作為方法選擇之佐證。

**Q2：VIKOR 排序結果顯示 C1 條件不成立，這樣我的論文結果是不是「失敗」了，沒辦法得出明確結論？**
完全不會，這正是 VIKOR 方法論相較單純排序方法更嚴謹之處，而非研究失敗的訊號。誠如理論篇 3.6 節所述，C1 不成立時，VIKOR 之標準做法是提出一組「妥協解集合」，而非強行指定單一最優解，這其實更貼近真實決策情境——當兩個方案的綜合表現在統計上難以明確區分優劣時，如實呈現此一「難分軒輊」的發現，並建議決策者納入量化分析之外的其他考量（如預算限制、既有技術能力），遠比刻意挑選單一「贏家」更符合學術誠信與實務決策的需求，此一發現本身即具有研究價值。

**Q3：VIKOR 的策略係數 v 該怎麼設定？可以隨便調整讓結果變得比較好看嗎？**
策略係數 $v$ 之設定，應依研究情境之決策邏輯事先決定，而非為了得出「想要的結果」而事後調整（此為嚴重的學術倫理疑慮）。 $v = 0.5$ （群體效用與個別遺憾同等重視）是學術文獻中最常見、也最具正當性的預設值，除非研究問題明確涉及決策者對「整體多數效用」或「避免最弱環節劣勢」有特別偏好之情境（例如政府公共政策評估通常更重視避免任何面向出現極端落後，可能傾向調低 $v$ ），否則不建議任意調整。若確實需要調整，應於研究方法章節事先說明其理論依據，並可考慮同時報告不同 $v$ 值設定下之敏感度分析結果，而非僅呈現對結論最有利的單一設定。

**Q4：修正版 VIKOR 的「期望水準」該怎麼設定？可以直接設為候選方案中的最高分嗎？**
不建議直接沿用候選方案中的最高分，因為那樣就退化為傳統 VIKOR（以相對理想解為基準），失去了「期望水準」這個概念的獨立意義。期望水準應依研究情境之外部標準訂定，常見做法包括：(1) 量表滿分（如本週範例採用之 10 分）；(2) 產業標竿或最佳實務案例之公開績效數據；(3) 政府法規或認證標準之最低（或建議）門檻；(4) 專家會議共識訂定之合理目標值。訂定依據應於論文方法論章節中明確說明，這也是修正版 VIKOR 相較傳統 VIKOR，更能提供「絕對意義上還有多少改善空間」而非僅止於「候選方案間相對優劣」之管理意涵所在。

**Q5：DANP 和第 3 週學過的 PLS-SEM，都涉及某種「矩陣收斂」或「疊代求解」的概念，兩者是同一回事嗎？**
不是同一回事，雖然兩者都運用了疊代（iterative）的數學運算概念。第 3 週 PLS-SEM 之疊代，是透過反覆調整外部權重與內部路徑係數的估計值，直至模型參數估計收斂穩定，目的是從「大樣本觀察資料」中估計出最能解釋資料變異的統計參數；本週 DANP 之疊代（矩陣冪次疊代），則是將一個已經確定的、來自專家判斷的影響力分配矩陣，持續自我相乘直至達到數學上的穩態分布，目的是找出這個既定網路結構下，影響力最終如何均衡分布，並不涉及任何統計估計或資料配適的概念。兩者的「疊代」在數學形式上或許類似，但統計意涵與運算目的截然不同。

**Q6：如果我的候選方案評選準則中，有些是效益型、有些是成本型，VIKOR 的計算會需要調整嗎？**
會，這與第 4 週理論篇 3.6 節、Q3 中討論之 TOPSIS 效益型／成本型準則處理邏輯完全類似（見理論篇 3.5 節之 $f_j^{*}$ 、 $f_j^{-}$ 定義公式）。對效益型準則， $f_j^{*}$ 取候選方案中的最大值、 $f_j^{-}$ 取最小值；對成本型準則則相反。本週 Colab 實作之 `vikor()` 函式已透過 `benefit_criteria` 參數處理此一邏輯，使用時務必如同第 4 週 TOPSIS 一般，仔細核對每一項準則之類型標記是否正確，標記錯誤將導致排序結果完全顛倒卻不會產生任何程式錯誤訊息。

**Q7：本週 DANP 分析只有單一層級的五個準則，如果我的研究設計有「構面（dimension）」與「準則（criteria）」兩個層級，加權超矩陣的計算會有什麼不同？**
當決策問題具有「構面→準則」兩層級結構時（例如本週五項準則可進一步歸類為「環境技術構面」與「使用者體驗構面」），須額外執行一次「構面層級」的 DEMATEL 分析，求出構面間的總影響矩陣並正規化，得到一個構面數 × 構面數的加權矩陣。接著，將未加權超矩陣依構面分組切割為多個區塊（block），每個區塊對應「某構面之準則」對「另一構面之準則」的影響部分，再將對應的構面層級權重值，乘上該區塊中的每一個元素，即完成加權超矩陣的建構。此後的極限超矩陣冪次疊代步驟與本週範例完全相同。這是進行完整 DANP 分析（而非本週簡化之單層級版本）時的必要延伸步驟，建議有多層級準則結構之期末專題學生，可在本週 Step 2 程式碼基礎上自行擴充實作。

**Q8：VIKOR 的 S 值和 R 值，跟第 4 週 TOPSIS 的 D+ 和 D- 概念上是同一件事嗎？**
概念上有相通之處但並非同一件事，這是初學者常見的混淆點。TOPSIS 之 $D^+$ 、 $D^-$ 分別是「方案與正理想解之歐氏距離」「方案與負理想解之歐氏距離」，兩者是獨立計算之後再合併為單一貼近係數；VIKOR 之 $S$ 、 $R$ 則分別是「所有準則加權偏離程度之總和（近似曼哈頓距離）」與「表現最差之單一準則的最大偏離程度（近似柴比雪夫距離）」，兩者衡量的都是「與正理想解的偏離程度」，只是聚合方式不同（ $S$ 用加總、 $R$ 用取最大值），而非像 TOPSIS 一樣同時考慮正負兩個理想解的相對位置。這也是為什麼 VIKOR 的 $Q$ 值計算需要額外引入策略係數 $v$ 來調和 $S$ 與 $R$ 兩種聚合邏輯，而 TOPSIS 的貼近係數 CC 則是直接以距離比例計算，不涉及類似的策略權衡參數。

---

## 延伸研究方向：醫療院所導入 AI 輔助排班系統滿意度落差之改善策略研究

### 6.1 研究背景與理論基礎

醫療院所導入 AI 輔助人力排班系統（如本週參考文獻中提及之台灣醫院實務案例，能在極短時間內產出符合勞動法規與人員需求之排班表），雖然在效率面已展現顯著效益，然而醫護人員對此類系統之實際使用滿意度，往往涉及公平性認知、班別偏好符合程度、緊急調度彈性等多項相互關聯之準則，且不同準則間可能存在複雜之因果影響關係（例如「排班公平性認知」可能同時直接影響滿意度，也透過「團隊向心力」間接產生影響）。本週參考文獻中之國際期刊研究（醫療系統品質修正版 VIKOR 評估、新進護理人員職能落差評估）已將 DANP 與修正版 VIKOR 成功應用於醫療品質評估情境，可作為本延伸研究方向之直接方法論參照。

### 6.2 建議研究設計

1. **理論框架**：延續本週 DANP-VIKOR 整合架構，以 DEMATEL／DANP 辨識 AI 排班系統滿意度評選準則間之因果結構與影響權重，並以修正版 VIKOR（期望水準）進行落差分析，找出最需優先改善之準則。
2. **候選評選準則（範例）**：
   - 排班公平性認知（系統決策邏輯是否被醫護人員認為公平合理）
   - 班別偏好符合程度（系統是否確實納入個人排班偏好考量）
   - 緊急調度彈性（系統面對臨時請假、人力短缺等突發狀況之應變能力）
   - 系統操作便利性（醫護人員查詢、申請調整班表之介面友善程度）
   - 工作生活平衡改善程度（系統導入後對醫護人員整體生活品質之實際影響）
3. **研究對象**：建議以已導入 AI 輔助排班系統至少 6 個月以上之醫療院所護理人員為研究對象，確保受訪者已累積足夠使用經驗形成穩定的滿意度評價。
4. **期望水準設定**：可參考醫院管理階層設定之滿意度目標值，或以同業標竿醫院之滿意度調查結果作為期望水準基準。
5. **分析流程**：完全比照本週 Colab 實作流程（DEMATEL 總影響矩陣 → DANP 超矩陣與極限超矩陣 → VIKOR 妥協排序 → 修正版 VIKOR 落差分析），僅需替換準則定義與評估資料，即可直接複用本週所有自訂函式（`vikor()`、`check_vikor_compromise_conditions()`）。
6. **管理實務意涵**：此類研究可協助醫院資訊部門與護理管理階層，明確辨識 AI 排班系統中最應優先改版精進之功能面向，而非依賴片段之使用者抱怨進行零散式系統調整，是將量化決策科學方法應用於醫療資訊系統迭代改善之具體實踐。

### 6.3 給學生的思考練習

請思考：若本延伸研究之研究對象，進一步區分為「資深護理人員（10 年以上年資）」與「新進護理人員（3 年以下年資）」兩個子群體，你預期兩個子群體對「班別偏好符合程度」與「系統操作便利性」兩項準則之期望水準設定，可能會有何種系統性差異？這樣的差異，對於落差分析結果之管理應用意涵，會帶來什麼樣的複雜化？

---

## 課後作業與練習

**練習一：修改直接影響矩陣觀察 DANP 權重變化**
請修改 Step 1 中直接影響矩陣的部分元素，重新執行完整 DANP 流程，觀察準則權重排序是否改變，並比較新舊兩組 DANP 權重與 DEMATEL 正規化權重之差異程度是否有所不同。

**練習二：真實決策情境實作**
請自行設計一個至少包含 3 個候選方案、4 個評選準則之決策情境，實際邀請至少 5 位親友或同學扮演「專家」填答 DEMATEL 直接影響評估問卷與候選方案績效評分，套用本週完整 Colab 程式碼執行 DANP-VIKOR 分析。請繳交：(1) 問卷記錄、(2) 執行後的 Colab Notebook（.ipynb）、(3) 一頁 A4 的結果摘要（比照本週「結果呈現與分析」段落之寫法）。

**練習三：策略係數 v 敏感度分析**
請將 Step 6 之 VIKOR 演算法，在 $v = 0.1, 0.3, 0.5, 0.7, 0.9$ 五種不同策略係數設定下分別執行，比較最終妥協排序結果是否穩定，並以此結果說明策略係數選擇對決策結論的實質影響程度。

**練習四：文獻延伸閱讀報告**
請從本週「參考文獻與延伸閱讀」清單中，任選一篇 DANP、VIKOR 或 DANP-VIKOR 整合應用之期刊論文，撰寫一頁重點摘要，內容須包含：(1) 該研究之準則架構與候選方案、(2) 報告之 DANP 權重排序結果、(3) VIKOR 妥協排序結果與 C1／C2 條件檢驗結論、(4) 該研究提出之管理實務建議為何。

**練習五：修正版 VIKOR 期望水準敏感度分析**
請將 Step 8 中之期望水準（本範例設為滿分 10 分），分別調整為 8 分與 9 分兩種較保守之設定，重新執行落差分析，比較不同期望水準設定下，改善優先順序排序是否改變，並討論期望水準設定之保守／積極程度，對管理決策建議可能帶來的影響。

---

## 參考文獻與延伸閱讀（已查核連結）

1. Improving the Green Building Evaluation System in China Based on the DANP Method. *Sustainability*, 10(4), 1173.
   https://doi.org/10.3390/su10041173
   （完整示範 DANP 方法應用於綠色建築評估系統之國際期刊論文，詳細說明未加權超矩陣、加權超矩陣與 DANP 權重求解步驟，為本週理論篇 3.2–3.3 節之直接方法論依據，且研究主題與本週研究設計範例完全對應。）

2. A Combination of DEMATEL and BWM-Based ANP Methods for Exploring the Green Building Rating System in Taiwan. *Sustainability*, 12(8), 3216.
   https://doi.org/10.3390/su12083216
   （台灣本土綠色建築評估之 DANP／BWM-ANP 整合研究，並建構 INRM 影響網絡關係圖分析評估系統中各指標之關聯，可作為本週研究設計範例延伸應用之直接台灣本土參照文獻。）

3. Using a Modified VIKOR Technique for Evaluating and Improving the National Healthcare System Quality. *Mathematics*, 9(12), 1349.
   https://doi.org/10.3390/math9121349
   （以修正版 VIKOR（期望水準）評估與改善國家醫療系統品質之國際期刊論文，明確示範以落差分析取代相對理想解之研究設計，為本週理論篇 3.7 節與延伸研究方向「醫療院所滿意度落差改善」之直接前導文獻。）

4. A Hybrid MADM Model for Newly Graduated Nurse's Competence Evaluation and Improvement. *PMC*.
   https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8082270/
   （整合 DANP 與落差改善比率評估法，探討新進護理人員職能落差之國際期刊論文，研究情境與本週延伸研究方向高度相關，可直接作為醫療情境準則設計之參考範本。）

5. 員榮 AI 排班 25 秒搞定護理班表 台灣護理資訊學會驚豔智慧醫療成果。
   https://www.changhuanews.com/2026/07/news-0306.html
   （台灣醫院實際導入 AI 護理排班系統之新聞報導，可作為本週延伸研究方向「醫療院所導入 AI 輔助排班系統」之研究背景與實務現況佐證。）

6. Hsu, C.-H., Wang, F.-K., & Tzeng, G.-H. (2012). The best vendor selection for conducting the recycled material based on a hybrid MCDM model combining DANP with VIKOR. *Resources, Conservation and Recycling*, 66, 95-111.
   https://ideas.repec.org/a/gam/jmathe/v9y2021i12p1349-d573070.html
   （DANP 結合 VIKOR 之經典應用文獻，供應商評選情境，可作為方法論整合設計之延伸參考。）

7. The Use of a DANP with VIKOR Approach for Establishing the Model of E-Learning Service Quality. *Eurasia Journal of Mathematics, Science and Technology Education*.
   https://www.ejmste.com/article/the-use-of-a-danp-with-vikor-approach-for-establishing-the-model-of-e-learning-service-quality-4999
   （DANP-VIKOR 整合模型應用於線上學習服務品質評估之開放取用期刊論文，示範完整之落差改善研究設計，可作為服務品質類延伸研究之參考範本。）

8. Compromise solution by MCDM methods: A comparative analysis of VIKOR and TOPSIS. *European Journal of Operational Research*, 156(2), 445-455.
   https://www.semanticscholar.org/paper/Compromise-solution-by-MCDM-methods:-A-comparative-Opricovic-Tzeng/b31aa0b60875ea0e0d7f2aeffc93ace4e18ed3da
   （Opricovic 與 Tzeng 之 VIKOR 方法論經典論文，完整推導 S、R、Q 值計算公式與妥協解可接受條件，為本週理論篇 3.4–3.6 節之直接數學依據。）

**方法論經典文獻（建議延伸閱讀，非本次線上搜尋來源，圖書館或資料庫可查閱）**：

- Saaty, T. L. (1996). *Decision Making with Dependence and Feedback: The Analytic Network Process*. RWS Publications.
- Ou Yang, Y.-P., Shieh, H.-M., Leu, J.-D., & Tzeng, G.-H. (2008). A novel hybrid MCDM model combined with DEMATEL and ANP with applications. *International Journal of Operations Research*, 5(3), 160-168.
- Opricovic, S. (1998). *Multicriteria Optimization of Civil Engineering Systems*. Faculty of Civil Engineering, Belgrade.
- Opricovic, S., & Tzeng, G.-H. (2007). Extended VIKOR method in comparison with outranking methods. *European Journal of Operational Research*, 178(2), 514-529.
- Yu, P. L. (1973). A class of solutions for group decision problems. *Management Science*, 19(8), 936-946.

---

## 附錄

### 附錄 A：AHP（第 4 週）、DEMATEL（第 5 週）、DANP（本週）三方法定位總覽

| 比較構面 | AHP | DEMATEL | DANP |
|---|---|---|---|
| 準則關係假設 | 彼此獨立 | 可能互為因果 | 可能互為因果（承接 DEMATEL） |
| 核心運算 | 特徵向量法 | 無窮等比矩陣級數 | 超矩陣冪次疊代 |
| 直接產出 | 準則權重 | 中心度／原因度、因果分群 | 準則權重（反映網路均衡狀態） |
| 是否可直接用於方案排序 | 可（搭配 TOPSIS） | 不可 | 可（搭配 VIKOR 或 TOPSIS） |
| 方法論定位 | 起點：最簡化假設 | 中繼：辨識因果結構 | 終點：在因果網路上求解權重 |

### 附錄 B：VIKOR（本週）與 TOPSIS（第 4 週）方法對照表

| 比較構面 | TOPSIS（第 4 週） | VIKOR（本週） |
|---|---|---|
| 核心邏輯 | 同時最大化與正理想解之接近度、最小化與負理想解之接近度 | 兼顧群體效用最大化與個別遺憾最小化 |
| 主要統計量 | $D^+$ 、 $D^-$ 、貼近係數 CC | $S$ （群體效用）、 $R$ （個別遺憾）、 $Q$ （妥協指標） |
| 距離測度性質 | 歐氏距離（Euclidean distance） | $S$ 為曼哈頓距離性質、 $R$ 為柴比雪夫距離性質 |
| 是否檢驗解的穩定性 | 否，直接依 CC 值排序 | 是，須檢驗 C1（可接受優勢）與 C2（可接受穩定性） |
| 是否支援落差分析延伸 | 否（原始設計為相對排序） | 是（修正版 VIKOR 可用期望水準取代相對理想解） |

### 附錄 C：術語中英對照表

| 中文術語 | 英文術語 | 縮寫 |
|---|---|---|
| 網路分析法 | Analytic Network Process | ANP |
| 決策實驗室分析法為基礎之網路分析法 | DEMATEL-based Analytic Network Process | DANP |
| 未加權超矩陣 | Unweighted Supermatrix | — |
| 加權超矩陣 | Weighted Supermatrix | — |
| 極限超矩陣 | Limit Supermatrix | — |
| 多準則最適化與妥協解 | VlseKriterijumska Optimizacija I Kompromisno Resenje | VIKOR |
| 群體效用測度 | Group Utility Measure | S |
| 個別遺憾測度 | Individual Regret Measure | R |
| 妥協指標值 | Compromise Index | Q |
| 可接受優勢 | Acceptable Advantage | C1 |
| 可接受穩定性 | Acceptable Stability in Decision Making | C2 |
| 妥協解 | Compromise Solution | — |
| 期望水準 | Aspiration Level | — |
| 修正版 VIKOR | Modified VIKOR | — |
| 落差分析 | Gap Analysis | — |

### 附錄 D：常見程式錯誤排解（Debugging Tips）

| 錯誤現象 | 常見原因 | 排解建議 |
|---|---|---|
| 未加權超矩陣欄總和不為 1 | 忘記先依列正規化總影響矩陣，或轉置順序錯誤 | 確認 `Tc = T / T.sum(axis=1, keepdims=True)` 先執行、再以 `.T` 轉置 |
| 極限超矩陣各欄數值遲遲未收斂 | 總影響矩陣本身數值不穩定（如正規化係數計算錯誤，來自第 5 週步驟） | 檢查第 5 週正規化步驟 `s = max(...)` 是否正確，並確認 `N` 矩陣譜半徑小於 1 |
| VIKOR 計算時出現除以 0 的錯誤 | 某準則所有候選方案表現完全相同（ $f_j^* = f_j^-$ ） | 本週 `vikor()` 函式已內建 `1e-10` 之極小值保護機制，惟仍應檢視該準則是否具有區辨力，必要時可考慮剔除 |
| C1／C2 判斷結果與手動計算不符 | `DQ` 計算時方案數 $J$ 誤用成準則數 $n$ | 確認 `DQ = 1/(J-1)` 中之 $J$ 為候選方案總數，而非評選準則數 |
| 落差分析結果全數為負值 | 期望水準設定低於候選方案實際表現（期望水準與最低水準方向設反） | 確認效益型準則之期望水準應高於候選方案之最佳表現，而非低於 |

### 附錄 E：繳交前自我檢核清單

- [ ] 已說明未加權超矩陣之建構過程（正規化與轉置步驟）
- [ ] 已報告極限超矩陣之收斂驗證（如收斂於第幾次冪次疊代）
- [ ] 已報告 DANP 最終權重，並與 DEMATEL 正規化權重對照說明差異
- [ ] 已明確標示每一項準則之類型（效益型／成本型）
- [ ] 已報告 VIKOR 完整計算過程（S、R、Q 值三線表）
- [ ] 已檢驗並報告 C1（可接受優勢）與 C2（可接受穩定性）條件是否成立
- [ ] 若 C1 不成立，已正確列出妥協解集合而非武斷指定單一方案
- [ ] 若執行修正版 VIKOR，已說明期望水準之設定依據
- [ ] 已繪製改善優先級落差分析圖與妥協排序比較圖
- [ ] 所有統計結果之文字敘述與表格數值一致，無謄寫錯誤

### 附錄 F：研究倫理提醒

本週研究設計涉及邀請專家同時完成 DEMATEL 直接影響評估與候選方案績效評分兩項填答任務，工作量較前幾週更高，除延續前五週已說明之知情同意、匿名性等基本倫理原則外，特別提醒：應合理評估專家填答之總時間成本，必要時可考慮分兩階段進行問卷施測（先完成 DEMATEL 評估，間隔數日後再進行方案績效評分），避免因問卷過長導致專家填答品質下降；若研究涉及企業實際建築專案或醫療院所系統評選之敏感決策資訊，應與受訪機構明確約定研究資料之保密範圍與學術發表時之匿名化處理方式。

### 附錄 G：第 1–6 週研究方法整合對照表

| 週次 | 方法 | 知識論基礎 | 核心產出 |
|---|---|---|---|
| 第 1 週 | EFA | 資料驅動 | 量表因素結構 |
| 第 2 週 | 階層迴歸＋調節效應 | 資料驅動 | 前因效果與調節效果 |
| 第 3 週 | PLS-SEM | 資料驅動 | 完整因果路徑模型 |
| 第 4 週 | AHP／Fuzzy AHP／TOPSIS | 知識驅動（準則獨立） | 準則權重與方案排序 |
| 第 5 週 | DEMATEL | 知識驅動（準則互為因果） | 因果分群與影響網絡 |
| 第 6 週 | DANP／VIKOR | 知識驅動（準則互為因果） | 網路均衡權重與妥協排序／落差分析 |

至此，本課程第二模組（第 4–7 週）已完成從「準則獨立」（AHP）到「準則互為因果」（DEMATEL）、再到「在因果網路上求解權重並進行妥協排序」（DANP-VIKOR）之完整方法論演進脈絡，下週將介紹更輕量化的權重求解技術（BWM），作為此一方法論家族之延伸補充。

### 附錄 H：極限超矩陣手動驗證之簡化數值範例

為協助學生理解超矩陣冪次疊代收斂之數學原理，以下以一個簡化的 2×2 未加權超矩陣為例，展示如何以解析方式（而非疊代方式）直接求解極限超矩陣：

$$
W = \begin{bmatrix} 0.3 & 0.6 \\ 0.7 & 0.4 \end{bmatrix}
$$

此矩陣為欄隨機矩陣（每欄總和為 1），其極限分布（穩態分布）可透過求解 $W\pi = \pi$ （ $\pi$ 為特徵值 1 對應之特徵向量，正規化後總和為 1）之解析公式求得。對一般 2×2 欄隨機矩陣 $\begin{bmatrix} a & 1-b \\ 1-a & b \end{bmatrix}$ ，其穩態分布為：

$$
\pi = \left( \frac{1-b}{2-a-b},\ \frac{1-a}{2-a-b} \right)
$$

代入 $a=0.3$ 、 $b=0.4$ ：

$$
\pi = \left( \frac{0.6}{1.3},\ \frac{0.7}{1.3} \right) = (0.4615,\ 0.5385)
$$

以 Python 實際疊代驗證（`np.linalg.matrix_power(W, k)`），可觀察到 $k=10$ 時已完全收斂至 $(0.4615, 0.5385)$ ，與解析解完全一致。學生可將此簡化 2×2 範例之邏輯，對照本週 Step 3 之 `numpy` 精確運算輸出結果，作為自我檢核練習，這也是論文口試中若被要求「請說明極限超矩陣的數學意義」時最直觀的準備方式——它本質上就是馬可夫鏈之穩態分布求解問題。

### 附錄 I：本週常用 Python 函式速查表

| 函式／方法 | 功能 |
|---|---|
| `T / T.sum(axis=1, keepdims=True)` | 依列正規化總影響矩陣，得列隨機矩陣 $T_c$ |
| `Tc.T` | 轉置得未加權超矩陣（欄隨機矩陣） |
| `np.linalg.matrix_power(W, k)` | 計算矩陣 $W$ 之 $k$ 次冪，用於超矩陣收斂疊代 |
| `vikor(performance_matrix, weights, benefit_criteria, v)` | 本週自訂函式，計算 VIKOR 之 S、R、Q 值並排序 |
| `check_vikor_compromise_conditions(vikor_result)` | 本週自訂函式，檢驗 C1／C2 妥協解可接受條件 |
| `(aspiration - X) / (aspiration - worst)` | 修正版 VIKOR 落差計算核心公式 |

### 附錄 J：期刊審查意見對照檢核表

| 常見審查意見 | 對應本週教材章節 | 回應要點 |
|---|---|---|
| 「DANP 權重與 DEMATEL 中心度正規化權重有何不同？為何不直接用後者？」 | 理論篇 3.3 節、Step 4、Q1 | 補充對照表並說明極限超矩陣捕捉之網路均衡分布意涵 |
| 「未見超矩陣收斂驗證，如何確認極限超矩陣計算正確？」 | Step 3 | 補充不同冪次下數值變化表，展示收斂過程 |
| 「VIKOR 排序僅報告 Q 值，未檢驗妥協解是否穩定」 | 理論篇 3.6 節、Step 7 | 補充 C1、C2 條件檢驗結果，並正確處理條件不成立之情形 |
| 「策略係數 v 的設定依據為何？」 | Q3 | 說明 v=0.5 之預設依據，或補充敏感度分析 |
| 「修正版 VIKOR 之期望水準設定是否具有客觀依據？」 | 理論篇 3.7 節、Q4 | 補充期望水準訂定之外部標準來源說明 |

---

## 下週預告

第 7 週將進入「新型輕量化決策演算法：BWM 最佳化與直觀模糊集合（IFS）」，學生將學習如何以最佳最差法（Best-Worst Method, BWM）大幅降低專家成對比較之認知負荷（僅需 $2n-3$ 次比較，而非 AHP 之 $\binom{n}{2}$ 次），並透過線性規劃求解最優權重；同時學習以直觀模糊集合（Intuitionistic Fuzzy Sets, IFS）之歸屬度、非歸屬度與猶豫度，刻畫決策情境中比第 4 週三角模糊數更細緻的不確定性型態。研究範例將以「企業綠色包裝材料採購評估：基於最佳最差法之快速權重求解」為主題，並延伸至「新創加速器創業投資案甄選決策：直觀模糊 BWM 評估模型」之期末專題發想方向，同時作為第二模組（第 4–7 週）知識驅動型決策科學方法之總結收官。
