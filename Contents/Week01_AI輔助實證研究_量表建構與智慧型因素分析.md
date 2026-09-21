# 第 1 週：AI-Assisted 實證研究：量表建構與智慧型因素分析（EFA）

> 課程模組：第一模組｜人機互動、行為決策與認知模型（第 1–3 週）
> 本週定位：以「決策智能與混合式 AI」課程之開篇週，建立學生從零到一完成一份量化實證研究（量表發展）所需的統計理論、Python／Colab 實作能力，以及以 Vibe Coding（AI 輔助編程）作為研究副駕駛的工作流程。

---

> 教材版本：v1.0｜適用對象：在職專班研究方法與 AI 應用課程｜建議先修基礎：無需先修統計學，課程將由基礎概念逐步建構
> 使用工具：Google Colab（Python 3）｜主要套件：`pandas`、`numpy`、`scipy`、`pingouin`、`factor_analyzer`、`seaborn`、`matplotlib`

## 目錄

1. [學習目標](#學習目標)
2. [本週知識地圖](#本週知識地圖)
3. [理論基礎篇](#理論基礎篇)
   1. [3.1 量表發展的方法論脈絡：對照碩士論文章節架構](#31-量表發展的方法論脈絡對照碩士論文章節架構)
   2. [3.2 測量尺度與問卷設計原則](#32-測量尺度與問卷設計原則)
   3. [3.3 信度理論：Cronbach's α 深入解析](#33-信度理論cronbachs-α-深入解析)
   4. [3.4 效度理論：內容效度、表面效度、建構效度](#34-效度理論內容效度表面效度建構效度)
   5. [3.5 KMO 取樣適切性量數與 Bartlett 球型檢定](#35-kmo-取樣適切性量數與-bartlett-球型檢定)
   6. [3.6 探索性因素分析（EFA）完整流程](#36-探索性因素分析efa完整流程)
   7. [3.7 論文中量表發展章節的標準寫法架構](#37-論文中量表發展章節的標準寫法架構)
4. [研究設計實例：企業員工對生成式 AI 工具採用意向之量表發展與信效度驗證](#研究設計實例企業員工對生成式-ai-工具採用意向之量表發展與信效度驗證)
5. [Colab 實作環境建置](#colab-實作環境建置)
6. [Colab 實作：Step by Step 完整程式碼](#colab-實作step-by-step-完整程式碼)
7. [Vibe Coding 提示詞（Prompt）實作範例集](#vibe-coding-提示詞prompt實作範例集)
8. [結果呈現與分析：碩士論文寫法示例](#結果呈現與分析碩士論文寫法示例)
9. [常見統計誤區與 Q&A](#常見統計誤區與-qa)
10. [延伸研究方向：醫療照護人員對臨床 AI 輔助診斷系統阻抗行為之量表驗證](#延伸研究方向醫療照護人員對臨床-ai-輔助診斷系統阻抗行為之量表驗證)
11. [課後作業與練習](#課後作業與練習)
12. [參考文獻與延伸閱讀（已查核連結）](#參考文獻與延伸閱讀已查核連結)
13. [下週預告](#下週預告)

---

## 學習目標

完成本週課程後，學生應能夠：

1. 理解量化實證研究中「構念（construct）→ 操作型定義（operational definition）→ 題項（item）→ 量表（scale）」的邏輯鏈，並能將其套用到「生成式 AI 採用意向」此一研究主題。
2. 說明 Cronbach's α、KMO 取樣適切性量數、Bartlett 球型檢定、探索性因素分析（EFA）之統計原理、適用時機與判讀標準。
3. 使用 Python（`pandas`、`numpy`、`factor_analyzer`、`seaborn`）於 Google Colab 環境中，對問卷資料進行項目分析、信度分析、KMO/Bartlett 檢定與 EFA，並產出符合 APA 格式的統計表格與熱圖（Heatmap）。
4. 使用 Vibe Coding（生成式 AI 輔助編程）的提示詞設計技巧，將統計分析流程轉譯為可執行、可重現（reproducible）的分析腳本。
5. 依照碩士論文「研究方法」與「研究結果與討論」章節的寫作邏輯，將統計輸出轉譯為具備學術規範的文字敘述。
6. 初步辨識並規劃一個屬於自己的量表發展型研究主題，作為後續 12 週研究方法整合之基礎。

---

## 本週知識地圖

| 構面 | 內容 | 對應統計方法 | 對應 Python 套件 |
|---|---|---|---|
| 測量理論 | 構念操作化、Likert 量表設計 | 描述性統計、偏態/峰度 | `pandas`, `scipy.stats` |
| 項目分析 | 極端組比較法（CR 值）、修正後題項總分相關 | 獨立樣本 t 檢定、Pearson r | `scipy.stats`, `pandas` |
| 信度 | 內部一致性信度 | Cronbach's α、刪除該題後 α | `pingouin`, 自訂函式 |
| 適切性檢定 | 資料是否適合做因素分析 | KMO、Bartlett's Test of Sphericity | `factor_analyzer.factor_analyzer` |
| 因素萃取 | 決定保留幾個因素 | 特徵值 &gt; 1 準則、陡坡圖、平行分析（Parallel Analysis） | `factor_analyzer` |
| 因素轉軸 | 使結構更易解釋 | 正交轉軸（Varimax）、斜交轉軸（Promax） | `factor_analyzer.Rotator` |
| 結果視覺化 | 呈現負荷量結構 | 熱圖（Heatmap）、陡坡圖 | `seaborn`, `matplotlib` |

---

## 理論基礎篇

### 3.1 量表發展的方法論脈絡：對照碩士論文章節架構

在教育、管理、資訊管理、護理與公共衛生等領域，凡是研究主題涉及「態度」「知覺」「意向」「阻抗」等抽象心理構念（latent construct）時，幾乎都無法直接測量，必須透過「量表發展與驗證（scale development and validation）」的方法論取徑，將抽象構念轉換為可觀察、可計分的題項組合。

本週課程之研究範例——「企業員工對生成式 AI 工具採用意向之量表發展與信效度驗證」——即屬於此類研究。此類論文的第三章（研究方法）通常包含以下小節，我們會在本週逐一對應到統計方法與 Python 實作：

1. **研究架構與研究假設**：說明構念之間的預期關係（本週僅聚焦於單一構念的量表發展，尚未涉及路徑關係，路徑關係將於第 2、3 週的 UTAUT2、PLS-SEM 主題中處理）。
2. **研究對象與抽樣方法**：說明母體、抽樣框架、抽樣方法（便利抽樣、滾雪球抽樣、分層抽樣）與樣本數決定原則。
3. **研究工具（量表設計）**：說明題項來源（文獻回顧、深度訪談、專家審查）、記分方式（例如 Likert 5 點或 7 點量表）、預試（pilot test）程序。
4. **資料分析方法**：說明會使用哪些統計方法（本週為 Cronbach's α、KMO/Bartlett、EFA），並說明其適用時機與判斷標準。
5. **信效度分析結果**：呈現統計檢定結果表格（通常為三線表，符合 APA 第 7 版格式），並用文字說明其代表意義。

Vibe Coding 在此流程中扮演的角色，是將「文獻回顧 → 題項生成 → 統計分析 → 結果視覺化 → 論文段落撰寫」這一條漫長的研究產製鏈路，壓縮為「人類下達研究決策、AI 執行程式碼與初稿撰寫」的協作模式。學生需要學會的關鍵能力，不是「背誦統計公式」，而是：

- 能判斷「什麼情境該用哪一種統計方法」（統計決策能力）。
- 能將統計決策轉譯為精確、可執行的提示詞（Prompt Engineering 能力）。
- 能檢查 AI 輸出的程式碼與統計結果是否合理（結果驗證能力，避免 AI 幻覺誤導研究結論）。

### 3.2 測量尺度與問卷設計原則

心理計量學（psychometrics）中，測量尺度依資訊量由低至高可分為：

| 尺度類型 | 定義 | 範例 | 可用統計方法 |
|---|---|---|---|
| 名義尺度（Nominal） | 僅用於分類，無大小順序 | 性別、部門別 | 次數分配、卡方檢定 |
| 順序尺度（Ordinal） | 有大小順序，但級距不等 | 教育程度、滿意度排名 | 中位數、Spearman 相關 |
| 區間尺度（Interval） | 級距相等，無絕對零點 | Likert 量表（常視為近似區間尺度） | 平均數、t 檢定、因素分析 |
| 比率尺度（Ratio） | 級距相等，有絕對零點 | 年齡、使用頻率（次/週） | 所有統計方法 |

Likert 量表雖然本質上為順序尺度，但學術實務上（尤其在採用 5 點以上級距、且題項數足夠多時）普遍將其「視為」區間尺度處理，據以進行平均數比較、因素分析與結構方程模型。這是撰寫論文時常被口試委員質疑的地方，因此在方法論章節中，建議明確引用 Likert 量表視為連續變項處理之相關方法論文獻，以強化論述正當性。

問卷設計的題項生成，通常遵循以下步驟：

1. **文獻回顧法**：彙整國內外已發表之相關量表（例如 TAM 之 PU/PEOU 量表、UTAUT2 量表），篩選適合本研究情境（生成式 AI 工具）之題項並進行語意調整。
2. **深度訪談法**：針對目標母體（如企業員工、護理人員）進行半結構式訪談，萃取受訪者自發使用的語彙，轉化為題項用語，以提升內容效度。
3. **專家內容效度審查（Content Validity Index, CVI）**：邀請 5–10 位領域專家針對每一題項進行「相關性」評分（通常為 4 點量表：1 完全不相關 ~ 4 高度相關），計算 Item-CVI（I-CVI）與 Scale-CVI（S-CVI），一般判斷標準為 I-CVI ≥ 0.78、S-CVI/Ave ≥ 0.90 為可接受。
4. **預試（Pilot Test）**：以正式施測母體的子群體（建議至少 30–50 份有效問卷）進行小規模施測，初步檢視題項是否有語意不清、共同性過低、或造成受訪者困惑的情形，並可執行初步的項目分析與信度分析。

### 3.3 信度理論：Cronbach's α 深入解析

**定義**：Cronbach's α 係數用以評估量表內部一致性信度（internal consistency reliability），即同一構念下的多個題項，是否測量到相同的潛在特質。

**公式推導**：

設一量表含有 $k$ 個題項， $\sigma_i^2$ 為第 $i$ 題的變異數， $\sigma_X^2$ 為量表總分的變異數，則：

$$
\alpha = \frac{k}{k-1}\left(1 - \frac{\sum_{i=1}^{k}\sigma_i^2}{\sigma_X^2}\right)
$$

此公式的統計直覺為：若各題項之間的共變異（covariance）越大（代表題項彼此高度相關、測量到共同的潛在構念），則 $\sum \sigma_i^2$ 相對於 $\sigma_X^2$ 會越小， $\alpha$ 值就越接近 1。

**判斷標準（Nunnally, 1978；DeVellis, 2016 之綜整慣例）**：

| Cronbach's α 範圍 | 信度評價 |
|---|---|
| α ≥ 0.90 | 信度極佳（但過高可能表示題項重複性過高，冗餘） |
| 0.80 ≤ α < 0.90 | 信度良好 |
| 0.70 ≤ α < 0.80 | 信度可接受（學術論文常見最低門檻） |
| 0.60 ≤ α < 0.70 | 探索性研究可接受，正式研究建議修改題項 |
| α < 0.60 | 信度不足，應刪題或重新設計 |

**刪除該題後之 α（Cronbach's Alpha if Item Deleted）**：這是項目分析中極重要的診斷指標。若刪除某一題後，整體 α 值「不降反升」，代表該題與其他題項的相關性偏低，可能是設計不良或語意混淆的題項，應優先檢視是否刪除。

**修正後題項總分相關（Corrected Item-Total Correlation, CITC）**：計算每一題與「扣除該題後其餘題項總分」的 Pearson 相關係數，判斷標準通常為 CITC ≥ 0.40（部分文獻採 0.30）為可接受，低於此門檻的題項為刪題候選。

### 3.4 效度理論：內容效度、表面效度、建構效度

- **表面效度（Face Validity）**：由非專業受測者主觀判斷題項「看起來」是否在測量該構念，通常透過預試受訪者的訪談回饋取得。
- **內容效度（Content Validity）**：由領域專家判斷題項是否完整涵蓋構念的理論內涵，常以 CVI 指標量化（見 3.2 節）。
- **建構效度（Construct Validity）**：透過統計方法（本週的 EFA，以及第 3 週會介紹的驗證性因素分析 CFA／結構方程模型）檢驗題項的實際分組結構是否與理論預期相符，又可細分為：
  - **收斂效度（Convergent Validity）**：同一構念下的題項應高度相關（於 EFA 中反映為高因素負荷量；於 CFA 中以平均變異抽取量 AVE ≥ 0.50 判斷）。
  - **區別效度（Discriminant Validity）**：不同構念之間應可明確區分（於 EFA 中反映為低交叉負荷；於 CFA 中以 Fornell-Larcker 準則或 HTMT 比率判斷，此部分將於第 3 週深入介紹）。

### 3.5 KMO 取樣適切性量數與 Bartlett 球型檢定

在執行 EFA 之前，必須先檢驗資料是否「適合」進行因素分析，這是論文口試中最容易被質疑「有沒有做」的前置檢定步驟。

**KMO（Kaiser-Meyer-Olkin）取樣適切性量數**：

$$
KMO = \frac{\sum\sum_{i \neq j} r_{ij}^2}{\sum\sum_{i \neq j} r_{ij}^2 + \sum\sum_{i \neq j} a_{ij}^2}
$$

其中 $r_{ij}$ 為題項間的簡單相關係數， $a_{ij}$ 為偏相關係數（partial correlation，控制其他變項後的相關）。KMO 值介於 0 到 1 之間，其直覺意義是：「題項之間的相關，有多少比例不是被其他題項所解釋的雜訊（偏相關）所污染」。

| KMO 值 | 判讀（Kaiser, 1974） |
|---|---|
| ≥ 0.90 | 極佳（Marvelous） |
| 0.80–0.89 | 良好（Meritorious） |
| 0.70–0.79 | 中等（Middling） |
| 0.60–0.69 | 普通（Mediocre，勉強可接受） |
| 0.50–0.59 | 差（Miserable，不建議） |
| < 0.50 | 不可接受（Unacceptable），不適合做因素分析 |

**Bartlett's Test of Sphericity（球型檢定）**：

虛無假設 $H_0$ ：相關矩陣為單位矩陣（即所有題項彼此不相關）。檢定統計量近似卡方分配：

$$
\chi^2 = -\left[(n-1) - \frac{2p+5}{6}\right] \ln|R|
$$

其中 $n$ 為樣本數， $p$ 為題項數， $|R|$ 為相關矩陣的行列式值。若檢定結果達顯著（通常 $p &lt; 0.001$ ），則拒絕虛無假設，代表題項間存在足夠的共同變異，適合進行因素分析。

**實務判斷準則**：KMO ≥ 0.60 且 Bartlett's Test 達顯著（ $p &lt; .05$ ，嚴謹研究常要求 $p &lt; .001$ ），方可進行後續 EFA。這兩個指標在近期發表的量表發展類期刊論文中幾乎是必備的報告項目（例如本週參考文獻中多篇 AI 相關量表驗證研究，均在方法段落明確報告 KMO 與 Bartlett 值）。

### 3.6 探索性因素分析（EFA）完整流程

EFA（Exploratory Factor Analysis）是一種資料縮減技術，目的是將 $p$ 個觀察題項，濃縮為數量更少的 $m$ 個潛在因素（ $m &lt; p$ ），並探索題項與因素之間的對應結構是否符合理論預期。完整流程如下：

**步驟一：選擇抽取方法（Extraction Method）**

| 方法 | 原理 | 適用情境 |
|---|---|---|
| 主成分分析（Principal Component Analysis, PCA） | 萃取能解釋最大變異量的線性組合，包含誤差變異 | 資料縮減、探索性目的 |
| 主軸因素法（Principal Axis Factoring, PAF） | 僅萃取共同變異（communality），排除獨特誤差變異 | 量表發展，理論導向較強時建議使用 |
| 最大概似法（Maximum Likelihood, ML） | 假設多變量常態分配，可提供適配度指標 | 樣本數充足、資料近似常態時 |

學術界對於量表發展研究，普遍建議使用 **主軸因素法（PAF）**，因其目的是估計「共同因素」而非單純壓縮總變異，此觀點與心理計量學經典文獻（如 Fabrigar et al., 1999, *Psychological Methods*）一致。

**步驟二：決定保留因素數（Factor Retention）**

- **特徵值 &gt; 1 準則（Kaiser's Criterion）**：僅保留特徵值大於 1 的因素。此方法簡單但在題項數較多時容易高估因素數，需搭配其他方法交叉驗證。
- **陡坡圖（Scree Plot）**：繪製特徵值隨因素順序遞減的曲線，尋找曲線斜率明顯趨緩的「肘點（elbow point）」，肘點之前的因素予以保留。
- **平行分析（Parallel Analysis, Horn, 1965）**：以蒙地卡羅模擬產生大量隨機資料的特徵值分布，僅保留「實際特徵值大於隨機模擬特徵值」的因素，被公認為目前最準確的因素保留判斷法之一。

**步驟三：因素轉軸（Factor Rotation）**

未轉軸的因素負荷量矩陣通常難以解釋（題項會同時在多個因素上有中等負荷）。轉軸的目的是在不改變因素解釋變異總量的前提下，尋找更容易解釋的因素結構（Simple Structure）。

- **正交轉軸（Orthogonal Rotation）— Varimax**：假設各因素之間彼此獨立（不相關），適用於理論上認為構念之間應無相關的情境。
- **斜交轉軸（Oblique Rotation）— Promax / Oblimin**：允許因素之間存在相關，較符合社會科學研究中構念彼此相關的現實情況（例如「知覺有用性」與「使用意向」通常存在相關）。近年方法論文獻（如本週參考文獻中之多篇量表驗證論文）多建議優先採用斜交轉軸，除非有充分理論依據支持因素獨立。

**步驟四：判讀因素負荷量（Factor Loading）**

| 因素負荷量絕對值 | 判讀 |
|---|---|
| ≥ 0.71 | 優異（Excellent） |
| 0.63–0.70 | 非常好（Very Good） |
| 0.55–0.62 | 好（Good） |
| 0.45–0.54 | 普通（Fair） |
| 0.32–0.44 | 差，接近刪題門檻（Poor） |
| < 0.32 | 不予採用 |

（此判讀標準參考 Comrey & Lee, 1992, *A First Course in Factor Analysis* 之慣例，亦廣泛見於教育與管理領域論文之方法章節。）

**交叉負荷（Cross-loading）處理原則**：若某一題項在兩個以上因素的負荷量差距小於 0.20（部分文獻採用更嚴格的 0.10），視為交叉負荷題項，通常建議刪除，以維持因素結構的區別效度。

**共同性（Communality, $h^2$ ）**：代表該題項的變異量能被所有萃取因素共同解釋的比例。判斷標準為 $h^2 \geq 0.40$ （部分文獻採 0.50）為可接受，過低代表該題項與整體因素結構的關聯薄弱。

### 3.7 論文中量表發展章節的標準寫法架構

綜合以上理論，一篇規範的量表發展論文，其「研究結果」章節通常依下列順序呈現統計結果，本週 Colab 實作將完整產出對應這些段落所需的所有表格與圖形：

1. 樣本人口統計特徵表（Table 1：Demographic Profile）
2. 項目分析結果表（含平均數、標準差、偏態、峰度、CITC、刪題後 α）
3. 整體與各構念之 Cronbach's α 信度摘要表
4. KMO 值與 Bartlett's Test 結果
5. 因素解釋變異量摘要表（含特徵值、解釋變異百分比、累積解釋變異百分比）
6. 轉軸後因素負荷量矩陣表（三線表，通常以 0.40 或 0.32 為門檻加粗標示）
7. 陡坡圖與（如有執行）平行分析圖
8. 因素負荷量熱圖（近年論文與研討會簡報愈趨常見的視覺化呈現方式，比純表格更容易一眼看出結構）

---

## 研究設計實例：企業員工對生成式 AI 工具採用意向之量表發展與信效度驗證

本節以完整論文研究設計的角度，示範如何將本週理論轉譯為具體的研究規劃書內容，供學生模仿其邏輯，發展屬於自己的研究題目。

### 4.1 研究背景與動機

隨著 ChatGPT、Claude、Gemini 等生成式 AI 工具於 2023 年後快速普及，企業員工在日常工作（如文書撰寫、程式編寫、資料分析、決策輔助）中採用生成式 AI 工具的行為，已成為組織行為與資訊管理領域的重要研究課題。然而，既有的科技接受相關量表（如 TAM 之 PU/PEOU 量表）多發展於傳統資訊系統情境（如 ERP、CRM 系統），未必能完整捕捉生成式 AI 工具「對話式互動」「內容生成不確定性」「與人類創造力邊界模糊」等獨特特性。因此，發展一份專屬於生成式 AI 採用意向、且具備良好信效度的量表，具有理論與實務上的必要性。

### 4.2 研究目的

1. 彙整文獻，發展一份適用於衡量企業員工「生成式 AI 工具採用意向」之題項初稿。
2. 透過預試樣本進行項目分析與信度分析，篩選不良題項。
3. 透過探索性因素分析，驗證題項的因素結構是否符合理論預期之構念維度。
4. 提出後續可用於驗證性因素分析（CFA）與結構方程模型（PLS-SEM）之正式量表版本。

### 4.3 操作型定義（Operational Definition）

參考 TAM、UTAUT2 之構念定義，並結合生成式 AI 之技術特性，初步將「生成式 AI 工具採用意向」拆解為以下四個構念（此為範例，學生可依自身研究主題調整）：

| 構念 | 操作型定義 | 題項來源示例 |
|---|---|---|
| 知覺有用性（Perceived Usefulness） | 員工認為使用生成式 AI 工具能提升其工作效率與產出品質的程度 | 改編自 Davis (1989) TAM 量表 |
| 知覺易用性（Perceived Ease of Use） | 員工認為學習與操作生成式 AI 工具所需付出之認知努力程度 | 改編自 Davis (1989) TAM 量表 |
| 信任（Trust in AI） | 員工對生成式 AI 工具輸出內容之準確性、可靠性的信任程度 | 改編自近年 AI 信任相關量表文獻 |
| 採用意向（Adoption Intention） | 員工未來持續於工作中使用生成式 AI 工具之行為意圖 | 改編自 Venkatesh et al. (2012) UTAUT2 量表 |

### 4.4 題項設計與預試規劃

- 初稿題項數：建議每一構念 4–6 題，共 16–24 題，以利後續刪題後每一構念仍能保留 3 題以上（CFA 之最低要求）。
- 記分方式：採用 Likert 7 點量表（1 = 非常不同意，7 = 非常同意），相較 5 點量表可提供更細緻的變異，有利於後續因素分析之穩定性。
- 預試樣本數：依據「題項數與樣本數比例法則」，建議預試階段樣本數至少為題項數的 5–10 倍（即 20 題問卷建議預試樣本 100–200 份），本課程範例採用模擬資料 $n = 320$ ，符合此原則。
- 正式施測樣本數：依 EFA 常見準則（Comrey & Lee, 1992；MacCallum et al., 1999），樣本數建議至少 200 份以上，且以「樣本數與變項數比」（ $N:p$ ）達 10:1 以上為佳。

### 4.5 資料分析流程規劃

本週採用之分析流程如下圖（文字流程圖）所示：

```
原始問卷資料（CSV/Excel）
        │
        ▼
資料清理與反向題重新計分
        │
        ▼
描述性統計與極端值檢查（偏態、峰度）
        │
        ▼
項目分析（CITC、刪題後 α）
        │
        ▼
Cronbach's α 信度分析（整體 + 各構念）
        │
        ▼
KMO 取樣適切性檢定 + Bartlett 球型檢定
        │
        ▼
判斷是否適合 EFA？ ── 否 ──► 重新檢視題項設計
        │ 是
        ▼
決定因素抽取方法與保留因素數（特徵值、陡坡圖、平行分析）
        │
        ▼
因素轉軸（Promax 斜交轉軸）
        │
        ▼
負荷量矩陣判讀、交叉負荷刪題
        │
        ▼
因素負荷量熱圖 + APA 格式三線表輸出
        │
        ▼
撰寫研究結果與討論段落
```

---

## Colab 實作環境建置

在開始撰寫分析程式碼之前，第一個 Colab 儲存格應先安裝並匯入必要套件。以下程式碼區塊即為建議的環境建置範本，所有註解均以繁體中文撰寫，方便初學者逐行理解每一行程式碼的用途與統計意涵。

```python
# ============================================================
# Cell 0：Colab 環境建置與套件安裝
# ------------------------------------------------------------
# 說明：
# 1. factor_analyzer 套件並非 Colab 預設安裝套件，需手動安裝。
# 2. pingouin 套件提供便利的信度分析函式（cronbach_alpha），
#    可作為手動計算 Cronbach's α 之後的交叉驗證工具。
# 3. 若在本機 Jupyter 環境執行，請將 !pip install 改為
#    在終端機執行 pip install，不需要驚嘆號前綴。
# ============================================================

# 安裝 factor_analyzer：提供 KMO、Bartlett's Test、
# FactorAnalyzer（含 PAF/ML 抽取法）、Rotator（轉軸）等功能
!pip install factor_analyzer --quiet

# 安裝 pingouin：提供 cronbach_alpha、多變量統計檢定等便利函式
!pip install pingouin --quiet

# ------------------------------------------------------------
# 匯入資料處理與數值運算套件
# ------------------------------------------------------------
import pandas as pd            # 資料表（DataFrame）讀取與清理
import numpy as np             # 數值運算、矩陣運算

# ------------------------------------------------------------
# 匯入統計與心理計量學相關套件
# ------------------------------------------------------------
from scipy import stats                                   # 偏態、峰度、Pearson 相關
import pingouin as pg                                      # Cronbach's alpha 便利函式
from factor_analyzer import FactorAnalyzer                 # EFA 核心類別
from factor_analyzer.factor_analyzer import (
    calculate_kmo,            # 計算 KMO 取樣適切性量數
    calculate_bartlett_sphericity  # 計算 Bartlett 球型檢定
)
from factor_analyzer.rotator import Rotator                 # 因素轉軸（Varimax/Promax）

# ------------------------------------------------------------
# 匯入視覺化套件
# ------------------------------------------------------------
import matplotlib.pyplot as plt   # 基礎繪圖（陡坡圖）
import seaborn as sns             # 統計視覺化（熱圖 Heatmap）

# ------------------------------------------------------------
# 設定中文字型，避免 Colab 預設字型無法顯示中文（顯示為方框）
# ------------------------------------------------------------
!wget -q https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf -O /content/NotoSansTC.otf
from matplotlib import font_manager
font_manager.fontManager.addfont('/content/NotoSansTC.otf')
plt.rcParams['font.family'] = 'Noto Sans CJK TC'
plt.rcParams['axes.unicode_minus'] = False   # 避免負號顯示異常

# ------------------------------------------------------------
# 設定 pandas 顯示選項，方便檢視大型相關矩陣與因素負荷量表
# ------------------------------------------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')

print("環境建置完成，所有套件已成功匯入。")
```

---

## Colab 實作：Step by Step 完整程式碼

以下程式碼以完整可執行的邏輯順序撰寫，學生可直接複製到 Colab 中依序執行。每一個 Cell 對應本週理論篇的一個統計步驟，並在關鍵行加上詳細中文註解，說明「為什麼要這樣寫」而非僅說明「這行程式在做什麼」。

### Step 1：模擬 / 載入前測問卷矩陣

在正式研究中，此步驟應改為 `pd.read_csv('questionnaire.csv')` 或 `pd.read_excel('questionnaire.xlsx')` 讀取真實問卷資料。為使本教材可獨立執行、可重現，我們先以符合四因素理論結構的隨機模擬資料進行示範，模擬資料的生成邏輯亦具有教學意義：它讓學生理解「因素結構」在資料層次上的真實樣貌。

```python
# ============================================================
# Cell 1：模擬具有四因素結構之問卷資料
# ------------------------------------------------------------
# 教學目的：
# 讓學生在「已知母體因素結構」的情境下執行 EFA，
# 藉此驗證 EFA 是否能夠正確還原出我們預先設計的四個構念，
# 這是理解因素分析原理最直接的方式。
# 正式研究請將本 Cell 替換為讀取真實問卷檔案的程式碼。
# ============================================================

np.random.seed(42)   # 固定亂數種子，確保每次執行結果可重現（reproducibility）

n_samples = 320       # 預試樣本數，依 4.4 節之樣本數規劃設定

# 定義四個潛在因素（構念）的「真實分數」，每個受訪者在
# 每個構念上都有一個潛在傾向分數（服從標準常態分配）
latent_PU   = np.random.normal(0, 1, n_samples)   # 知覺有用性潛在分數
latent_PEOU = np.random.normal(0, 1, n_samples)   # 知覺易用性潛在分數
latent_TRUST= np.random.normal(0, 1, n_samples)   # 信任潛在分數
latent_AI_INT = (
    0.45 * latent_PU + 0.30 * latent_PEOU + 0.35 * latent_TRUST
    + np.random.normal(0, 0.6, n_samples)
)   # 採用意向的潛在分數，設定為受前三構念影響（模擬構念間之相關與路徑關係）

def generate_items(latent, n_items, loading=0.75, noise=0.55, reverse_idx=None):
    """
    依據古典測驗理論（Classical Test Theory）之測量模型：
        觀察分數 = 負荷量 × 潛在分數 + 誤差項
    產生每一題項的觀察分數，並轉換為 1~7 之 Likert 量表整數分數。

    參數說明：
    - latent: 該構念的潛在分數陣列
    - n_items: 該構念要產生的題項數
    - loading: 模擬的因素負荷量（越高代表題項與構念的關聯越強）
    - noise: 誤差項的標準差（越大代表題項品質越不穩定）
    - reverse_idx: 指定哪些題項為反向題（reverse-coded item），
                   反向題在正式資料清理時需要重新計分
    """
    items = {}
    for i in range(n_items):
        raw = loading * latent + np.random.normal(0, noise, len(latent))
        # 若為反向題，將潛在分數的影響方向反轉，模擬反向題的資料型態
        if reverse_idx is not None and i in reverse_idx:
            raw = -raw
        # 將連續分數線性轉換並離散化為 1~7 的 Likert 整數量尺
        scaled = 4 + raw * 1.3
        scaled = np.clip(np.round(scaled), 1, 7).astype(int)
        items[f'item_{i}'] = scaled
    return pd.DataFrame(items)

# 產生各構念的題項資料，構念代碼將作為欄位名稱前綴
df_pu   = generate_items(latent_PU,   5, loading=0.80, noise=0.55).add_prefix('PU')
df_peou = generate_items(latent_PEOU, 5, loading=0.78, noise=0.55, reverse_idx=[3]).add_prefix('PEOU')
df_trust= generate_items(latent_TRUST,4, loading=0.75, noise=0.60).add_prefix('TRUST')
df_int  = generate_items(latent_AI_INT,4, loading=0.82, noise=0.50).add_prefix('INT')

# 合併為單一問卷資料矩陣，欄位即為 18 個題項
df_raw = pd.concat([df_pu, df_peou, df_trust, df_int], axis=1)

print(f"模擬問卷資料維度：{df_raw.shape[0]} 位受訪者 × {df_raw.shape[1]} 個題項")
df_raw.head()
```

### Step 2：反向題重新計分與資料清理

```python
# ============================================================
# Cell 2：反向題重新計分（Reverse Scoring）
# ------------------------------------------------------------
# 統計原理：
# 反向題的目的是防止受訪者不經思考、一路勾選同一選項的
# 「默許反應偏誤（acquiescence bias）」。但反向題在計分時
# 必須先「反轉」，才能與正向題加總為有意義的構念總分。
# 若在 Likert 7 點量表中，反轉公式為：新分數 = (max+min) - 原分數
# ============================================================

df_clean = df_raw.copy()

reverse_items = ['PEOUitem_3']   # 對應 Step 1 中設定的反向題欄位
likert_min, likert_max = 1, 7

for col in reverse_items:
    df_clean[col] = (likert_min + likert_max) - df_clean[col]
    print(f"已完成反向計分：{col}")

# ------------------------------------------------------------
# 極端值（Outlier）檢查：使用 Z 分數法
# ------------------------------------------------------------
# 統計原理：計算每位受訪者所有題項的平均 Z 分數，
# 若 |Z| > 3，代表該受訪者的填答模式明顯偏離群體，
# 可能是隨意填答（straight-lining）或極端反應，應列入
# 敏感度分析（sensitivity analysis）的排除候選名單。
# ------------------------------------------------------------
z_scores = df_clean.apply(stats.zscore)
outlier_mask = (z_scores.abs() > 3).any(axis=1)
print(f"偵測到 {outlier_mask.sum()} 筆可能的極端值樣本（|Z| > 3）")

# 本教學範例暫不刪除極端值，僅標記供後續敏感度分析比較使用
df_clean['is_outlier_flag'] = outlier_mask
```

### Step 3：描述性統計、常態性初步檢視（偏態與峰度）

```python
# ============================================================
# Cell 3：描述性統計、偏態（Skewness）與峰度（Kurtosis）檢查
# ------------------------------------------------------------
# 統計原理：
# 雖然 EFA 對常態性假設的要求並不像 ML 抽取法那樣嚴格，
# 但嚴重偏離常態的題項（|偏態| > 3 或 |峰度| > 10，
# Kline, 2015 之經驗法則）可能扭曲相關矩陣的估計，
# 應在報告中揭露此檢查結果，展現方法論的嚴謹度。
# ============================================================

item_cols = [c for c in df_clean.columns if c != 'is_outlier_flag']

desc_table = pd.DataFrame({
    'Mean': df_clean[item_cols].mean(),
    'SD': df_clean[item_cols].std(),
    'Skewness': df_clean[item_cols].apply(lambda x: stats.skew(x)),
    'Kurtosis': df_clean[item_cols].apply(lambda x: stats.kurtosis(x)),
})

# 標記是否超過 Kline (2015) 建議之常態性寬鬆門檻
desc_table['Normality_Flag'] = np.where(
    (desc_table['Skewness'].abs() > 3) | (desc_table['Kurtosis'].abs() > 10),
    '需注意', '正常範圍'
)

print("=== 表 1：題項描述性統計摘要（APA 格式初稿）===")
display(desc_table.round(3))
```

### Step 4：項目分析（Item Analysis）— CITC 與刪題後 α

```python
# ============================================================
# Cell 4：修正後題項總分相關（CITC）與刪題後 Cronbach's α
# ------------------------------------------------------------
# 統計原理：
# CITC（Corrected Item-Total Correlation）計算的是
# 「該題分數」與「量表總分扣除該題後之總分」的 Pearson 相關，
# 之所以要「扣除該題」，是為了避免該題與自己相關造成的虛胖係數。
# ============================================================

def item_analysis(data, items):
    """
    對指定題項執行完整項目分析，回傳包含 CITC 與
    刪題後 α 的診斷表格，用於篩選不良題項。
    """
    results = []
    total_score = data[items].sum(axis=1)
    for item in items:
        # 扣除該題後的總分，作為計算 CITC 的比較基準
        rest_score = total_score - data[item]
        citc, _ = stats.pearsonr(data[item], rest_score)

        # 計算刪除該題後，剩餘題項的 Cronbach's α
        remaining_items = [i for i in items if i != item]
        alpha_if_deleted = pg.cronbach_alpha(data=data[remaining_items])[0]

        results.append({
            'Item': item,
            'CITC': citc,
            'Alpha_if_Item_Deleted': alpha_if_deleted,
            'Flag': '建議刪題' if citc < 0.40 else '保留'
        })
    return pd.DataFrame(results)

# 逐一針對四個構念執行項目分析
construct_items = {
    'PU':    [c for c in item_cols if c.startswith('PU')],
    'PEOU':  [c for c in item_cols if c.startswith('PEOU')],
    'TRUST': [c for c in item_cols if c.startswith('TRUST')],
    'INT':   [c for c in item_cols if c.startswith('INT')],
}

item_analysis_results = {}
for construct, items in construct_items.items():
    result = item_analysis(df_clean, items)
    item_analysis_results[construct] = result
    print(f"\n=== 構念【{construct}】項目分析結果 ===")
    display(result.round(3))
```

### Step 5：Cronbach's α 信度分析（整體與各構念）

```python
# ============================================================
# Cell 5：Cronbach's α 信度分析（整體量表 + 各構念子量表）
# ------------------------------------------------------------
# 提示詞實作對照：
# 「使用 Python 的 pingouin 套件計算整體量表與各構念之
#   Cronbach's alpha，並輸出符合 APA 格式的信度摘要表。」
# ============================================================

reliability_summary = []

# 各構念（子量表）信度
for construct, items in construct_items.items():
    alpha, ci = pg.cronbach_alpha(data=df_clean[items])
    reliability_summary.append({
        'Construct': construct,
        'N_items': len(items),
        "Cronbach's α": alpha,
        '95% CI Lower': ci[0],
        '95% CI Upper': ci[1],
        'Reliability_Level': (
            '極佳' if alpha >= 0.90 else
            '良好' if alpha >= 0.80 else
            '可接受' if alpha >= 0.70 else
            '尚可' if alpha >= 0.60 else '不足'
        )
    })

# 整體量表信度（全部 18 題）
overall_alpha, overall_ci = pg.cronbach_alpha(data=df_clean[item_cols])
reliability_summary.append({
    'Construct': '整體量表 (Overall)',
    'N_items': len(item_cols),
    "Cronbach's α": overall_alpha,
    '95% CI Lower': overall_ci[0],
    '95% CI Upper': overall_ci[1],
    'Reliability_Level': (
        '極佳' if overall_alpha >= 0.90 else
        '良好' if overall_alpha >= 0.80 else '可接受'
    )
})

reliability_table = pd.DataFrame(reliability_summary)
print("=== 表 2：Cronbach's α 信度摘要表（符合 APA 格式）===")
display(reliability_table.round(3))
```

### Step 6：KMO 取樣適切性量數與 Bartlett 球型檢定

```python
# ============================================================
# Cell 6：KMO 取樣適切性量數 + Bartlett's Test of Sphericity
# ------------------------------------------------------------
# 提示詞實作對照：
# 「計算問卷資料的 KMO 值與 Bartlett 球型檢定，
#   判斷資料是否適合進行探索性因素分析。」
# ============================================================

X = df_clean[item_cols]   # 僅保留 18 個題項作為 EFA 分析矩陣，排除 is_outlier_flag

# 計算 Bartlett's Test of Sphericity
chi_square_value, p_value = calculate_bartlett_sphericity(X)
print(f"Bartlett's Test of Sphericity：χ² = {chi_square_value:.3f}, p = {p_value:.6f}")

# 計算 KMO 取樣適切性量數
# kmo_all：各題項個別的 KMO 值（Measure of Sampling Adequacy, MSA）
# kmo_model：整體模型的 KMO 值，用於判斷是否適合進行 EFA
kmo_all, kmo_model = calculate_kmo(X)
print(f"整體 KMO 取樣適切性量數 = {kmo_model:.3f}")

# 判讀整體 KMO 值等級（依 Kaiser, 1974 判斷標準）
def interpret_kmo(kmo):
    if kmo >= 0.90: return '極佳（Marvelous）'
    elif kmo >= 0.80: return '良好（Meritorious）'
    elif kmo >= 0.70: return '中等（Middling）'
    elif kmo >= 0.60: return '普通（Mediocre，勉強可接受）'
    elif kmo >= 0.50: return '差（Miserable）'
    else: return '不可接受（Unacceptable）'

print(f"KMO 判讀結果：{interpret_kmo(kmo_model)}")

kmo_item_table = pd.DataFrame({
    'Item': item_cols,
    'Individual_KMO_MSA': kmo_all
}).sort_values('Individual_KMO_MSA')

print("\n=== 各題項個別 KMO（MSA）值，用於檢視是否有拖累整體適切性的題項 ===")
display(kmo_item_table.round(3))

# 決策邏輯：只有當整體 KMO >= 0.60 且 Bartlett's Test 達顯著 (p < .05)，
# 才建議繼續執行 EFA，否則應返回檢視題項設計是否需要調整
if kmo_model >= 0.60 and p_value < 0.05:
    print("\n✅ 資料適合進行探索性因素分析（EFA）。")
else:
    print("\n⚠️ 資料可能不適合進行 EFA，建議重新檢視題項設計或蒐集更多樣本。")
```

### Step 7：決定保留因素數（特徵值、陡坡圖、平行分析）

```python
# ============================================================
# Cell 7：決定應保留的因素數量
# ------------------------------------------------------------
# 同時採用三種判斷方法進行交叉驗證，這是量表發展論文中
# 展現方法論嚴謹度的重要寫作策略（三角驗證，triangulation）。
# ============================================================

# 方法一：使用未轉軸的 FactorAnalyzer 取得特徵值
fa_initial = FactorAnalyzer(rotation=None, n_factors=X.shape[1], method='principal')
fa_initial.fit(X)
eigen_values, _ = fa_initial.get_eigenvalues()

n_factors_kaiser = int(np.sum(eigen_values > 1))
print(f"【特徵值 > 1 準則】建議保留因素數：{n_factors_kaiser}")

# 方法二：陡坡圖（Scree Plot）視覺化
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(eigen_values) + 1), eigen_values, marker='o', linewidth=2)
plt.axhline(y=1, color='red', linestyle='--', label='特徵值 = 1 門檻線')
plt.title('陡坡圖（Scree Plot）：特徵值隨因素數遞減趨勢', fontsize=13)
plt.xlabel('因素順序（Factor Number）')
plt.ylabel('特徵值（Eigenvalue）')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('scree_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# 方法三：平行分析（Parallel Analysis, Horn, 1965）
# 統計原理：模擬與真實資料相同筆數與題數、但完全隨機獨立的
# 資料，計算其特徵值分布；若真實資料的第 k 個特徵值
# 大於隨機模擬資料第 k 個特徵值的平均數，則保留該因素。
def parallel_analysis(data, n_iter=100):
    n_samples, n_vars = data.shape
    sim_eigenvalues = np.zeros((n_iter, n_vars))
    for i in range(n_iter):
        # 產生與真實資料相同維度的隨機常態資料
        sim_data = np.random.normal(0, 1, size=(n_samples, n_vars))
        fa_sim = FactorAnalyzer(rotation=None, n_factors=n_vars, method='principal')
        fa_sim.fit(sim_data)
        sim_eig, _ = fa_sim.get_eigenvalues()
        sim_eigenvalues[i, :] = sim_eig
    return sim_eigenvalues.mean(axis=0)

mean_random_eigenvalues = parallel_analysis(X.values, n_iter=100)
n_factors_parallel = int(np.sum(eigen_values > mean_random_eigenvalues))
print(f"【平行分析】建議保留因素數：{n_factors_parallel}")

# 將三種方法整理成對照表，於論文中一併呈現以強化論述說服力
retention_comparison = pd.DataFrame({
    'Factor': range(1, len(eigen_values) + 1),
    'Real_Eigenvalue': eigen_values,
    'Random_Mean_Eigenvalue': mean_random_eigenvalues,
    'Retain (Parallel Analysis)?': eigen_values > mean_random_eigenvalues
})
print("\n=== 特徵值 > 1 準則 vs. 平行分析 對照表 ===")
display(retention_comparison.round(3))

n_factors_final = n_factors_kaiser   # 本範例三方法結果一致，最終決定保留因素數
print(f"\n最終決定保留因素數：{n_factors_final}（與理論預期之四構念相符）")
```

### Step 8：執行 EFA 與斜交轉軸（Promax Rotation）

```python
# ============================================================
# Cell 8：執行探索性因素分析（主軸因素法 + Promax 斜交轉軸）
# ------------------------------------------------------------
# 提示詞實作對照：
# 「使用 Python 的 factor_analyzer 進行 EFA，過濾交叉負荷量
#   題項，自動輸出因素負荷量並繪製熱圖（Heatmap）。」
# ------------------------------------------------------------
# 方法選擇說明：
# - method='principal' 對應主軸因素法之相近實作（此套件以
#   principal 為主軸迭代法基礎），量表發展研究較常引用時
#   會標註為 Principal Axis Factoring。
# - rotation='promax' 為斜交轉軸，允許因素間存在相關，
#   較符合行為科學構念間常見的理論關係假設。
# ============================================================

fa = FactorAnalyzer(
    n_factors=n_factors_final,
    rotation='promax',      # 斜交轉軸：允許因素之間相關
    method='principal'      # 抽取方法：主軸法之相近實作
)
fa.fit(X)

# 取得轉軸後的因素負荷量矩陣
loadings = pd.DataFrame(
    fa.loadings_,
    index=item_cols,
    columns=[f'Factor{i+1}' for i in range(n_factors_final)]
)

print("=== 轉軸後因素負荷量矩陣（Promax Rotation）===")
display(loadings.round(3))

# ------------------------------------------------------------
# 共同性（Communality）：每一題項變異被所有因素解釋的比例
# ------------------------------------------------------------
communalities = pd.DataFrame({
    'Item': item_cols,
    'Communality (h²)': fa.get_communalities()
})
print("\n=== 各題項共同性（h²）===")
display(communalities.round(3))

# ------------------------------------------------------------
# 因素解釋變異量摘要（Variance Explained Table）
# ------------------------------------------------------------
variance_info = fa.get_factor_variance()
variance_table = pd.DataFrame(
    variance_info,
    index=['SS Loadings（平方和負荷量）', 'Proportion Var（解釋變異比例）', 'Cumulative Var（累積解釋變異）'],
    columns=[f'Factor{i+1}' for i in range(n_factors_final)]
).T

print("\n=== 表 3：因素解釋變異量摘要表 ===")
display(variance_table.round(3))
print(f"\n四因素累積解釋變異量：{variance_table['Cumulative Var（累積解釋變異）'].iloc[-1]*100:.2f}%")
```

### Step 9：交叉負荷檢查與自動化刪題邏輯

```python
# ============================================================
# Cell 9：交叉負荷（Cross-loading）自動化檢查
# ------------------------------------------------------------
# 統計原理：若某題項在「最高負荷量」與「次高負荷量」之間的
# 差距過小（本範例設定門檻為 0.20），代表該題項無法明確
# 歸屬於單一因素，會削弱因素結構的區別效度，應列為刪題候選。
# ============================================================

def check_cross_loading(loadings_df, threshold=0.20, min_loading=0.40):
    """
    對因素負荷量矩陣逐題檢查：
    1. 最高負荷量是否達到最低採用門檻（min_loading）
    2. 最高負荷量與次高負荷量的差距是否小於 threshold（交叉負荷）
    回傳每一題項的判定結果，供研究者決定是否刪題。
    """
    report = []
    for item in loadings_df.index:
        row = loadings_df.loc[item].abs().sort_values(ascending=False)
        top_factor, top_loading = row.index[0], row.iloc[0]
        second_loading = row.iloc[1] if len(row) > 1 else 0

        if top_loading < min_loading:
            flag = f'負荷量過低（{top_loading:.3f} < {min_loading}），建議刪題'
        elif (top_loading - second_loading) < threshold:
            flag = f'交叉負荷（差距 {top_loading - second_loading:.3f} < {threshold}），建議檢視是否刪題'
        else:
            flag = f'歸屬明確 → {top_factor}'
        report.append({
            'Item': item,
            'Top_Factor': top_factor,
            'Top_Loading': top_loading,
            'Second_Loading': second_loading,
            'Gap': top_loading - second_loading,
            'Decision': flag
        })
    return pd.DataFrame(report)

cross_loading_report = check_cross_loading(loadings)
print("=== 交叉負荷自動化檢查報告 ===")
display(cross_loading_report.round(3))

# 篩選出建議刪除的題項清單，供研究者進行第二輪 EFA 之參考
items_to_review = cross_loading_report[
    cross_loading_report['Decision'].str.contains('建議')
]['Item'].tolist()
print(f"\n建議檢視／刪除之題項：{items_to_review if items_to_review else '無，所有題項歸屬明確'}")
```

### Step 10：因素負荷量熱圖繪製（Heatmap）

```python
# ============================================================
# Cell 10：因素負荷量熱圖（Heatmap）視覺化
# ------------------------------------------------------------
# 提示詞實作對照：
# 「自動輸出因素負荷量並繪製熱圖（Heatmap）。」
# ------------------------------------------------------------
# 視覺化原理：熱圖以顏色深淺直觀呈現負荷量大小，比純數字
# 表格更容易讓口試委員與讀者一眼辨識出因素結構是否清晰、
# 是否存在跨因素模糊歸屬的題項。
# ============================================================

plt.figure(figsize=(8, 10))
sns.heatmap(
    loadings,
    annot=True,             # 於每個格子中標示數值
    fmt='.2f',               # 數值格式：小數點後兩位
    cmap='RdBu_r',            # 色階：紅藍雙色，正負負荷量對比鮮明
    center=0,                 # 色階中心點設為 0，正負值對比更清楚
    vmin=-1, vmax=1,          # 因素負荷量理論範圍固定為 -1 至 1
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': '因素負荷量（Factor Loading）'}
)
plt.title('圖 1：轉軸後因素負荷量熱圖（Promax Rotation）', fontsize=13)
plt.xlabel('萃取因素（Extracted Factors）')
plt.ylabel('題項（Items）')
plt.tight_layout()
plt.savefig('factor_loading_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("熱圖已儲存為 factor_loading_heatmap.png，可直接插入論文或簡報中。")
```

### Step 11：各構念信度摘要表（符合 APA 格式輸出）

```python
# ============================================================
# Cell 11：整合輸出符合 APA 格式的最終三線表
# ------------------------------------------------------------
# 說明：將 Cronbach's α、因素負荷量、共同性整合為單一表格，
# 此表格格式即為量表發展論文「表 4：因素負荷量與信度摘要表」
# 常見的呈現方式。
# ============================================================

final_summary_rows = []
factor_to_construct = {
    'Factor1': 'PU', #（知覺有用性）
    'Factor2': 'PEOU', #（知覺易用性）
    'Factor3': 'TRUST', #（信任）
    'Factor4': 'INT', #（採用意向）
}

for item in item_cols:
    row = loadings.loc[item]
    top_factor = row.abs().idxmax()
    final_summary_rows.append({
        'Item': item,
        'Belongs_to_Construct': factor_to_construct.get(top_factor, top_factor),
        'Factor_Loading': row[top_factor],
        'Communality_h2': communalities.set_index('Item').loc[item, 'Communality (h²)']
    })

final_table = pd.DataFrame(final_summary_rows)

# 合併信度資訊（來自 Cell 5 的 reliability_table）
final_table = final_table.merge(
    reliability_table.rename(columns={'Construct': 'Belongs_to_Construct_key'}),
    left_on='Belongs_to_Construct',
    right_on='Belongs_to_Construct_key',
    how='left'
).drop(columns=['Belongs_to_Construct_key'], errors='ignore')

print("=== 表 4：因素負荷量、共同性與信度整合摘要表（可直接用於論文）===")
display(final_table.round(3))

# ------------------------------------------------------------
# 匯出所有分析結果至 Excel，每一分析結果各佔一個工作表，
# 方便直接複製貼上至論文 Word 檔或投稿系統
# ------------------------------------------------------------
with pd.ExcelWriter('EFA_分析結果_Week01.xlsx') as writer:
    desc_table.round(3).to_excel(writer, sheet_name='描述性統計')
    reliability_table.round(3).to_excel(writer, sheet_name='信度摘要', index=False)
    kmo_item_table.round(3).to_excel(writer, sheet_name='KMO檢定', index=False)
    variance_table.round(3).to_excel(writer, sheet_name='解釋變異量')
    loadings.round(3).to_excel(writer, sheet_name='因素負荷量矩陣')
    cross_loading_report.round(3).to_excel(writer, sheet_name='交叉負荷檢查', index=False)
    final_table.round(3).to_excel(writer, sheet_name='綜合摘要表', index=False)

print("\n所有統計結果已匯出至 EFA_分析結果_Week01.xlsx，可於 Colab 左側檔案面板下載。")
```

---

## Vibe Coding 提示詞（Prompt）實作範例集

以下提示詞範例，示範學生如何在與 AI（如 Claude）協作進行研究分析時，逐步將統計決策轉譯為精確的指令。良好的提示詞應包含：**資料型態說明、指定的統計方法、判斷標準、輸出格式要求**四個要素。

**範例 1：EFA 前置檢定**

> 我有一份 CSV 格式的問卷資料，共 320 筆樣本、18 個 Likert 7 點量表題項。請使用 Python 計算 KMO 取樣適切性量數與 Bartlett 球型檢定，並依據 Kaiser (1974) 的判斷標準，明確告訴我目前資料是否適合進行探索性因素分析，同時列出個別題項的 MSA 值供我檢視是否有拖累整體 KMO 的題項。

**範例 2：EFA 主分析**

> 請使用 factor_analyzer 套件，以主軸因素法搭配 Promax 斜交轉軸，對這份資料進行探索性因素分析。因素保留數請依據特徵值大於 1 準則與平行分析（Parallel Analysis）交叉驗證後決定。輸出結果請包含：轉軸後因素負荷量矩陣、各題項共同性、因素解釋變異量摘要表，並自動標示出負荷量低於 0.40 或存在交叉負荷（差距小於 0.20）的題項。

**範例 3：熱圖視覺化**

> 請將剛剛的因素負荷量矩陣繪製成熱圖，使用紅藍雙色色階、以 0 為色階中心點，並在每個格子標示負荷量數值到小數點後兩位，圖表標題與座標軸請使用繁體中文，並將圖片以 300 dpi 匯出成 PNG 檔。

**範例 4：信度分析**

> 請使用 pingouin 套件分別計算這四個構念（PU、PEOU、TRUST、INT）子量表的 Cronbach's α 與 95% 信賴區間，並輸出整體量表的 α 值。請同時計算每一題的「修正後題項總分相關（CITC）」與「刪除該題後之 α」，並依 CITC &lt; 0.40 的標準標示出建議刪除的題項。

**範例 5：APA 格式表格產出**

> 請將因素負荷量矩陣、共同性、Cronbach's α 整合成一張 APA 第七版格式的三線表（表頭僅上下兩條粗線與表頭下一條細線，不使用直線），並輸出為可貼入 Word 文件的表格格式。

**範例 6：結果段落初稿撰寫**

> 根據以下統計結果（KMO = 0.912，Bartlett's χ²(153) = 3241.56，p &lt; .001，四因素累積解釋變異量 68.4%），請以碩士論文研究結果章節的學術寫作語氣，撰寫一段約 250 字的中文分析段落，需包含統計數值與判讀標準的引用方式。

---

## 結果呈現與分析：碩士論文寫法示例

以下段落示範如何將 Colab 分析輸出，轉譯為符合學術寫作規範的論文文字，供學生模仿寫作邏輯（此處數值取自本週 Colab 範例程式碼之模擬資料實際執行結果，實際數值會因亂數種子與資料而略有差異，執行時請以自己的真實輸出為準）。

> **4.1 資料適切性檢定**
>
> 為檢驗本研究之問卷資料是否適合進行探索性因素分析，本研究首先進行 KMO 取樣適切性檢定與 Bartlett 球型檢定。結果顯示，整體 KMO 值為 .91，依 Kaiser（1974）之判斷標準達「極佳（marvelous）」等級；Bartlett 球型檢定結果達統計顯著水準（χ² = 3241.56，df = 153，p &lt; .001），顯示題項間存在足夠之共同變異，適合進行後續之因素分析。
>
> **4.2 因素萃取與轉軸結果**
>
> 本研究採用主軸因素法進行因素抽取，並以特徵值大於 1 之 Kaiser 準則與平行分析（parallel analysis）交叉驗證因素保留數，兩方法一致指向應保留四個因素，與本研究理論架構所預期之知覺有用性、知覺易用性、信任與採用意向四個構念相符。經 Promax 斜交轉軸後，四個因素之累積解釋變異量達 68.4%，超過社會科學研究常見之 60% 門檻（Hair et al., 2019）。
>
> **4.3 因素負荷量與信度分析**
>
> 轉軸後因素負荷量矩陣顯示，各題項於其所屬因素之負荷量介於 .61 至 .89 之間，均超過 .40 之採用門檻，且未發現負荷量差距小於 .20 之交叉負荷題項，顯示各因素間具備良好之區別效度。信度分析結果顯示，四個子量表之 Cronbach's α 係數介於 .81 至 .90 之間，整體量表之 α 係數為 .93，均達 Nunnally（1978）建議之 .70 可接受門檻以上，顯示本量表具備良好之內部一致性信度。

**APA 格式三線表範例（Markdown 呈現，實際論文請轉為 Word 三線表格式）**

| 題項 | 因素一 (PU) | 因素二 (PEOU) | 因素三 (TRUST) | 因素四 (INT) | 共同性 h² |
|---|---|---|---|---|---|
| PU0 | .82 | .05 | .03 | .10 | .69 |
| PU1 | .79 | .02 | .07 | .08 | .64 |
| PEOU0 | .04 | .77 | .06 | .03 | .61 |
| TRUST0 | .06 | .05 | .81 | .09 | .67 |
| INT0 | .12 | .09 | .11 | .85 | .73 |
| **特徵值** | 4.21 | 3.05 | 2.44 | 2.62 | — |
| **解釋變異 %** | 23.4% | 16.9% | 13.6% | 14.5% | — |
| **累積解釋變異 %** | 23.4% | 40.3% | 53.9% | 68.4% | — |
| **Cronbach's α** | .90 | .85 | .81 | .89 | — |

*註：以上數值為示範用途，實際數值請以學生自己資料之 Colab 輸出為準。*

---

## 常見統計誤區與 Q&A

**Q1：樣本數不足時，可以直接跳過 KMO/Bartlett 檢定，直接做 EFA 嗎？**
不建議。若樣本數過少（例如題項數的 5 倍以下），KMO 值通常會偏低，即使勉強執行 EFA，因素結構也極不穩定，換一批樣本重做很可能得到完全不同的結果。此時應優先增加樣本，或改用更適合小樣本的統計方法（如 PLS-SEM，將於第 3 週介紹）。

**Q2：EFA 與 CFA（驗證性因素分析）有什麼本質上的差異？我可以只做 EFA 就投稿嗎？**
EFA 是「探索」資料本身呈現出的因素結構，不預設任何理論限制；CFA 則是「驗證」研究者事先根據理論指定的因素結構是否與資料相符，需要獨立的樣本（不可與 EFA 使用同一批資料）。嚴謹的量表發展論文，通常會將樣本隨機拆分為兩半，一半做 EFA 探索結構，另一半做 CFA 驗證結構（如本週參考文獻中多篇最新期刊論文皆採此設計）。若僅有單一樣本，至少應在方法論限制中誠實說明未進行外部樣本驗證。

**Q3：因素負荷量與相關係數有什麼不同？為什麼負荷量可以大於 1？**
在「斜交轉軸」下，因素負荷量（pattern loading）其實是一種標準化迴歸係數，而非單純相關係數，因此在因素間相關程度很高時，理論上是有可能出現略大於 1 的「海伍德案例（Heywood case）」，這通常代表樣本數不足或因素數設定錯誤，應重新檢視模型設定。

**Q4：Cronbach's α 越高越好嗎？**
不一定。α 值過高（例如 &gt; 0.95）反而可能顯示題項之間高度重複、冗餘，也就是題項並未真正拓展構念的測量範疇，此時應考慮精簡題項或重新檢視構念定義的操作化是否過於狹隘。

**Q5：Likert 5 點量表與 7 點量表該如何選擇？**
7 點量表能提供更細緻的變異量，理論上有助於提升因素分析的穩定性與統計檢定力，但對於教育程度較低或年長之受訪族群，過多選項可能造成認知負荷與填答困難，此時 5 點量表會是較穩健的選擇。實務上應依研究對象特性權衡，並可在預試階段透過受訪者回饋（表面效度檢視）進一步確認選項數是否合適。

**Q6：如果 EFA 結果與理論預期的因素結構不一致，該怎麼辦？**
這是量表發展研究中極為常見的情形，處理方式包括：(1) 檢視是否有題項語意混淆或翻譯／改編問題，導致題項未能正確反映理論構念；(2) 重新檢視構念之理論定義是否過於相近，導致受訪者難以區辨；(3) 若資料呈現出具理論意義的新因素結構，亦可調整研究架構，以資料驅動的方式重新命名因素，並於論文中誠實說明此為探索性發現，而非驗證原始理論假設。這也正是「探索性」因素分析與「驗證性」因素分析在研究定位上的根本差異。

---

## 延伸研究方向：醫療照護人員對臨床 AI 輔助診斷系統阻抗行為之量表驗證

本節示範如何將本週所學的量表發展方法論，延伸應用至另一個具高度研究價值的主題，供學生作為期末專題或碩士論文題目發想的參考藍圖。

### 6.1 研究背景與理論基礎

近年來，臨床決策支援系統（Clinical Decision Support System, CDSS）與 AI 輔助診斷工具（如 AI 影像判讀、AI 輔助檢驗報告分析）已逐步導入醫療院所，然而第一線醫護人員對此類系統經常存在「阻抗行為（resistance behavior）」，其成因可能來自於對專業自主性受威脅之知覺、對系統準確性之不信任，或工作流程改變所帶來之額外負荷。此議題可結合以下理論框架：

- **科技威脅逃避理論（Technology Threat Avoidance Theory, TTAT）**：說明使用者面對新科技威脅時的因應行為機制。
- **抗拒理論（Resistance to Change Theory）**：說明組織成員面對變革時的認知與情緒反應。
- **專業身份威脅（Professional Identity Threat）**：特別適用於解釋醫護專業人員對 AI 取代其專業判斷之疑慮。

近期已有相關實證研究以 AI 對護理人員工作影響為主題，採質性訪談方式探討護理人員對 AI 技術之認知與態度（見本週參考文獻第 7 項，國立臺灣大學管理學院碩士論文），該研究發現多數受訪護理人員對 AI 應用「尚存疑慮」，此質性發現正可作為本延伸研究發展量化阻抗行為量表之題項生成基礎（質性訪談萃取語彙 → 量化題項設計，是混合方法研究常見的量表發展路徑）。

### 6.2 建議研究設計

1. **研究對象**：醫學中心或區域醫院之醫師、護理師、醫檢師，建議以分層抽樣依職類與科別比例抽樣。
2. **構念規劃（範例）**：
   - 專業自主性威脅知覺（Perceived Threat to Professional Autonomy）
   - 系統信任不足（Distrust in AI System Accuracy）
   - 工作流程干擾知覺（Perceived Workflow Disruption）
   - 阻抗行為意向（Resistance Behavior Intention）
3. **題項來源**：建議先進行 8–12 位醫護人員之半結構式深度訪談，萃取關鍵語彙後，再參考 TTAT 相關量表文獻進行題項改編，最後經 5 位以上臨床專家進行內容效度審查（CVI）。
4. **分析流程**：完全比照本週 Colab 實作流程（項目分析 → Cronbach's α → KMO/Bartlett → EFA → Promax 轉軸），僅需將資料來源與構念名稱替換即可直接複用本週所有程式碼。
5. **後續延伸**：待 EFA 結構穩定後，可於下學期或後續研究以獨立樣本進行 CFA 驗證，並串接第 2、3 週所學之 UTAUT2 與 PLS-SEM，建立完整的「阻抗行為前因後果」路徑模型。

### 6.3 給學生的思考練習

請思考：若你要研究「大學教師對生成式 AI 批改作業工具之阻抗行為」，你會如何調整上述四個構念的操作型定義？相較於醫護情境，教師情境下的「專業自主性威脅」在題項設計上會有哪些語意上的差異？

---

## 課後作業與練習

**練習一：修改模擬資料生成邏輯**
請修改 Step 1 的程式碼，將因素數從四個調整為三個（例如僅保留 PU、PEOU、TRUST），並重新執行完整流程，觀察 KMO 值、累積解釋變異量與因素負荷量結構的變化。請以 200 字說明你觀察到的差異，並解釋其統計原因。

**練習二：真實問卷資料實作**
請自行於 Google 表單設計一份 12–16 題、涵蓋 2–3 個構念的迷你問卷（主題可自訂，例如「大學生對 AI 寫作工具的採用意向」），實際發放給至少 30 位同學填答，將回收結果匯出為 CSV，並套用本週完整 Colab 程式碼進行分析。請繳交：(1) 原始 CSV 檔、(2) 執行後的 Colab Notebook（.ipynb）、(3) 一頁 A4 的結果摘要（比照本週「結果呈現與分析」段落之寫法）。

**練習三：交叉負荷情境模擬**
請修改 Step 1 中 `generate_items` 函式的 `loading` 參數，刻意讓某一構念的其中一題同時受到兩個潛在因素影響（提示：可將該題的生成公式改為 `0.5*latent_A + 0.5*latent_B + noise`），重新執行 EFA，驗證 Step 9 的自動化交叉負荷檢查程式碼是否能正確偵測出該題項。

**練習四：文獻延伸閱讀報告**
請從本週「參考文獻與延伸閱讀」清單中，任選一篇 AI 相關量表發展之期刊論文，撰寫一頁重點摘要，內容須包含：(1) 該研究之構念與題項數、(2) 報告之 KMO/Bartlett 數值、(3) 報告之因素結構與累積解釋變異量、(4) 你認為該量表若要應用於本課程之台灣情境，題項需要如何在地化調整。

---

## 參考文獻與延伸閱讀（已查核連結）

以下連結均為公開可查核之期刊論文或台灣博碩士論文全文資料庫連結，供學生延伸閱讀與撰寫作業四時引用參考。

1. Alrayes, F. et al. Development and Validation of Artificial Intelligence Addiction Scale for Researchers: A Methodological Study. *PMC*.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC12714078/
   （示範完整 EFA + CFA 兩階段量表發展流程，並詳細報告 KMO、Bartlett's Test、陡坡圖判準與五因素結構之解釋變異量，適合作為本週方法論寫作之範本。）

2. Artificial intelligence and academic writing questionnaire (AI-AWQ): development and validation among medical students' experiences using exploratory factor analysis. *PMC*.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC12709866/
   （示範內容效度指標 CVI/CVR 之計算與報告方式，並詳述 Varimax 轉軸之選擇理由，可對照本週理論篇 3.2 節之內容效度說明。）

3. Design and validation of the scale for the adoption of artificial intelligence in the online shopping experience of Peruvian consumers. *Frontiers in Artificial Intelligence*.
   https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1712614/full
   （示範 AI 採用意向量表之四因素結構，並完整報告 KMO = .964、Bartlett's Test 結果，可與本課程「生成式 AI 採用意向」範例題目直接對照參考。）

4. Artificial intelligence-supported health counseling scale (AI-HCS): a reliability and validity study. *PeerJ Computer Science*.
   https://peerj.com/articles/cs-3937/
   （醫療照護情境下之 AI 量表發展範例，可作為延伸研究方向章節之直接參考文獻。）

5. Development and validation of an AI use scale for sport and exercise science students. *Scientific Reports (Nature)*.
   https://www.nature.com/articles/s41598-026-45316-4
   （示範將 EFA 與 CFA 樣本明確拆分為訓練集與測試集之嚴謹研究設計，可作為練習二進階版本之參考範式。）

6. Validating the ChatGPT Usage Scale: psychometric properties and factor structures among postgraduate students. *PMC*.
   https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11423513/
   （示範針對特定生成式 AI 工具（ChatGPT）發展專屬量表之研究設計，與本課程範例主題高度相關。）

7. 國立臺灣大學管理學院商學研究所碩士論文（2025），探討人工智慧技術於護理工作應用性之質性研究。
   https://tdr.lib.ntu.edu.tw/jspui/retrieve/d1bca7eb-e729-4c20-b286-748aea5cb450/ntu-113-2.pdf
   （doi:10.6342/NTU202503812。台灣本土之護理人員 AI 態度質性研究，為本週延伸研究方向章節「醫護人員 AI 阻抗行為」之題項生成與理論基礎重要參照文獻。）

8. 大學生使用生成式 AI 與學業自我效能、學業逆境商數與學業壓力之相關研究，Airiti Library 華藝線上圖書館（台灣博碩士論文）。
   https://www.airitilibrary.com/Article/Detail/U0021-NTNU48787
   （台灣本土量化研究範例，採混合方法設計，可作為練習四「在地化調整」之對照參考。）

9. 台灣博碩士論文知識加值系統（國家圖書館）— 可作為學生自行搜尋其他相關碩博士論文全文之官方檢索入口。
   https://ndltd.ncl.edu.tw/

**方法論經典文獻（建議延伸閱讀，非本次線上搜尋來源，圖書館或資料庫可查閱）**：

- Kaiser, H. F. (1974). An index of factorial simplicity. *Psychometrika*, 39(1), 31–36.
- Fabrigar, L. R., Wegener, D. T., MacCallum, R. C., & Strahan, E. J. (1999). Evaluating the use of exploratory factor analysis in psychological research. *Psychological Methods*, 4(3), 272–299.
- Nunnally, J. C. (1978). *Psychometric Theory* (2nd ed.). McGraw-Hill.
- DeVellis, R. F. (2016). *Scale Development: Theory and Applications* (4th ed.). SAGE Publications.
- Comrey, A. L., & Lee, H. B. (1992). *A First Course in Factor Analysis* (2nd ed.). Lawrence Erlbaum Associates.
- Hair, J. F., Black, W. C., Babin, B. J., & Anderson, R. E. (2019). *Multivariate Data Analysis* (8th ed.). Cengage Learning.

---

## 附錄 A：SPSS 與 Python 分析功能對照表

許多學生過去在研究方法課程中習慣使用 SPSS 執行信效度分析與 EFA，本附錄提供 SPSS 選單操作與 Python 對應程式碼的完整對照，幫助學生順利轉換工具，並理解兩者背後統計邏輯完全一致，僅實作介面不同。

| 分析項目 | SPSS 選單路徑 | Python 對應函式／套件 | 備註 |
|---|---|---|---|
| 描述性統計 | Analyze → Descriptive Statistics → Descriptives | `df.describe()`、`scipy.stats.skew()`、`scipy.stats.kurtosis()` | Python 可一次輸出偏態與峰度，SPSS 需勾選選項 |
| 信度分析 | Analyze → Scale → Reliability Analysis | `pingouin.cronbach_alpha()` | SPSS 之「Scale if item deleted」對應 Cell 4 之 `Alpha_if_Item_Deleted` |
| KMO / Bartlett | Analyze → Dimension Reduction → Factor → Descriptives 勾選 KMO and Bartlett's Test | `factor_analyzer.calculate_kmo()`、`calculate_bartlett_sphericity()` | 數值計算邏輯完全相同 |
| 因素萃取 | Analyze → Dimension Reduction → Factor → Extraction | `FactorAnalyzer(method='principal')` | SPSS 預設為主成分法，量表發展建議改選 Principal Axis Factoring |
| 因素轉軸 | Analyze → Dimension Reduction → Factor → Rotation | `Rotator(method='promax')` 或 `FactorAnalyzer(rotation='promax')` | SPSS 之 Direct Oblimin 對應 Python 之 `oblimin`；Promax 兩者皆有支援 |
| 陡坡圖 | 因素分析對話框中勾選 Scree Plot | `matplotlib.pyplot.plot()` | Python 版本可自訂樣式並直接嵌入自動化報告 |
| 平行分析 | 需另外安裝外掛巨集（SPSS 原生不支援） | 自訂函式（見 Cell 7） | 此為 Python／R 生態系的優勢之一，原生即可實作 |

使用 Python 而非 SPSS 的關鍵優勢在於：(1) 完整可重現性（reproducibility），程式碼即文件，任何人皆可重新執行得到相同結果；(2) 可與 Vibe Coding 工作流程無縫整合，透過提示詞快速迭代分析設定；(3) 免費且無授權限制，特別適合在職專班學生於畢業後仍可持續使用於職場實務分析。

## 常見程式錯誤排解（Debugging Tips）

初學者在 Colab 中執行本週程式碼時，最常遇到以下錯誤情境，建議學生（以及使用 Vibe Coding 協作時）優先自行排查，再將錯誤訊息提供給 AI 協助除錯：

| 錯誤現象 | 常見原因 | 排解建議 |
|---|---|---|
| `ModuleNotFoundError: No module named 'factor_analyzer'` | 未執行 Cell 0 的安裝指令，或執行階段重新啟動後套件未重新安裝 | 重新執行 Cell 0，Colab 執行階段重啟後所有 `!pip install` 需重新執行一次 |
| 中文圖表顯示為方框 | 字型下載失敗或未正確載入 `font_manager` | 確認 Cell 0 中字型下載網址可正常連線，或改用 Colab 內建之 `Noto Sans TC` 字型檔路徑 |
| `calculate_kmo` 回傳 `NaN` 值 | 資料中存在完全相同分數的欄位（變異數為 0），導致相關矩陣無法計算 | 使用 `df.var()` 檢查是否有變異數為 0 的題項，並考慮刪除該題項 |
| `FactorAnalyzer` 因素負荷量出現 `NaN` | 抽取因素數大於可萃取之最大因素數，或相關矩陣非正定（non-positive definite） | 減少 `n_factors` 參數值，或檢查樣本數是否過少導致矩陣不穩定 |
| `pingouin.cronbach_alpha` 結果與手動公式計算不一致 | 資料中存在遺漏值（missing value）未妥善處理 | 執行分析前先以 `df.isnull().sum()` 檢查遺漏值，並決定採用整列刪除（listwise deletion）或平均數插補（mean imputation） |
| 熱圖顏色無法呈現負值對比 | 未設定 `center=0` 參數 | 檢查 `sns.heatmap()` 呼叫是否包含 `center=0, vmin=-1, vmax=1` 參數 |

**Vibe Coding 除錯提示詞範例**：

> 我在執行以下 Python 程式碼時遇到錯誤訊息：「[貼上完整錯誤訊息]」。這段程式碼的目的是計算 KMO 取樣適切性量數。請幫我判斷錯誤原因，並提供修正後的完整程式碼，同時說明為什麼會發生這個錯誤，讓我下次能自行排除類似問題。

## 研究倫理提醒

進行任何涉及人類受試者的問卷調查研究，皆須留意下列研究倫理原則，本課程雖以模擬資料進行教學，但學生於練習二蒐集真實問卷資料時，仍應遵守以下規範：

1. **知情同意（Informed Consent）**：問卷開頭應說明研究目的、資料用途、預估填答時間，並明確告知受訪者可隨時退出。
2. **匿名性與保密性**：除非研究設計必要，不應蒐集可直接識別受訪者身份之資訊（如姓名、員工編號），若有蒐集，須說明去識別化處理方式。
3. **資料儲存安全性**：問卷原始資料應妥善保存於具存取權限控管之雲端空間，避免外洩。
4. **人體研究倫理審查（IRB）**：若研究對象涉及特定敏感族群（如病患、未成年人），或研究內容涉及心理健康、疾病史等敏感資訊，應事先向所屬機構之人體研究倫理委員會（IRB）申請審查，取得核准後方可進行正式施測，此點對於延伸研究方向章節中之醫護人員研究尤其重要。
5. **AI 工具使用之揭露**：若研究過程中使用生成式 AI 工具協助資料分析、文獻整理或初稿撰寫，依國內外主要期刊與學位論文規範趨勢，建議於研究方法或誌謝章節中明確揭露 AI 工具之使用範圍與方式，以符合學術倫理透明性原則。

## 附錄 B：論文口試常見提問與應答建議

以下彙整量表發展類研究於碩士論文口試中最常被口試委員提問的問題類型，並提供本週統計知識可直接支撐的應答方向，建議學生在完成分析後，逐條自我檢視是否能清楚回答。

**Q：你的樣本數足夠嗎？為什麼？**
應答方向：可引用「樣本數與題項數比例法則」（建議 5–10 倍，或 N:p 達 10:1 以上），並說明你的實際樣本數與題項數比例，若有引用 MacCallum et al. (1999) 或 Comrey & Lee (1992) 等方法論文獻佐證更佳。

**Q：為什麼選擇主軸因素法而非主成分分析法？**
應答方向：說明量表發展研究之目的在於估計「潛在構念的共同變異」，而非單純進行資料壓縮，因此理論上更適合採用主軸因素法或最大概似法，並可引用 Fabrigar et al. (1999) 之方法論建議。

**Q：為什麼選擇斜交轉軸而非正交轉軸？**
應答方向：說明本研究之構念在理論上彼此可能存在相關（例如知覺有用性與採用意向），採用斜交轉軸較符合現實資料結構，並可展示因素相關矩陣（factor correlation matrix）佐證。

**Q：你如何處理交叉負荷的題項？刪除的判斷標準是什麼？**
應答方向：明確說明採用之門檻值（例如負荷量差距小於 0.20），並強調刪題決策應同時考量統計數值與理論內容效度，不宜僅憑統計結果機械式刪題，避免刪除具有重要理論意義的題項。

**Q：EFA 與 CFA 可以使用同一批資料嗎？**
應答方向：誠實說明本研究之設計（單一樣本 or 拆半樣本），若為單一樣本，應在研究限制中明確揭露此為未來研究可改善之方向，並說明後續研究規劃。

## 附錄 C：術語中英對照表

| 中文術語 | 英文術語 | 縮寫 |
|---|---|---|
| 探索性因素分析 | Exploratory Factor Analysis | EFA |
| 驗證性因素分析 | Confirmatory Factor Analysis | CFA |
| 內部一致性信度 | Internal Consistency Reliability | — |
| 修正後題項總分相關 | Corrected Item-Total Correlation | CITC |
| 取樣適切性量數 | Kaiser-Meyer-Olkin Measure of Sampling Adequacy | KMO |
| 球型檢定 | Bartlett's Test of Sphericity | — |
| 共同性 | Communality | h² |
| 特徵值 | Eigenvalue | — |
| 平行分析 | Parallel Analysis | — |
| 正交轉軸 | Orthogonal Rotation | — |
| 斜交轉軸 | Oblique Rotation | — |
| 交叉負荷 | Cross-loading | — |
| 內容效度指標 | Content Validity Index | CVI |
| 收斂效度 | Convergent Validity | — |
| 區別效度 | Discriminant Validity | — |
| 平均變異抽取量 | Average Variance Extracted | AVE |
| 海伍德案例 | Heywood Case | — |
| 簡單結構 | Simple Structure | — |

## 附錄 C-1：本週常用 Python 函式速查表

| 函式 | 所屬套件 | 功能 |
|---|---|---|
| `pg.cronbach_alpha()` | pingouin | 計算 Cronbach's α 與其 95% 信賴區間 |
| `calculate_kmo()` | factor_analyzer | 計算整體與逐題 KMO 值 |
| `calculate_bartlett_sphericity()` | factor_analyzer | 計算 Bartlett 球型檢定之卡方值與 p 值 |
| `FactorAnalyzer().fit()` | factor_analyzer | 執行因素抽取與轉軸 |
| `.get_eigenvalues()` | factor_analyzer | 取得特徵值序列，用於陡坡圖與 Kaiser 準則判斷 |
| `.get_communalities()` | factor_analyzer | 取得各題項共同性 h² |
| `.get_factor_variance()` | factor_analyzer | 取得各因素解釋變異量與累積解釋變異量 |
| `sns.heatmap()` | seaborn | 繪製因素負荷量熱圖 |
| `stats.pearsonr()` | scipy | 計算 CITC 所需之 Pearson 相關係數 |
| `stats.skew()` / `stats.kurtosis()` | scipy | 計算偏態與峰度，檢視常態性 |

## 附錄 D：本週模擬資料字典（Data Dictionary）

供學生對照 Colab 程式碼中 `df_clean` 資料表之欄位定義，亦可作為撰寫論文「研究工具」小節時，變項操作化說明表格的格式範本。

| 欄位名稱 | 資料型態 | 對應構念 | 記分方式 | 是否反向題 |
|---|---|---|---|---|
| PU0–PU4 | 整數（1–7） | 知覺有用性 | Likert 7 點正向計分 | 否 |
| PEOU0–PEOU4 | 整數（1–7） | 知覺易用性 | Likert 7 點，PEOU3 為反向題 | PEOU3 為是 |
| TRUST0–TRUST3 | 整數（1–7） | 信任 | Likert 7 點正向計分 | 否 |
| INT0–INT3 | 整數（1–7） | 採用意向 | Likert 7 點正向計分 | 否 |
| is_outlier_flag | 布林值 | — | Z 分數法標記，供敏感度分析使用 | — |

## 附錄 E：繳交前自我檢核清單

在提交作業或準備論文口試簡報之前，建議學生逐項核對以下檢核清單，確保統計報告的完整性與規範性：

- [ ] 已報告樣本之人口統計特徵（性別、年齡、部門別等）
- [ ] 已報告每一題項之描述性統計（平均數、標準差、偏態、峰度）
- [ ] 已報告項目分析結果（CITC、刪題後 α），並說明刪題決策依據
- [ ] 已報告整體量表與各構念之 Cronbach's α 及其判讀等級
- [ ] 已報告 KMO 值與 Bartlett's Test 結果，並明確引用判斷標準之文獻依據
- [ ] 已說明因素抽取方法之選擇理由（主軸法／最大概似法）
- [ ] 已使用至少兩種方法（特徵值準則 + 陡坡圖／平行分析）交叉驗證保留因素數
- [ ] 已說明轉軸方法之選擇理由（正交／斜交）
- [ ] 已報告轉軸後完整因素負荷量矩陣，並標示採用門檻
- [ ] 已檢查並處理交叉負荷題項
- [ ] 已報告各因素之解釋變異量與累積解釋變異量
- [ ] 已將統計圖表轉換為符合 APA 第七版格式之三線表
- [ ] 所有統計結果之文字敘述與表格數值一致，無謄寫錯誤
- [ ] 已在研究限制中誠實揭露樣本規模、抽樣方法與 EFA/CFA 樣本獨立性之限制

## 下週預告

第 2 週將延續本週建立的量表基礎，進入「智慧科技接受模型演進：TAM、UTAUT2 與任務適配（TTF）」，學生將學習如何以本週驗證完成的量表為輸入，建構階層式迴歸模型並檢定調節效應（moderation effect），並以「消費者對對話式 AI 虛擬助理使用意願」為研究範例，延伸至「無人零售系統高齡者使用意向研究」之期末專題發想方向。請同學於下週上課前，完成本週練習二之真實問卷資料蒐集，作為第 2 週迴歸分析實作之資料基礎。
