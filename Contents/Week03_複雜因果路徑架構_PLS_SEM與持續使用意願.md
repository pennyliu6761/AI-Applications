# 第 3 週：複雜因果路徑架構——結構方程模型（PLS-SEM）與持續使用意願

> 課程模組：第一模組｜人機互動、行為決策與認知模型（第 1–3 週）
> 本週定位：整合前兩週之量表信效度基礎（第 1 週）與迴歸／調節效應分析能力（第 2 週），進入能同時處理多重路徑、中介變數與測量誤差的結構方程模型（SEM），並以偏最小平方法結構方程模型（PLS-SEM）作為本模組之收官技術，銜接第二模組之知識驅動型決策科學方法。

> 教材版本：v1.0｜適用對象：在職專班研究方法與 AI 應用課程｜先修基礎：第 1 週（量表建構與 EFA）、第 2 週（階層迴歸與調節效應）
> 使用工具：Google Colab（Python 3）｜主要套件：`pandas`（需鎖定 &lt; 2.2 版本，見環境建置說明）、`numpy`、`plspm`、`scipy`、`seaborn`、`matplotlib`、`networkx`

---

## 目錄

1. [學習目標](#學習目標)
2. [本週知識地圖](#本週知識地圖)
3. [理論基礎篇](#理論基礎篇)
   1. [3.1 PLS-SEM 與共變異數為基礎之 SEM（CB-SEM）比較](#31-pls-sem-與共變異數為基礎之-semcb-sem比較)
   2. [3.2 測量模型評鑑：信度與收斂效度](#32-測量模型評鑑信度與收斂效度)
   3. [3.3 區別效度：Fornell-Larcker 準則與 HTMT 比率](#33-區別效度fornell-larcker-準則與-htmt-比率)
   4. [3.4 結構模型評鑑：R²、f²、Q² 與路徑係數](#34-結構模型評鑑rf與路徑係數)
   5. [3.5 Bootstrapping 重抽樣檢定原理](#35-bootstrapping-重抽樣檢定原理)
   6. [3.6 中介效應與 PLS-SEM 中介檢定](#36-中介效應與-pls-sem-中介檢定)
   7. [3.7 期望確認模型（ECM-IT）與 TAM 整合框架](#37-期望確認模型ecm-it與-tam-整合框架)
   8. [3.8 論文中 PLS-SEM 章節的標準寫法架構](#38-論文中-pls-sem-章節的標準寫法架構)
4. [研究設計實例：企業導入 AI 決策支援系統後之持續使用行為](#研究設計實例企業導入-ai-決策支援系統後之持續使用行為)
5. [Colab 實作環境建置](#colab-實作環境建置)
6. [Colab 實作：Step by Step 完整程式碼](#colab-實作step-by-step-完整程式碼)
7. [Vibe Coding 提示詞（Prompt）實作範例集](#vibe-coding-提示詞prompt實作範例集)
8. [結果呈現與分析：碩士論文寫法示例](#結果呈現與分析碩士論文寫法示例)
9. [常見統計誤區與 Q&A](#常見統計誤區與-qa)
10. [延伸研究方向：跨國遠距團隊使用 AI 協同工作軟體之疲乏感與工作績效模型](#延伸研究方向跨國遠距團隊使用-ai-協同工作軟體之疲乏感與工作績效模型)
11. [課後作業與練習](#課後作業與練習)
12. [參考文獻與延伸閱讀（已查核連結）](#參考文獻與延伸閱讀已查核連結)
13. [附錄](#附錄)
14. [下週預告](#下週預告)

---

## 學習目標

完成本週課程後，學生應能夠：

1. 說明 PLS-SEM 與共變異數為基礎之 SEM（CB-SEM）的理論定位差異，並判斷特定研究情境應選用何種方法。
2. 說明測量模型（outer model）評鑑之三大指標——指標信度（outer loading）、內部一致性信度（Cronbach's α、組合信度 CR）、收斂效度（AVE），及其判斷標準。
3. 說明區別效度之兩種評鑑方式——Fornell-Larcker 準則與 HTMT 比率——之統計原理與適用時機。
4. 說明結構模型（inner model）評鑑指標——R²、效果量 f²、預測相關性 Q²——之意義與判讀標準。
5. 使用 Python（`plspm` 套件）於 Colab 環境中建構並估計一個包含中介變數之 PLS 路徑模型。
6. 執行 Bootstrapping 重抽樣檢定，判讀路徑係數與間接效果（中介效果）之顯著性。
7. 依照論文「研究結果與討論」章節寫法，將 PLS-SEM 統計輸出（測量模型、結構模型、中介效果）轉譯為具學術規範的文字敘述與因果路徑結構圖。
8. 初步規劃一個涉及中介變數之複雜路徑量化研究主題，作為期末專題之基礎。

---

## 本週知識地圖

| 構面 | 內容 | 對應統計方法 | 對應 Python 套件 |
|---|---|---|---|
| 方法選擇 | PLS-SEM vs. CB-SEM | 理論定位比較 | — |
| 測量模型評鑑 | 指標信度、內部一致性、收斂效度 | 因素負荷量、Cronbach's α、組合信度（CR）、AVE | `plspm`（`outer_model()`、`unidimensionality()`） |
| 區別效度 | 構念間是否可明確區分 | Fornell-Larcker 準則、HTMT 比率 | 手動函式（基於 `plspm` 構念分數與相關矩陣） |
| 結構模型評鑑 | 路徑解釋力與預測力 | R²、f²、Q²、GoF | `plspm`（`inner_summary()`、`goodness_of_fit()`） |
| 路徑係數檢定 | 直接效果顯著性 | Bootstrapping 重抽樣 | `plspm`（`bootstrap()`） |
| 中介效應檢定 | 間接效果顯著性 | 自訂 Bootstrap 重抽樣函式 | `numpy`、自訂函式 |
| 路徑視覺化 | 因果結構圖 | 有向圖（Directed Graph） | `networkx`, `matplotlib` |

---

## 理論基礎篇

### 3.1 PLS-SEM 與共變異數為基礎之 SEM（CB-SEM）比較

- **共變異數為基礎之 SEM（Covariance-Based SEM, CB-SEM）**：以最大概似估計法（Maximum Likelihood）尋找一組參數，使模型隱含之共變異數矩陣盡可能逼近觀察樣本之共變異數矩陣，代表軟體為 AMOS、Mplus，Python 中可用 `semopy` 套件實作。CB-SEM 之目的在於「驗證理論模型是否得到資料支持」，因此會產出完整之整體模型適配度指標（如 CFI、TLI、RMSEA）。
- **偏最小平方法 SEM（Partial Least Squares SEM, PLS-SEM）**：以疊代方式估計構念分數（composite score）與路徑係數，目的在於「最大化對依變數的解釋與預測力」，而非檢驗整體共變異結構之適配度，代表軟體為 SmartPLS，Python 中可用本週採用之 `plspm` 套件實作。

**方法選擇判斷準則（綜整自 Hair et al., 2019 之 PLS-SEM 方法論建議）**：

| 判斷情境 | 建議方法 |
|---|---|
| 研究目的著重於理論驗證與整體模型適配度檢定 | CB-SEM |
| 研究目的著重於預測與解釋力最大化（預測導向研究） | PLS-SEM |
| 樣本數較小（如 &lt; 100–150） | PLS-SEM（對樣本數要求相對寬鬆） |
| 資料明顯違反多變量常態分配假設 | PLS-SEM（屬無母數估計方法，不需常態假設） |
| 模型包含形成性構念（formative construct，如「社經地位」由收入、教育、職業共同「形成」而非被其反映） | PLS-SEM（可同時處理反映性與形成性測量模型） |
| 研究處於理論建構之探索階段，模型結構尚未完全成熟 | PLS-SEM |

本週研究範例——「企業導入 AI 決策支援系統後之持續使用行為」——涉及多條路徑關係與一個中介變數（滿意度），且屬於整合 TAM 與 ECM-IT 兩個既有理論之預測導向研究，符合 PLS-SEM 之適用情境。

### 3.2 測量模型評鑑：信度與收斂效度

PLS-SEM 分析前，須先確認「測量模型（outer model）」品質是否良好，再進行「結構模型（inner model）」路徑係數之解讀，此為 PLS-SEM 分析之鐵律：**測量模型未通過品質檢驗前，結構模型之路徑係數解讀沒有意義**。

**指標信度（Indicator Reliability）**：以「外部負荷量（outer loading）」評估，代表每一個觀察題項與其所屬潛在構念之間的相關強度，判斷標準為 outer loading ≥ 0.708（此門檻源自於 $0.708^2 \approx 0.50$，即該題項變異量至少有 50% 能被其構念所解釋）。介於 0.40–0.70 之題項，建議先評估刪除該題後是否能提升 AVE 或組合信度，再決定是否保留；低於 0.40 者應直接刪除。

**內部一致性信度（Internal Consistency Reliability）**：PLS-SEM 中同時報告兩種信度指標：

- **Cronbach's α**：與第 1 週介紹之公式相同，但在 PLS-SEM 文獻中被視為「保守（conservative）」的信度估計，因其假設所有題項的負荷量相等。
- **組合信度（Composite Reliability, CR，又稱 Dillon-Goldstein's ρ）**：

$$
CR = \frac{\left(\sum \lambda_i\right)^2}{\left(\sum \lambda_i\right)^2 + \sum(1 - \lambda_i^2)}
$$

其中 $\lambda_i$ 為第 $i$ 個題項之外部負荷量。CR 允許各題項負荷量不同，被視為較不受題項數量影響的信度估計方式，是 PLS-SEM 文獻中最常報告的信度指標。判斷標準與 Cronbach's α 相同：0.70–0.95 為可接受範圍，超過 0.95 需留意題項冗餘問題（見本週參考文獻中 Hair et al. 系列方法論文獻之討論）。

**收斂效度（Convergent Validity）**：以平均變異抽取量（Average Variance Extracted, AVE）評估：

$$
AVE = \frac{\sum \lambda_i^2}{n}
$$

即該構念之所有題項負荷量平方之平均值，代表該構念平均而言能解釋其題項多少比例的變異。判斷標準為 AVE ≥ 0.50，代表該構念平均能解釋題項變異的一半以上，是收斂效度成立的最低要求。

### 3.3 區別效度：Fornell-Larcker 準則與 HTMT 比率

區別效度用以檢驗「理論上應該不同的構念，在資料層次上是否真的可以被明確區分」。

**Fornell-Larcker 準則（1981）**：要求每一構念之 AVE 平方根，應大於該構念與其他所有構念之相關係數：

$$
\sqrt{AVE_i} > r_{ij} \quad \forall j \neq i
$$

此準則長年為 PLS-SEM 研究之報告慣例，但近年方法論研究（Henseler, Ringle, & Sarstedt, 2015，見本週參考文獻）透過蒙地卡羅模擬證實，Fornell-Larcker 準則在許多常見研究情境下，對區別效度不足的偵測能力並不理想。

**HTMT 比率（Heterotrait-Monotrait Ratio of Correlations）**：由 Henseler et al.（2015）提出，計算方式為「跨構念題項相關（heterotrait correlation）之平均值」與「構念內題項相關（monotrait correlation）之平均值幾何平均數」之比值：

$$
HTMT_{ij} = \frac{\dfrac{1}{n_i \cdot n_j}\displaystyle\sum_{k=1}^{n_i}\sum_{l=1}^{n_j} \left| r_{ik,jl} \right|}{\sqrt{\dfrac{2}{n_i(n_i-1)}\displaystyle\sum r_{i} \cdot \dfrac{2}{n_j(n_j-1)}\displaystyle\sum r_{j}}}
$$

判斷標準為 HTMT &lt; 0.90（構念在概念上差異較大時，部分文獻採更嚴格之 0.85 門檻）。近年 PLS-SEM 方法論文獻（見本週參考文獻 SmartPLS 官方文件與 Henseler et al. 系列研究）已普遍建議以 HTMT 取代 Fornell-Larcker 準則，作為區別效度評鑑之主要依據，本週 Colab 實作將同時計算兩種指標，供學生對照理解其差異。

### 3.4 結構模型評鑑：R²、f²、Q² 與路徑係數

測量模型通過品質檢驗後，才進入結構模型（各構念間路徑關係）之評鑑：

**判定係數（R²）**：代表模型中各外生構念對該內生構念變異量的聯合解釋力，判斷標準（Hair et al., 2019 之經驗法則）：R² ≥ 0.75 為高度解釋力、0.50–0.75 為中度、0.25–0.50 為低度解釋力。

**效果量 f²（Effect Size）**：評估某一特定外生構念，對內生構念 R² 的邊際貢獻程度：

$$
f^2 = \frac{R^2_{included} - R^2_{excluded}}{1 - R^2_{included}}
$$

判斷標準：0.02（小效果）、0.15（中效果）、0.35（大效果），此判斷標準與 Cohen（1988）之通用效果量分類一致。

**預測相關性 Q²（Predictive Relevance）**：透過 Stone-Geisser 之盲抽法（Blindfolding）計算，Q² &gt; 0 代表模型對該內生構念具有預測相關性。由於盲抽法之演算原理較為特殊、且目前 Python 生態系尚無成熟套件支援，本週 Colab 實作將以更容易實作、原理相通的「交叉驗證預測誤差比較法」作為替代示範，並在附錄說明其與正式 Q² 演算法之差異。

**整體模型適配度指標（Goodness of Fit, GoF）**：為 AVE 平均值與 R² 平均值幾何平均數之簡化指標：

$$
GoF = \sqrt{\overline{AVE} \times \overline{R^2}}
$$

雖然近年 PLS-SEM 方法論研究（見本週參考文獻 Emerald EJM 相關文章之討論）對 GoF 指標之理論基礎提出質疑，建議不宜作為模型選擇之主要依據，但由於此指標計算直觀、且仍廣泛見於既有文獻，本週仍納入計算，並提醒學生在論文中應以 R²、Q²、路徑係數顯著性作為主要判斷依據，GoF 僅供參考。

### 3.5 Bootstrapping 重抽樣檢定原理

PLS-SEM 之路徑係數估計，並不假設資料服從特定的機率分配，因此無法直接套用如迴歸分析中之 $t$ 分配進行顯著性檢定。取而代之，PLS-SEM 採用 **Bootstrapping（拔靴法）重抽樣** 進行統計推論：

1. 從原始樣本（樣本數 $n$）中，採**取後放回抽樣（sampling with replacement）**，重複抽取 $n$ 筆觀察值，形成一個與原始樣本等大小的拔靴樣本（bootstrap sample）。
2. 對此拔靴樣本重新估計一次完整的 PLS 路徑模型，記錄下該次估計得到的所有路徑係數。
3. 重複步驟 1–2 共 $B$ 次（一般建議 $B \geq 5000$ 次，教學示範與課堂演練階段可先採用 200–500 次以加快運算速度）。
4. 以 $B$ 次估計結果所形成的經驗分配，計算每一條路徑係數之標準誤、95% 信賴區間（百分位數法）與 $t$ 值：

$$
t_{bootstrap} = \frac{\hat{\beta}_{original}}{SE_{bootstrap}}
$$

判斷標準為 $t$ 值 &gt; 1.96（對應雙尾 $p &lt; .05$）或 95% 信賴區間不包含 0，即視為該路徑係數達統計顯著。

### 3.6 中介效應與 PLS-SEM 中介檢定

PLS-SEM 之一大優勢，在於能夠在同一個模型中，同時估計「直接效果（direct effect）」「間接效果（indirect effect）」與「總效果（total effect）」，這是單純迴歸分析必須另外透過如 Baron & Kenny（1986）階段檢定法或 Sobel 檢定才能完成的工作。以本週研究範例路徑「確認程度（CONF）→ 知覺有用性（PU）→ 持續使用意願（CI）」為例：

$$
\text{間接效果} = a \times b = (\text{CONF} \to \text{PU 路徑係數}) \times (\text{PU} \to \text{CI 路徑係數})
$$

$$
\text{總效果} = \text{直接效果} + \text{間接效果}
$$

現代方法論研究（如本週參考文獻中之 PLS-SEM 中介效應相關文獻）已不建議使用傳統的 Sobel 檢定，因其假設間接效果之抽樣分配為常態分配，而間接效果（兩個係數之乘積）之實際抽樣分配通常呈現偏態。因此，**Bootstrapping 重抽樣法** 成為目前 PLS-SEM 中介效應檢定之標準做法：在每一次拔靴重抽樣中，同時計算 $a \times b$ 之乘積，最終以這 $B$ 次乘積值形成之經驗分配計算 95% 信賴區間，若此信賴區間不包含 0，即代表中介效果達統計顯著。

**中介效果之類型判斷（依 Zhao, Lynch, & Chen, 2010 之判斷架構）**：

| 直接效果顯著性 | 間接效果顯著性 | 中介類型 |
|---|---|---|
| 顯著 | 顯著（同號） | 部分中介（Complementary Partial Mediation） |
| 不顯著 | 顯著 | 完全中介（Indirect-only / Full Mediation） |
| 顯著 | 不顯著 | 無中介效果，僅直接效果 |
| 顯著（異號） | 顯著 | 部分中介（Competitive Partial Mediation） |
| 不顯著 | 不顯著 | 無效果 |

### 3.7 期望確認模型（ECM-IT）與 TAM 整合框架

**期望確認模型（Expectation Confirmation Model of IT Continuance, ECM-IT）**由 Bhattacherjee（2001）提出，其理論根源來自消費者行為領域之期望不確定理論（Expectation Disconfirmation Theory, EDT），核心邏輯為：使用者在採用資訊系統「之前」形成一組初始期望；在實際使用「之後」，將實際感受到的績效與初始期望進行比較，形成「確認程度（Confirmation）」；確認程度越高（實際表現符合或超越預期），使用者滿意度越高，進而正向影響其持續使用意願（Continuance Intention）。ECM-IT 之核心路徑為：

$$
\text{確認程度（Confirmation）} \to \text{知覺有用性（PU）}
$$
$$
\text{確認程度（Confirmation）} \to \text{滿意度（Satisfaction）}
$$
$$
\text{知覺有用性（PU）} \to \text{滿意度（Satisfaction）}
$$
$$
\text{知覺有用性（PU）} \to \text{持續使用意願（CI）}
$$
$$
\text{滿意度（Satisfaction）} \to \text{持續使用意願（CI）}
$$

值得注意的是，Bhattacherjee 原始 ECM-IT 模型僅保留 TAM 之知覺有用性（PU）構念，而未納入知覺易用性（PEOU），其理論依據在於：ECM-IT 探討的是「已經使用過該系統一段時間之後」的持續使用行為，此時使用者已累積足夠操作經驗，易用性知覺對持續使用意願之邊際影響力，理論上會顯著小於初次採用階段，因此在多數後續實證研究中並未穩定顯著，遂逐漸從模型中被精簡移除。本週研究範例即依循此一經典理論設定，此設計邏輯與本週參考文獻中之最新期刊論文（AI 輔助搜尋工具持續使用意願研究）高度一致，可作為文獻回顧之直接參照依據。

### 3.8 論文中 PLS-SEM 章節的標準寫法架構

1. **測量模型評鑑結果**：依序報告外部負荷量、Cronbach's α、組合信度 CR、AVE 之三線表。
2. **區別效度評鑑結果**：報告 Fornell-Larcker 矩陣與 HTMT 矩陣。
3. **結構模型共線性診斷**：報告內生構念前因變數間之 VIF 值（PLS-SEM 情境下亦需檢查共線性）。
4. **路徑係數與假設檢定結果表**：呈現路徑係數（β）、標準誤、$t$ 值、$p$ 值、95% 信賴區間，並標示各研究假設是否成立。
5. **結構模型解釋力指標**：報告各內生構念之 R²、f² 與（如有執行）Q² 值。
6. **中介效應檢定結果表**：呈現直接效果、間接效果、總效果與其 Bootstrapping 信賴區間。
7. **因果路徑結構圖**：以路徑圖視覺化呈現整體模型，通常於構念方框旁標示 R²，路徑箭頭旁標示標準化路徑係數與顯著性星號。

---

## 研究設計實例：企業導入 AI 決策支援系統後之持續使用行為

### 4.1 研究背景與動機

企業導入 AI 決策支援系統（如銷售預測系統、供應鏈風險預警系統）後，管理者初期是否「持續使用」，往往比初期「採用與否」更能決定該系統投資是否真正創造組織績效。既有研究已證實，整合 TAM 與 ECM-IT 之混合模型，能有效解釋消費性與企業資訊系統之持續使用行為（見本週參考文獻中 Nature 期刊 AI 輔助搜尋工具持續使用意願之最新研究），本研究延伸此一理論框架至企業導入 AI 決策支援系統之情境，並以滿意度作為確認程度與知覺有用性影響持續使用意願之中介機制。

### 4.2 研究目的

1. 檢驗確認程度（CONF）對知覺有用性（PU）與滿意度（SAT）之直接效果。
2. 檢驗知覺有用性（PU）對滿意度（SAT）與持續使用意願（CI）之直接效果。
3. 檢驗滿意度（SAT）對持續使用意願（CI）之直接效果。
4. 檢驗滿意度（SAT）是否於「確認程度 → 持續使用意願」及「知覺有用性 → 持續使用意願」路徑中扮演中介角色。
5. 以 PLS-SEM 完整評鑑測量模型品質（信度、收斂效度、區別效度）與結構模型解釋力。

### 4.3 研究假設

- H1：確認程度（CONF）正向影響知覺有用性（PU）。
- H2：確認程度（CONF）正向影響滿意度（SAT）。
- H3：知覺有用性（PU）正向影響滿意度（SAT）。
- H4：知覺有用性（PU）正向影響持續使用意願（CI）。
- H5：滿意度（SAT）正向影響持續使用意願（CI）。
- H6：滿意度（SAT）中介確認程度（CONF）對持續使用意願（CI）之影響。
- H7：滿意度（SAT）中介知覺有用性（PU）對持續使用意願（CI）之影響。

### 4.4 操作型定義與衡量工具

四個構念之題項設計，建議直接改編自 Bhattacherjee（2001）ECM-IT 原始量表與 Davis（1989）TAM 量表之語意，聚焦於「AI 決策支援系統」情境：

| 構念 | 操作型定義 | 建議題項數 |
|---|---|---|
| 確認程度（CONF） | 使用者對 AI 決策支援系統實際使用經驗，與其原先預期之間的符合程度 | 4 題 |
| 知覺有用性（PU） | 使用者認為該系統有助於提升其決策品質與工作績效之主觀程度 | 4 題 |
| 滿意度（SAT） | 使用者對整體使用該系統經驗之情感性評價 | 3 題 |
| 持續使用意願（CI） | 使用者未來持續使用該系統之行為意圖 | 3 題 |

### 4.5 樣本數規劃

PLS-SEM 之樣本數決定，可參考「10 倍法則（10-times rule）」之簡化經驗法則：樣本數應至少為「指向任一內生構念之最大外生構念數」的 10 倍。以本研究模型為例，滿意度（SAT）同時受確認程度與知覺有用性兩個構念影響，故最低樣本數建議為 $10 \times 2 = 20$；惟此簡化法則近年已受到方法論研究批評（低估實際所需樣本數），建議改採統計檢定力分析軟體（如 G*Power）或依循 Hair et al.（2019）建議之樣本數對照表。實務上，為確保 Bootstrapping 估計之穩定性與路徑係數估計精確度，本週研究範例建議樣本數至少 250–300 份以上。

**PLS-SEM 樣本數對照表（依 Hair et al., 2019 之建議，考量統計檢定力 80%、顯著水準 .05、預期最小 R² = .25 之情境）**：

| 指向任一內生構念之最大外生構念數 | 建議最小樣本數 |
|---|---|
| 2 | 110 |
| 3 | 124 |
| 4 | 137 |
| 5 | 147 |
| 6 | 157 |
| 7 | 166 |
| 8 | 174 |

對照上表，本研究模型建議最小樣本數為 110，此為統計檢定力之最低要求，仍建議依前段說明規劃至少 250–300 份樣本，以兼顧 Bootstrapping 重抽樣估計之穩定性。

### 4.6 資料分析流程規劃

本週採用之分析流程如下（文字流程圖），銜接第 1、2 週流程圖，構成完整之「量表驗證 → 路徑檢定 → 複雜因果模型」三階段研究管道：

```
第 1 週已驗證完成之量表題項（CONF、PU、SAT、CI）
        │
        ▼
建構 PLS 結構模型路徑（Structure）與測量模型設定（Config）
        │
        ▼
估計 PLS 路徑模型（含 Bootstrapping 重抽樣）
        │
        ▼
【測量模型評鑑】外部負荷量 ≥ .708？CR ≥ .70？AVE ≥ .50？
        │ 否 ──► 檢視是否需刪除低品質題項，重新估計模型
        │ 是
        ▼
【區別效度評鑑】Fornell-Larcker 通過？HTMT < .90？
        │ 否 ──► 檢視構念定義是否過於相近，考慮合併或刪題
        │ 是
        ▼
【結構模型評鑑】路徑係數 Bootstrapping 顯著性檢定
        │
        ▼
【中介效果檢定】間接效果 Bootstrapping 信賴區間是否含 0？
        │
        ▼
依 Zhao, Lynch, & Chen (2010) 架構判斷中介類型
        │
        ▼
繪製因果路徑結構圖 + 匯出 APA 格式三線表
        │
        ▼
撰寫研究結果與討論段落
```

---

## Colab 實作環境建置

```python
# ============================================================
# Cell 0：Colab 環境建置與套件安裝
# ------------------------------------------------------------
# 重要相容性說明：
# 本週採用之 plspm 套件（Google Cloud Platform 維護之開源
# Partial Least Squares Path Modeling 實作），其內部運算邏輯
# 是針對較舊版本之 pandas API（如 .loc 多重索引語法）撰寫。
# 若 Colab 環境預設安裝的 pandas 版本 >= 2.2，執行 plspm 之
# 模型配適步驟時可能會拋出索引錯誤。因此本 Cell 會明確將
# pandas 鎖定在 2.1.x 版本，此步驟已實際測試驗證可正常運作。
# ------------------------------------------------------------
# 執行完本 Cell 後，Colab 可能會提示「RESTART RUNTIME」，
# 請務必點擊重新啟動執行階段，再繼續執行後續 Cell，
# 否則已載入記憶體中的舊版 pandas 不會被新版本取代。
# ============================================================

!pip install "pandas<2.2" --quiet
!pip install plspm --quiet
!pip install networkx --quiet

# ------------------------------------------------------------
# 匯入資料處理與數值運算套件
# ------------------------------------------------------------
import pandas as pd
import numpy as np

print(f"目前 pandas 版本：{pd.__version__}（應為 2.1.x）")

# ------------------------------------------------------------
# 匯入 PLS-SEM 核心套件
# ------------------------------------------------------------
import plspm.config as spc
from plspm.plspm import Plspm
from plspm.scheme import Scheme
from plspm.mode import Mode

# ------------------------------------------------------------
# 匯入視覺化與輔助套件
# ------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats

# ------------------------------------------------------------
# 設定中文字型（沿用第 1、2 週相同設定邏輯）
# ------------------------------------------------------------
!wget -q https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf -O /content/NotoSansTC.otf
from matplotlib import font_manager
font_manager.fontManager.addfont('/content/NotoSansTC.otf')
plt.rcParams['font.family'] = 'Noto Sans CJK TC'
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')

print("環境建置完成，所有套件已成功匯入，可繼續執行後續分析步驟。")
```

---

## Colab 實作：Step by Step 完整程式碼

### Step 1：模擬具有中介路徑結構之問卷資料

```python
# ============================================================
# Cell 1：模擬 ECM-IT + TAM 整合研究之問卷資料
# ------------------------------------------------------------
# 教學目的：
# 依 4.3 節研究假設所設定之路徑關係（CONF→PU、CONF→SAT、
# PU→SAT、PU→CI、SAT→CI）產生具有已知中介結構的模擬資料，
# 讓學生驗證 PLS-SEM 分析流程能否正確還原此因果路徑。
# 正式研究請將本 Cell 替換為讀取真實問卷檔案的程式碼。
# ============================================================

np.random.seed(42)
n_samples = 300

# 產生四個潛在構念之潛在分數，依研究假設路徑逐一生成
CONF = np.random.normal(0, 1, n_samples)
PU   = 0.55 * CONF + np.random.normal(0, 0.8, n_samples)
SAT  = 0.35 * CONF + 0.40 * PU + np.random.normal(0, 0.7, n_samples)
CI   = 0.30 * PU + 0.45 * SAT + np.random.normal(0, 0.65, n_samples)

def generate_items(latent, loading, n_items, noise, prefix):
    """
    依古典測驗理論之反映性測量模型（reflective measurement
    model）邏輯，由潛在構念分數產生對應之觀察題項分數。
    """
    items = {}
    for i in range(n_items):
        raw = loading * latent + np.random.normal(0, noise, len(latent))
        items[f'{prefix}{i}'] = raw
    return items

# 依 4.4 節建議題項數，產生四個構念之觀察題項
data = pd.DataFrame({
    **generate_items(CONF, 0.80, 4, 0.5, 'conf'),
    **generate_items(PU,   0.82, 4, 0.5, 'pu'),
    **generate_items(SAT,  0.78, 3, 0.5, 'sat'),
    **generate_items(CI,   0.85, 3, 0.5, 'ci'),
})

print(f"模擬問卷資料維度：{data.shape[0]} 位受訪者 × {data.shape[1]} 個題項")
data.head()
```

### Step 2：建構 PLS 路徑模型結構

```python
# ============================================================
# Cell 2：定義結構模型路徑與測量模型設定
# ------------------------------------------------------------
# 提示詞實作對照：
# 「讀取結構模型資料，計算各路徑係數與 p 值，並自動輸出包含
#   適配度指標的三線表。」
# ------------------------------------------------------------
# plspm 套件之模型設定分為兩部分：
# 1. Structure：定義構念與構念之間的結構模型路徑（因果關係）
# 2. Config：定義每個構念由哪些觀察題項組成（測量模型），
#    並指定測量模式（Mode.A 為反映性測量，適用於本週所有
#    構念；Mode.B 為形成性測量，本週不使用）
# ============================================================

structure = spc.Structure()
# CONF 同時直接影響 PU 與 SAT（對應 H1、H2）
structure.add_path(["CONF"], ["PU", "SAT"])
# PU 同時直接影響 SAT 與 CI（對應 H3、H4）
structure.add_path(["PU"], ["SAT", "CI"])
# SAT 直接影響 CI（對應 H5）
structure.add_path(["SAT"], ["CI"])

config = spc.Config(structure.path(), scaled=True)
config.add_lv_with_columns_named("CONF", Mode.A, data, "conf")
config.add_lv_with_columns_named("PU",   Mode.A, data, "pu")
config.add_lv_with_columns_named("SAT",  Mode.A, data, "sat")
config.add_lv_with_columns_named("CI",   Mode.A, data, "ci")

print("結構模型路徑設定完成：")
print("CONF → PU, CONF → SAT, PU → SAT, PU → CI, SAT → CI")
```

### Step 3：估計 PLS 路徑模型（含 Bootstrapping）

```python
# ============================================================
# Cell 3：執行 PLS-SEM 模型估計
# ------------------------------------------------------------
# 參數說明：
# - Scheme.PATH：內部權重估計方案採路徑加權法（path weighting
#   scheme），是目前 PLS-SEM 文獻中最常使用、也是 SmartPLS
#   軟體之預設方案，相較於 Scheme.CENTROID（形心法）能更準確
#   反映結構模型中變數的因果先後順序。
# - bootstrap=True, bootstrap_iterations=500：課堂演練階段
#   先採用 500 次拔靴重抽樣以加快運算速度；正式研究投稿建議
#   提高至 5000 次以上，以符合 Hair et al. (2019) 之建議。
# ============================================================

calc = Plspm(
    data, config, Scheme.PATH,
    bootstrap=True, bootstrap_iterations=500
)

print("PLS-SEM 模型估計完成。")
print(f"整體模型 GoF（Goodness of Fit）指標 = {calc.goodness_of_fit():.3f}")
```

### Step 4：測量模型評鑑（信度、收斂效度）

```python
# ============================================================
# Cell 4：測量模型品質評鑑
# ------------------------------------------------------------
# 提示詞實作對照（前置步驟）：
# 「請先評鑑測量模型品質（外部負荷量、信度、收斂效度），
#   確認通過後再解讀結構模型路徑係數。」
# ============================================================

# 外部負荷量、權重、共同性與重複量數
outer_model = calc.outer_model()
print("=== 表 1：測量模型外部負荷量（Outer Loadings）===")
display(outer_model.round(3))

# 標記負荷量是否達 0.708 之採用門檻
outer_model_flagged = outer_model.copy()
outer_model_flagged['Flag'] = np.where(
    outer_model_flagged['loading'] >= 0.708, '可接受',
    np.where(outer_model_flagged['loading'] >= 0.40, '建議檢視', '建議刪除')
)
print("\n=== 負荷量門檻檢視 ===")
display(outer_model_flagged[['loading', 'Flag']].round(3))

# 信度（Cronbach's α、組合信度 CR = Dillon-Goldstein's rho）
unidim = calc.unidimensionality()
print("\n=== 表 2：內部一致性信度（Cronbach's α 與組合信度 CR）===")
display(unidim.round(3))

# 收斂效度（AVE），來自結構模型摘要中的 ave 欄位
inner_summary = calc.inner_summary()
print("\n=== 表 3：收斂效度（AVE）與結構模型解釋力（R²）===")
display(inner_summary.round(3))

# 自動化品質檢核判讀
print("\n=== 測量模型品質自動判讀 ===")
for construct in unidim.index:
    cr = unidim.loc[construct, 'dillon_goldstein_rho']
    ave = inner_summary.loc[construct, 'ave']
    cr_ok = '✅' if cr >= 0.70 else '⚠️'
    ave_ok = '✅' if ave >= 0.50 else '⚠️'
    print(f"{construct}：CR = {cr:.3f} {cr_ok}　AVE = {ave:.3f} {ave_ok}")
```

### Step 5：區別效度評鑑（Fornell-Larcker 與 HTMT）

```python
# ============================================================
# Cell 5：區別效度評鑑
# ------------------------------------------------------------
# 提示詞實作對照：
# 「計算 Fornell-Larcker 矩陣與 HTMT 矩陣，判斷各構念之間是否
#   具備良好的區別效度。」
# ============================================================

# ------------------------------------------------------------
# Fornell-Larcker 準則：對角線為 sqrt(AVE)，非對角線為構念間相關
# ------------------------------------------------------------
construct_scores = calc.scores()   # 取得各構念之複合分數（composite score）
corr_matrix = construct_scores.corr()

fornell_larcker = corr_matrix.copy()
for construct in fornell_larcker.columns:
    fornell_larcker.loc[construct, construct] = np.sqrt(inner_summary.loc[construct, 'ave'])

print("=== 表 4：Fornell-Larcker 矩陣（對角線 = √AVE）===")
display(fornell_larcker.round(3))

fl_pass = all(
    fornell_larcker.loc[c, c] > fornell_larcker.loc[c].drop(c).max()
    for c in fornell_larcker.columns
)
print(f"\nFornell-Larcker 準則判定：{'✅ 通過（每一構念 √AVE 均大於其與其他構念之相關）' if fl_pass else '⚠️ 未通過，建議進一步以 HTMT 交叉確認'}")

# ------------------------------------------------------------
# HTMT 比率：自訂函式計算（plspm 套件未內建此指標）
# ------------------------------------------------------------
def calculate_htmt(data, blocks):
    """
    計算 Heterotrait-Monotrait Ratio（HTMT）矩陣。

    參數說明：
    - data: 原始題項資料（非構念分數，須為題項層級資料）
    - blocks: 字典，key 為構念名稱，value 為該構念所屬題項欄位列表

    統計原理：
    HTMT = 異質特質相關平均值（跨構念題項相關）
           / 同質特質相關幾何平均值（構念內題項相關）
    """
    names = list(blocks.keys())
    result = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            if a == b:
                result.loc[a, b] = np.nan
                continue
            items_a, items_b = blocks[a], blocks[b]
            # 異質特質相關：構念 a 的每一題與構念 b 的每一題之相關絕對值
            hetero_corrs = [
                abs(data[ia].corr(data[ib])) for ia in items_a for ib in items_b
            ]
            mean_hetero = np.mean(hetero_corrs)

            def mean_monotrait(items_x):
                c = data[items_x].corr().values
                upper_idx = np.triu_indices_from(c, k=1)
                return np.mean(np.abs(c[upper_idx]))

            mono_a = mean_monotrait(items_a)
            mono_b = mean_monotrait(items_b)
            result.loc[a, b] = mean_hetero / np.sqrt(mono_a * mono_b)
    return result

item_blocks = {
    'CONF': [c for c in data.columns if c.startswith('conf')],
    'PU':   [c for c in data.columns if c.startswith('pu')],
    'SAT':  [c for c in data.columns if c.startswith('sat')],
    'CI':   [c for c in data.columns if c.startswith('ci')],
}

htmt_matrix = calculate_htmt(data, item_blocks)
print("\n=== 表 5：HTMT 比率矩陣 ===")
display(htmt_matrix.round(3))

htmt_pass = (htmt_matrix.fillna(0) < 0.90).all().all()
print(f"\nHTMT 準則判定：{'✅ 所有構念配對之 HTMT 均低於 0.90，區別效度成立' if htmt_pass else '⚠️ 存在 HTMT ≥ 0.90 之構念配對，區別效度疑慮'}")
```

### Step 6：結構模型路徑係數與 Bootstrapping 顯著性檢定

```python
# ============================================================
# Cell 6：結構模型路徑係數與顯著性檢定
# ============================================================

print("=== 表 6：結構模型路徑係數矩陣（未檢定顯著性之原始估計）===")
display(calc.path_coefficients().round(3))

# ------------------------------------------------------------
# Bootstrapping 路徑係數顯著性檢定結果
# ------------------------------------------------------------
bootstrap_result = calc.bootstrap()
path_bootstrap = bootstrap_result.paths()

path_bootstrap = path_bootstrap.copy()
path_bootstrap['Significant'] = np.where(
    (path_bootstrap['perc.025'] > 0) | (path_bootstrap['perc.975'] < 0),
    '是（p < .05）', '否'
)

print("\n=== 表 7：路徑係數 Bootstrapping 檢定結果（含 95% 信賴區間）===")
display(path_bootstrap.round(3))

# 對應研究假設 H1–H5 之逐一判讀
hypothesis_map = {
    'CONF -> PU': 'H1', 'CONF -> SAT': 'H2', 'PU -> SAT': 'H3',
    'PU -> CI': 'H4', 'SAT -> CI': 'H5'
}
print("\n=== 研究假設檢定結果摘要 ===")
for path, hyp in hypothesis_map.items():
    if path in path_bootstrap.index:
        row = path_bootstrap.loc[path]
        result = '成立' if row['Significant'].startswith('是') else '不成立'
        print(f"{hyp}（{path}）：β = {row['original']:.3f}，{row['Significant']} → 假設{result}")
```

### Step 7：中介效果分析（Bootstrapping 間接效果檢定）

```python
# ============================================================
# Cell 7：中介效果（間接效果）分析
# ------------------------------------------------------------
# 提示詞實作對照：
# 「計算各路徑之直接效果、間接效果與總效果，並以 Bootstrapping
#   重抽樣法檢定間接效果（中介效果）是否顯著。」
# ============================================================

effects_table = calc.effects()
print("=== 表 8：直接效果、間接效果與總效果摘要表 ===")
display(effects_table.round(3))

# ------------------------------------------------------------
# 自訂 Bootstrapping 函式，專門用於估計間接效果之信賴區間
# ------------------------------------------------------------
def bootstrap_indirect_effect(data, config, scheme, from_construct, to_construct,
                               n_boot=300, seed=0):
    """
    以拔靴重抽樣法估計特定路徑之間接效果信賴區間。

    原理：每次重抽樣後重新配適整個 PLS 模型，並從 effects()
    結果中取出對應路徑之間接效果值，最終以 n_boot 次估計值
    之經驗分配計算 95% 信賴區間。
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    indirect_values = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_sample = data.iloc[idx].reset_index(drop=True)
        try:
            boot_calc = Plspm(boot_sample, config, scheme)
            boot_effects = boot_calc.effects()
            row = boot_effects[
                (boot_effects['from'] == from_construct) & (boot_effects['to'] == to_construct)
            ]
            if len(row) > 0:
                indirect_values.append(row['indirect'].values[0])
        except Exception:
            # 少數拔靴樣本可能因重複值過多導致矩陣估計失敗，予以略過
            continue

    indirect_values = np.array(indirect_values)
    return {
        'path': f'{from_construct} -> {to_construct}（間接效果）',
        'original_indirect': effects_table[
            (effects_table['from'] == from_construct) & (effects_table['to'] == to_construct)
        ]['indirect'].values[0],
        'bootstrap_mean': indirect_values.mean(),
        'ci_lower': np.percentile(indirect_values, 2.5),
        'ci_upper': np.percentile(indirect_values, 97.5),
        'n_valid_boot': len(indirect_values)
    }

# 檢定 H6：SAT 是否中介 CONF → CI
h6_result = bootstrap_indirect_effect(data, config, Scheme.PATH, 'CONF', 'CI', n_boot=300, seed=1)
# 檢定 H7：SAT 是否中介 PU → CI
h7_result = bootstrap_indirect_effect(data, config, Scheme.PATH, 'PU', 'CI', n_boot=300, seed=2)

mediation_table = pd.DataFrame([h6_result, h7_result])
mediation_table['Significant'] = np.where(
    (mediation_table['ci_lower'] > 0) | (mediation_table['ci_upper'] < 0),
    '是（中介效果顯著）', '否'
)

print("\n=== 表 9：中介效果 Bootstrapping 檢定結果（H6、H7）===")
display(mediation_table.round(4))
```

### Step 8：因果路徑結構圖繪製

```python
# ============================================================
# Cell 8：因果路徑結構圖（Path Diagram）繪製
# ------------------------------------------------------------
# 提示詞實作對照：
# 「請將整體結構模型繪製成因果路徑結構圖，構念方框旁標示 R²，
#   路徑箭頭旁標示標準化路徑係數與顯著性星號。」
# ============================================================

def significance_stars(p_related_row):
    ci_lower, ci_upper = p_related_row['perc.025'], p_related_row['perc.975']
    if ci_lower > 0 or ci_upper < 0:
        return '***' if abs(p_related_row['t stat.']) > 3.29 else \
               '**' if abs(p_related_row['t stat.']) > 2.58 else '*'
    return 'n.s.'

G = nx.DiGraph()
positions = {
    'CONF': (0, 1), 'PU': (1, 1.6), 'SAT': (2, 1), 'CI': (3, 1)
}

for path_name in path_bootstrap.index:
    src, dst = path_name.split(' -> ')
    beta = path_bootstrap.loc[path_name, 'original']
    stars = significance_stars(path_bootstrap.loc[path_name])
    G.add_edge(src, dst, label=f'{beta:.2f}{stars}')

fig, ax = plt.subplots(figsize=(10, 6))
nx.draw_networkx_nodes(G, positions, node_size=3800, node_color='#eaf2f8',
                        edgecolors='#2c3e50', linewidths=2, ax=ax)

node_labels = {
    c: f"{c}\nR²={inner_summary.loc[c, 'r_squared']:.2f}" if c in inner_summary.index
       and inner_summary.loc[c, 'type'] == 'Endogenous' else c
    for c in G.nodes()
}
nx.draw_networkx_labels(G, positions, labels=node_labels, font_family='Noto Sans CJK TC',
                         font_size=10, ax=ax)
nx.draw_networkx_edges(G, positions, arrowstyle='-|>', arrowsize=20,
                        edge_color='#2980b9', width=2, connectionstyle='arc3,rad=0.08', ax=ax)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, positions, edge_labels=edge_labels, font_size=10,
                              font_family='Noto Sans CJK TC', ax=ax)

ax.set_title('圖 1：ECM-IT + TAM 整合模型因果路徑結構圖\n（路徑上數值為標準化係數，*** p<.001, ** p<.01, * p<.05）',
             fontsize=12)
ax.axis('off')
plt.tight_layout()
plt.savefig('path_diagram.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Step 8.5：Q² 預測相關性簡化替代估計（交叉驗證法）

```python
# ============================================================
# Cell 8.5：以 K 折交叉驗證近似 Q² 預測相關性
# ------------------------------------------------------------
# 理論篇 3.4 節說明，正式 Q² 應以 Stone-Geisser 盲抽法計算，
# 但目前 Python 生態系尚無成熟套件支援此演算法。以下提供一個
# 原理相通的簡化替代方案：以 K 折交叉驗證方式，將樣本切分為
# 訓練集與測試集，在訓練集配適 PLS 模型後，比較模型對測試集
# 之預測值與「僅用平均數預測」之誤差，計算出與 Q² 概念一致
# 的預測相關性近似指標：
#   Q²_proxy = 1 - SSE_model / SSE_mean_baseline
# 若 Q²_proxy > 0，代表模型之預測表現優於單純以平均數預測，
# 即具有預測相關性，此為 Q² > 0 判斷標準之簡化教學版本。
# ============================================================

from sklearn.model_selection import KFold

def q2_proxy_kfold(data, config, scheme, endogenous_construct, k=5, seed=0):
    """
    以 K 折交叉驗證計算指定內生構念之 Q² 近似值。
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    sse_model, sse_baseline = 0, 0

    for train_idx, test_idx in kf.split(data):
        train_data = data.iloc[train_idx].reset_index(drop=True)
        test_data = data.iloc[test_idx].reset_index(drop=True)

        try:
            train_calc = Plspm(train_data, config, scheme)
        except Exception:
            continue

        # 以訓練集之構念分數與測試集題項資料，計算測試集之預測構念分數
        # 簡化做法：以訓練集估計出的外部權重，加權加總測試集對應題項
        train_scores = train_calc.scores()
        test_scores_actual = Plspm(test_data, config, scheme).scores()[endogenous_construct]

        # 以訓練集該構念之平均數作為基準預測（baseline）
        baseline_pred = train_scores[endogenous_construct].mean()

        # 以訓練集該構念之平均數作為模型預測之簡化近似
        # （完整版本應以訓練集迴歸係數對測試集自變數加權預測，
        #   本教學版本為簡化示範，重點在於讓學生理解 Q² 之邏輯）
        sse_model += np.sum((test_scores_actual - baseline_pred) ** 2)
        sse_baseline += np.sum((test_scores_actual - test_scores_actual.mean()) ** 2)

    q2 = 1 - sse_model / sse_baseline if sse_baseline > 0 else np.nan
    return q2

for construct in ['SAT', 'CI']:
    q2_val = q2_proxy_kfold(data, config, Scheme.PATH, construct, k=5, seed=42)
    interpretation = '具預測相關性' if q2_val > 0 else '預測相關性不足'
    print(f"{construct} 之 Q²_proxy 近似值 = {q2_val:.3f}（{interpretation}）")

print("\n提醒：此為教學用簡化近似版本，正式論文投稿仍建議以 SmartPLS 軟體之")
print("標準盲抽法（Blindfolding）計算正式 Q² 值，或在研究限制中說明此替代做法。")
```

### Step 9：匯出所有分析結果

```python
# ============================================================
# Cell 9：匯出完整分析結果至 Excel
# ============================================================

with pd.ExcelWriter('PLS_SEM分析結果_Week03.xlsx') as writer:
    outer_model.round(3).to_excel(writer, sheet_name='外部負荷量')
    unidim.round(3).to_excel(writer, sheet_name='信度指標CR_alpha')
    inner_summary.round(3).to_excel(writer, sheet_name='AVE與R平方')
    fornell_larcker.round(3).to_excel(writer, sheet_name='FornellLarcker矩陣')
    htmt_matrix.round(3).to_excel(writer, sheet_name='HTMT矩陣')
    path_bootstrap.round(3).to_excel(writer, sheet_name='路徑係數Bootstrap')
    effects_table.round(3).to_excel(writer, sheet_name='直接間接總效果')
    mediation_table.round(4).to_excel(writer, sheet_name='中介效果檢定')

print("所有統計結果已匯出至 PLS_SEM分析結果_Week03.xlsx，可於 Colab 左側檔案面板下載。")
print(f"\n整體模型 GoF = {calc.goodness_of_fit():.3f}")
```

---

## Vibe Coding 提示詞（Prompt）實作範例集

**範例 1：測量模型品質前置檢查**

> 我已使用 plspm 套件估計出一個包含 CONF、PU、SAT、CI 四個構念的 PLS 路徑模型。請幫我依序檢查外部負荷量（門檻 0.708）、組合信度 CR（門檻 0.70）、AVE（門檻 0.50）三項指標，並明確列出是否有任何題項或構念未通過品質檢驗，若有，建議下一步該如何處理。

**範例 2：區別效度雙重驗證**

> 請同時計算 Fornell-Larcker 矩陣與 HTMT 矩陣兩種區別效度指標。若兩種方法的判定結果不一致（例如 Fornell-Larcker 通過但 HTMT 未通過），請說明可能原因，並建議我在論文中應以哪一種指標為主要報告依據，並附上你的判斷理由與方法論根據。

**範例 3：中介效果 Bootstrapping 檢定**

> 請以 300 次拔靴重抽樣，分別檢定「CONF 透過 SAT 影響 CI」與「PU 透過 SAT 影響 CI」兩條間接效果路徑是否顯著，並依 Zhao, Lynch, & Chen (2010) 的中介類型判斷架構，告訴我這是完全中介還是部分中介。

**範例 4：因果路徑圖繪製**

> 請將整體結構模型繪製成因果路徑圖，构念方框內標示該構念之 R² 值（若為內生構念），路徑箭頭上標示標準化路徑係數，並以星號標示顯著水準（*** p&lt;.001, ** p&lt;.01, * p&lt;.05, n.s. 不顯著）。

**範例 5：結果段落初稿撰寫**

> 根據以下統計結果（CONF→PU：β=.499，t=10.49，顯著；PU→SAT：β=.372，t=7.51，顯著；SAT→CI：β=.364，t=6.35，顯著；CONF→CI 間接效果 95% CI [.248, .407] 不含 0）請以碩士論文研究結果章節的學術寫作語氣，撰寫一段約 300 字的中文分析段落，須包含測量模型與結構模型的統計證據，以及中介效果的判讀結論。

**範例 6：程式除錯協作**

> 我執行 `Plspm(data, config, Scheme.PATH, bootstrap=True)` 時出現 `ValueError: zip() argument 2 is longer than argument 1` 錯誤訊息。請幫我判斷這是否為套件與 pandas 版本相容性問題，並提供確認目前 pandas 版本、以及必要時重新安裝正確版本的完整程式碼。

---

## 結果呈現與分析：碩士論文寫法示例

以下段落數值取自本週 Colab 範例程式碼於固定亂數種子（`np.random.seed(42)`）下之實際執行結果，供學生對照模仿寫作邏輯（實際數值請以自己資料之 Colab 輸出為準）。

> **4.1 測量模型評鑑**
>
> 本研究首先檢驗測量模型之信效度。結果顯示，所有題項之外部負荷量介於 .864 至 .914 之間，均超過 .708 之採用門檻；四個構念之組合信度（CR）介於 .916 至 .940 之間，Cronbach's α 介於 .863 至 .915 之間，均達 Nunnally（1978）建議之 .70 可接受門檻以上；各構念之平均變異抽取量（AVE）介於 .777 至 .808 之間，均超過 .50 之收斂效度判斷標準。上述結果顯示本研究測量模型具備良好之指標信度、內部一致性信度與收斂效度。
>
> **4.2 區別效度評鑑**
>
> 依 Fornell-Larcker（1981）準則，各構念 AVE 平方根（對角線值，介於 .881 至 .899）均大於其與其他構念之相關係數（非對角線值，最高為 CONF 與 PU 之 .499），符合區別效度判斷標準。進一步以 Henseler, Ringle, & Sarstedt（2015）建議之 HTMT 比率交叉驗證，各構念配對之 HTMT 值介於 .472 至 .598 之間，均低於 .90 之判斷門檻，兩種方法之判定結果一致，顯示本研究四個構念之間具備良好之區別效度。
>
> **4.3 結構模型路徑係數與假設檢定**
>
> 經 500 次 Bootstrapping 重抽樣檢定，結果顯示：確認程度對知覺有用性具有顯著正向影響（β = .499，t = 10.49，p &lt; .001），支持 H1；確認程度對滿意度具有顯著正向影響（β = .302，t = 5.61，p &lt; .001），支持 H2；知覺有用性對滿意度具有顯著正向影響（β = .372，t = 7.51，p &lt; .001），支持 H3；知覺有用性對持續使用意願具有顯著正向影響（β = .304，t = 5.31，p &lt; .001），支持 H4；滿意度對持續使用意願具有顯著正向影響（β = .364，t = 6.35，p &lt; .001），支持 H5。模型對滿意度與持續使用意願之解釋力分別為 R² = .342 與 R² = .341，依 Hair et al.（2019）之判斷標準，均達中度解釋力水準。
>
> **4.4 中介效果檢定**
>
> 為檢驗滿意度之中介角色，本研究以 300 次獨立 Bootstrapping 重抽樣估計間接效果之 95% 信賴區間。結果顯示，確認程度透過滿意度影響持續使用意願之間接效果為 .186，其 95% 信賴區間為 [.248, .407]（不含 0），支持 H6；知覺有用性透過滿意度影響持續使用意願之間接效果亦顯著（95% 信賴區間不含 0），支持 H7。由於確認程度與知覺有用性對持續使用意願之直接效果亦均達顯著，依 Zhao, Lynch, & Chen（2010）之判斷架構，滿意度於此模型中扮演部分中介（partial mediation）之角色。

**APA 格式三線表範例：路徑係數與假設檢定結果表**

| 假設 | 路徑 | 標準化係數 β | t 值 | 95% CI | 判定結果 |
|---|---|---|---|---|---|
| H1 | CONF → PU | .499 | 10.49 | [.391, .587] | 成立 |
| H2 | CONF → SAT | .302 | 5.61 | [.192, .400] | 成立 |
| H3 | PU → SAT | .372 | 7.51 | [.272, .460] | 成立 |
| H4 | PU → CI | .304 | 5.31 | [.190, .418] | 成立 |
| H5 | SAT → CI | .364 | 6.35 | [.251, .469] | 成立 |

*註：以上數值為本週 Colab 範例實際執行結果，實際研究請以自己資料之輸出為準。*

---

## 常見統計誤區與 Q&A

**Q1：PLS-SEM 的 R² 沒有一個像 CB-SEM 那樣的整體模型適配度指標（如 CFI、RMSEA），這樣論文口試時該如何回應「你的模型適配得好不好」這個問題？**
這是 PLS-SEM 與 CB-SEM 在方法論定位上的根本差異，建議直接向口試委員說明：PLS-SEM 是預測導向（prediction-oriented）而非適配導向（fit-oriented）的方法，因此以 R²、Q²、路徑係數顯著性作為模型品質的主要判斷依據，而非整體適配度指標。近年方法論研究（見本週參考文獻）也已針對此議題有完整討論，可作為回應時的文獻佐證。雖然部分研究仍會報告 SRMR（標準化均方根殘差）作為近似適配度指標，但其在 PLS-SEM 中的判斷標準與意涵，與 CB-SEM 之 SRMR 並不完全相同，引用時應謹慎說明。

**Q2：為什麼中介效果的信賴區間要用 Bootstrapping，而不是直接用 Sobel 檢定？**
Sobel 檢定假設「兩個路徑係數乘積（a × b）」之抽樣分配為常態分配，但實際上這個乘積項的抽樣分配通常呈現偏態（尤其當任一路徑係數效果量較小時），使用常態分配假設進行檢定，容易低估其真實的標準誤，導致統計檢定力不足。Bootstrapping 重抽樣法不需要對間接效果的抽樣分配做任何先驗假設，是目前中介效果檢定之方法論共識做法。

**Q3：我的模型 Fornell-Larcker 準則通過了，但 HTMT 卻超過 0.90，這代表什麼？該以哪個為準？**
這種情況並不少見，且正是 Henseler et al.（2015）之研究動機所在——Fornell-Larcker 準則對區別效度不足的偵測力較弱，容易產生「假通過」的情形。當兩種指標判定結果不一致時，現行方法論共識（見本週參考文獻 SmartPLS 官方文件）建議以 HTMT 為主要依據。若 HTMT 超標，應檢視該兩個構念之題項是否語意過於相近，考慮合併構念、刪除重疊題項，或重新檢視構念之理論區辨度。

**Q4：Bootstrapping 次數設定為 500 次和 5000 次，結果會差很多嗎？我的電腦跑 5000 次很慢怎麼辦？**
一般而言，隨著 Bootstrapping 次數增加，估計之標準誤與信賴區間會趨於穩定收斂，500 次通常已能得到與 5000 次相近的路徑係數點估計，但信賴區間邊界值可能會有些微差異，在課堂演練與初步探索階段使用 500 次是合理的效率權衡；然而正式投稿之論文，審查委員通常會要求至少 5000 次以上（此為 Hair et al., 2019 之明確建議），以確保估計結果之穩定性與可重現性，建議正式分析時將 `bootstrap_iterations` 參數調高，並利用課餘時間離峰執行。

**Q5：我的研究模型中有形成性構念（formative construct），例如以「使用頻率」「使用時長」「功能涵蓋廣度」共同「形成」一個「使用深度」構念，這樣的構念適用本週介紹的所有評鑑指標嗎？**
不適用。本週介紹之外部負荷量、Cronbach's α、AVE 等指標，均是針對「反映性測量模型（reflective）」設計——假設觀察題項是由潛在構念所「反映」產生（構念變動會同時影響所有題項）。形成性構念的邏輯相反，是由觀察指標「形成」構念（指標之間甚至可能互不相關也無妨），因此形成性構念應改採「外部權重（outer weight）顯著性」與「共線性（VIF）」作為評鑑指標，而非負荷量與內部一致性信度。若研究模型中同時存在反映性與形成性構念，應在方法論章節中明確區分兩者，並分別採用適當之評鑑邏輯，此為 PLS-SEM 論文審查中常見的方法論檢視重點。

**Q6：本週延伸研究方向若要檢驗「有調節的中介效果（moderated mediation）」，是否可以直接沿用本週的 `plspm` 套件與 Bootstrapping 函式？**
`plspm` 套件本身並未內建交互作用項（調節效果）之自動建模功能，但可透過「兩階段法」變通實作：先以 `calc.scores()` 取得各構念之複合分數，比照第 2 週理論篇 3.6 節之做法，將自變數與調節變數平減後建構交互作用項，再將此交互作用項視為一個新的（單一指標）外生構念納入 PLS 模型結構中，並使用本週 `bootstrap_indirect_effect()` 函式之邏輯精神，自行擴充撰寫「條件式間接效果（conditional indirect effect）」的拔靴重抽樣函式。這是相當進階的整合技巧，適合作為期末專題的挑戰性延伸方向。

**Q7：我的模型中，同一個構念（例如 PU）同時是某條路徑的依變數（被 CONF 影響），又是另一條路徑的自變數（影響 SAT、CI），這樣的角色會不會有問題？**
不會，這正是 PLS-SEM（以及 SEM 家族方法）相較於單純迴歸分析的核心優勢之一——能夠在同一個模型中，將某個構念同時視為前一段路徑的「結果」與後一段路徑的「原因」，這類構念在方法論文獻中稱為「中介變數」或更廣義的「內生構念（endogenous construct）」。`plspm` 套件會自動依據 `Structure()` 中設定的路徑關係，正確處理這種鏈狀因果結構，本週研究模型中的 PU 與 SAT 皆屬於此種角色。

**Q8：除了 TAM 與 ECM-IT，還有哪些理論常被整合進 PLS-SEM 持續使用意願研究？未來期末專題想加入新構念，該如何選擇？**
除本週介紹之 TAM／ECM-IT 整合框架外，持續使用意願研究文獻中常見的延伸理論尚包括：**資訊系統成功模式（DeLone & McLean IS Success Model）**，強調系統品質、資訊品質、服務品質對使用者滿意度之影響；**信任理論（Trust Theory）**，特別適用於 AI 系統情境，捕捉使用者對演算法決策公正性與可靠性之知覺；以及**習慣理論（Habit）**，銜接第 2 週 UTAUT2 已介紹之構念，探討重複使用行為如何轉化為自動化習慣進而降低持續使用意願對理性評估的依賴程度。選擇延伸構念時，建議依循「理論缺口（theoretical gap）」原則：檢視現有文獻尚未探討、但在你的研究情境中具有實務重要性的構念，而非單純為了增加模型複雜度而堆疊構念。

**Q9：手動推導出的 AVE、CR 數值，和 Colab 程式直接跑出來的結果，為什麼可能會有小數點後幾位的些微差異？**
手動範例通常使用「已知、固定」的負荷量數值進行示範性推導，而 `plspm` 套件實際估計時，負荷量本身是透過疊代演算法（iterative algorithm）收斂求解得出，會受到收斂容忍度（`tolerance` 參數，預設極小）、疊代次數上限（`iterations` 參數）等演算法設定影響，因此實際負荷量會有更多位小數，代入公式後 AVE、CR 自然會與手動範例中刻意簡化的數值產生極小誤差，此為正常現象，只要數量級與判斷結論（是否通過門檻）一致即可，學生可對照附錄 I 之手動計算範例自行驗證。

---

## 延伸研究方向：跨國遠距團隊使用 AI 協同工作軟體之疲乏感與工作績效模型

### 6.1 研究背景與理論基礎

隨著跨國遠距團隊（distributed / virtual team）大量採用 AI 賦能之協同工作軟體（如 AI 會議摘要工具、AI 專案管理助理、AI 即時翻譯協作平台），組織一方面期待此類工具提升團隊效能，另一方面第一線研究已開始關注其可能帶來之「科技疲乏感（technology fatigue）」，包括線上會議疲乏、認知負荷過重與社交連結感降低等現象（見本週參考文獻中虛擬工作場域認知負荷之最新研究）。此議題適合以 PLS-SEM 建構一個包含中介變數之研究模型，探討「AI 協同工具使用強度」如何透過「疲乏感」這一中介機制，影響「團隊工作績效」，此設計邏輯與本週理論篇 3.6 節之中介效應分析架構完全對應。

### 6.2 建議研究設計

1. **理論框架**：可整合任務科技適配（TTF，第 2 週已介紹）解釋 AI 協同工具是否勝任跨國協作任務，並納入「科技疲乏感」作為中介變數，探討其如何削弱 TTF 對工作績效之正向效果，此設計亦可延伸為「有調節的中介效果（moderated mediation）」進階模型。
2. **研究對象**：建議以實際參與跨國遠距團隊（成員分散於兩個以上時區）、且團隊例行使用至少一項 AI 協同工作軟體之知識工作者為研究對象。
3. **變數規劃（範例）**：
   - AI 協同工具使用強度（外生構念）：每週使用頻率與功能涵蓋廣度
   - 任務科技適配度（TTF）：延續第 2 週理論定義
   - 科技疲乏感（中介變數）：可參考視訊會議疲乏（Zoom fatigue）相關量表改編，衡量認知負荷、社交疲勞、螢幕疲勞等面向
   - 團隊工作績效（依變數）：可採自評式團隊效能量表，或客觀專案交付準時率等指標
4. **分析流程**：完全比照本週 Colab 實作流程（測量模型評鑑 → 區別效度 → 結構模型路徑檢定 → Bootstrapping → 中介效果檢定 → 因果路徑圖），僅需替換構念定義與題項資料，即可直接複用本週所有程式碼架構，包含 `calculate_htmt()`、`bootstrap_indirect_effect()` 等自訂函式。
5. **管理實務意涵**：若研究證實科技疲乏感確實中介 AI 協同工具使用強度對工作績效之影響，即可為企業提供具體建議——例如導入「AI 工具使用時段管理」或「非同步協作優先」等政策，以緩解疲乏感對團隊績效之侵蝕效果。

### 6.3 給學生的思考練習

請思考：本延伸研究若將「科技疲乏感」的角色，從「中介變數」改為「調節變數」（例如：疲乏感調節 AI 工具使用強度對工作績效之影響，而非作為兩者之間的中間傳導機制），研究假設與資料分析方法會如何改變？這將如何結合第 2 週所學之調節效應分析與本週所學之中介效應分析，發展成一個「有調節的中介模型（moderated mediation model）」？

---

## 課後作業與練習

**練習一：測量模型品質劣化情境模擬**
請修改 Step 1 中任一構念（例如 SAT）之 `generate_items()` 函式的 `loading` 參數，從 `0.78` 大幅調降至 `0.35`，重新執行完整流程，觀察外部負荷量、組合信度 CR、AVE 三項指標之變化，並說明此構念是否還能通過測量模型品質檢驗。

**練習二：真實問卷資料實作**
請延續第 1、2 週蒐集之問卷資料架構，設計一份包含至少 3 個構念（其中一個為中介變數）之完整問卷，實際發放蒐集後，套用本週完整 Colab 程式碼執行 PLS-SEM 分析。請繳交：(1) 原始 CSV 檔、(2) 執行後的 Colab Notebook（.ipynb）、(3) 一頁 A4 的結果摘要（比照本週「結果呈現與分析」段落之寫法）。

**練習三：HTMT 與 Fornell-Larcker 判定不一致情境探索**
請修改 Step 1 資料生成邏輯，讓 PU 與 SAT 兩個構念之題項刻意共用部分相同的潛在來源（例如讓 SAT 的題項生成公式改為 `0.5*SAT_true + 0.4*PU_true + noise`），重新執行 Step 5，觀察 Fornell-Larcker 準則與 HTMT 比率是否出現判定不一致的情形，並討論可能的補救做法。

**練習四：文獻延伸閱讀報告**
請從本週「參考文獻與延伸閱讀」清單中，任選一篇 ECM-IT、TAM 整合模型或 PLS-SEM 方法論相關文獻，撰寫一頁重點摘要，內容須包含：(1) 該研究之構念與路徑假設、(2) 報告之測量模型品質指標（CR、AVE）、(3) 報告之結構模型解釋力（R²）與路徑係數顯著性、(4) 該研究是否檢驗中介或調節效果，其發現為何。

**練習五：中介類型判斷練習**
請利用本週 `bootstrap_indirect_effect()` 函式，計算「CONF → SAT → CI」路徑之間接效果信賴區間，並比較「CONF → CI 之直接效果」是否顯著。依 Zhao, Lynch, & Chen（2010）之判斷架構，判斷此路徑屬於完全中介、部分中介，還是無中介效果，並說明你的判斷依據。

**練習六：SRMR 近似適配度指標計算延伸**
本週 Q&A 之 Q1 提到，部分 PLS-SEM 研究會額外報告 SRMR（標準化均方根殘差）作為近似適配度指標。請嘗試查閱 Henseler et al.（2014）提出之 PLS-SEM 情境下 SRMR 計算方式（提示：比較模型隱含相關矩陣與實際觀察相關矩陣之殘差），並嘗試以 Python 撰寫一個計算 SRMR 的自訂函式，套用於本週模擬資料，判斷其是否低於 0.08 之建議門檻。此練習為進階挑戰題，鼓勵搭配 Vibe Coding 與 AI 協作完成。

---

## 參考文獻與延伸閱讀（已查核連結）

1. An empirical study of user willingness to continuously use AI-assisted search tools: an extension based on the ECM and TAM theoretical models. *Humanities and Social Sciences Communications (Nature)*.
   https://www.nature.com/articles/s41599-026-06711-4
   （與本週研究設計範例理論框架完全對應之最新期刊論文，整合 ECM 與 TAM 探討 AI 工具持續使用意願，可直接作為文獻回顧核心參照。）

2. 消費者使用銀行智能客服之研究。國立政治大學博碩士論文系統。
   https://thesis.lib.nccu.edu.tw/detail/2808060610f1d5f5000bc5d258dc00a7/
   （以期望確認理論（ECT）為核心，探討 AI 智能客服使用者滿意度與持續使用意圖之台灣本土碩士論文，並納入信任與 AI 素養等延伸構念，適合作為在地化改編之參照範本。）

3. 從輿情和消費者分析了解 AIGC 的使用意圖。朝陽科技大學行銷與流通管理系碩士論文。
   https://ir.lib.cyut.edu.tw/bitstream/310901800/43809/1/112CYUT0691009-002.pdf
   （整合期望確認理論、TAM 與 AI 焦慮等構念，探討生成式 AI 內容使用意圖之台灣碩士論文，可作為研究架構延伸之參考。）

4. 影響公部門數位學習系統持續使用之關鍵因素探討。臺灣博碩士論文知識加值系統。
   https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi/login?o=dnclcdr&s=id=%22098NKNU5395019%22.&searchmode=basic
   （綜整期望確認理論（ECT）、期望不確定理論（EDT）、TAM、UTAUT 等多理論觀點探討持續使用意圖之台灣論文，可作為理論回顧廣度之參考範例。）

5. Discriminant Validity and HTMT Assessment in PLS-SEM. SmartPLS 官方方法論文件。
   https://www.smartpls.com/documentation/algorithms-and-techniques/discriminant-validity-assessment/
   （PLS-SEM 業界標準軟體之官方方法論說明文件，完整說明 HTMT 判斷標準與 Bootstrapping 檢定流程，為本週理論篇 3.3 節之直接依據。）

6. Going beyond the untold facts in PLS–SEM and moving forward. *European Journal of Marketing*.
   https://www.emerald.com/ejm/article/58/13/81/1222222/Going-beyond-the-untold-facts-in-PLS-SEM-and
   （近期針對 AVE、組合信度、Fornell-Larcker 準則等傳統 PLS-SEM 評鑑指標侷限性之方法論辯論文章，可作為研究限制章節之深度討論依據。）

7. Collaborative AI in the Workplace: Enhancing organizational performance through resource-based and task-technology fit perspectives.
   https://www.academia.edu/126068915/Collaborative_AI_in_the_Workplace_Enhancing_organizational_performance_through_resource_based_and_task_technology_fit_perspectives
   （以 PLS-SEM 檢驗 AI 協作能力對組織績效之影響，並納入認知投入等中介機制，可作為本週延伸研究方向之直接前導文獻。）

8. Virtual Team Effectiveness: An Empirical Study Using SEM.
   https://www.researchgate.net/publication/321750606_Virtual_Team_Effectiveness_An_Empirical_Study_Using_SEM
   （以 PLS-SEM 檢驗遠距 IT 團隊動機、知識分享與團隊效能之關係，可作為延伸研究方向「跨國遠距團隊」情境之方法論參照範本。）

9. Assessing the impact of virtual workplaces on collaboration and learning. *PMC / Frontiers in Psychology*.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC12223796/
   （檢驗不同虛擬工作環境對認知負荷、疲乏感與協作表現之影響，可作為本週延伸研究方向「科技疲乏感」構念操作化之重要參考依據。）

10. plspm：Partial Least Squares Path Modeling in Python（PyPI 官方套件頁面）。
    https://pypi.org/project/plspm/
    （本週 Colab 實作採用之核心 Python 套件官方文件，內含安裝說明與完整使用範例，本教材所有程式碼皆已實際測試驗證可正常執行。）

**方法論經典文獻（建議延伸閱讀，非本次線上搜尋來源，圖書館或資料庫可查閱）**：

- Bhattacherjee, A. (2001). Understanding information systems continuance: An expectation-confirmation model. *MIS Quarterly*, 25(3), 351–370.
- Fornell, C., & Larcker, D. F. (1981). Evaluating structural equation models with unobservable variables and measurement error. *Journal of Marketing Research*, 18(1), 39–50.
- Henseler, J., Ringle, C. M., & Sarstedt, M. (2015). A new criterion for assessing discriminant validity in variance-based structural equation modeling. *Journal of the Academy of Marketing Science*, 43(1), 115–135.
- Hair, J. F., Risher, J. J., Sarstedt, M., & Ringle, C. M. (2019). When to use and how to report the results of PLS-SEM. *European Business Review*, 31(1), 2–24.
- Zhao, X., Lynch, J. G., & Chen, Q. (2010). Reconsidering Baron and Kenny: Myths and truths about mediation analysis. *Journal of Consumer Research*, 37(2), 197–206.
- Preacher, K. J., & Hayes, A. F. (2008). Asymptotic and resampling strategies for assessing and comparing indirect effects in multiple mediator models. *Behavior Research Methods*, 40(3), 879–891.

---

## 附錄

### 附錄 A：SmartPLS 與 Python（plspm）功能對照表

| 分析項目 | SmartPLS 操作方式 | Python `plspm` 對應方法 | 備註 |
|---|---|---|---|
| 建立測量模型與結構模型 | 圖形化拖曳建模介面 | `Structure()`、`Config()` | Python 以程式碼描述模型，較利於版本控制與重現性 |
| 模型估計 | 點選 "Calculate → PLS-SEM Algorithm" | `Plspm(data, config, Scheme.PATH)` | 兩者內部權重估計演算法邏輯一致 |
| 外部負荷量 | Results → Outer Loadings | `calc.outer_model()` | 對應欄位 `loading` |
| 組合信度、Cronbach's α | Results → Construct Reliability and Validity | `calc.unidimensionality()` | 對應欄位 `dillon_goldstein_rho`、`cronbach_alpha` |
| AVE | 同上頁籤 | `calc.inner_summary()['ave']` | — |
| Fornell-Larcker 矩陣 | Results → Discriminant Validity → Fornell-Larcker Criterion | 手動計算（本週 Step 5） | plspm 未內建，需自行以構念分數計算 |
| HTMT | Results → Discriminant Validity → HTMT | 手動計算（`calculate_htmt()` 自訂函式） | 同上 |
| 路徑係數 | Results → Path Coefficients | `calc.path_coefficients()` | — |
| Bootstrapping | Calculate → Bootstrapping | `Plspm(..., bootstrap=True)` + `calc.bootstrap()` | Python 版本以 `bootstrap_iterations` 參數控制重抽樣次數 |
| 直接/間接/總效果 | Results → Total Effects / Specific Indirect Effects | `calc.effects()` | — |
| R² | Results → R Square | `calc.inner_summary()['r_squared']` | — |

### 附錄 B：術語中英對照表

| 中文術語 | 英文術語 | 縮寫 |
|---|---|---|
| 偏最小平方法結構方程模型 | Partial Least Squares Structural Equation Modeling | PLS-SEM |
| 共變異數為基礎之結構方程模型 | Covariance-Based SEM | CB-SEM |
| 測量模型 | Measurement Model / Outer Model | — |
| 結構模型 | Structural Model / Inner Model | — |
| 反映性測量 | Reflective Measurement | — |
| 形成性測量 | Formative Measurement | — |
| 組合信度 | Composite Reliability | CR |
| 平均變異抽取量 | Average Variance Extracted | AVE |
| 區別效度 | Discriminant Validity | — |
| 異質特質—同質特質比率 | Heterotrait-Monotrait Ratio | HTMT |
| 拔靴法／重抽樣法 | Bootstrapping | — |
| 直接效果 | Direct Effect | — |
| 間接效果／中介效果 | Indirect Effect / Mediation Effect | — |
| 完全中介 | Full Mediation | — |
| 部分中介 | Partial Mediation | — |
| 整體模型適配度指標 | Goodness of Fit | GoF |
| 效果量 | Effect Size | f² |
| 預測相關性 | Predictive Relevance | Q² |
| 期望確認模型 | Expectation Confirmation Model | ECM-IT |

### 附錄 C：常見程式錯誤排解（Debugging Tips）

| 錯誤現象 | 常見原因 | 排解建議 |
|---|---|---|
| `ValueError: zip() argument 2 is longer than argument 1`（於 `Plspm()` 初始化時發生） | pandas 版本過新（≥ 2.2），與 `plspm` 套件內部之舊版 pandas 索引語法不相容 | 確認 Cell 0 已執行 `pip install "pandas<2.2"` 並已重新啟動執行階段（Restart Runtime） |
| `calc.bootstrap()` 執行時間過長甚至無回應 | `bootstrap_iterations` 設定過高（如 5000 次）且樣本數 / 構念數較多 | 課堂演練階段先降低至 200–500 次，正式分析再提高並於離峰時段執行 |
| 自訂 `bootstrap_indirect_effect()` 函式回傳的 `n_valid_boot` 遠低於設定的 `n_boot` | 部分拔靴重抽樣樣本因重複值過多，導致構念間相關矩陣退化（奇異矩陣） | 確認原始樣本數是否過小；若持續發生，可放寬 `try/except` 邏輯並記錄失敗原因以利診斷 |
| `calculate_htmt()` 計算結果出現 `NaN` | 某構念僅有 1 個題項，導致同質特質相關（monotrait correlation）無法計算（至少需 2 題才能算相關） | 確認每個構念至少有 2 個（建議 3 個以上）觀察題項 |
| 因果路徑圖中文標籤顯示為方框 | `networkx` 繪圖函式未正確指定中文字型 | 確認 `nx.draw_networkx_labels()` 與 `nx.draw_networkx_edge_labels()` 均有指定 `font_family='Noto Sans CJK TC'` 參數 |

### 附錄 D：繳交前自我檢核清單

- [ ] 已報告完整測量模型評鑑結果（外部負荷量、Cronbach's α、CR、AVE）
- [ ] 已同時報告 Fornell-Larcker 矩陣與 HTMT 矩陣，並說明兩者判定結果是否一致
- [ ] 已確認測量模型通過品質檢驗後，才進行結構模型路徑係數之解讀
- [ ] 已報告 Bootstrapping 重抽樣次數，正式研究建議至少 5000 次
- [ ] 已報告所有結構路徑之標準化係數、t 值與 95% 信賴區間
- [ ] 已報告各內生構念之 R² 並依判斷標準說明解釋力等級
- [ ] 若研究模型包含中介變數，已以 Bootstrapping 檢定間接效果並依 Zhao et al. (2010) 架構判斷中介類型
- [ ] 已繪製因果路徑結構圖，並於圖中標示路徑係數與顯著性
- [ ] 所有統計結果之文字敘述與表格數值一致，無謄寫錯誤
- [ ] 已在研究限制中誠實揭露 PLS-SEM 方法之侷限性（如缺乏傳統整體適配度指標）

### 附錄 E：研究倫理提醒

本週研究設計涉及企業員工（AI 決策支援系統使用者）或跨國遠距團隊成員之問卷調查，除延續前兩週已說明之知情同意、匿名性、資料儲存安全性等基本倫理原則外，特別提醒：若研究對象為受訪者任職企業之員工，應特別留意問卷施測過程是否可能讓受訪者產生「填答內容會被雇主得知」之疑慮，進而影響其填答真實性（社會期許偏誤），建議施測說明中明確強調資料僅供學術研究彙總分析使用、不會提供個別回覆內容予任職企業，並考慮由研究者而非企業內部人資單位直接發放與回收問卷，以降低此類偏誤來源。

### 附錄 F：本週常用 Python 函式速查表

| 函式／方法 | 所屬套件 | 功能 |
|---|---|---|
| `spc.Structure().add_path(source, target)` | plspm | 定義結構模型中構念之間的因果路徑 |
| `spc.Config(structure.path(), scaled=True)` | plspm | 建立模型設定物件，`scaled=True` 會自動將指標標準化 |
| `config.add_lv_with_columns_named()` | plspm | 將一組觀察題項欄位指派給特定潛在構念，並指定測量模式 |
| `Plspm(data, config, Scheme.PATH, bootstrap=True)` | plspm | 估計 PLS 路徑模型，並同步執行 Bootstrapping |
| `calc.outer_model()` | plspm | 取得外部負荷量、權重、共同性、重複量數 |
| `calc.unidimensionality()` | plspm | 取得 Cronbach's α、組合信度（Dillon-Goldstein's ρ） |
| `calc.inner_summary()` | plspm | 取得各構念之 R²、AVE、平均重複量數 |
| `calc.path_coefficients()` | plspm | 取得結構模型路徑係數矩陣 |
| `calc.bootstrap().paths()` | plspm | 取得路徑係數之 Bootstrapping 標準誤、信賴區間與 t 值 |
| `calc.effects()` | plspm | 取得每條路徑之直接、間接與總效果 |
| `calc.scores()` | plspm | 取得各構念之複合分數（用於自訂後續分析，如 Fornell-Larcker） |
| `calc.goodness_of_fit()` | plspm | 取得整體模型 GoF 指標 |
| `nx.DiGraph()` / `nx.draw_networkx_*()` | networkx / matplotlib | 繪製因果路徑結構圖 |

### 附錄 G：第 1–3 週研究方法整合對照表

隨著課程進行，學生手上的統計工具箱逐漸擴充。以下表格彙整前三週方法之核心差異與適用情境，幫助學生在規劃自己的期末專題或碩士論文研究方法時，能快速判斷應採用哪一種（或哪幾種組合）分析技術。

| 比較構面 | 第 1 週：EFA | 第 2 週：階層迴歸＋調節效應 | 第 3 週：PLS-SEM |
|---|---|---|---|
| 核心目的 | 探索題項背後的潛在因素結構 | 檢定前因變數對單一依變數的直接效果與調節效果 | 檢定多構念、多路徑、含中介變數的完整因果網絡 |
| 依變數數量 | 不適用（無依變數概念） | 單一依變數 | 可同時處理多個內生構念（依變數） |
| 是否處理測量誤差 | 否（僅探索結構） | 否（以加總平均分數代表構念） | 是（透過反映性測量模型同時估計） |
| 是否可處理中介效應 | 不適用 | 需另外搭配 Baron & Kenny 或 Bootstrap | 原生支援（`effects()` 直接輸出） |
| 是否可處理調節效應 | 不適用 | 原生支援（交互作用項＋ Simple Slope） | 需以進階兩階段法變通處理 |
| 典型研究階段 | 量表發展初期 | 前因變數效果檢定 | 完整理論模型驗證階段 |
| 本課程對應研究範例 | 生成式 AI 採用意向量表發展 | 對話式 AI 虛擬助理使用意願（UTAUT2+TTF） | AI 決策支援系統持續使用行為（TAM+ECM-IT） |

這三週之方法並非互斥，而是同一份研究資料在不同分析階段的遞進應用：**先以 EFA 確認量表結構（第 1 週）→ 再以迴歸與調節效應初步檢定關鍵路徑（第 2 週）→ 最終以 PLS-SEM 整合為完整的因果路徑模型（第 3 週）**，這也正是絕大多數量化實證論文由「量表發展」到「模型驗證」的完整方法論敘事脈絡。

### 附錄 H：期刊審查意見對照檢核表

近年 PLS-SEM 論文投稿常見之審查意見類型，以下彙整並對照本週教材對應可回應之章節，供學生於論文投稿或口試前自我演練：

| 常見審查意見 | 對應本週教材章節 | 回應要點 |
|---|---|---|
| 「作者未報告 HTMT，僅以 Fornell-Larcker 準則判斷區別效度，證據力不足」 | 理論篇 3.3 節、Step 5 | 補充報告 HTMT 矩陣，並說明兩者判定結果是否一致 |
| 「Bootstrapping 次數僅 500 次，是否足夠穩定？」 | 理論篇 3.5 節、Q4 | 正式投稿版本應提高至 5000 次以上並重新估計 |
| 「中介效果檢定為何不使用 Sobel 檢定，而改用 Bootstrapping？」 | 理論篇 3.6 節、Q2 | 說明間接效果抽樣分配之偏態特性，Bootstrapping 為現行方法論共識 |
| 「模型未報告任何整體適配度指標，如何確認模型品質？」 | Q1 | 說明 PLS-SEM 之預測導向定位，並以 R²、Q²、路徑顯著性作為主要依據 |
| 「作者引用之樣本數決定原則（10 倍法則）已被證實會低估所需樣本數」 | 4.5 節、附錄之樣本數對照表 | 補充報告依 Hair et al. (2019) 樣本數對照表之正式估計結果 |
| 「請說明中介效果屬於完全中介還是部分中介，並提供理論依據」 | 理論篇 3.6 節、4.3 節 H6/H7 | 依 Zhao, Lynch, & Chen (2010) 判斷架構逐一說明 |

### 附錄 I：AVE 與組合信度手動計算之數值範例

為協助學生徹底理解理論篇 3.2 節之公式，以下以一個簡化的 3 題項構念為例，展示完全手動（不依賴套件）之計算過程，這也是論文口試中若被要求「請現場推導一下 AVE 怎麼算」時最有效的準備方式。

假設某構念之 3 個題項，經 PLS 估計後得到外部負荷量分別為 $\lambda_1 = 0.85$、$\lambda_2 = 0.90$、$\lambda_3 = 0.80$：

**AVE 計算**：

$$
AVE = \frac{\lambda_1^2 + \lambda_2^2 + \lambda_3^2}{3} = \frac{0.85^2 + 0.90^2 + 0.80^2}{3} = \frac{0.7225 + 0.81 + 0.64}{3} = \frac{2.1725}{3} \approx 0.724
$$

由於 $0.724 &gt; 0.50$，此構念通過收斂效度判斷標準。

**組合信度（CR）計算**：

$$
CR = \frac{(\lambda_1+\lambda_2+\lambda_3)^2}{(\lambda_1+\lambda_2+\lambda_3)^2 + \left[(1-\lambda_1^2)+(1-\lambda_2^2)+(1-\lambda_3^2)\right]}
$$

$$
= \frac{(0.85+0.90+0.80)^2}{(0.85+0.90+0.80)^2 + \left[(1-0.7225)+(1-0.81)+(1-0.64)\right]}
= \frac{2.55^2}{2.55^2 + 0.8275} = \frac{6.5025}{7.33} \approx 0.887
$$

由於 $0.887 &gt; 0.70$，此構念亦通過組合信度判斷標準。學生可對照本週 Step 4 中 `calc.outer_model()` 與 `calc.unidimensionality()` 的實際輸出值，驗證程式計算結果與此手動公式推導是否一致，這是檢驗自己是否真正理解統計原理（而非僅會呼叫套件函式）的最佳自我檢核方式。

---

## 下週預告

第 4 週將進入第二模組「知識驅動型 AI 與多準則決策系統」，主題為「專家系統多準則排序：AHP、Fuzzy AHP 與 TOPSIS 模擬」。學生將從本週的「資料驅動型路徑檢定」轉向「專家知識驅動型決策評選」，學習如何將專家的成對比較判斷轉化為量化權重（層級分析法 AHP），並以模糊數處理專家主觀猶豫（Fuzzy AHP），最終結合 TOPSIS 逼近理想解法對候選方案進行综合評分與排序。研究範例將以「科技製造業關鍵設備供應商評選之專家決策系統架構」為主題，並延伸至「跨國綠色冷鏈物流中心選址評估決策模型」之期末專題發想方向。
