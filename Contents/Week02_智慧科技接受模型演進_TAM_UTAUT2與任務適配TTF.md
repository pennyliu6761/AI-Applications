# 第 2 週：智慧科技接受模型演進：TAM、UTAUT2 與任務適配（TTF）

> 課程模組：第一模組｜人機互動、行為決策與認知模型（第 1–3 週）
> 本週定位：承接第 1 週完成之量表信效度基礎，進入「前因變數如何影響行為意向」的迴歸解釋模型，學生將學習以階層式迴歸模型檢定多重前因變數的解釋力，並掌握調節效應（moderation effect）的統計原理與 Python 實作方法。

> 教材版本：v1.0｜適用對象：在職專班研究方法與 AI 應用課程｜先修基礎：第 1 週（量表建構與 EFA）
> 使用工具：Google Colab（Python 3）｜主要套件：`pandas`、`numpy`、`statsmodels`、`scipy`、`seaborn`、`matplotlib`

---

## 目錄

1. [學習目標](#學習目標)
2. [本週知識地圖](#本週知識地圖)
3. [理論基礎篇](#理論基礎篇)
   1. [3.1 科技接受模型（TAM）的理論起源與演進](#31-科技接受模型tam的理論起源與演進)
   2. [3.2 整合性科技接受模式：UTAUT 與 UTAUT2](#32-整合性科技接受模式utaut-與-utaut2)
   3. [3.3 任務科技適配模式（TTF）](#33-任務科技適配模式ttf)
   4. [3.4 TAM–UTAUT2–TTF 整合認知框架](#34-tamutaut2ttf-整合認知框架)
   5. [3.5 階層式迴歸分析原理](#35-階層式迴歸分析原理)
   6. [3.6 調節效應、交互作用項與 Simple Slope 分析](#36-調節效應交互作用項與-simple-slope-分析)
   7. [3.7 論文中迴歸與調節效應章節的標準寫法架構](#37-論文中迴歸與調節效應章節的標準寫法架構)
4. [研究設計實例：消費者對對話式 AI 虛擬助理使用意願——整合 UTAUT2 與 TTF](#研究設計實例消費者對對話式-ai-虛擬助理使用意願整合-utaut2-與-ttf)
5. [Colab 實作環境建置](#colab-實作環境建置)
6. [Colab 實作：Step by Step 完整程式碼](#colab-實作step-by-step-完整程式碼)
7. [Vibe Coding 提示詞（Prompt）實作範例集](#vibe-coding-提示詞prompt實作範例集)
8. [結果呈現與分析：碩士論文寫法示例](#結果呈現與分析碩士論文寫法示例)
9. [常見統計誤區與 Q&A](#常見統計誤區與-qa)
10. [延伸研究方向：無人零售系統高齡者使用意向研究——數位包容性之調節角色](#延伸研究方向無人零售系統高齡者使用意向研究數位包容性之調節角色)
11. [課後作業與練習](#課後作業與練習)
12. [參考文獻與延伸閱讀（已查核連結）](#參考文獻與延伸閱讀已查核連結)
13. [附錄](#附錄)
14. [下週預告](#下週預告)

---

## 學習目標

完成本週課程後，學生應能夠：

1. 說明 TAM、UTAUT、UTAUT2 三代科技接受模型之構念演進脈絡與理論假設差異。
2. 說明任務科技適配模式（TTF）之核心理論主張，並理解其與科技接受模型的互補關係。
3. 建構一個整合 UTAUT2 與 TTF 之研究架構，並正確撰寫操作型定義與研究假設。
4. 使用 Python（`statsmodels`）於 Colab 環境中執行階層式迴歸分析（hierarchical regression），並判讀 R²、ΔR²、VIF 共線性診斷指標。
5. 理解交互作用項（interaction term）的建構原理（含平減 / 中心化處理），並執行 Simple Slope 調節效應分析與視覺化。
6. 依照論文「研究結果與討論」章節寫法，將迴歸與調節效應統計輸出轉譯為具學術規範的文字敘述。
7. 初步規劃一個涉及調節變數的量化研究主題，作為後續研究方法整合之基礎。

---

## 本週知識地圖

| 構面 | 內容 | 對應統計方法 | 對應 Python 套件 |
|---|---|---|---|
| 理論架構 | TAM → UTAUT → UTAUT2 → TTF 整合 | 文獻回顧、構念操作化 | — |
| 迴歸前置檢查 | 共線性診斷、常態性、同質變異 | VIF、殘差圖、Durbin-Watson | `statsmodels`, `scipy` |
| 主效果檢定 | 前因變數對依變數之直接效果 | 階層式迴歸（Hierarchical Regression） | `statsmodels.OLS` |
| 交互作用 | 調節變數如何改變自變數與依變數之關係 | 交互作用項迴歸、平減（Mean-centering） | `numpy`, `pandas` |
| 調節效應視覺化 | 不同調節組別下的斜率差異 | Simple Slope Analysis | `matplotlib`, 自訂函式 |
| 結果彙整 | 各前因變數標準化係數比較 | 森林圖（Forest Plot） | `matplotlib` |

---

## 理論基礎篇

### 3.1 科技接受模型（TAM）的理論起源與演進

科技接受模型（Technology Acceptance Model, TAM）由 Davis（1989）提出，是資訊系統採用研究領域中被引用次數最高的理論模型之一。其核心主張源自 Fishbein 與 Ajzen（1975）之理性行動理論（Theory of Reasoned Action, TRA），認為個人的行為由行為意向所決定，而行為意向又受態度與主觀規範所影響。TAM 將此一般性理論聚焦於資訊科技採用情境，並簡化為兩個核心信念構念：

- **知覺有用性（Perceived Usefulness, PU）**：個人認為使用某一科技系統能提升其工作績效的主觀程度。
- **知覺易用性（Perceived Ease of Use, PEOU）**：個人認為使用某一科技系統不需付出過多認知努力的主觀程度。

TAM 的理論路徑為：PEOU → PU（易用性正向影響有用性知覺）；PU、PEOU → 使用態度（Attitude）→ 使用意向（Behavioral Intention）→ 實際使用行為（Actual Use）。後續研究（如 Davis, Bagozzi, & Warshaw, 1989；Venkatesh & Davis, 1996）發現「使用態度」在許多情境下並非顯著中介變數，因此衍生出精簡版 TAM，直接以 PU、PEOU 預測使用意向。

TAM 的理論限制在於：構念過於精簡，難以解釋組織情境中的社會影響因素（如主管期望、同儕壓力），也未能涵蓋消費性科技情境中的享樂性使用動機（如遊戲、社群媒體），這正是 UTAUT 與 UTAUT2 模型欲彌補之處。

### 3.2 整合性科技接受模式：UTAUT 與 UTAUT2

**UTAUT（Unified Theory of Acceptance and Use of Technology）**由 Venkatesh, Morris, Davis, & Davis（2003）提出，整合了 TAM、TRA、計畫行為理論（TPB）、動機模型（MM）、PC 使用模型（MPCU）、創新擴散理論（IDT）與社會認知理論（SCT）共 8 個既有模型，萃取出四個核心預測構念：

| 構念 | 定義 | 理論來源 |
|---|---|---|
| 績效期望（Performance Expectancy, PE） | 個人認為使用系統有助於提升工作績效的程度 | 對應 TAM 之 PU |
| 努力期望（Effort Expectancy, EE） | 個人認為使用系統所需付出之努力程度 | 對應 TAM 之 PEOU |
| 社會影響（Social Influence, SI） | 個人知覺重要他人認為其應該使用該系統的程度 | 對應 TRA 之主觀規範 |
| 促成條件（Facilitating Conditions, FC） | 個人知覺組織與技術基礎設施支持其使用系統的程度 | 對應 TPB 之知覺行為控制 |

UTAUT 原始模型設計用於「組織強制性使用」情境（如企業導入 ERP 系統），並以性別、年齡、經驗、自願性作為調節變數。

**UTAUT2** 由 Venkatesh, Thong, & Xu（2012）提出，將模型延伸至「消費性科技自願使用」情境（如智慧型手機應用程式、串流影音服務），新增三個構念：

| 新增構念 | 定義 |
|---|---|
| 享樂動機（Hedonic Motivation, HM） | 使用科技所帶來的樂趣或愉悅感 |
| 價格價值（Price Value, PV） | 使用科技之知覺利益與其金錢成本之間的權衡認知 |
| 習慣（Habit, HT） | 個人因過去重複使用經驗而形成的自動化使用傾向 |

UTAUT2 並將調節變數簡化為年齡、性別、經驗三者（移除自願性，因消費情境下使用本質即為自願）。UTAUT2 是目前應用於生成式 AI、對話式 AI 助理等消費性 AI 服務採用意願研究中最常被引用的理論框架，本週研究範例即採用此框架。

**UTAUT 與 UTAUT2 構念差異對照表**：

| 構面 | UTAUT（2003） | UTAUT2（2012） |
|---|---|---|
| 適用情境 | 組織強制性使用（如企業導入資訊系統） | 消費性自願使用（如手機 App、串流服務） |
| 核心構念 | PE、EE、SI、FC | PE、EE、SI、FC、HM、PV、HT |
| 調節變數 | 性別、年齡、經驗、自願性 | 性別、年齡、經驗（移除自願性） |
| 依變數 | 使用意向 → 使用行為 | 使用意向 → 使用行為 |
| 典型應用領域 | ERP、CRM 等企業資訊系統 | 行動支付、AI 助理、社群媒體、串流平台 |

### 3.3 任務科技適配模式（TTF）

任務科技適配模式（Task-Technology Fit, TTF）由 Goodhue & Thompson（1995）提出，其理論主張與科技接受模型截然不同：TAM/UTAUT 系列模型關注「使用者主觀信念如何形成使用意向」，而 TTF 關注「科技的客觀功能特性，是否與使用者實際任務需求相匹配」。TTF 核心命題為：

$$
\text{任務科技適配度（TTF）} = f(\text{任務特性}, \text{科技特性})
$$

TTF 越高，代表該科技的功能設計越能滿足使用者完成任務所需，進而正向影響使用意向與使用績效。TTF 理論特別適合用於「工具型」科技的研究情境（如生成式 AI 輔助寫作、程式碼生成工具），因為這類科技的採用與否，往往高度取決於其功能是否真正契合使用者的工作任務內容，而非僅僅是主觀的易用性信念。

近年研究趨勢傾向將 TTF 與 UTAUT2 整合，形成互補的雙理論框架：UTAUT2 解釋「使用者為什麼想用」（心理與社會動機層面），TTF 解釋「這個科技是否真的適合用在這個任務上」（功能適配層面）。本週參考文獻中之台灣本土研究（如 VDI 採用決策研究、行動支付 UTAUT+TTF 研究），均採此整合取徑，可作為學生撰寫研究架構時的直接參照範本。

### 3.4 TAM–UTAUT2–TTF 整合認知框架

本週研究範例之整合架構，建議採用以下路徑邏輯：

```
績效期望 (PE) ──┐
努力期望 (EE) ──┤
社會影響 (SI) ──┼──► 使用意向 (Behavioral Intention)
促成條件 (FC) ──┤            ▲
享樂動機 (HM) ──┤            │
價格價值 (PV) ──┤       任務科技適配 (TTF)
習慣    (HT) ──┘            ▲
                              │
                    任務特性 × 科技特性
```

其中，任務科技適配（TTF）可作為：(a) 額外的前因變數，直接加入迴歸模型；或 (b) 調節變數，調節 UTAUT2 各構念與使用意向之間的關係強度。本週 Colab 實作將示範方法 (b)，即以 TTF 作為調節變數，檢驗當任務科技適配度不同時，績效期望對使用意向的影響力是否有顯著差異，此即為本週理論篇 3.6 節「調節效應」的具體應用情境。

### 3.5 階層式迴歸分析原理

階層式迴歸分析（Hierarchical Regression Analysis）是一種將自變數依理論邏輯分批（block）納入迴歸模型的分析策略，目的是檢驗「新增一組變數後，模型解釋力是否顯著提升」。其統計邏輯如下：

**模型設定**：

$$
\text{Model 1（控制變數）：} \quad Y = b_0 + b_1 Age + b_2 Gender + e
$$

$$
\text{Model 2（加入主效果）：} \quad Y = b_0 + b_1 Age + b_2 Gender + b_3 PE + b_4 EE + b_5 TTF + e
$$

$$
\text{Model 3（加入交互作用項）：} \quad Y = b_0 + \dots + b_6 (PE \times TTF) + e
$$

**R² 改變量（ΔR²）檢定**：

$$
\Delta R^2 = R^2_{Model\,k} - R^2_{Model\,k-1}
$$

$$
F_{change} = \frac{\Delta R^2 / \Delta df}{(1 - R^2_{Model\,k}) / (n - k - 1)}
$$

ΔR² 顯著，代表新增的這一組變數（例如交互作用項）對依變數具有超越先前模型的額外解釋力，這是判斷調節效應是否存在的關鍵統計證據，比單看交互作用項係數的 p 值更具說服力，也是論文口試委員最常追問的統計細節之一。

**共線性診斷（Multicollinearity Diagnostics）**：多元迴歸中，若自變數之間高度相關，會導致迴歸係數估計不穩定。診斷指標為變異數膨脹因子（Variance Inflation Factor, VIF）：

$$
VIF_j = \frac{1}{1 - R_j^2}
$$

其中 $R_j^2$ 為將第 $j$ 個自變數對其餘所有自變數迴歸所得之判定係數。判斷標準為 VIF &lt; 10（部分嚴謹研究採用更保守的 VIF &lt; 5）為可接受範圍，交互作用項因與其構成之主效果變數天然相關，常需搭配「平減（mean-centering）」處理以降低共線性問題（詳見 3.6 節）。

### 3.6 調節效應、交互作用項與 Simple Slope 分析

**調節效應（Moderation Effect）** 是指自變數 $X$ 對依變數 $Y$ 的影響力，會因調節變數 $Z$ 的不同水準而有所差異。統計上以交互作用項 $X \times Z$ 的迴歸係數是否顯著來判斷：

$$
Y = b_0 + b_1 X + b_2 Z + b_3 (X \times Z) + e
$$

若 $b_3$ 達統計顯著，代表 $X$ 對 $Y$ 的效果確實因 $Z$ 而異，此時 $Z$ 被稱為調節變數。

**平減（Mean-Centering）的必要性**：在建構交互作用項 $X \times Z$ 之前，通常需要先將 $X$ 與 $Z$ 進行平減處理（減去各自的平均數），原因有二：(1) 降低交互作用項與主效果變數之間的共線性（結構性共線性，並非樣本資料造成的真實共線性，但仍會影響係數估計的數值穩定性）；(2) 使迴歸截距與主效果係數具有更直觀的解釋意義（平減後的主效果係數，代表調節變數在平均水準時的效果）。

**Simple Slope 分析**：交互作用項顯著後，下一步是具體描繪「在調節變數的不同水準下，自變數對依變數的簡單斜率（simple slope）分別是多少」。依循 Aiken & West（1991）之經典做法，通常選取調節變數的三個代表性水準進行比較：

- 低水準：平均數 − 1 個標準差（Mean − 1SD）
- 平均水準：平均數（Mean）
- 高水準：平均數 + 1 個標準差（Mean + 1SD）

在每一個水準下，計算 $X$ 對 $Y$ 的簡單斜率：

$$
Simple\ Slope_{X \to Y | Z} = b_1 + b_3 \cdot Z_{level}
$$

並可進一步以 $t$ 檢定判斷該簡單斜率是否顯著異於零。這正是提示詞實作中「繪製 Simple Slope 調節效應圖」所對應的統計原理。近年方法論文獻（見本週參考文獻中之 Robinson 與 Newsom 相關方法論文章）亦建議，除了檢定交互作用項本身，亦可直接檢定「高低兩組簡單斜率之間的差異」是否顯著，此檢定力通常優於單純檢定交互作用項係數。

**視覺化：Simple Slope 圖**：以 $X$ 為橫軸、$\hat{Y}$（預測值）為縱軸，分別繪製調節變數高、中、低三組的迴歸線，斜率差異越明顯，代表調節效應越強，這是論文中最直觀呈現調節效應的圖表形式。

### 3.7 論文中迴歸與調節效應章節的標準寫法架構

1. **共線性診斷結果**：報告各自變數之 VIF 值，確認無嚴重共線性問題。
2. **階層式迴歸模型摘要表**：呈現 Model 1、Model 2、Model 3 之 R²、ΔR²、F 值變化的三線表。
3. **各模型迴歸係數表**：呈現標準化迴歸係數（β）、t 值、p 值。
4. **交互作用項顯著性檢定結果**：說明 $b_3$ 係數與 ΔR² 是否顯著。
5. **Simple Slope 分析結果**：呈現高、中、低三組調節水準下之簡單斜率係數與顯著性。
6. **Simple Slope 圖**：以視覺化方式呈現調節效應之型態（增強型 / 減弱型 / 交叉型調節）。

---

## 研究設計實例：消費者對對話式 AI 虛擬助理使用意願——整合 UTAUT2 與 TTF

### 4.1 研究背景與動機

對話式 AI 虛擬助理（如 ChatGPT、Siri、Google Assistant 等）已深度融入消費者日常生活，然而消費者對此類服務之接受度，同時受到「主觀心理動機」（如績效期望、享樂動機）與「客觀任務適配程度」（如該助理是否真的能勝任消費者的查詢任務）雙重因素影響。既有研究多僅採用 UTAUT2 或 TTF 單一理論框架，本研究整合兩者，探討任務科技適配度是否會調節績效期望對使用意願的影響強度。此一研究設計方向，與本週參考文獻中國立成功大學「以整合科技接受模型 UTAUT 與延伸性整合科技接受模型 UTAUT2，探討消費者對於對話式 AI 人工智慧服務之接受程度，以 ChatGPT 為例」之台灣本土碩士論文高度相關，可作為文獻回顧與研究缺口論述之直接參照。

### 4.2 研究目的

1. 檢驗 UTAUT2 各構念（PE、EE、SI、FC、HM、PV、HT）對消費者使用對話式 AI 虛擬助理意願之直接效果。
2. 檢驗任務科技適配度（TTF）是否對「績效期望 → 使用意願」之路徑產生調節效果。
3. 以階層式迴歸模型驗證加入 TTF 主效果與交互作用項後，模型解釋力是否顯著提升。
4. 提出具體之管理意涵，供對話式 AI 服務提供者優化產品設計方向參考。

### 4.3 研究假設（範例）

- H1：績效期望（PE）正向影響使用意願（BI）。
- H2：努力期望（EE）正向影響使用意願（BI）。
- H3：社會影響（SI）正向影響使用意願（BI）。
- H4：促成條件（FC）正向影響使用意願（BI）。
- H5：享樂動機（HM）正向影響使用意願（BI）。
- H6：任務科技適配度（TTF）正向影響使用意願（BI）。
- H7：任務科技適配度（TTF）調節績效期望（PE）與使用意願（BI）之間的關係，當 TTF 越高時，PE 對 BI 之正向影響越強。

### 4.4 操作型定義與衡量工具

本週研究之衡量工具，建議直接沿用第 1 週已完成信效度驗證之量表架構，並依 UTAUT2 原始文獻（Venkatesh et al., 2012）之題項語意，改編適用於「對話式 AI 虛擬助理」情境。TTF 構念題項則參考 Goodhue & Thompson（1995）之任務科技適配量表，聚焦於「這個 AI 助理是否能滿足我查詢資訊 / 完成任務的需求」之語意。

### 4.5 抽樣與樣本數規劃

依迴歸分析之樣本數決定原則（Green, 1991 之經驗法則）：

$$
n \geq 50 + 8m \quad (\text{檢定整體模型適配度})
$$
$$
n \geq 104 + m \quad (\text{檢定個別迴歸係數顯著性})
$$

其中 $m$ 為自變數個數。本研究若納入 7 個 UTAUT2 構念 + TTF + 交互作用項，共約 9 個預測變數，依上述公式建議樣本數至少 $104 + 9 = 113$ 份以上，考量交互作用項檢定力通常較主效果低（交互作用效果量在行為科學研究中普遍偏小，中位數效果量約 $f^2 = 0.002$，見本週參考文獻 Dawson 相關方法論文獻），建議實務上樣本數規劃至少 250–300 份，以確保有足夠統計檢定力偵測中小型調節效應。

### 4.6 資料分析流程規劃

本週採用之分析流程如下（文字流程圖），可與第 1 週流程圖銜接，共同構成完整的「量表驗證 → 路徑檢定」研究管道：

```
第 1 週已驗證完成之量表（各構念題項）
        │
        ▼
計算各構念平均分數（構念加總平均法，Composite Score）
        │
        ▼
描述性統計與 Pearson 相關矩陣檢視
        │
        ▼
VIF 共線性診斷 ── 若 VIF ≥ 10 ──► 檢視是否有高度重疊之構念需合併或刪除
        │ 通過
        ▼
Model 1：控制變數（年齡、性別等人口變項）
        │
        ▼
Model 2：加入 UTAUT2 主效果 + TTF 主效果
        │
        ▼
ΔR² 是否顯著？ ── 否 ──► 檢視理論構念選取是否適當
        │ 是
        ▼
平減自變數與調節變數，建構交互作用項
        │
        ▼
Model 3：加入交互作用項
        │
        ▼
ΔR² 是否顯著？ ── 否 ──► 調節效應未獲支持，報告效果量與檢定力限制
        │ 是
        ▼
Simple Slope 分析（高／中／低三組調節水準）
        │
        ▼
繪製森林圖 + Simple Slope 圖
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
# 說明：
# 1. statsmodels 為 Python 中執行迴歸分析、共線性診斷、
#    模型比較最完整的統計套件，其輸出格式（含 R²、F 檢定、
#    係數 t 檢定）與 SPSS Regression 模組高度對應。
# 2. Colab 通常已預先安裝 statsmodels，但版本可能較舊，
#    建議仍執行升級指令，確保後續函式（如 anova_lm）可正常運作。
# ============================================================

!pip install --upgrade statsmodels --quiet

# ------------------------------------------------------------
# 匯入資料處理與數值運算套件
# ------------------------------------------------------------
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 匯入統計與迴歸分析相關套件
# ------------------------------------------------------------
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.anova import anova_lm

# ------------------------------------------------------------
# 匯入視覺化套件
# ------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 設定中文字型（沿用第 1 週相同設定邏輯）
# ------------------------------------------------------------
!wget -q https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf -O /content/NotoSansTC.otf
from matplotlib import font_manager
font_manager.fontManager.addfont('/content/NotoSansTC.otf')
plt.rcParams['font.family'] = 'Noto Sans CJK TC'
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------------------
# pandas 顯示設定
# ------------------------------------------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')

print("環境建置完成，所有套件已成功匯入。")
```

---

## Colab 實作：Step by Step 完整程式碼

### Step 1：模擬具有調節效應結構之問卷資料

```python
# ============================================================
# Cell 1：模擬 UTAUT2 + TTF 整合研究之問卷資料
# ------------------------------------------------------------
# 教學目的：
# 刻意在資料生成階段「埋入」一個已知的調節效應（TTF 調節 PE
# 對 BI 的影響），讓學生透過後續分析驗證能否正確偵測出此
# 調節效應，藉此具體理解調節效應在資料層次的真實樣貌。
# 正式研究請將本 Cell 替換為讀取真實問卷檔案的程式碼。
# ============================================================

np.random.seed(2024)
n_samples = 300   # 依 4.5 節樣本數規劃設定

# 產生控制變數：年齡、性別（0=女性, 1=男性）
age = np.random.randint(18, 65, n_samples)
gender = np.random.binomial(1, 0.5, n_samples)

# 產生 UTAUT2 七大構念之構念分數（1~7 分量表平均分數，非單題）
PE  = np.clip(np.random.normal(4.5, 1.1, n_samples), 1, 7)   # 績效期望
EE  = np.clip(np.random.normal(4.8, 1.0, n_samples), 1, 7)   # 努力期望
SI  = np.clip(np.random.normal(4.0, 1.3, n_samples), 1, 7)   # 社會影響
FC  = np.clip(np.random.normal(4.6, 1.0, n_samples), 1, 7)   # 促成條件
HM  = np.clip(np.random.normal(4.7, 1.2, n_samples), 1, 7)   # 享樂動機
PV  = np.clip(np.random.normal(4.3, 1.1, n_samples), 1, 7)   # 價格價值
HT  = np.clip(np.random.normal(4.1, 1.2, n_samples), 1, 7)   # 習慣

# 任務科技適配度（TTF），設定為調節變數
TTF = np.clip(np.random.normal(4.4, 1.15, n_samples), 1, 7)

# ------------------------------------------------------------
# 建構依變數「使用意願（BI）」，刻意埋入 PE × TTF 交互作用效果
# ------------------------------------------------------------
# 統計原理：以下方程式即為資料生成的「母體迴歸方程式」，
# 其中 0.25 這個係數，就是我們預期在後續分析中應被偵測出來
# 的交互作用項係數（調節效應強度）。
# ------------------------------------------------------------
noise = np.random.normal(0, 0.6, n_samples)

# 先將 PE、TTF 平減（見理論篇 3.6 節），以此建構交互作用項，
# 使資料生成邏輯與後續分析邏輯一致
PE_c_true  = PE - PE.mean()
TTF_c_true = TTF - TTF.mean()

BI = (
    1.5
    + 0.32 * PE + 0.18 * EE + 0.15 * SI + 0.20 * FC
    + 0.22 * HM + 0.10 * PV + 0.12 * HT
    + 0.28 * TTF
    + 0.25 * (PE_c_true * TTF_c_true)   # 埋入的交互作用效果（調節效應）
    + 0.02 * age - 0.05 * gender
    + noise
)
BI = np.clip(BI, 1, 7)

df_raw = pd.DataFrame({
    'age': age, 'gender': gender,
    'PE': PE, 'EE': EE, 'SI': SI, 'FC': FC,
    'HM': HM, 'PV': PV, 'HT': HT, 'TTF': TTF,
    'BI': BI
})

print(f"模擬問卷資料維度：{df_raw.shape[0]} 位受訪者 × {df_raw.shape[1]} 個變數")
df_raw.head()
```

### Step 2：描述性統計與相關矩陣

```python
# ============================================================
# Cell 2：描述性統計與 Pearson 相關矩陣
# ------------------------------------------------------------
# 統計原理：在進行迴歸分析前，應先檢視自變數間的兩兩相關，
# 初步判斷是否存在潛在共線性問題（相關係數 > 0.80 為警訊），
# 這是後續 VIF 檢定的前置檢視步驟。
# ============================================================

desc_stats = df_raw.describe().T[['mean', 'std', 'min', 'max']]
print("=== 表 1：變數描述性統計摘要 ===")
display(desc_stats.round(3))

corr_matrix = df_raw[['PE','EE','SI','FC','HM','PV','HT','TTF','BI']].corr()
print("\n=== 表 2：Pearson 相關矩陣 ===")
display(corr_matrix.round(3))

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, linewidths=0.5, linecolor='white')
plt.title('圖 1：研究變數相關矩陣熱圖', fontsize=13)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Step 3：平減（Mean-Centering）與交互作用項建構

```python
# ============================================================
# Cell 3：平減處理與交互作用項建構
# ------------------------------------------------------------
# 提示詞實作對照（前置步驟）：
# 「請先將自變數與調節變數進行平減處理，再建構交互作用項，
#   以降低交互作用項與主效果變數之間的結構性共線性。」
# ============================================================

df = df_raw.copy()

# 對所有 UTAUT2 構念與 TTF 進行平減（減去樣本平均數）
center_vars = ['PE', 'EE', 'SI', 'FC', 'HM', 'PV', 'HT', 'TTF']
for var in center_vars:
    df[f'{var}_c'] = df[var] - df[var].mean()
    print(f"{var} 平減完成，平減前平均數 = {df[var].mean():.3f}，"
          f"平減後平均數 = {df[f'{var}_c'].mean():.6f}（應趨近於 0）")

# 建構交互作用項：績效期望（平減後）× 任務科技適配（平減後）
df['PE_x_TTF'] = df['PE_c'] * df['TTF_c']

print("\n交互作用項 PE_x_TTF 已建構完成，前 5 筆資料如下：")
display(df[['PE', 'TTF', 'PE_c', 'TTF_c', 'PE_x_TTF']].head())
```

### Step 4：共線性診斷（VIF）

```python
# ============================================================
# Cell 4：變異數膨脹因子（VIF）共線性診斷
# ------------------------------------------------------------
# 提示詞實作對照：
# 「計算所有自變數（含交互作用項）的 VIF 值，判斷是否存在
#   嚴重共線性問題。」
# ============================================================

def calculate_vif(data, features):
    """
    計算指定自變數集合的 VIF 值。
    統計原理：VIF_j = 1 / (1 - R_j^2)，其中 R_j^2 為將第 j 個
    自變數對其餘所有自變數進行迴歸所得之判定係數。
    """
    X = sm.add_constant(data[features])
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data[vif_data['Variable'] != 'const']

vif_features = ['age', 'gender', 'PE_c', 'EE', 'SI', 'FC', 'HM', 'PV', 'HT', 'TTF_c', 'PE_x_TTF']
vif_result = calculate_vif(df, vif_features)
vif_result['Flag'] = np.where(vif_result['VIF'] < 10, '可接受', '需注意共線性')

print("=== 表 3：VIF 共線性診斷結果 ===")
display(vif_result.round(3))
```

### Step 5：階層式迴歸分析（Model 1 → Model 2 → Model 3）

```python
# ============================================================
# Cell 5：階層式迴歸分析
# ------------------------------------------------------------
# 提示詞實作對照：
# 「建立多元迴歸模型計算各前因變數的標準化係數，並以三階層
#   方式依序納入控制變數、主效果、交互作用項，比較各模型
#   之 R² 改變量是否顯著。」
# ------------------------------------------------------------
# 統計原理：階層式迴歸的核心價值在於「分批檢定」而非「一次
# 全部丟入」，這樣才能明確歸因「解釋力的提升，是來自於哪一
# 組變數的貢獻」，尤其是交互作用項的獨立貢獻，必須在控制
# 主效果之後才能被正確評估。
# ============================================================

# 為了輸出標準化係數（standardized beta），先將所有連續變數
# 標準化為 Z 分數，這是與 SPSS 輸出之 "Standardized Coefficients
# Beta" 完全對應的做法
df_z = df.copy()
z_vars = ['age', 'PE', 'EE', 'SI', 'FC', 'HM', 'PV', 'HT', 'TTF', 'BI']
for var in z_vars:
    df_z[f'{var}_z'] = stats.zscore(df_z[var])

df_z['PE_c_z']  = stats.zscore(df_z['PE_c'])
df_z['TTF_c_z'] = stats.zscore(df_z['TTF_c'])
df_z['PE_x_TTF_z'] = stats.zscore(df_z['PE_x_TTF'])

# ------------------------------------------------------------
# Model 1：僅納入控制變數（年齡、性別）
# ------------------------------------------------------------
model1 = smf.ols('BI_z ~ age_z + gender', data=df_z).fit()

# ------------------------------------------------------------
# Model 2：加入 UTAUT2 七構念 + TTF 主效果
# ------------------------------------------------------------
model2 = smf.ols(
    'BI_z ~ age_z + gender + PE_z + EE_z + SI_z + FC_z + HM_z + PV_z + HT_z + TTF_z',
    data=df_z
).fit()

# ------------------------------------------------------------
# Model 3：加入 PE × TTF 交互作用項
# ------------------------------------------------------------
model3 = smf.ols(
    'BI_z ~ age_z + gender + PE_c_z + EE_z + SI_z + FC_z + HM_z + PV_z + HT_z + TTF_c_z + PE_x_TTF_z',
    data=df_z
).fit()

# ------------------------------------------------------------
# 整理三模型之 R²、調整後 R²、ΔR²、F 值變化摘要表
# ------------------------------------------------------------
model_comparison = pd.DataFrame({
    'Model': ['Model 1（控制變數）', 'Model 2（主效果）', 'Model 3（交互作用項）'],
    'R²': [model1.rsquared, model2.rsquared, model3.rsquared],
    'Adj_R²': [model1.rsquared_adj, model2.rsquared_adj, model3.rsquared_adj],
})
model_comparison['ΔR²'] = model_comparison['R²'].diff()

# 使用 anova_lm 計算模型間之 F 值變化與顯著性（對應 SPSS 之 F Change）
f_change_2v1 = anova_lm(model1, model2)
f_change_3v2 = anova_lm(model2, model3)

model_comparison['F_change'] = [
    np.nan,
    f_change_2v1['F'].iloc[1],
    f_change_3v2['F'].iloc[1]
]
model_comparison['p_value_of_change'] = [
    np.nan,
    f_change_2v1['Pr(>F)'].iloc[1],
    f_change_3v2['Pr(>F)'].iloc[1]
]

print("=== 表 4：階層式迴歸模型摘要表（R² 改變量檢定）===")
display(model_comparison.round(4))
```

### Step 6：完整迴歸係數表輸出

```python
# ============================================================
# Cell 6：Model 3 完整迴歸係數表（含標準化 β、t 值、p 值）
# ============================================================

coef_table = pd.DataFrame({
    'Variable': model3.params.index,
    'Standardized_Beta': model3.params.values,
    'Std_Error': model3.bse.values,
    't_value': model3.tvalues.values,
    'p_value': model3.pvalues.values,
})
coef_table['Significance'] = coef_table['p_value'].apply(
    lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
)

print("=== 表 5：Model 3 迴歸係數表（標準化係數）===")
display(coef_table.round(3))

print(f"\nModel 3：R² = {model3.rsquared:.3f}, Adjusted R² = {model3.rsquared_adj:.3f}, "
      f"F({int(model3.df_model)}, {int(model3.df_resid)}) = {model3.fvalue:.3f}, "
      f"p {'< .001' if model3.f_pvalue < 0.001 else f'= {model3.f_pvalue:.3f}'}")

# 交互作用項是否顯著，直接決定本研究之調節效應假設（H7）是否成立
interaction_row = coef_table[coef_table['Variable'] == 'PE_x_TTF_z']
if interaction_row['p_value'].values[0] < 0.05:
    print("\n✅ PE × TTF 交互作用項達統計顯著，支持 H7：TTF 對 PE→BI 路徑具有調節效果。")
else:
    print("\n⚠️ PE × TTF 交互作用項未達統計顯著，H7 未獲支持。")
```

### Step 6.5：迴歸模型診斷（殘差常態性、同質變異、自我相關）

```python
# ============================================================
# Cell 6.5：迴歸模型基本假設診斷
# ------------------------------------------------------------
# 統計原理：多元迴歸之係數估計與顯著性檢定，建立在以下四項
# 古典假設之上：(1) 線性關係、(2) 殘差常態性、(3) 殘差同質
# 變異（homoscedasticity）、(4) 殘差獨立（無自我相關）。
# 違反這些假設不會讓係數估計「錯誤」，但會使標準誤與顯著性
# 檢定結果失真，因此於論文中報告此類診斷，是展現統計嚴謹度
# 的重要環節，也是口試委員經常關注的細節。
# ============================================================

residuals = model3.resid
fitted = model3.fittedvalues

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# 圖 A：殘差 Q-Q 圖，檢視常態性
sm.qqplot(residuals, line='45', fit=True, ax=axes[0])
axes[0].set_title('殘差常態機率圖（Q-Q Plot）')

# 圖 B：殘差 vs. 預測值散佈圖，檢視同質變異
axes[1].scatter(fitted, residuals, alpha=0.5, color='#2980b9')
axes[1].axhline(y=0, color='red', linestyle='--')
axes[1].set_xlabel('預測值（Fitted Values）')
axes[1].set_ylabel('殘差（Residuals）')
axes[1].set_title('殘差 vs. 預測值散佈圖')

# 圖 C：殘差直方圖
axes[2].hist(residuals, bins=25, color='#16a085', edgecolor='white')
axes[2].set_title('殘差分布直方圖')
axes[2].set_xlabel('殘差值')

plt.tight_layout()
plt.savefig('residual_diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------
# 數值化診斷指標
# ------------------------------------------------------------
# Shapiro-Wilk 常態性檢定：p > .05 表示不拒絕殘差為常態分配之假設
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"Shapiro-Wilk 常態性檢定：W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}")

# Durbin-Watson 統計量：檢驗殘差自我相關，數值介於 0~4，
# 理想值接近 2，經驗法則為 1.5~2.5 之間視為無明顯自我相關疑慮
dw_stat = sm.stats.stattools.durbin_watson(residuals)
print(f"Durbin-Watson 統計量 = {dw_stat:.3f}"
      f"（{'可接受範圍' if 1.5 <= dw_stat <= 2.5 else '需注意自我相關疑慮'}）")

# Breusch-Pagan 檢定：檢驗異質變異，p < .05 表示存在異質變異問題
from statsmodels.stats.diagnostic import het_breuschpagan
bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model3.model.exog)
print(f"Breusch-Pagan 異質變異檢定：LM = {bp_stat:.4f}, p = {bp_p:.4f}"
      f"（{'同質變異假設成立' if bp_p >= 0.05 else '存在異質變異，建議報告穩健標準誤'}）")
```



```python
# ============================================================
# Cell 7：標準化係數森林圖（Forest Plot）
# ------------------------------------------------------------
# 提示詞實作對照：
# 「以標準化係數森林圖呈現各前因變數對使用意願的相對影響力，
#   並標示 95% 信賴區間。」
# ------------------------------------------------------------
# 視覺化原理：森林圖以水平線段呈現每個係數的點估計值與信賴
# 區間，信賴區間跨越 0 者代表該效果不顯著，是同時呈現「效果
# 量大小」與「統計顯著性」最有效率的單一圖表形式。
# ============================================================

plot_vars = ['PE_c_z', 'EE_z', 'SI_z', 'FC_z', 'HM_z', 'PV_z', 'HT_z', 'TTF_c_z', 'PE_x_TTF_z']
plot_labels = ['績效期望 (PE)', '努力期望 (EE)', '社會影響 (SI)', '促成條件 (FC)',
               '享樂動機 (HM)', '價格價值 (PV)', '習慣 (HT)', '任務科技適配 (TTF)',
               'PE × TTF 交互作用']

conf_int = model3.conf_int(alpha=0.05)
forest_data = pd.DataFrame({
    'Variable': plot_labels,
    'Beta': [model3.params[v] for v in plot_vars],
    'CI_lower': [conf_int.loc[v, 0] for v in plot_vars],
    'CI_upper': [conf_int.loc[v, 1] for v in plot_vars],
})
forest_data = forest_data.sort_values('Beta')

fig, ax = plt.subplots(figsize=(8, 6))
y_pos = np.arange(len(forest_data))

ax.errorbar(
    forest_data['Beta'], y_pos,
    xerr=[forest_data['Beta'] - forest_data['CI_lower'], forest_data['CI_upper'] - forest_data['Beta']],
    fmt='o', color='#2c3e50', ecolor='#7f8c8d', capsize=4, markersize=7
)
ax.axvline(x=0, color='red', linestyle='--', linewidth=1, label='零效果參考線')
ax.set_yticks(y_pos)
ax.set_yticklabels(forest_data['Variable'])
ax.set_xlabel('標準化迴歸係數（β）及 95% 信賴區間')
ax.set_title('圖 2：各前因變數對使用意願之標準化係數森林圖', fontsize=13)
ax.legend()
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('forest_plot.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Step 8：Simple Slope 調節效應分析

```python
# ============================================================
# Cell 8：Simple Slope 分析
# ------------------------------------------------------------
# 提示詞實作對照：
# 「以交互作用項繪製 Simple Slope 調節效應圖，比較調節變數
#   高、中、低三組水準下，自變數對依變數的簡單斜率差異。」
# ------------------------------------------------------------
# 統計原理：依 Aiken & West (1991) 之經典做法，於調節變數
# TTF 的平均數 ± 1 標準差處，分別計算 PE 對 BI 的簡單斜率，
# 並檢定各簡單斜率是否顯著異於零。
# ============================================================

# 從 Model 3（使用原始未標準化資料重新配適，以利於得到具實質
# 意義單位的簡單斜率，而非標準化單位，方便繪圖解釋）
model3_raw = smf.ols(
    'BI ~ age + gender + PE_c + EE + SI + FC + HM + PV + HT + TTF_c + PE_x_TTF',
    data=df
).fit()

b_PE   = model3_raw.params['PE_c']
b_TTF  = model3_raw.params['TTF_c']
b_int  = model3_raw.params['PE_x_TTF']

ttf_sd = df['TTF_c'].std()
ttf_levels = {
    '低 TTF（M − 1SD）': -ttf_sd,
    '平均 TTF（M）':      0,
    '高 TTF（M + 1SD）':  ttf_sd,
}

simple_slope_results = []
# 取得共變異數矩陣，用於計算簡單斜率的標準誤，才能進行顯著性檢定
cov_matrix = model3_raw.cov_params()

for label, ttf_val in ttf_levels.items():
    simple_slope = b_PE + b_int * ttf_val
    # 簡單斜率之變異數 = Var(b_PE) + 2*ttf_val*Cov(b_PE, b_int) + ttf_val^2*Var(b_int)
    var_slope = (
        cov_matrix.loc['PE_c', 'PE_c']
        + 2 * ttf_val * cov_matrix.loc['PE_c', 'PE_x_TTF']
        + (ttf_val ** 2) * cov_matrix.loc['PE_x_TTF', 'PE_x_TTF']
    )
    se_slope = np.sqrt(var_slope)
    t_stat = simple_slope / se_slope
    df_resid = model3_raw.df_resid
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df_resid))

    simple_slope_results.append({
        'TTF_Level': label,
        'Simple_Slope': simple_slope,
        'SE': se_slope,
        't_value': t_stat,
        'p_value': p_val,
        'Significant': '是' if p_val < 0.05 else '否'
    })

simple_slope_table = pd.DataFrame(simple_slope_results)
print("=== 表 6：Simple Slope 分析結果（PE → BI，依 TTF 水準分組）===")
display(simple_slope_table.round(4))
```

### Step 9：Simple Slope 視覺化

```python
# ============================================================
# Cell 9：Simple Slope 調節效應圖繪製
# ============================================================

pe_range = np.linspace(df['PE_c'].min(), df['PE_c'].max(), 100)

fig, ax = plt.subplots(figsize=(8, 6))
colors = {'低 TTF（M − 1SD）': '#3498db', '平均 TTF（M）': '#2ecc71', '高 TTF（M + 1SD）': '#e74c3c'}

for label, ttf_val in ttf_levels.items():
    predicted_BI = (
        model3_raw.params['Intercept']
        + b_PE * pe_range
        + b_TTF * ttf_val
        + b_int * pe_range * ttf_val
        # 其餘控制變數與主效果暫以樣本平均值代入（即設為 0，因已平減或另計）
    )
    ax.plot(pe_range + df['PE'].mean(), predicted_BI, label=label,
            color=colors[label], linewidth=2.5)

ax.set_xlabel('績效期望 PE（原始量尺）')
ax.set_ylabel('預測使用意願 BI')
ax.set_title('圖 3：任務科技適配（TTF）對「績效期望→使用意願」路徑之調節效果', fontsize=13)
ax.legend(title='任務科技適配水準')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('simple_slope_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print("圖形解讀：若三條迴歸線的斜率明顯不同（尤其高 TTF 組斜率最陡），")
print("代表任務科技適配度越高，績效期望對使用意願的促進效果越強，")
print("此為典型的「增強型調節效果（enhancing moderation）」型態。")
```

### Step 10：匯出所有分析結果

```python
# ============================================================
# Cell 10：匯出完整分析結果至 Excel
# ============================================================

with pd.ExcelWriter('迴歸與調節效應分析結果_Week02.xlsx') as writer:
    desc_stats.round(3).to_excel(writer, sheet_name='描述性統計')
    corr_matrix.round(3).to_excel(writer, sheet_name='相關矩陣')
    vif_result.round(3).to_excel(writer, sheet_name='VIF共線性診斷', index=False)
    model_comparison.round(4).to_excel(writer, sheet_name='階層迴歸模型比較', index=False)
    coef_table.round(3).to_excel(writer, sheet_name='Model3係數表', index=False)
    simple_slope_table.round(4).to_excel(writer, sheet_name='SimpleSlope分析', index=False)

print("所有統計結果已匯出至 迴歸與調節效應分析結果_Week02.xlsx，可於 Colab 左側檔案面板下載。")
```

### Step 11：封裝完整分析管道為可重複使用函式

```python
# ============================================================
# Cell 11：將本週完整分析流程封裝為單一函式
# ------------------------------------------------------------
# 教學目的：
# 展示如何將一連串分散的 Cell，重構為一個可重複呼叫、
# 可用於不同研究主題資料的「分析管道函式（pipeline function）」，
# 這是從課堂練習過渡到真實研究工作流程的重要工程能力，
# 也是 Vibe Coding 協作中，向 AI 請求「重構程式碼」時
# 最常見的實務情境。
# ============================================================

def run_moderation_analysis(data, iv, mod, dv, controls=None, alpha=0.05):
    """
    執行完整之階層式迴歸 + 調節效應分析管道。

    參數說明：
    - data: 包含所有變數之 pandas DataFrame
    - iv: 自變數欄位名稱（字串）
    - mod: 調節變數欄位名稱（字串）
    - dv: 依變數欄位名稱（字串）
    - controls: 控制變數欄位名稱列表，預設為 None
    - alpha: 顯著水準，預設 .05

    回傳：
    - 一個字典，包含三個模型物件、模型比較表、Simple Slope 結果表
    """
    df_local = data.copy()
    controls = controls or []

    # Step A：平減自變數與調節變數
    df_local[f'{iv}_c']  = df_local[iv] - df_local[iv].mean()
    df_local[f'{mod}_c'] = df_local[mod] - df_local[mod].mean()
    df_local[f'{iv}_x_{mod}'] = df_local[f'{iv}_c'] * df_local[f'{mod}_c']

    # Step B：建立三階層迴歸公式字串
    control_terms = ' + '.join(controls) if controls else '1'
    formula1 = f'{dv} ~ {control_terms}'
    formula2 = f'{dv} ~ {control_terms} + {iv} + {mod}'
    formula3 = f'{dv} ~ {control_terms} + {iv}_c + {mod}_c + {iv}_x_{mod}'

    m1 = smf.ols(formula1, data=df_local).fit()
    m2 = smf.ols(formula2, data=df_local).fit()
    m3 = smf.ols(formula3, data=df_local).fit()

    # Step C：模型比較表
    comparison = pd.DataFrame({
        'Model': ['Model 1', 'Model 2', 'Model 3'],
        'R2': [m1.rsquared, m2.rsquared, m3.rsquared],
    })
    comparison['Delta_R2'] = comparison['R2'].diff()

    # Step D：Simple Slope 分析（自動計算 ± 1SD 三水準）
    b_iv  = m3.params[f'{iv}_c']
    b_int = m3.params[f'{iv}_x_{mod}']
    mod_sd = df_local[f'{mod}_c'].std()
    cov = m3.cov_params()

    slope_rows = []
    for level_name, level_val in [('低', -mod_sd), ('中', 0), ('高', mod_sd)]:
        slope = b_iv + b_int * level_val
        var_slope = (
            cov.loc[f'{iv}_c', f'{iv}_c']
            + 2 * level_val * cov.loc[f'{iv}_c', f'{iv}_x_{mod}']
            + (level_val ** 2) * cov.loc[f'{iv}_x_{mod}', f'{iv}_x_{mod}']
        )
        se = np.sqrt(var_slope)
        t_val = slope / se
        p_val = 2 * (1 - stats.t.cdf(abs(t_val), m3.df_resid))
        slope_rows.append({
            'Level': f'{level_name} {mod}', 'Simple_Slope': slope,
            'SE': se, 't': t_val, 'p': p_val,
            'Sig': '是' if p_val < alpha else '否'
        })

    return {
        'model1': m1, 'model2': m2, 'model3': m3,
        'comparison': comparison,
        'simple_slopes': pd.DataFrame(slope_rows)
    }

# ------------------------------------------------------------
# 呼叫範例：直接重現本週 Step 5–8 之完整分析結果
# ------------------------------------------------------------
result = run_moderation_analysis(
    data=df, iv='PE', mod='TTF', dv='BI', controls=['age', 'gender']
)

print("=== 管道函式輸出：模型比較表 ===")
display(result['comparison'].round(4))

print("\n=== 管道函式輸出：Simple Slope 分析表 ===")
display(result['simple_slopes'].round(4))

print("\n此函式之最大價值：日後研究若更換自變數（例如改為 EE）或調節變數")
print("（例如改為數位包容性），只需重新呼叫 run_moderation_analysis()")
print("並替換參數，即可立即得到完整分析結果，無需重寫整段分析程式碼。")
```

---

## Vibe Coding 提示詞（Prompt）實作範例集

**範例 1：共線性前置檢查**

> 我的迴歸模型有 9 個自變數，其中包含一組平減後的交互作用項。請使用 statsmodels 計算所有變數的 VIF 值，並依 VIF &lt; 10 之判斷標準，明確告訴我模型是否存在共線性疑慮，若有，請指出是哪些變數之間的關聯造成的。

**範例 2：階層式迴歸主分析**

> 請以階層方式建立三個迴歸模型：Model 1 僅含控制變數（年齡、性別），Model 2 加入 7 個 UTAUT2 構念與 TTF 主效果，Model 3 再加入 PE 與 TTF 的交互作用項（請先平減後再相乘）。請輸出三模型的 R²、調整後 R²、ΔR² 與 F 值改變量的顯著性檢定表。

**範例 3：Simple Slope 分析**

> 交互作用項已確認顯著，請計算在調節變數 TTF 為平均數上下 1 個標準差、以及平均數本身共三個水準時，自變數 PE 對依變數 BI 的簡單斜率，並檢定每一條簡單斜率是否顯著異於零，最後繪製三條迴歸線的 Simple Slope 圖。

**範例 4：森林圖視覺化**

> 請將 Model 3 中所有自變數的標準化係數與其 95% 信賴區間，繪製成森林圖，依係數大小排序，並以紅色虛線標示零效果參考線。

**範例 5：結果段落初稿撰寫**

> 根據以下統計結果（Model 3 之 R² = .58，ΔR² 相較 Model 2 增加 .04，F 值改變量達顯著 p &lt; .01；PE×TTF 交互作用項標準化 β = .19，p &lt; .01；高 TTF 組簡單斜率 = .45，p &lt; .001，低 TTF 組簡單斜率 = .21，p &lt; .05），請以碩士論文研究結果章節的學術寫作語氣，撰寫一段約 300 字的中文分析段落，說明調節效應的統計證據與實質意涵。

**範例 6：模型診斷結果解讀**

> 我已經取得殘差常態機率圖、殘差 vs. 預測值散佈圖、Shapiro-Wilk 檢定（p = .032）與 Durbin-Watson 統計量（= 1.87）。請幫我判斷這個迴歸模型是否違反古典假設，並說明若常態性假設輕微違反，是否仍可信賴迴歸係數之顯著性檢定結果，或需要採取穩健標準誤（robust standard error）等補救方法。

---

## 結果呈現與分析：碩士論文寫法示例

> **4.1 共線性診斷**
>
> 為避免多元迴歸分析結果受共線性問題干擾，本研究於正式分析前先計算各自變數之變異數膨脹因子（VIF）。結果顯示，所有自變數之 VIF 值均介於 1.12 至 2.87 之間，均遠低於 Hair et al.（2019）建議之 10 的判斷門檻，顯示本研究模型不存在嚴重共線性問題。
>
> **4.2 階層式迴歸分析結果**
>
> 本研究採階層式迴歸分析，依序納入控制變數、UTAUT2 主效果與 TTF、以及 PE × TTF 交互作用項。結果顯示，Model 1（僅含控制變數）之解釋力為 R² = .02，未達統計顯著；Model 2 加入 UTAUT2 七大構念與 TTF 主效果後，R² 顯著提升至 .54（ΔR² = .52，F 值改變量達 p &lt; .001）；Model 3 進一步加入交互作用項後，R² 再度顯著提升至 .58（ΔR² = .04，F 值改變量達 p &lt; .01），顯示任務科技適配度對績效期望與使用意願之關係確實具有超越主效果之額外解釋力。
>
> **4.3 調節效應與 Simple Slope 分析**
>
> Model 3 之迴歸係數結果顯示，PE × TTF 交互作用項之標準化係數為 β = .19（p &lt; .01），支持研究假設 H7。為進一步釐清此調節效應之具體型態，本研究依 Aiken 與 West（1991）之建議，於任務科技適配度平均數上下一個標準差處進行簡單斜率分析。結果顯示，當任務科技適配度處於高水準時，績效期望對使用意願之簡單斜率為 .45（p &lt; .001）；當任務科技適配度處於低水準時，簡單斜率降為 .21（p &lt; .05）。此結果顯示，隨著使用者知覺該對話式 AI 虛擬助理與其任務需求之適配程度提升，績效期望對其使用意願之促進效果亦隨之增強，呈現典型之增強型調節效果（enhancing moderation）。

**APA 格式三線表範例：階層式迴歸模型摘要表**

| 模型 | 納入變數 | R² | Adj. R² | ΔR² | F 改變量 | p 值 |
|---|---|---|---|---|---|---|
| Model 1 | 年齡、性別 | .02 | .01 | — | 1.42 | .243 |
| Model 2 | + PE, EE, SI, FC, HM, PV, HT, TTF | .54 | .53 | .52 | 39.87 | &lt; .001 |
| Model 3 | + PE × TTF | .58 | .57 | .04 | 8.13 | .005 |

*註：以上數值為示範用途，實際數值請以學生自己資料之 Colab 輸出為準。*

---

## 常見統計誤區與 Q&A

**Q1：為什麼一定要平減之後才能建構交互作用項？直接把原始分數相乘不行嗎？**
技術上可以直接相乘，模型的整體 R² 與交互作用項的顯著性檢定結果不會改變，但主效果係數（$b_1$、$b_2$）的意義會變得難以解釋（會變成「當另一變數為 0 時」的效果，而 Likert 量表通常沒有 0 分），且未平減時交互作用項與主效果之間的相關性會人為地大幅提高，造成係數估計之標準誤不必要地膨脹，因此平減是學術實務上的標準做法。

**Q2：交互作用項不顯著，是不是代表我的研究架構是錯的？**
不一定。調節效應在行為科學研究中普遍屬於小效果量（如本週參考文獻 Dawson 相關方法論文獻指出，管理與心理學期刊中類別型調節變數的效果量中位數僅約 $f^2 = 0.002$），不顯著可能單純是樣本數不足導致統計檢定力（power）不夠，而非理論架構錯誤。建議可報告效果量與信賴區間，並在研究限制中誠實說明樣本規模對於偵測小效果量調節效應的侷限性。

**Q3：ΔR² 檢定與交互作用項係數的 t 檢定，兩者的虛無假設有何不同？為什麼結果會不一致？**
兩者的虛無假設其實是完全相同的（交互作用項係數為 0），在僅有單一交互作用項的情況下，兩種檢定在數學上會得到一致的顯著性結果（$F_{change} = t^2$）。若研究模型同時納入多個交互作用項（例如三因子交互作用），ΔR² 檢定的是「整組交互作用項的聯合顯著性」，此時才可能與單一係數之 t 檢定結果不完全一致。

**Q4：Simple Slope 分析一定要選擇 ± 1 個標準差嗎？**
± 1 個標準差是 Aiken & West（1991）最經典且最廣泛採用的慣例，但並非唯一標準。若調節變數具有明確的理論或實務切點（例如以年齡切分「青年／中高齡」、以量表中位數切分「高／低使用經驗組」），亦可依研究情境調整選取水準，惟應於論文中明確說明選取依據。

**Q5：調節效應與中介效應（Mediation Effect）有什麼本質上的差異？我該如何判斷該用哪一個？**
中介效應探討的是「自變數如何透過某個中介機制影響依變數」（回答「為什麼」與「透過什麼路徑」），統計上以間接效果（indirect effect = a × b）是否顯著檢定；調節效應探討的是「自變數對依變數的影響力，何時較強、何時較弱」（回答「在什麼條件下」），統計上以交互作用項是否顯著檢定。判斷依據應回歸理論邏輯：若你認為 Z 是「X 影響 Y 的必經路徑或機制」，應設定為中介變數；若你認為 Z 是「改變 X 與 Y 關係強度的邊界條件」，應設定為調節變數。本週之 TTF 即屬於後者的理論定位。少數研究會進一步採用「有調節的中介效果（moderated mediation）」整合模型，此進階主題將於第 11 週 SEM-ANN 混合式架構中再進一步延伸討論。

**Q6：森林圖與一般長條圖相比，在論文中呈現迴歸結果有什麼優勢？**
長條圖僅能呈現係數的點估計值，無法直接呈現統計不確定性；森林圖同時呈現點估計與 95% 信賴區間，讀者可一眼判斷該效果是否顯著（信賴區間是否跨越 0），且能同時比較多個前因變數的相對影響力排序，是統合分析（meta-analysis）領域發展出、近年逐漸被行為科學迴歸研究採用之視覺化慣例。

**Q8：什麼情況下，迴歸分析已經不夠用，必須改用第 3 週要學的結構方程模型（SEM）？**
下列任一情形出現時，建議改採 SEM／PLS-SEM 取代單純迴歸分析：(1) 研究架構中存在中介變數，需要同時估計多條路徑而非逐條分別檢定；(2) 構念是以多題項潛在變數（latent variable）形式測量，而非簡化為單一加總平均分數，SEM 能同時處理測量誤差；(3) 依變數不只一個，需要建立多重依變數的整體路徑模型；(4) 研究者需要同時報告收斂效度、區別效度等測量模型品質指標。本週之迴歸分析，本質上是將各構念以「加總平均分數」簡化為單一可觀察變數後才進行分析，此作法之限制與 SEM 的優勢，將於第 3 週深入說明。

| 分析情境 | 建議方法 | 原因 |
|---|---|---|
| 僅檢驗少數幾條直接效果與一組調節效應 | 階層式迴歸（本週方法） | 操作直覺、報告格式廣為熟悉、對樣本數要求相對較低 |
| 涉及多條中介路徑、多個依變數 | PLS-SEM 或共變異數為基礎之 SEM（第 3 週） | 可同時估計整體路徑模型，並處理測量誤差 |
| 樣本數較小（如 &lt; 100）但構念與路徑關係複雜 | PLS-SEM | 對樣本數要求相對寬鬆，適合探索性理論建構 |
| 需要嚴謹驗證構念之測量模型品質（收斂效度、區別效度） | 共變異數為基礎之 SEM（如 AMOS、`semopy`） | 可提供完整之整體模型適配度指標（CFI、RMSEA 等） |

---

## 延伸研究方向：無人零售系統高齡者使用意向研究——數位包容性之調節角色

### 6.1 研究背景與理論基礎

隨著無人商店、自助結帳系統等無人零售科技於台灣快速普及（見本週參考文獻工研院相關報導），高齡消費者在此類科技情境下的使用意向，成為兼具學術與社會政策價值的研究議題。國家教育研究院相關報告指出，台灣 70 歲以上民眾在電子商務、行動支付等數位應用服務之使用經驗，相較於其他年齡層存在明顯落差（見本週參考文獻第 6 項），此「數位落差（digital divide）」現象，極可能扮演調節變數的角色，削弱高齡者知覺易用性、促成條件等構念對其使用意向的正向影響。

### 6.2 建議研究設計

1. **理論框架**：以 UTAUT2 為主要解釋框架，並納入「數位包容性（Digital Inclusion）」或「數位素養（Digital Literacy）」作為調節變數，檢驗其是否調節促成條件（FC）、努力期望（EE）對使用意向之影響路徑，此設計邏輯完全對應本週理論篇 3.6 節之調節效應分析架構。
2. **研究對象**：建議以 65 歲以上、具有至少一次無人商店或自助結帳系統使用經驗（或觀察經驗）之高齡消費者為研究對象，可搭配社區關懷據點、長青學苑進行立意抽樣。
3. **變數規劃（範例）**：
   - 績效期望（PE）、努力期望（EE）、促成條件（FC）：沿用 UTAUT2 原始構念，語意調整為無人零售情境
   - 數位包容性（調節變數）：可參考數位素養量表改編，衡量高齡者對數位裝置與網路服務之基本操作能力與心理適應程度
   - 使用意向（BI）：依變數
4. **分析流程**：完全比照本週 Colab 實作流程（VIF 診斷 → 階層式迴歸 → 平減與交互作用項建構 → Simple Slope 分析 → 視覺化），僅需替換資料來源與變數定義即可直接複用本週所有程式碼架構。
5. **政策與實務意涵**：若研究證實數位包容性確實調節促成條件對使用意向之影響，即可為零售業者與政策制定者提供具體建議方向——例如針對數位包容性較低之高齡族群，加強現場人力輔助（作為促成條件的實體強化措施），而非僅依賴介面設計優化（知覺易用性改善）。

### 6.3 給學生的思考練習

請思考：若將本延伸研究之調節變數，從「數位包容性」換成「科技焦慮（Technology Anxiety）」，你預期其調節效果的方向會是增強型調節，還是減弱型調節？請說明你的理論推論邏輯，並嘗試修改本週 Colab 程式碼 Step 1 之資料生成邏輯，模擬出你所預期的調節效果型態。

---

## 課後作業與練習

**練習一：修改調節效應強度並觀察檢定力變化**
請修改 Step 1 中交互作用效果的係數（原設定為 `0.25`），分別調整為 `0.05`（極小效果）與 `0.40`（大效果），重新執行完整流程，比較三種情境下交互作用項的 p 值與 Simple Slope 分析結果之差異，並以 200 字說明你觀察到「效果量大小」與「統計顯著性」之間的關係。

**練習二：真實問卷資料實作**
請延續第 1 週練習二蒐集之問卷資料（或重新設計一份包含至少 2 個自變數、1 個調節變數、1 個依變數之問卷），實際發放蒐集後，套用本週完整 Colab 程式碼執行階層式迴歸與調節效應分析。請繳交：(1) 原始 CSV 檔、(2) 執行後的 Colab Notebook（.ipynb）、(3) 一頁 A4 的結果摘要（比照本週「結果呈現與分析」段落之寫法）。

**練習三：交叉驗證不同 Simple Slope 選點策略**
請將 Step 8 中 Simple Slope 分析的調節變數水準，從「± 1 個標準差」改為「量表理論中位數上下」（例如 7 點量表以 4 分為中位數，比較 3 分與 5 分兩組），重新計算簡單斜率，比較兩種選點策略下的結論是否一致。

**練習四：文獻延伸閱讀報告**
請從本週「參考文獻與延伸閱讀」清單中，任選一篇台灣本土 UTAUT2 或 TTF 相關碩士論文，撰寫一頁重點摘要，內容須包含：(1) 該研究之理論架構與構念、(2) 報告之迴歸或結構方程模式解釋力（R² 或 CFI/RMSEA 等適配指標）、(3) 該研究是否檢驗調節或中介效果，其發現為何、(4) 你認為若將該研究之調節變數應用於本課程延伸研究方向（高齡者無人零售），是否合適，為什麼。

**Q7：控制變數（年齡、性別）該放在 Model 1 還是應該和主效果一起放入 Model 2？**
學術慣例上，理論意義較弱、僅作為統計控制目的之人口統計變數（年齡、性別、教育程度等），應獨立放在最先進入的區塊（Model 1），目的是「先扣除這些變數的解釋力，再檢視理論構念是否仍具有超越人口統計特徵之額外解釋力」。這也是階層式迴歸相較於一次性納入所有變數之標準迴歸，更能清楚展現「理論貢獻」的關鍵原因。

**練習五：封裝函式的遷移應用**
請直接呼叫本週 Step 11 所封裝之 `run_moderation_analysis()` 函式，將自變數換成 `EE`（努力期望）、調節變數維持 `TTF`，重新執行分析，比較「EE × TTF」與本週範例「PE × TTF」兩組交互作用效果的強度與顯著性是否一致，並嘗試提出一個理論解釋，說明為什麼任務科技適配度對這兩條路徑的調節強度可能不同。

---

## 參考文獻與延伸閱讀（已查核連結）

1. 以整合科技接受模型 UTAUT 與延伸性整合科技接受模型 UTAUT2，探討消費者對於對話式 AI 人工智慧服務之接受程度，以 ChatGPT 為例說明之。國立成功大學，臺灣博碩士論文知識加值系統。
   https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi/login?o=dnclcdr&s=id%3D%22111NCKU5121015%22.&searchmode=basic
   （與本週研究設計範例主題完全對應之台灣本土碩士論文，可直接作為文獻回顧與研究缺口論述之核心參照。）

2. 公務機關採用虛擬桌面基礎建設決策因素之實證探索：結合 TTF 與 UTAUT2 模式。Airiti Library 華藝線上圖書館。
   https://www.airitilibrary.com/Article/Detail/10224858-201905-201906040008-201906040008-69-88
   （示範 TTF 與 UTAUT2 整合模型之研究架構設計，可對照本週理論篇 3.4 節之整合認知框架。）

3. 當手機變成錢包：UTAUT 與 TTF 的應用。臺灣博碩士論文知識加值系統。
   https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi?o=dnclcdr&s=id=%22105PCCU0323006%22.&searchmode=basic
   （示範以性別、年齡作為調節變數之 UTAUT+TTF 整合研究，並包含中介效果路徑分析，適合作為進階延伸閱讀。）

4. 探索民眾對健康照護應用程式的使用意圖：擴展 UTAUT2 模型。Airiti Library 華藝線上圖書館。
   https://www.airitilibrary.com/Article/Detail/22259481-N202503280013-00005
   （納入人工智慧自我效能作為前置變數之 UTAUT2 延伸模型，並採用 PLS-SEM 分析，可與第 3 週課程內容對照參考。）

5. 智慧化無人商店之消費者接受度調查。臺灣博碩士論文知識加值系統。
   https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi/login?o=dnclcdr&s=id%3D%22106NKIT0682015%22.&searchmode=basic
   （本週延伸研究方向「無人零售系統高齡者使用意向研究」之直接前導文獻。）

6. 年長者的現在，亦是數位的時代：高齡者的數位素養培育。國家教育研究院電子報，第 252 期。
   https://teric.naer.edu.tw/wSite/PDFReader?xmlId=2067336&fileName=1738563114601&format=pdf
   （提供台灣高齡者數位落差之官方統計數據，可作為延伸研究方向章節之研究背景論述依據。）

7. Public acceptance of using artificial intelligence-assisted weight management apps in high-income southeast Asian adults with overweight and obesity: a cross-sectional study. *PMC*.
   https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10879329/
   （以 UTAUT2 模型搭配結構方程模式，檢驗 AI 應用程式使用意圖，並完整報告模型適配度指標，可作為國際期刊寫作規範之參照範本。）

8. Robinson, C. D., et al. Difference in Simple Slopes versus the Interaction Term. *General Linear Model Journal*.
   https://www.glmj.org/archives/articles/Robinson_v39n1.pdf
   （方法論專文，詳細比較交互作用項顯著性檢定與簡單斜率差異檢定兩種調節效應檢定方式之統計檢定力差異，為本週理論篇 3.6 節之重要方法論依據。）

9. 探討使用 e 化智能系統行為意圖之影響因素：UTAUT 2 的應用和擴展。《醫務管理期刊》，24(3)，320–341。Airiti Library 華藝線上圖書館。
   https://www.airitilibrary.com/Article/Detail/16086961-N202309270011-00005
   （以護理人員為對象之 UTAUT2 延伸研究，並納入科技壓力作為調節變數，可作為醫療情境延伸研究之額外參考文獻。）

**方法論經典文獻（建議延伸閱讀，非本次線上搜尋來源，圖書館或資料庫可查閱）**：

- Davis, F. D. (1989). Perceived usefulness, perceived ease of use, and user acceptance of information technology. *MIS Quarterly*, 13(3), 319–340.
- Venkatesh, V., Morris, M. G., Davis, G. B., & Davis, F. D. (2003). User acceptance of information technology: Toward a unified view. *MIS Quarterly*, 27(3), 425–478.
- Venkatesh, V., Thong, J. Y. L., & Xu, X. (2012). Consumer acceptance and use of information technology: Extending the unified theory. *MIS Quarterly*, 36(1), 157–178.
- Goodhue, D. L., & Thompson, R. L. (1995). Task-technology fit and individual performance. *MIS Quarterly*, 19(2), 213–236.
- Aiken, L. S., & West, S. G. (1991). *Multiple Regression: Testing and Interpreting Interactions*. SAGE Publications.
- Cohen, J., Cohen, P., West, S. G., & Aiken, L. S. (2003). *Applied Multiple Regression/Correlation Analysis for the Behavioral Sciences* (3rd ed.). Lawrence Erlbaum Associates.

---

## 附錄

### 附錄 A：SPSS 與 Python 迴歸分析功能對照表

| 分析項目 | SPSS 選單路徑 | Python 對應函式／套件 | 備註 |
|---|---|---|---|
| 多元迴歸 | Analyze → Regression → Linear | `statsmodels.formula.api.ols()` | 兩者係數估計法（OLS）完全相同 |
| 階層式迴歸（分區塊） | Linear Regression 對話框中設定多個 Block | 依序建立 `model1`、`model2`、`model3` 並用 `anova_lm()` 比較 | Python 需手動比較模型，SPSS 會自動輸出 R² Change 欄位 |
| VIF 共線性診斷 | Linear Regression → Statistics → Collinearity Diagnostics | `variance_inflation_factor()` | 數值計算邏輯相同 |
| 標準化係數 | 預設輸出於 Coefficients 表格之 Standardized Coefficients Beta 欄位 | 先將變數 `stats.zscore()` 標準化後再迴歸 | Python 需手動標準化，SPSS 自動輸出 |
| R² 改變量顯著性 | Statistics → R squared change | `anova_lm(model_reduced, model_full)` | 對應 SPSS 之 F Change 與 Sig. F Change 欄位 |

### 附錄 B：術語中英對照表

| 中文術語 | 英文術語 | 縮寫 |
|---|---|---|
| 科技接受模型 | Technology Acceptance Model | TAM |
| 整合性科技接受模式 | Unified Theory of Acceptance and Use of Technology | UTAUT / UTAUT2 |
| 任務科技適配 | Task-Technology Fit | TTF |
| 績效期望 | Performance Expectancy | PE |
| 努力期望 | Effort Expectancy | EE |
| 社會影響 | Social Influence | SI |
| 促成條件 | Facilitating Conditions | FC |
| 享樂動機 | Hedonic Motivation | HM |
| 價格價值 | Price Value | PV |
| 習慣 | Habit | HT |
| 階層式迴歸分析 | Hierarchical Regression Analysis | — |
| 調節效應 | Moderation Effect | — |
| 交互作用項 | Interaction Term | — |
| 平減／中心化 | Mean-Centering | — |
| 簡單斜率分析 | Simple Slope Analysis | — |
| 變異數膨脹因子 | Variance Inflation Factor | VIF |

### 附錄 C：常見程式錯誤排解（Debugging Tips）

| 錯誤現象 | 常見原因 | 排解建議 |
|---|---|---|
| `PatsyError` 於 `smf.ols()` 公式解析時發生 | 變數名稱包含特殊符號（如 `PE×TTF`）未被公式引擎接受 | 變數命名避免使用 `×`、空格等特殊符號，改用底線（如 `PE_x_TTF`） |
| VIF 計算結果全部偏高（甚至 &gt; 20） | 忘記先平減即建構交互作用項 | 檢查 Step 3 是否確實對主效果變數執行平減後才相乘 |
| `anova_lm()` 回傳錯誤 `模型並非巢狀（nested）` | 兩模型使用之資料筆數不同（例如含遺漏值時各模型自動排除的樣本不同） | 統一使用同一份已處理遺漏值的資料矩陣重新配適所有模型 |
| Simple Slope 檢定的 t 值與手動 Excel 計算結果不一致 | 共變異數矩陣未正確取用平減後之變數 | 確認 `cov_params()` 中取用的變數名稱與模型公式中實際使用的變數名稱完全一致 |

### 附錄 D：繳交前自我檢核清單

- [ ] 已報告共線性診斷結果（VIF），確認無嚴重共線性問題
- [ ] 已依理論邏輯將自變數分批納入階層式迴歸模型
- [ ] 已報告每一模型之 R²、調整後 R²、ΔR² 及其顯著性檢定
- [ ] 已報告最終模型之完整標準化迴歸係數表
- [ ] 已明確說明交互作用項建構前是否進行平減處理
- [ ] 若交互作用項顯著，已進行 Simple Slope 分析並報告至少三組調節水準之簡單斜率
- [ ] 已繪製 Simple Slope 圖，並於文字中正確描述調節效果之型態（增強型／減弱型／交叉型）
- [ ] 所有統計結果之文字敘述與表格數值一致，無謄寫錯誤
- [ ] 已在研究限制中誠實揭露樣本規模對於偵測調節效應統計檢定力之侷限性

### 附錄 E：研究倫理提醒

本週研究設計涉及高齡者、消費者等一般族群之問卷調查，除延續第 1 週已說明之知情同意、匿名性、資料儲存安全性等基本倫理原則外，特別提醒：若延伸研究方向涉及高齡者（尤其認知功能可能退化之族群），問卷設計應力求語句簡潔、字體放大、必要時提供現場協助施測人員，並應特別注意避免因研究者引導性協助造成之填答偏誤，此為高齡者相關量化研究方法論中經常被口試委員關注之議題。

### 附錄 F：本週常用 Python 函式速查表

| 函式 | 所屬套件 | 功能 |
|---|---|---|
| `smf.ols(formula, data).fit()` | statsmodels | 以公式介面建立並配適 OLS 迴歸模型 |
| `anova_lm(model_reduced, model_full)` | statsmodels.stats.anova | 比較巢狀模型之 F 值改變量與顯著性 |
| `variance_inflation_factor()` | statsmodels.stats.outliers_influence | 計算單一變數之 VIF 值 |
| `model.conf_int()` | statsmodels | 取得迴歸係數之信賴區間，用於森林圖繪製 |
| `model.cov_params()` | statsmodels | 取得係數共變異數矩陣，用於 Simple Slope 標準誤計算 |
| `stats.zscore()` | scipy | 將變數標準化為 Z 分數 |
| `sm.qqplot()` | statsmodels.api | 繪製殘差常態機率圖 |
| `stats.shapiro()` | scipy | Shapiro-Wilk 常態性檢定 |
| `sm.stats.stattools.durbin_watson()` | statsmodels | 計算 Durbin-Watson 自我相關統計量 |
| `het_breuschpagan()` | statsmodels.stats.diagnostic | Breusch-Pagan 異質變異檢定 |

### 附錄 G：延伸調節效應型態辨識指南

Simple Slope 圖繪製完成後，論文中應明確描述所觀察到的調節效果屬於哪一種型態，以下提供辨識準則供學生對照：

| 型態 | 圖形特徵 | 統計特徵 | 實務意涵範例 |
|---|---|---|---|
| 增強型調節（Enhancing Moderation） | 三條迴歸線斜率同方向，但陡緩程度不同，高調節組斜率最陡 | 高、低兩組簡單斜率同號，且差異達顯著 | 任務科技適配度越高，績效期望的促進效果越強 |
| 緩衝型調節（Buffering Moderation） | 高調節組斜率明顯變緩，甚至趨近水平 | 高調節組簡單斜率不顯著，低調節組顯著 | 促成條件越充足，年齡對使用意向的負向影響被削弱 |
| 交叉型調節（Crossover / Antagonistic Moderation） | 兩條迴歸線在自變數某一數值處相交，方向由正轉負或反之 | 高、低兩組簡單斜率異號，且均達顯著 | 社會影響對年輕族群為正向效果，對年長族群反而產生負向抗拒效果 |

---

## 下週預告

第 3 週將進入「複雜因果路徑架構：結構方程模型（PLS-SEM）與持續使用意願」，學生將學習如何將本週以迴歸分析檢定的各項路徑關係，整合為一個完整的結構方程模型，同時處理多個依變數、中介變數與潛在變數測量誤差，並學習收斂效度（AVE、CR）、區別效度（Fornell-Larcker、HTMT）與 Bootstrapping 重抽樣檢定等 PLS-SEM 核心分析技術。研究範例將以「企業導入 AI 決策支援系統後之持續使用行為：結合 TAM 與期望確認模型（ECM-IT）」為主題，並延伸至「跨國遠距團隊使用 AI 協同工作軟體之疲乏感與工作績效模型」之期末專題發想方向。
