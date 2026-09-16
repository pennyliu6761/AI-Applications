# 第 4 週：專家系統多準則排序——AHP、Fuzzy AHP 與 TOPSIS 模擬

> 課程模組：第二模組｜知識驅動型 AI 與多準則決策系統（第 4–7 週）
> 本週定位：從第一模組「資料驅動型」的量表與路徑分析，轉向「專家知識驅動型」的決策科學方法。學生將學習如何將決策者（或領域專家）的主觀成對比較判斷，透過嚴謹的數學程序轉化為客觀化的準則權重（AHP），並處理專家判斷中固有的模糊與猶豫（Fuzzy AHP），最終結合 TOPSIS 逼近理想解排序法，對候選方案進行綜合評分與排序，這是模擬企業高階主管複雜決策邏輯的核心方法論工具箱。

> 教材版本：v1.0｜適用對象：在職專班研究方法與 AI 應用課程｜先修基礎：無須先修統計學，本週為全新的決策科學方法論脈絡
> 使用工具：Google Colab（Python 3）｜主要套件：`numpy`、`pandas`、`scipy`、`matplotlib`（本週核心演算法皆以 `numpy` 手動實作，不依賴特定商用套件，所有程式碼已實際測試驗證可正常執行）

---

## 目錄

1. [學習目標](#學習目標)
2. [本週知識地圖](#本週知識地圖)
3. [理論基礎篇](#理論基礎篇)
   1. [3.1 多準則決策（MCDM）與專家系統決策架構總覽](#31-多準則決策mcdm與專家系統決策架構總覽)
   2. [3.2 層級分析法（AHP）：成對比較與 Saaty 九點量表](#32-層級分析法ahp成對比較與-saaty-九點量表)
   3. [3.3 特徵向量法與最大特徵值](#33-特徵向量法與最大特徵值)
   4. [3.4 一致性指標（CI）與一致性比率（CR）](#34-一致性指標ci與一致性比率cr)
   5. [3.5 模糊層級分析法（Fuzzy AHP）：三角模糊數與 Chang 擴展分析法](#35-模糊層級分析法fuzzy-ahp三角模糊數與-chang-擴展分析法)
   6. [3.6 TOPSIS 逼近理想解排序法](#36-topsis-逼近理想解排序法)
   7. [3.7 AHP-TOPSIS 整合流程與敏感度分析](#37-ahp-topsis-整合流程與敏感度分析)
   8. [3.8 群體決策整合方法：多位專家判斷如何合併](#38-群體決策整合方法多位專家判斷如何合併)
   9. [3.9 論文中 AHP／Fuzzy AHP／TOPSIS 章節的標準寫法架構](#39-論文中-ahpfuzzy-ahptopsis-章節的標準寫法架構)
4. [研究設計實例：科技製造業關鍵設備供應商評選之專家決策系統架構](#研究設計實例科技製造業關鍵設備供應商評選之專家決策系統架構)
5. [Colab 實作環境建置](#colab-實作環境建置)
6. [Colab 實作：Step by Step 完整程式碼](#colab-實作step-by-step-完整程式碼)
7. [Vibe Coding 提示詞（Prompt）實作範例集](#vibe-coding-提示詞prompt實作範例集)
8. [結果呈現與分析：碩士論文寫法示例](#結果呈現與分析碩士論文寫法示例)
9. [常見統計誤區與 Q&A](#常見統計誤區與-qa)
10. [延伸研究方向：跨國綠色冷鏈物流中心選址評估決策模型](#延伸研究方向跨國綠色冷鏈物流中心選址評估決策模型)
11. [課後作業與練習](#課後作業與練習)
12. [參考文獻與延伸閱讀（已查核連結）](#參考文獻與延伸閱讀已查核連結)
13. [附錄](#附錄)
14. [下週預告](#下週預告)

---

## 學習目標

完成本週課程後，學生應能夠：

1. 說明多準則決策（MCDM）方法之理論定位，並理解其與第一模組資料驅動方法在知識論基礎上的根本差異。
2. 說明層級分析法（AHP）之成對比較原理、Saaty 九點量表語意，並能自行設計一份專家成對比較問卷。
3. 使用 Python（`numpy`）手動實作特徵向量法求解 AHP 權重，並計算一致性指標（CI）與一致性比率（CR），判斷專家判斷之邏輯一致性是否可接受。
4. 說明模糊層級分析法（Fuzzy AHP）處理專家主觀猶豫之數學原理，並使用 Chang（1996）擴展分析法實作模糊權重求解。
5. 理解並能辨識 Chang 擴展分析法之已知方法論限制（零權重問題），並知悉學術界對此之批判與替代做法。
6. 使用 Python 實作 TOPSIS 演算法，包含向量正規化、加權、正負理想解計算與相對貼近係數（Closeness Coefficient）排序。
7. 執行 AHP-TOPSIS 整合分析之敏感度分析，檢驗排序結果之穩健性。
8. 依照論文「研究方法」與「研究結果與討論」章節寫法，將 AHP、Fuzzy AHP、TOPSIS 之統計輸出轉譯為具學術規範的文字敘述、準則權重雷達圖與方案排序長條圖。

---

## 本週知識地圖

| 構面 | 內容 | 對應方法 | 對應 Python 實作 |
|---|---|---|---|
| 問題結構化 | 將決策問題拆解為目標、準則、方案之層級架構 | 層級架構圖（Hierarchy Structure） | 手動定義 |
| 準則權重求解（精確） | 將專家成對比較判斷轉化為客觀化權重 | AHP 特徵向量法 | `numpy.linalg.eig` |
| 邏輯一致性檢驗 | 檢驗專家判斷是否存在邏輯矛盾 | CI、CR、RI 對照表 | 自訂函式 |
| 準則權重求解（模糊） | 處理專家判斷之主觀猶豫與模糊性 | 三角模糊數、Chang 擴展分析法 | 自訂函式 |
| 方案綜合評分排序 | 依多項準則對候選方案進行綜合排序 | TOPSIS（正負理想解、貼近係數） | 自訂函式 |
| 穩健性檢驗 | 檢驗排序結果對權重變動之敏感程度 | 敏感度分析（Sensitivity Analysis） | 迴圈模擬 |
| 結果視覺化 | 呈現準則權重結構與方案排序 | 雷達圖、長條圖 | `matplotlib` |

---

## 理論基礎篇

### 3.1 多準則決策（MCDM）與專家系統決策架構總覽

多準則決策（Multi-Criteria Decision Making, MCDM）是決策科學（Decision Science）領域的核心方法論家族，處理的問題型態為：**在多個（通常互相衝突）的評估準則下，從有限個候選方案中選出最適方案或進行優劣排序**。這與第一模組（第 1–3 週）所學之資料驅動方法，在知識論基礎上有根本性的差異：

| 比較構面 | 資料驅動型方法（第一模組） | 知識驅動型方法（第二模組，本週起） |
|---|---|---|
| 知識來源 | 大樣本問卷調查資料 | 少數（通常 3–10 位）領域專家之判斷 |
| 適用情境 | 探討普遍母體之心理與行為規律 | 探討特定決策情境下之最適方案選擇 |
| 樣本規模 | 需要大樣本以確保統計檢定力 | 不需大樣本，重視專家代表性與判斷品質 |
| 核心統計邏輯 | 機率推論、假設檢定 | 數學規劃、距離測度、模糊集合論 |
| 典型研究產出 | 驗證或拒絕一組理論假設 | 產出一組可直接應用之決策權重或排序結果 |

本週學習之 AHP、Fuzzy AHP、TOPSIS，皆屬於「專家系統（Expert System）」取徑之決策支援方法，其核心精神是將人類專家難以量化的「經驗」與「直覺判斷」，透過結構化的比較程序與數學運算，轉化為可重現、可溝通、可稽核的量化決策依據，這正是本課程「決策智能與混合式 AI」定位下，知識驅動型 AI 的具體實踐。

**典型 MCDM 問題的層級結構**：

```
                總目標（Goal）
                     │
        ┌────────────┼────────────┐
     準則一         準則二         準則三 …（Criteria）
        │            │            │
   ┌────┴────┐  ┌────┴────┐  ┌────┴────┐
 方案A 方案B 方案C ...（Alternatives，各方案在每個準則下皆須評估）
```

### 3.2 層級分析法（AHP）：成對比較與 Saaty 九點量表

層級分析法（Analytic Hierarchy Process, AHP）由 Saaty（1980）提出，核心邏輯是：與其要求專家「直接」對 $n$ 個準則同時進行權重分配（此對人類認知負荷極大，且準確度低），不如將問題拆解為 $\binom{n}{2}$ 次的「兩兩比較」，每次僅需回答「準則 A 相對於準則 B，重要程度為何」，透過此種局部化判斷，可大幅降低專家認知負荷，並藉由後續的一致性檢驗程序，確保這些局部判斷組合起來具有整體邏輯一致性。

**Saaty 九點量表（Saaty's 1–9 Scale）**：

| 尺度值 | 語意定義 | 尺度值 | 語意定義 |
|---|---|---|---|
| 1 | 同等重要（Equal Importance） | 2, 4, 6, 8 | 相鄰尺度之中間值 |
| 3 | 稍微重要（Moderate Importance） | 1/3, 1/5, 1/7, 1/9 | 對應之倒數判斷（B 相對 A 重要） |
| 5 | 頗為重要（Strong Importance） | — | — |
| 7 | 極為重要（Very Strong Importance） | — | — |
| 9 | 絕對重要（Extreme Importance） | — | — |

**成對比較矩陣（Pairwise Comparison Matrix）**：設有 $n$ 個準則，專家針對每一配對 $(C_i, C_j)$ 給出比較值 $a_{ij}$ ，代表「準則 $i$ 相對於準則 $j$ 的重要程度」，並依定義 $a_{ji} = 1/a_{ij}$ （倒數性質）與 $a_{ii} = 1$ （對角線恆為 1），形成一個 $n \times n$ 的正倒值矩陣（reciprocal matrix）：

$$
A = \begin{bmatrix}
1 & a_{12} & \cdots & a_{1n} \\
1/a_{12} & 1 & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
1/a_{1n} & 1/a_{2n} & \cdots & 1
\end{bmatrix}
$$

若專家之判斷完全邏輯一致（完美傳遞性，即 $a_{ik} = a_{ij} \times a_{jk}$ 對所有 $i,j,k$ 成立），則此矩陣理論上為秩 1（rank 1）矩陣，其唯一非零特徵值恰等於 $n$ 。然而現實中專家判斷難免存在不一致，因此矩陣之最大特徵值 $\lambda_{max}$ 通常會略大於 $n$ ，此一差距正是後續一致性檢驗之基礎。

### 3.3 特徵向量法與最大特徵值

AHP 求解準則權重之標準方法為**特徵向量法（Eigenvector Method）**：求解成對比較矩陣 $A$ 之最大特徵值 $\lambda_{max}$ 所對應之特徵向量 $w$ ，並將其正規化（使各元素總和為 1），即為各準則之權重向量：

$$
Aw = \lambda_{max} w
$$

此方法之數學直覺為：權重向量 $w$ 應滿足「以矩陣 $A$ 對 $w$ 做線性變換後，結果應與原本的 $w$ 成比例（僅相差一個純量 $\lambda_{max}$ ）」，這代表 $w$ 是整個成對比較矩陣所隱含之「相對重要性結構」的最佳數學代表。雖然實務上也存在近似解法（如列向量幾何平均法、列和正規化平均法），但特徵向量法因具有嚴謹之數學理論基礎與 Saaty 本人之原始推導，至今仍是學術論文中最廣泛採用、最具方法論正當性的權重求解方式。

### 3.4 一致性指標（CI）與一致性比率（CR）

**一致性指標（Consistency Index, CI）**：

$$
CI = \frac{\lambda_{max} - n}{n - 1}
$$

$\lambda_{max}$ 越接近 $n$ （矩陣維度），代表專家判斷越接近完全邏輯一致，CI 值也就越接近 0。

**隨機一致性指標（Random Index, RI）**：Saaty 透過大量隨機產生正倒值矩陣的蒙地卡羅模擬，計算出不同矩陣維度 $n$ 下，「純粹隨機判斷」所產生的平均 CI 值，作為比較基準：

| $n$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| RI | 0 | 0 | 0.58 | 0.90 | 1.12 | 1.24 | 1.32 | 1.41 | 1.45 | 1.49 |

**一致性比率（Consistency Ratio, CR）**：

$$
CR = \frac{CI}{RI}
$$

**判斷標準（Saaty, 1980 之經典建議）**：CR ≤ 0.10 時，代表專家判斷之邏輯一致性在可接受範圍內，可繼續採用該組權重進行後續分析；若 CR &gt; 0.10，代表專家判斷存在顯著邏輯矛盾（例如認為「A 比 B 重要」、「B 比 C 重要」，卻又認為「C 比 A 重要」），應請專家重新檢視並修正其成對比較判斷，此為 AHP 分析流程中不可省略的品質控管步驟，也是論文口試中最常被要求現場說明的統計細節之一。

### 3.5 模糊層級分析法（Fuzzy AHP）：三角模糊數與 Chang 擴展分析法

傳統 AHP 要求專家給出「精確」的成對比較數值（如恰好是 3 或恰好是 5），但現實中專家之判斷往往帶有猶豫與模糊性（例如「A 大概比 B 重要一些，但沒有很確定」）。**模糊層級分析法（Fuzzy AHP）** 以三角模糊數（Triangular Fuzzy Number, TFN）取代精確數值，刻畫此種語意上的不確定性。

**三角模糊數（TFN）**：以三元組 $\tilde{a} = (l, m, u)$ 表示，其中 $l$ 為最悲觀估計、 $m$ 為最可能估計、 $u$ 為最樂觀估計，其隸屬函數（membership function）為以 $m$ 為峰值、向兩側線性遞減之三角形分布。

**Saaty 尺度轉換為三角模糊數之對照表（綜合自 van Laarhoven & Pedrycz, 1983；Chang, 1996 之慣例）**：

| Saaty 尺度 | 語意 | 對應三角模糊數 $(l, m, u)$ |
|---|---|---|
| 1 | 同等重要 | (1, 1, 1) |
| 3 | 稍微重要 | (2, 3, 4) |
| 5 | 頗為重要 | (4, 5, 6) |
| 7 | 極為重要 | (6, 7, 8) |
| 9 | 絕對重要 | (9, 9, 9) |
| 2, 4, 6, 8 | 相鄰尺度之中間值 | (1,2,3)、(3,4,5)、(5,6,7)、(7,8,9) |

倒數關係亦相應轉換：若 $\tilde{a}_{ij} = (l, m, u)$ ，則 $\tilde{a}_{ji} = (1/u, 1/m, 1/l)$ 。

**Chang（1996）擴展分析法（Extent Analysis Method）求解流程**：

1. **計算模糊合成擴展值（Fuzzy Synthetic Extent Value）**：對每一準則 $i$ ，計算：

$$
S_i = \sum_{j=1}^{n} \tilde{a}_{ij} \otimes \left[ \sum_{i=1}^{n}\sum_{j=1}^{n} \tilde{a}_{ij} \right]^{-1}
$$

即該準則所有列之模糊數總和，除以整個矩陣所有元素之模糊數總和（模糊除法運算對三角模糊數而言，即為對應之 $l, m, u$ 分別運算）。

2. **計算模糊數之可能度（Degree of Possibility）**：比較任兩個模糊合成擴展值 $S_i = (l_1,m_1,u_1)$ 與 $S_j = (l_2,m_2,u_2)$ ， $S_i \geq S_j$ 之可能度定義為：

$$
V(S_i \geq S_j) =
\begin{cases}
1 & \text{若 } m_1 \geq m_2 \\
0 & \text{若 } l_2 \geq u_1 \\
\dfrac{l_2 - u_1}{(m_1 - u_1) - (m_2 - l_2)} & \text{其他情況}
\end{cases}
$$

3. **求解權重向量**：第 $i$ 個準則之權重（未正規化）為其對其餘所有準則之可能度的最小值：

$$
w_i' = \min_{j \neq i} V(S_i \geq S_j)
$$

最後將所有 $w_i'$ 正規化（除以總和），即得到 Fuzzy AHP 之最終權重向量。

**Chang 擴展分析法之已知方法論限制**：值得特別注意的是，Wang, Luo, & Hua（2008，發表於 *European Journal of Operational Research*，見本週參考文獻）透過數學證明與實例分析指出，Chang 擴展分析法在特定資料型態下（尤其是準則間重要性差異較懸殊時），容易產生某些準則權重被計算為「恰好等於 0」的不合理結果，此現象並非程式錯誤，而是該方法「取最小可能度」這一運算步驟之數學特性所導致的已知缺陷。本週 Colab 實作將刻意呈現此一現象，讓學生實際觀察並理解此方法論限制，而非僅止於文獻上的抽象認識。

### 3.6 TOPSIS 逼近理想解排序法

TOPSIS（Technique for Order Preference by Similarity to Ideal Solution）由 Hwang & Yoon（1981）提出，核心邏輯為：**最佳方案應同時具備「與正理想解（最優方案組合）距離最近」且「與負理想解（最劣方案組合）距離最遠」兩項特性**。完整計算流程如下：

**步驟一：建構決策矩陣並進行向量正規化（Vector Normalization）**

$$
r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{k=1}^{m} x_{kj}^2}}
$$

此正規化方式可消除不同準則之間單位與量綱不一致的問題（例如「價格」以新台幣計、「交期」以天數計）。

**步驟二：建構加權正規化決策矩陣**

$$
v_{ij} = w_j \times r_{ij}
$$

其中 $w_j$ 即為 AHP（或 Fuzzy AHP）求解得出之準則權重，這正是 AHP 與 TOPSIS 兩方法整合串接的關鍵接口。

**步驟三：決定正理想解（Positive Ideal Solution, PIS）與負理想解（Negative Ideal Solution, NIS）**

對「效益型準則」（數值越大越好，如品質、服務）：PIS 取該準則之最大值，NIS 取最小值；對「成本型準則」（數值越小越好，如價格、風險）：PIS 取最小值，NIS 取最大值。

**步驟四：計算各方案與正／負理想解之歐氏距離**

$$
D_i^+ = \sqrt{\sum_{j=1}^{n} (v_{ij} - v_j^+)^2} \qquad D_i^- = \sqrt{\sum_{j=1}^{n} (v_{ij} - v_j^-)^2}
$$

**步驟五：計算相對貼近係數（Closeness Coefficient, CC）**

$$
CC_i = \frac{D_i^-}{D_i^+ + D_i^-}
$$

$CC_i$ 介於 0 到 1 之間，值越接近 1，代表該方案越接近正理想解、越遠離負理想解，即為越優方案。所有候選方案依 $CC_i$ 值由大到小排序，即為 TOPSIS 之最終決策建議排序。

### 3.7 AHP-TOPSIS 整合流程與敏感度分析

AHP 與 TOPSIS 之整合，本質上是一種「兩階段混合決策模型」：**第一階段以 AHP（或 Fuzzy AHP）求解「準則」的相對權重，第二階段以 TOPSIS 將此權重套用於「方案」的綜合評分與排序**。此種整合方式讓 AHP 專注於處理「準則重要性」此一較適合人類專家直覺判斷的問題，同時讓 TOPSIS 處理「多方案、多準則、數值型」資料之系統化排序，截長補短、分工明確，是 MCDM 文獻中應用最廣泛的整合模式之一（見本週參考文獻中之台灣本土供應商評選實證研究）。

**敏感度分析（Sensitivity Analysis）之必要性**：由於 AHP 權重來自專家主觀判斷，論文審查與口試委員經常關心「若準則權重有所變動，最終排序結果是否依然穩健？」。標準做法是逐一調整某一準則之權重（例如上下浮動 10%、20%），並等比例調整其餘準則權重以維持總和為 1，重新執行 TOPSIS 計算，觀察方案排序是否發生變化。若排序結果在合理權重擾動範圍內依然穩定，即可增強研究結論之說服力；若排序對權重變動高度敏感，則應在研究討論中誠實揭露此一限制。

### 3.8 群體決策整合方法：多位專家判斷如何合併

當研究設計涉及多位專家（如本週範例之 5 位主管）分別填答成對比較問卷時，需要一套原則明確的方法將個別判斷矩陣整合為單一之群體判斷矩陣，再進行後續之特徵向量求解。學術文獻上常見兩種整合策略：

**幾何平均整合法（Aggregation of Individual Judgments, AIJ）**：先將每位專家針對同一組準則配對之判斷值，取幾何平均數，形成一個整合後的群體成對比較矩陣，再對此矩陣求解特徵向量權重與一致性檢驗：

$$
a_{ij}^{group} = \left(\prod_{k=1}^{K} a_{ij}^{(k)}\right)^{1/K}
$$

其中 $K$ 為專家人數。此法之優點是計算簡潔、且整合後之矩陣仍保有正倒值性質，是目前應用最廣泛之群體決策整合方式，本週 Step 1 之範例矩陣即可視為已完成此一整合程序之結果。

**個別優先向量整合法（Aggregation of Individual Priorities, AIP）**：先讓每一位專家各自完成獨立的 AHP 分析（各自求解出一組權重向量與 CR 值），再將 $K$ 組權重向量取算術平均或加權平均，整合為最終權重。此法之優點是可個別檢視每位專家之判斷一致性（CR 值），並可依專家之決策地位或專業程度賦予不同之整合權重，但計算流程相對繁瑣。

實務選擇建議：若研究目的著重於「反映整體專家群體的共識判斷」，建議採用幾何平均整合法（AIJ）；若研究設計刻意希望保留個別專家判斷之異質性（例如比較不同部門專家意見之差異），則應採用個別優先向量整合法（AIP），並可進一步分析不同專家群體間之權重差異是否具有統計或實務意義。

### 3.9 論文中 AHP／Fuzzy AHP／TOPSIS 章節的標準寫法架構

1. **決策層級架構圖**：呈現目標、準則、方案之層級關係圖。
2. **專家成對比較矩陣與一致性檢驗結果**：呈現 CR 值，說明是否通過一致性檢驗。
3. **準則權重排序表**：呈現 AHP（與 Fuzzy AHP，如有執行）之權重值與排序。
4. **準則權重雷達圖**：視覺化呈現各準則之相對權重結構。
5. **TOPSIS 決策矩陣與正負理想解**：呈現原始決策矩陣、正規化矩陣、加權矩陣。
6. **方案排序結果表**：呈現各方案之 $D^+$ 、 $D^-$ 、 $CC$ 值與最終排序。
7. **方案相對貼近度排序長條圖**：視覺化呈現各方案之 $CC$ 值排序。
8. **敏感度分析結果**：呈現權重擾動下排序結果之穩健性檢驗。

---

## 研究設計實例：科技製造業關鍵設備供應商評選之專家決策系統架構

### 4.1 研究背景與動機

科技製造業（如半導體、面板、精密機械製造）之關鍵生產設備採購決策，往往涉及鉅額資本支出與長期供應鏈依存關係，錯誤的供應商評選決策可能導致產能損失、品質風險與交期延誤等重大經營衝擊。既有供應商評選研究多聚焦於一般製造業或傳統產業情境（見本週參考文獻中之台灣本土碩士論文），本研究延伸此一方法論至科技製造業關鍵設備採購之高風險、高資本密集決策情境，建構一套結合 AHP 與 TOPSIS 之專家決策系統架構。

### 4.2 研究目的

1. 建構科技製造業關鍵設備供應商評選之準則層級架構。
2. 透過專家成對比較問卷，以 AHP 特徵向量法求解各評選準則之相對權重，並檢驗其一致性。
3. 以 Fuzzy AHP 作為 AHP 之穩健性交叉驗證，處理專家判斷之主觀猶豫。
4. 以 TOPSIS 法整合準則權重，對候選供應商進行綜合評分與排序。
5. 執行敏感度分析，檢驗最終排序結果對準則權重變動之穩健程度。

### 4.3 評選準則架構

參考本週參考文獻中畢威寧（2005）之經典供應商績效評估研究架構，並考量科技製造業設備採購之特殊性，本研究採用以下四項評選準則：

| 準則代碼 | 準則名稱 | 操作型定義 | 準則類型 |
|---|---|---|---|
| C1 | 品質（Quality） | 設備之製程良率、精密度與可靠度表現 | 效益型（越高越好） |
| C2 | 價格（Price） | 設備採購總成本（含後續維護費用） | 成本型（越低越好） |
| C3 | 交期（Delivery） | 從下單至完成安裝驗收之時間表現 | 效益型（以評分表示，越高越好） |
| C4 | 服務（Service） | 售後技術支援、教育訓練與備品供應能力 | 效益型（越高越好） |

### 4.4 候選方案與研究對象

本研究以四家通過初步資格審查之關鍵設備候選供應商（以下以 Supplier_A ~ Supplier_D 代稱，實際研究應以真實廠商匿名代號呈現）作為候選方案，並邀請該企業採購部門、製程工程部門、品保部門等 5 位具決策參與資格之資深主管作為 AHP 成對比較問卷之填答專家。

### 4.5 專家問卷設計與資料蒐集程序

AHP 成對比較問卷之填答，建議依循以下程序：(1) 先向專家說明各準則之操作型定義，確保理解一致；(2) 以結構化問卷形式，逐一詢問專家對 $\binom{4}{2}=6$ 組準則配對之相對重要性判斷（採 Saaty 九點量表）；(3) 若為多位專家填答，可採幾何平均法（Geometric Mean）整合多位專家之個別判斷矩陣為一組群體判斷矩陣；(4) 針對每一份個別與整合後之判斷矩陣，均應計算 CR 值進行一致性檢驗，CR &gt; 0.10 者應請該專家重新檢視填答。

### 4.6 資料分析流程規劃

```
決策層級架構定義（目標、準則、方案）
        │
        ▼
專家成對比較問卷設計與資料蒐集（Saaty 九點量表）
        │
        ▼
【AHP】特徵向量法求解準則權重 + CR 一致性檢驗
        │
        ├── CR > 0.10 ──► 請專家重新檢視判斷矩陣
        │
        ▼ CR ≤ 0.10
【Fuzzy AHP】三角模糊數轉換 + Chang 擴展分析法求解模糊權重（穩健性交叉驗證）
        │
        ▼
蒐集各候選方案於各準則下之實際績效數據（決策矩陣）
        │
        ▼
【TOPSIS】向量正規化 → 加權 → 正負理想解 → 貼近係數排序
        │
        ▼
敏感度分析（權重擾動 ± 10%～20%）
        │
        ▼
繪製準則權重雷達圖 + 方案排序長條圖
        │
        ▼
撰寫研究結果與討論段落
```

### 4.7 準則獨立性假設之限制

AHP 方法論存在一項重要的前提假設：**各準則彼此獨立，不存在交互影響關係**。然而在本研究情境中，「品質」與「服務」兩項準則之間，實務上可能存在一定程度的關聯（例如設備品質越高，可能意味著該供應商技術實力越強，連帶影響其售後服務能力），此種準則間的潛在關聯性，是 AHP 方法論在應用上經常被口試委員質疑之處。若研究者認為準則間交互影響關係是研究問題的核心（而非可忽略的次要因素），則應考慮改採第 5、6 週將介紹之 DEMATEL 與 ANP（Analytic Network Process，網路分析法）等進階方法，此為本課程刻意將 AHP（假設準則獨立）安排在 DEMATEL（處理準則間因果關係）之前的教學設計邏輯——先讓學生掌握假設較單純的方法，再逐步引入更貼近決策現實複雜度的進階技術。

---

## Colab 實作環境建置

```python
# ============================================================
# Cell 0：Colab 環境建置與套件匯入
# ------------------------------------------------------------
# 說明：本週核心演算法（AHP 特徵向量法、Fuzzy AHP Chang 擴展
# 分析法、TOPSIS）皆以 numpy 手動實作，不依賴任何特定商用或
# 小眾套件，這麼做的教學考量是：AHP／TOPSIS 之計算邏輯本身
# 並不複雜，親手實作有助於學生徹底理解每一個統計量背後的
# 數學意義，而非僅止於呼叫黑盒函式。
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 設定中文字型（沿用第 1–3 週相同設定邏輯）
# ------------------------------------------------------------
!wget -q https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf -O /content/NotoSansTC.otf
from matplotlib import font_manager
font_manager.fontManager.addfont('/content/NotoSansTC.otf')
plt.rcParams['font.family'] = 'Noto Sans CJK TC'
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

print("環境建置完成，本週所有分析皆以 numpy／pandas 手動實作。")
```

---

## Colab 實作：Step by Step 完整程式碼

### Step 1：建構專家成對比較矩陣

```python
# ============================================================
# Cell 1：建構 4 準則之專家成對比較矩陣（已整合多位專家判斷）
# ------------------------------------------------------------
# 說明：以下矩陣代表已透過幾何平均法整合 5 位專家判斷後之
# 群體成對比較矩陣。列與欄依序為 品質(C1)、價格(C2)、
# 交期(C3)、服務(C4)。矩陣元素 a[i][j] 代表「準則 i 相對於
# 準則 j 的重要程度」，依 Saaty 九點量表填答。
# ============================================================

criteria = ['品質(C1)', '價格(C2)', '交期(C3)', '服務(C4)']

# 專家群體判斷：品質 > 價格 > 服務 > 交期
A = np.array([
    [1,    2,    4,    3   ],   # 品質 vs 價格/交期/服務
    [1/2,  1,    3,    2   ],   # 價格 vs 品質/交期/服務
    [1/4,  1/3,  1,    1/2 ],   # 交期 vs 品質/價格/服務
    [1/3,  1/2,  2,    1   ],   # 服務 vs 品質/價格/交期
])

pairwise_df = pd.DataFrame(A, index=criteria, columns=criteria)
print("=== 表 1：專家群體成對比較矩陣 ===")
display(pairwise_df.round(3))
```

### Step 2：AHP 特徵向量法求解權重

```python
# ============================================================
# Cell 2：AHP 特徵向量法求解準則權重
# ------------------------------------------------------------
# 提示詞實作對照：
# 「計算專家成對比較矩陣的最大特徵向量並檢驗 C.R.」
# ============================================================

def ahp_eigenvector_weights(matrix):
    """
    以特徵向量法求解 AHP 成對比較矩陣之準則權重。

    回傳：
    - weights: 正規化後之權重向量
    - lambda_max: 最大特徵值
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    # 理論上最大特徵值必為實數且為正，取其實部並找出最大者
    max_idx = np.argmax(eigenvalues.real)
    lambda_max = eigenvalues[max_idx].real
    weights = eigenvectors[:, max_idx].real
    weights = weights / weights.sum()   # 正規化，使權重總和為 1
    return weights, lambda_max

ahp_weights, lambda_max = ahp_eigenvector_weights(A)

ahp_weight_table = pd.DataFrame({
    'Criterion': criteria,
    'AHP_Weight': ahp_weights
}).sort_values('AHP_Weight', ascending=False)

print(f"最大特徵值 λmax = {lambda_max:.4f}（理論上限值為 n = {len(criteria)}）")
print("\n=== 表 2：AHP 準則權重排序 ===")
display(ahp_weight_table.round(4))
```

### Step 3：一致性檢驗（CI、RI、CR）

```python
# ============================================================
# Cell 3：一致性指標與一致性比率檢驗
# ------------------------------------------------------------
# 提示詞實作對照（延續 Step 2）：
# 「並檢驗 C.R.」
# ============================================================

def consistency_check(matrix, lambda_max):
    """
    計算一致性指標（CI）與一致性比率（CR），並依 Saaty (1980)
    建議之 CR <= 0.10 判斷標準，回傳是否通過一致性檢驗。
    """
    n = matrix.shape[0]
    CI = (lambda_max - n) / (n - 1)

    # Saaty (1980) 隨機一致性指標對照表
    RI_table = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12,
                6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = RI_table.get(n, 1.49)   # n > 10 時採用最大已知值作保守估計

    CR = CI / RI if RI > 0 else 0
    is_consistent = CR <= 0.10
    return CI, RI, CR, is_consistent

CI, RI, CR, is_consistent = consistency_check(A, lambda_max)

print(f"一致性指標 CI = {CI:.4f}")
print(f"隨機一致性指標 RI（n={len(criteria)}）= {RI}")
print(f"一致性比率 CR = {CR:.4f}")
print(f"\n{'✅ CR ≤ 0.10，專家判斷通過一致性檢驗，可繼續使用此組權重。' if is_consistent else '⚠️ CR > 0.10，建議請專家重新檢視成對比較判斷。'}")
```

### Step 4：Fuzzy AHP——三角模糊數轉換與 Chang 擴展分析法

```python
# ============================================================
# Cell 4：Fuzzy AHP（三角模糊數 + Chang 擴展分析法）
# ------------------------------------------------------------
# 教學目的：以與 Step 1 完全相同之成對比較判斷為基礎，改採
# 三角模糊數重新求解權重，讓學生對照 AHP（精確數值）與
# Fuzzy AHP（模糊數值）在同一組專家判斷下的權重差異。
# ============================================================

# Saaty 尺度 → 三角模糊數對照表（見理論篇 3.5 節）
TFN_TABLE = {
    1: (1, 1, 1), 2: (1, 2, 3), 3: (2, 3, 4), 4: (3, 4, 5),
    5: (4, 5, 6), 6: (5, 6, 7), 7: (6, 7, 8), 8: (7, 8, 9), 9: (9, 9, 9)
}

def to_tfn(value):
    """將 Saaty 精確尺度值轉換為對應之三角模糊數 (l, m, u)。"""
    if value >= 1:
        v = int(round(value))
        return TFN_TABLE.get(v, (1, 1, 1))
    else:
        v = int(round(1 / value))
        l, m, u = TFN_TABLE.get(v, (1, 1, 1))
        return (1 / u, 1 / m, 1 / l)   # 倒數關係之模糊數轉換

n = len(criteria)
fuzzy_matrix = [[to_tfn(A[i, j]) for j in range(n)] for i in range(n)]

print("=== 模糊化成對比較矩陣（每格為三角模糊數 l, m, u）===")
for i, row in enumerate(fuzzy_matrix):
    print(f"{criteria[i]}: {[tuple(round(x, 2) for x in cell) for cell in row]}")

def chang_extent_analysis(fuzzy_matrix):
    """
    Chang (1996) 擴展分析法求解 Fuzzy AHP 權重。
    詳細步驟說明請見理論篇 3.5 節公式推導。
    """
    n = len(fuzzy_matrix)

    # 步驟一：計算每個準則之模糊合成擴展值 S_i
    row_sums = []
    l_total = m_total = u_total = 0
    for i in range(n):
        l = sum(fuzzy_matrix[i][j][0] for j in range(n))
        m = sum(fuzzy_matrix[i][j][1] for j in range(n))
        u = sum(fuzzy_matrix[i][j][2] for j in range(n))
        row_sums.append((l, m, u))
        l_total += l; m_total += m; u_total += u

    S = [(l / u_total, m / m_total, u / l_total) for (l, m, u) in row_sums]

    # 步驟二：計算兩兩模糊合成擴展值之可能度 V(Si >= Sj)
    def degree_of_possibility(S1, S2):
        l1, m1, u1 = S1
        l2, m2, u2 = S2
        if m1 >= m2:
            return 1.0
        elif l2 >= u1:
            return 0.0
        else:
            return (l2 - u1) / ((m1 - u1) - (m2 - l2))

    V = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                V[i, j] = degree_of_possibility(S[i], S[j])

    # 步驟三：取每個準則對其餘所有準則可能度之最小值，作為未正規化權重
    raw_weights = np.array([
        min(V[i, j] for j in range(n) if j != i) for i in range(n)
    ])
    normalized_weights = raw_weights / raw_weights.sum()

    return S, V, raw_weights, normalized_weights

S_values, V_matrix, raw_w, fuzzy_weights = chang_extent_analysis(fuzzy_matrix)

print("\n=== 模糊合成擴展值 Si（l, m, u）===")
for c, s in zip(criteria, S_values):
    print(f"{c}: ({s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f})")

fuzzy_weight_table = pd.DataFrame({
    'Criterion': criteria,
    'Raw_Weight_minV': raw_w,
    'Fuzzy_AHP_Weight': fuzzy_weights
}).sort_values('Fuzzy_AHP_Weight', ascending=False)

print("\n=== 表 3：Fuzzy AHP（Chang 擴展分析法）準則權重 ===")
display(fuzzy_weight_table.round(4))
```

### Step 5：AHP 與 Fuzzy AHP 權重比較（含零權重現象討論）

```python
# ============================================================
# Cell 5：AHP vs. Fuzzy AHP 權重對照與已知方法論限制討論
# ============================================================

comparison_table = pd.DataFrame({
    'Criterion': criteria,
    'AHP_Weight': ahp_weights,
    'Fuzzy_AHP_Weight': fuzzy_weights,
    'Difference': ahp_weights - fuzzy_weights
})

print("=== 表 4：AHP 與 Fuzzy AHP 權重對照表 ===")
display(comparison_table.round(4))

zero_weight_criteria = comparison_table[comparison_table['Fuzzy_AHP_Weight'] < 0.001]['Criterion'].tolist()
if zero_weight_criteria:
    print(f"\n⚠️ 注意：{zero_weight_criteria} 在 Chang 擴展分析法下權重被計算為趨近 0。")
    print("這並非程式錯誤，而是 Wang, Luo, & Hua (2008) 已於方法論文獻中")
    print("證明之 Chang 擴展分析法已知限制（詳見本週 Q&A 與參考文獻）：")
    print("當準則間重要性差異較懸殊時，'取最小可能度' 之運算步驟")
    print("容易將部分準則權重歸零，即使該準則在專家原始判斷中並非完全不重要。")
    print("實務建議：可同時報告 AHP 與 Fuzzy AHP 兩組結果並交叉比對，")
    print("或改採 Buckley (1985) 之模糊幾何平均法等替代性模糊權重求解方法。")
```

### Step 6：建構候選供應商決策矩陣

```python
# ============================================================
# Cell 6：候選供應商於各準則下之實際績效決策矩陣
# ------------------------------------------------------------
# 資料說明：品質、交期、服務為 1-10 分之專家評分（效益型，
# 越高越好）；價格為新台幣萬元之報價（成本型，越低越好）。
# ============================================================

suppliers = ['Supplier_A', 'Supplier_B', 'Supplier_C', 'Supplier_D']

decision_matrix = pd.DataFrame({
    '品質(C1)': [8, 7, 9, 6],
    '價格(C2)': [70, 60, 85, 50],
    '交期(C3)': [7, 8, 6, 9],
    '服務(C4)': [8, 7, 8, 6],
}, index=suppliers)

# 標示每項準則之類型：True = 效益型（越大越好），False = 成本型（越小越好）
benefit_criteria = [True, False, True, True]

print("=== 表 5：候選供應商決策矩陣（原始績效數據）===")
display(decision_matrix)
```

### Step 7：TOPSIS 演算法完整實作

```python
# ============================================================
# Cell 7：TOPSIS 逼近理想解排序法
# ------------------------------------------------------------
# 提示詞實作對照：
# 「隨後將權重套入 TOPSIS 演算法對候選方案進行評分與長條圖
#   排序。」
# ============================================================

def topsis(decision_matrix, weights, benefit_criteria):
    """
    執行完整 TOPSIS 演算法。

    參數：
    - decision_matrix: pandas DataFrame，列為方案，欄為準則
    - weights: 各準則權重（應與欄位順序對應，總和為 1）
    - benefit_criteria: 布林值列表，True 代表效益型準則

    回傳：包含 D+、D-、CC、排名之結果 DataFrame
    """
    X = decision_matrix.values.astype(float)

    # 步驟一：向量正規化
    norm_X = X / np.sqrt((X ** 2).sum(axis=0))

    # 步驟二：加權正規化決策矩陣
    weighted_X = norm_X * weights

    # 步驟三：決定正理想解（PIS）與負理想解（NIS）
    pis = np.where(benefit_criteria, weighted_X.max(axis=0), weighted_X.min(axis=0))
    nis = np.where(benefit_criteria, weighted_X.min(axis=0), weighted_X.max(axis=0))

    # 步驟四：計算與正／負理想解之歐氏距離
    D_plus = np.sqrt(((weighted_X - pis) ** 2).sum(axis=1))
    D_minus = np.sqrt(((weighted_X - nis) ** 2).sum(axis=1))

    # 步驟五：計算相對貼近係數
    CC = D_minus / (D_plus + D_minus)

    result = pd.DataFrame({
        'D+': D_plus, 'D-': D_minus, 'CC（相對貼近係數）': CC
    }, index=decision_matrix.index)
    result['Rank'] = result['CC（相對貼近係數）'].rank(ascending=False).astype(int)
    return result.sort_values('CC（相對貼近係數）', ascending=False), norm_X, weighted_X, pis, nis

topsis_result, norm_matrix, weighted_matrix, pis, nis = topsis(
    decision_matrix, ahp_weights, benefit_criteria
)

print("=== 表 6：TOPSIS 方案排序結果（採用 AHP 權重）===")
display(topsis_result.round(4))

print(f"\n最終決策建議：{topsis_result.index[0]} 為相對貼近係數最高之候選供應商，建議優先評選。")
```

### Step 8：準則權重雷達圖繪製

```python
# ============================================================
# Cell 8：準則權重雷達圖（Radar Chart）
# ============================================================

angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False).tolist()
ahp_vals = ahp_weights.tolist() + [ahp_weights[0]]
fuzzy_vals = fuzzy_weights.tolist() + [fuzzy_weights[0]]
angles_closed = angles + [angles[0]]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.plot(angles_closed, ahp_vals, 'o-', linewidth=2, label='AHP 權重', color='#2980b9')
ax.fill(angles_closed, ahp_vals, alpha=0.15, color='#2980b9')
ax.plot(angles_closed, fuzzy_vals, 's--', linewidth=2, label='Fuzzy AHP 權重', color='#e74c3c')
ax.fill(angles_closed, fuzzy_vals, alpha=0.10, color='#e74c3c')

ax.set_xticks(angles)
ax.set_xticklabels(criteria, fontsize=11)
ax.set_title('圖 1：AHP 與 Fuzzy AHP 準則權重雷達圖對照', fontsize=13, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig('criteria_weight_radar.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Step 9：方案相對貼近度排序長條圖

```python
# ============================================================
# Cell 9：方案相對貼近度排序長條圖
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))
sorted_result = topsis_result.sort_values('CC（相對貼近係數）', ascending=True)
colors = plt.cm.RdYlGn(sorted_result['CC（相對貼近係數）'] / sorted_result['CC（相對貼近係數）'].max())

bars = ax.barh(sorted_result.index, sorted_result['CC（相對貼近係數）'], color=colors, edgecolor='#2c3e50')
for bar, val in zip(bars, sorted_result['CC（相對貼近係數）']):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
            va='center', fontsize=10)

ax.set_xlabel('相對貼近係數（Closeness Coefficient, CC）')
ax.set_title('圖 2：候選供應商 TOPSIS 相對貼近係數排序圖', fontsize=13)
ax.set_xlim(0, sorted_result['CC（相對貼近係數）'].max() * 1.2)
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('topsis_ranking_bar.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Step 10：敏感度分析

```python
# ============================================================
# Cell 10：敏感度分析——檢驗排序結果對權重變動之穩健性
# ------------------------------------------------------------
# 分析邏輯：逐一調整某一準則權重（在合理範圍內上下浮動），
# 並將其餘準則權重依原比例等比例縮放以維持總和為 1，重新
# 執行 TOPSIS，記錄每次調整後之方案排名是否改變。
# ============================================================

def sensitivity_analysis(decision_matrix, base_weights, benefit_criteria,
                          criterion_idx, delta_range=np.linspace(-0.3, 0.3, 13)):
    """
    對指定準則之權重進行敏感度分析。

    參數：
    - criterion_idx: 欲調整之準則在權重向量中的索引位置
    - delta_range: 權重調整幅度（比例），例如 -0.3 代表降低 30%
    """
    records = []
    for delta in delta_range:
        new_weights = base_weights.copy()
        adjusted = new_weights[criterion_idx] * (1 + delta)
        adjusted = np.clip(adjusted, 0.01, 0.95)   # 避免權重變為負值或超過 1

        remaining_idx = [i for i in range(len(new_weights)) if i != criterion_idx]
        remaining_sum = new_weights[remaining_idx].sum()
        scale_factor = (1 - adjusted) / remaining_sum if remaining_sum > 0 else 0

        new_weights[remaining_idx] = new_weights[remaining_idx] * scale_factor
        new_weights[criterion_idx] = adjusted

        result, *_ = topsis(decision_matrix, new_weights, benefit_criteria)
        top_supplier = result.index[0]
        records.append({
            'Delta': delta,
            f'{criteria[criterion_idx]}_Weight': adjusted,
            'Top_Ranked_Supplier': top_supplier
        })
    return pd.DataFrame(records)

# 以權重最高之準則「品質(C1)」為例，進行敏感度分析
sensitivity_result = sensitivity_analysis(decision_matrix, ahp_weights, benefit_criteria, criterion_idx=0)

print("=== 表 7：品質(C1)準則權重敏感度分析結果 ===")
display(sensitivity_result.round(4))

unique_top = sensitivity_result['Top_Ranked_Supplier'].nunique()
print(f"\n在權重調整範圍 ±30% 內，最優方案共出現 {unique_top} 種不同結果。")
print(f"{'✅ 排序結果相對穩健，未因合理權重擾動而改變最優方案。' if unique_top == 1 else '⚠️ 排序結果對此準則權重變動較為敏感，建議於研究討論中說明。'}")
```

### Step 11：匯出所有分析結果

```python
# ============================================================
# Cell 11：匯出完整分析結果至 Excel
# ============================================================

with pd.ExcelWriter('AHP_FuzzyAHP_TOPSIS分析結果_Week04.xlsx') as writer:
    pairwise_df.round(4).to_excel(writer, sheet_name='專家成對比較矩陣')
    ahp_weight_table.round(4).to_excel(writer, sheet_name='AHP權重', index=False)
    fuzzy_weight_table.round(4).to_excel(writer, sheet_name='FuzzyAHP權重', index=False)
    comparison_table.round(4).to_excel(writer, sheet_name='權重對照', index=False)
    decision_matrix.to_excel(writer, sheet_name='決策矩陣')
    topsis_result.round(4).to_excel(writer, sheet_name='TOPSIS排序結果')
    sensitivity_result.round(4).to_excel(writer, sheet_name='敏感度分析')

print("所有統計結果已匯出至 AHP_FuzzyAHP_TOPSIS分析結果_Week04.xlsx，可於 Colab 左側檔案面板下載。")
print(f"\nAHP 一致性比率 CR = {CR:.4f}（{'通過' if is_consistent else '未通過'}一致性檢驗）")
print(f"最終建議供應商：{topsis_result.index[0]}")
```

---

## Vibe Coding 提示詞（Prompt）實作範例集

**範例 1：AHP 權重與一致性檢驗**

> 我有一個 4×4 的專家成對比較矩陣（Saaty 九點量表），請使用 Python numpy 以特徵向量法求解準則權重，計算最大特徵值 λmax，並依 Saaty (1980) 之隨機一致性指標對照表計算 CI、RI、CR，明確告訴我這組專家判斷是否通過一致性檢驗（CR ≤ 0.10）。

**範例 2：Fuzzy AHP 完整實作**

> 請將同一組成對比較矩陣，依 Saaty 尺度轉換為三角模糊數，並使用 Chang (1996) 擴展分析法求解模糊權重。請完整實作模糊合成擴展值、可能度矩陣、最小可能度取值三個步驟，並提醒我若有任何準則權重被計算為 0，這代表什麼方法論意涵。

**範例 3：TOPSIS 排序與視覺化**

> 請將 AHP 求得的準則權重套入 TOPSIS 演算法，對這四家候選供應商進行評分排序，準則中價格為成本型（越低越好），其餘為效益型。請輸出正規化矩陣、加權矩陣、正負理想解、D+/D-/CC 值，並繪製一張依 CC 值排序的水平長條圖。

**範例 4：準則權重雷達圖**

> 請將 AHP 與 Fuzzy AHP 兩組準則權重，繪製在同一張雷達圖上以利對照比較，兩組數據請用不同顏色與不同線型（實線 vs 虛線）區分，並加上圖例。

**範例 5：敏感度分析**

> 請針對權重最高的準則，在其權重 ±30% 範圍內以 13 個等距離間隔進行敏感度分析，其餘準則權重依原比例等比例調整以維持總和為 1，每次調整後重新執行 TOPSIS，記錄最優方案是否改變，並幫我判斷目前的排序結果是否穩健。

**範例 6：結果段落初稿撰寫**

> 根據以下統計結果（AHP 權重：品質 .467、價格 .277、服務 .160、交期 .095，CR = .011；TOPSIS 排序：Supplier_A CC=.579 第一、Supplier_C CC=.561 第二），請以碩士論文研究結果章節的學術寫作語氣，撰寫一段約 250 字的中文分析段落。

---

## 結果呈現與分析：碩士論文寫法示例

以下段落數值取自本週 Colab 範例程式碼之實際執行結果，供學生對照模仿寫作邏輯（實際數值請以自己資料之 Colab 輸出為準）。

> **4.1 專家判斷一致性檢驗**
>
> 本研究以 5 位具決策參與資格之資深主管，針對品質、價格、交期、服務四項評選準則進行成對比較問卷填答，經幾何平均法整合為群體判斷矩陣後，計算得最大特徵值 λmax = 4.031，一致性指標 CI = .010，依 Saaty（1980）建議之隨機一致性指標（n=4 時 RI = 0.90），求得一致性比率 CR = .011，遠低於 .10 之可接受門檻，顯示專家群體之判斷具有良好之邏輯一致性，可繼續採用此組權重進行後續分析。
>
> **4.2 準則權重分析**
>
> AHP 特徵向量法求解結果顯示，四項準則之權重由高至低依序為：品質（.467）、價格（.277）、服務（.160）、交期（.095），顯示專家群體最重視設備之品質表現，其次為採購成本考量。為交叉驗證此結果之穩健性，本研究進一步以 Chang（1996）擴展分析法求解 Fuzzy AHP 權重，結果顯示品質（.503）、價格（.345）、服務（.152）之權重排序與 AHP 結果一致，惟交期準則之權重被計算為 0。此一現象與 Wang, Luo, & Hua（2008）於方法論文獻中指出之 Chang 擴展分析法已知限制相符——當準則間重要性差異較為懸殊時，「取最小可能度」之運算步驟容易產生零權重結果。本研究因此以 AHP 特徵向量法之權重結果作為後續 TOPSIS 分析之主要依據，並將此一方法論限制於研究限制章節中明確說明。
>
> **4.3 TOPSIS 方案排序結果**
>
> 將 AHP 權重代入 TOPSIS 演算法後，四家候選供應商之相對貼近係數（CC）由高至低依序為：Supplier_A（CC = .579）、Supplier_C（CC = .561）、Supplier_B（CC = .484）、Supplier_D（CC = .439）。Supplier_A 雖非四家供應商中品質評分最高者（品質評分次於 Supplier_C），但因其在價格與服務兩項準則之綜合表現優異，最終獲得最高之相對貼近係數，顯示 TOPSIS 排序結果係綜合多項準則之權衡結果，而非單一準則之直接反映，此正是多準則決策方法相較於單一指標評選之核心價值所在。
>
> **4.4 敏感度分析**
>
> 為檢驗上述排序結果之穩健性，本研究針對權重最高之品質準則，在其權重 ±30% 範圍內、以 13 個等距離間隔進行敏感度分析，其餘準則權重依原比例等比例調整以維持總和為 1。結果顯示，當品質權重調降 20% 以上時，最優方案由 Supplier_A 轉移為 Supplier_B；當品質權重調升 10% 以上時，最優方案則轉移為 Supplier_C；僅在權重調整幅度介於 -15% 至 +5% 之間的相對窄幅區間內，Supplier_A 維持為最優方案。此結果顯示本研究之排序結論對品質準則之權重設定具有中度敏感性，Supplier_A、Supplier_B、Supplier_C 三者之相對貼近係數差距並不算大，此發現提醒研究者：最終決策建議應同時參考 TOPSIS 排序結果與敏感度分析之穩健區間，而非僅呈現單一權重情境下的排序結論，這也凸顯了敏感度分析在多準則決策研究中作為結果穩健性佐證的重要性。

**APA 格式三線表範例：AHP 準則權重與一致性檢驗結果**

| 準則 | AHP 權重 | 排序 | Fuzzy AHP 權重 | 排序 |
|---|---|---|---|---|
| 品質（C1） | .467 | 1 | .503 | 1 |
| 價格（C2） | .277 | 2 | .345 | 2 |
| 服務（C4） | .160 | 3 | .152 | 3 |
| 交期（C3） | .095 | 4 | .000 | 4* |

*註：Fuzzy AHP 之交期準則權重因 Chang 擴展分析法之已知方法論限制被計算為 0，詳見本文第 4.2 節與 Q&A 說明。λmax = 4.031，CI = .010，RI = .90，CR = .011。*

---

## 常見統計誤區與 Q&A

**Q1：我的 CR 值是 0.12，超過了 0.10 的門檻，這代表我的研究就做不下去了嗎？**
不會，這是 AHP 分析流程中的正常品質控管步驟，並非研究失敗。標準處理方式是：回頭檢視該份成對比較矩陣中，哪些配對判斷可能存在邏輯矛盾（可觀察矩陣中是否有明顯與其他判斷不一致的儲存格），將此資訊回饋給填答專家，請其重新檢視並視情況修正判斷後，重新計算 CR 值，直到通過 .10 之門檻為止。這個「檢視—回饋—修正」的過程，本身就是 AHP 方法論確保決策品質的核心機制，論文中如實記錄此一修正過程，反而能展現研究方法的嚴謹度。

**Q2：AHP 的權重和 Fuzzy AHP 的權重不完全一樣，我該用哪一組進行後續的 TOPSIS 分析？**
兩者並非互斥，而是「精確判斷」與「模糊判斷」兩種不同認知假設下的權重估計結果，實務上常見的處理方式有三種：(1) 以其中一種方法（通常是理論基礎更成熟、應用更廣泛的 AHP 特徵向量法）為主要分析依據，另一方法作為穩健性交叉驗證；(2) 若兩組權重排序高度一致，可在論文中同時報告並說明其一致性作為研究結論穩健性之佐證；(3) 若兩組結果出現重大分歧（如本週範例中之零權重現象），應如實討論其成因與方法論限制，而非隱匿其中一組結果。

**Q3：TOPSIS 分析中「效益型」與「成本型」準則的區分很重要嗎？如果標錯了會怎樣？**
非常重要，這是 TOPSIS 計算中最容易出錯、卻也最容易被忽略檢查的環節。若將成本型準則（如價格，應越低越好）誤標為效益型，計算正負理想解時會直接反轉該準則的評價方向，導致「價格最貴的方案」反而被判定為在該準則上表現最優，使最終排序結果完全錯誤卻不會產生任何程式錯誤訊息（因為程式邏輯運算完全正確，只是研究者的類型標記有誤）。建議在正式分析前，務必列出每一項準則之類型判斷表，並請至少一位共同研究者或指導教授複核確認。

**Q4：我的研究只有 3 位專家填答成對比較問卷，這樣的專家人數會不會太少？**
AHP 方法論之專家人數要求，與統計抽樣之樣本數邏輯不同，並非「越多越好」，而是強調「專家代表性與判斷品質」。文獻上並無統一的最低專家人數規定，常見做法是 3–15 位具備該決策領域專業資格與豐富實務經驗之代表性專家，重點在於專家遴選標準之明確性與正當性（例如：具備 5 年以上採購決策經驗、具相關專業證照等），並於論文方法論章節中明確說明專家遴選依據，而非單純以人數多寡作為研究品質之判準。

**Q5：敏感度分析要對「每一個」準則都做一次嗎？工作量會不會太大？**
不一定需要對每一個準則都執行完整敏感度分析，實務上優先建議針對「權重最高（對最終排序影響最大）」與「排序結果中方案間 CC 值差距最小（最接近排序反轉臨界點）」的準則進行敏感度分析，這兩種情境最能反映排序結果的穩健程度。若時間與篇幅允許，涵蓋全部準則的完整敏感度分析當然更為嚴謹，但並非每一篇論文的必要條件，可依研究規模與口試委員要求彈性調整分析範圍。

**Q6：AHP 與第 1 週學過的 EFA、第 3 週的 PLS-SEM，在方法論定位上有什麼根本不同？我什麼時候該用哪一種？**
EFA 與 PLS-SEM 都屬於「資料驅動型」方法，其權重（因素負荷量或路徑係數）是從大樣本問卷「資料本身」統計估計而來，回答的是「母體中普遍存在的心理與行為規律是什麼」；AHP／Fuzzy AHP／TOPSIS 屬於「知識驅動型」方法，其權重是從少數專家的「主觀判斷」結構化萃取而來，回答的是「在這個特定決策情境下，應該選擇哪一個方案」。當研究問題是「哪些因素影響使用者的採用意願」時，適合資料驅動型方法；當研究問題是「在這幾個候選方案中，應該選擇哪一個」時，適合知識驅動型方法。許多混合式研究（如本課程第 11 週之 SEM-ANN）會同時整合兩種取徑，截長補短。

**Q7：我在敏感度分析中發現，排序結果其實對品質準則的權重相當敏感（Supplier_A、B、C 會隨權重調整而互換名次），這樣是不是代表我的研究「失敗」了，不能用了？**
完全不會，這反而是一個誠實且具有研究價值的發現。多準則決策研究的目的，並非強求得到一個「無論如何調整都不會改變」的排序結果（現實中極少有這麼「巧」的決策情境），而是要清楚呈現「在什麼樣的權重假設下，哪一個方案會勝出」，讓決策者能充分理解排序結論的邊界條件。本週研究範例的敏感度分析結果顯示，Supplier_A、B、C 三者之相對貼近係數差距不大，這個發現本身就具有重要的管理意涵：提醒決策者三家供應商在綜合表現上其實相當接近，最終選擇可能需要納入 TOPSIS 分析框架之外的其他考量（如既有合作關係、長期策略夥伴發展潛力等），這正是量化分析結果應該用來「輔助」而非「取代」管理決策判斷的具體展現，誠實報告此一發現，遠比刻意挑選一組能得出「穩健」結果的權重情境更符合學術倫理。

---

## 延伸研究方向：跨國綠色冷鏈物流中心選址評估決策模型

### 6.1 研究背景與理論基礎

跨國企業在建置綠色冷鏈物流中心（Green Cold Chain Logistics Center）時，選址決策同時涉及傳統物流選址考量（運輸成本、市場涵蓋範圍、勞動力供給）與綠色永續考量（能源效率、碳排放、在地環保法規遵循），是典型的多準則、多利害關係人之複雜決策問題，與本週理論篇 3.1 節所述之 MCDM 適用情境高度吻合。本週參考文獻中之台灣本土研究（應用模糊 AHP 探討工廠區位選擇之研究）已將 Fuzzy AHP 成功應用於工廠區位選擇此一相近決策情境，可作為本延伸研究方向之直接方法論參照。

### 6.2 建議研究設計

1. **理論框架**：延續本週 AHP-TOPSIS 整合架構，並可進一步引入第 5 週將介紹之 DEMATEL 方法，處理選址準則之間可能存在的因果影響關係（例如「當地綠能供給充足度」可能同時直接影響「能源成本」與「碳排放表現」兩項準則），突破傳統 AHP 假設準則間彼此獨立之限制。
2. **候選方案**：建議以企業已完成初步可行性評估之 3–6 個候選城市或園區作為候選方案。
3. **準則規劃（範例）**：
   - 傳統物流構面：運輸網絡便利性、市場涵蓋腹地、勞動力供給與成本
   - 綠色永續構面：再生能源供給穩定性、在地環保法規嚴謹度、既有綠色供應鏈生態系完整度
   - 風險構面：地緣政治穩定性、匯率波動風險、極端氣候災害風險
4. **分析流程**：完全比照本週 Colab 實作流程（AHP 權重求解 → 一致性檢驗 → Fuzzy AHP 穩健性驗證 → TOPSIS 排序 → 敏感度分析），僅需替換準則定義與候選方案評分資料，即可直接複用本週所有自訂函式（`ahp_eigenvector_weights()`、`chang_extent_analysis()`、`topsis()`、`sensitivity_analysis()`）。
5. **管理實務意涵**：此類研究可為跨國企業之區位決策委員會提供一套結構化、可稽核、可向董事會清楚說明決策依據的量化評選架構，相較於純粹依賴少數高階主管之直覺判斷，更能降低決策偏誤並提升決策過程之透明度。

### 6.3 給學生的思考練習

請思考：若跨國冷鏈物流選址決策，需要同時徵詢「總部策略規劃部門」與「當地營運團隊」兩組專家意見，而這兩組專家對「綠色永續」相關準則的重視程度可能存在系統性差異（總部可能更重視 ESG 形象、當地團隊可能更重視實際營運成本），你會如何調整本週的 AHP 群體決策整合方式（例如是否應賦予不同專家群體不同的整合權重），以及這樣的調整在論文方法論章節中應該如何清楚說明？

**Q8：我可以用第 1 週學過的 Cronbach's α 來檢驗 AHP 成對比較矩陣的可靠度嗎？**
不建議直接套用。Cronbach's α 是設計用來檢驗「多個題項是否測量到同一個潛在構念」的內部一致性信度指標，其統計假設與資料結構（大樣本、多題項、Likert 量表）與 AHP 成對比較矩陣（少數專家、結構化比較、正倒值矩陣）完全不同。AHP 有其專屬的、且在方法論上更適配的一致性檢驗工具——CR 一致性比率（見理論篇 3.4 節），這正是本課程刻意將第 1 週 EFA／信度分析與第 4 週 AHP 安排在不同教學脈絡下的原因：資料驅動與知識驅動兩類方法，各自擁有其對應之品質檢驗邏輯，不應混用。

---

## 課後作業與練習

**練習一：修改成對比較矩陣觀察 CR 變化**
請修改 Step 1 中成對比較矩陣的任一元素，刻意製造一個邏輯矛盾（例如將 `A[2,3]`〔交期 vs 服務〕的值從 `1/2` 改為 `5`，使交期反而被認為遠比服務重要，這與其他判斷矛盾），重新執行 Step 2、3，觀察 CR 值是否超過 .10 之門檻，並說明你觀察到的變化。

**練習二：真實專家決策情境實作**
請自行設計一個至少包含 3 個候選方案、3 個評選準則的決策情境（可以是選擇筆記型電腦、選擇實習公司、選擇研究所論文題目等貼近生活的決策），實際邀請至少 3 位親友或同學扮演「專家」填答成對比較問卷，套用本週完整 Colab 程式碼執行 AHP-TOPSIS 分析。請繳交：(1) 成對比較問卷截圖或記錄、(2) 執行後的 Colab Notebook（.ipynb）、(3) 一頁 A4 的結果摘要（比照本週「結果呈現與分析」段落之寫法）。

**練習三：零權重現象重現與替代方法查證**
請調整 Step 4 中成對比較矩陣的準則重要性差距（可嘗試更懸殊或更接近的判斷值），觀察 Chang 擴展分析法在哪些情境下容易產生零權重現象、哪些情境下不會。並請查閱 Buckley（1985）之模糊幾何平均法（Fuzzy Geometric Mean Method），簡述其與 Chang 擴展分析法在權重求解邏輯上的差異。

**練習四：文獻延伸閱讀報告**
請從本週「參考文獻與延伸閱讀」清單中，任選一篇 AHP、Fuzzy AHP 或 TOPSIS 相關之期刊論文或台灣碩士論文，撰寫一頁重點摘要，內容須包含：(1) 該研究之決策層級架構（目標、準則、方案）、(2) 報告之一致性檢驗結果（CR 值）、(3) 最終方案排序結果、(4) 你認為該研究之評選準則架構，若應用於本課程延伸研究方向（跨國綠色冷鏈物流選址），需要如何調整。

**練習五：敏感度分析延伸**
請將 Step 10 之敏感度分析函式，擴充為同時對「所有」準則逐一進行敏感度分析（而非僅針對單一準則），並將結果整合為一張總表，找出「排序結果最容易因權重變動而改變」的準則，並說明這對於研究結論的討論意涵。

**練習六：AHP 與敏感度分析結果的管理決策應用**
延續練習五之全準則敏感度分析結果，請以決策者的角度撰寫一段約 150 字的「管理建議」，說明面對排序結果對某些準則權重較為敏感的情況，企業在實際進行供應商評選決策時，除了量化排序結果之外，還應該額外考量哪些質性因素（例如既有合作關係、供應商配合度、緊急應變能力等），以及量化分析與管理判斷應如何相輔相成。

---

## 參考文獻與延伸閱讀（已查核連結）

1. 畢威寧（2005）。結合 AHP 與 TOPSIS 法於供應商績效評估之研究。《科學與工程技術期刊》，1(1)，75-83。DOI: 10.7117/JSET.200506.0075。
   https://www.airitilibrary.com/Publication/alDetailedMesh?docid=18166563-200506-1-1-75-83-a
   （PDF 全文：http://journal.dyu.edu.tw/dyujo/document/setjournal/s1-1-75-83.pdf）
   （台灣經典之 AHP+TOPSIS 供應商績效評估期刊論文，本週研究設計範例之四項評選準則即參照此文獻架構，被引用達 27 次，方法論定位清晰，適合作為文獻回顧核心參照。）

2. 使用 AHP-TOPSIS 方法選擇綠色精實製造供應商。臺灣博碩士論文知識加值系統。
   https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi/login?o=dnclcdr&s=id%3D%22111NYPI0030010%22.&searchmode=basic
   （結合綠色與精實製造觀點之 AHP-TOPSIS 供應商評選台灣碩士論文，可作為延伸研究方向「綠色」準則設計之參考範本。）

3. AHP 層級分析法探討工具機業供應商評選機制－以 O 科技公司為例。臺灣博碩士論文知識加值系統。
   https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi/login?o=dnclcdr&s=id%3D%22112YUNT0031074%22.&searchmode=basic
   （以科技製造業（工具機業）為研究對象之 AHP 供應商評選台灣碩士論文，與本週研究設計範例產業情境高度相關，評選準則排序（品質、技術、價格、服務）可直接對照參考。）

4. 貿易商選擇評估之研究－以某汽車零件工廠為例。臺灣博碩士論文知識加值系統。
   https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi?o=dnclcdr&s=id=%22097DYU01030039%22.&searchmode=basic
   （整合 AHP 與 TOPSIS 之貿易商評選台灣碩士論文，示範完整之決策模式應用流程。）

5. 應用模糊 AHP 探討工廠區位選擇之研究。國立高雄科技大學。
   https://dba.nkust.edu.tw/uploads/asset/data/6232cc3a2e63562407371059/33.pdf
   （台灣本土 Fuzzy AHP 應用於區位選擇之研究論文，與本週延伸研究方向「跨國綠色冷鏈物流中心選址」高度相關，可直接作為方法論與準則設計之參照範本。）

6. A Fuzzy AHP Approach for Supplier Selection Problem: A Case Study in a Gear Motor Company. *International Journal of Managing Value and Supply Chains*.
   https://arxiv.org/pdf/1311.2886
   （完整說明 Buckley 模糊優先權求解法與 Saaty 尺度轉三角模糊數對照表之國際期刊論文，為本週理論篇 3.5 節之直接方法論依據。）

7. Wang, Y.-M., Luo, Y., & Hua, Z. (2008). On the extent analysis method for fuzzy AHP and its applications. *European Journal of Operational Research*, 186(2), 735-747.
   https://www.researchgate.net/publication/4939952_On_the_extent_analysis_method_for_fuzzy_AHP_and_its_applications
   （高度被引用之方法論文獻，以數學證明指出 Chang（1996）擴展分析法容易產生零權重之已知限制，為本週 Step 4–5 實際觀察到之零權重現象的直接理論依據，務必於論文中誠實引用討論此方法論限制。）

8. Fuzzy analytic hierarchy process: Fallacy of the popular methods. *European Journal of Operational Research*（相關討論）。
   https://www.sciencedirect.com/science/article/abs/pii/S0377221713008576
   （近年對 Fuzzy AHP 主流方法（含 Chang 擴展分析法）之系統性方法論批判文獻回顧，可作為研究限制章節之深度討論依據。）

**方法論經典文獻（建議延伸閱讀，非本次線上搜尋來源，圖書館或資料庫可查閱）**：

- Saaty, T. L. (1980). *The Analytic Hierarchy Process: Planning, Priority Setting, Resource Allocation*. McGraw-Hill.
- Hwang, C. L., & Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and Applications*. Springer-Verlag.
- Chang, D.-Y. (1996). Applications of the extent analysis method on fuzzy AHP. *European Journal of Operational Research*, 95(3), 649-655.
- Buckley, J. J. (1985). Fuzzy hierarchical analysis. *Fuzzy Sets and Systems*, 17(3), 233-247.
- van Laarhoven, P. J. M., & Pedrycz, W. (1983). A fuzzy extension of Saaty's priority theory. *Fuzzy Sets and Systems*, 11(1-3), 229-241.
- Zadeh, L. A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338-353.

---

## 附錄

### 附錄 A：AHP／TOPSIS 常用 Python 函式速查表

| 函式 | 功能 | 本週對應 Cell |
|---|---|---|
| `ahp_eigenvector_weights()` | 以特徵向量法求解 AHP 準則權重與 λmax | Step 2 |
| `consistency_check()` | 計算 CI、RI、CR 並判斷是否通過一致性檢驗 | Step 3 |
| `to_tfn()` | 將 Saaty 精確尺度轉換為三角模糊數 | Step 4 |
| `chang_extent_analysis()` | Chang 擴展分析法求解 Fuzzy AHP 權重 | Step 4 |
| `topsis()` | 完整 TOPSIS 演算法（正規化、加權、理想解、CC 值） | Step 7 |
| `sensitivity_analysis()` | 權重擾動下之排序穩健性檢驗 | Step 10 |

### 附錄 B：術語中英對照表

| 中文術語 | 英文術語 | 縮寫 |
|---|---|---|
| 多準則決策 | Multi-Criteria Decision Making | MCDM |
| 層級分析法 | Analytic Hierarchy Process | AHP |
| 模糊層級分析法 | Fuzzy AHP | FAHP |
| 逼近理想解排序法 | Technique for Order Preference by Similarity to Ideal Solution | TOPSIS |
| 成對比較矩陣 | Pairwise Comparison Matrix | — |
| 一致性指標 | Consistency Index | CI |
| 隨機一致性指標 | Random Index | RI |
| 一致性比率 | Consistency Ratio | CR |
| 三角模糊數 | Triangular Fuzzy Number | TFN |
| 擴展分析法 | Extent Analysis Method | — |
| 正理想解 | Positive Ideal Solution | PIS |
| 負理想解 | Negative Ideal Solution | NIS |
| 相對貼近係數 | Closeness Coefficient | CC |
| 敏感度分析 | Sensitivity Analysis | — |

### 附錄 C：常見程式錯誤排解（Debugging Tips）

| 錯誤現象 | 常見原因 | 排解建議 |
|---|---|---|
| `np.linalg.eig()` 回傳複數特徵值 | 正倒值矩陣理論上應有實數最大特徵值，但浮點數運算可能產生極小虛部 | 使用 `.real` 取實部即可，本週程式碼已內建此處理 |
| Fuzzy AHP 權重全部計算為相同值 | `to_tfn()` 函式可能未正確處理倒數關係，導致模糊矩陣退化 | 檢查 `A[i,j] < 1` 時是否正確呼叫倒數轉換邏輯 |
| TOPSIS 排序結果與直覺嚴重不符 | 效益型／成本型準則標記錯誤（見 Q3） | 逐一核對 `benefit_criteria` 列表與準則實際性質是否一致 |
| 敏感度分析中權重總和不等於 1 | 調整後未正確等比例縮放其餘準則權重 | 檢查 `scale_factor` 計算是否正確，可加入 `assert abs(new_weights.sum()-1)<1e-6` 除錯 |
| CR 值計算結果為負數 | λmax 小於 n，理論上不應發生，通常為矩陣輸入錯誤（如忘記設定倒數關係） | 檢查成對比較矩陣是否滿足 `A[j,i] = 1/A[i,j]` 之正倒值性質 |

### 附錄 D：繳交前自我檢核清單

- [ ] 已報告完整之決策層級架構（目標、準則、方案）
- [ ] 已報告專家成對比較矩陣與其一致性檢驗結果（CI、RI、CR）
- [ ] 若 CR > 0.10，已說明如何請專家重新檢視並修正判斷
- [ ] 已明確標示每一項準則之類型（效益型／成本型）
- [ ] 已報告 TOPSIS 完整計算過程（正規化矩陣、加權矩陣、正負理想解）
- [ ] 已報告各方案之 D+、D-、CC 值與最終排序
- [ ] 已繪製準則權重雷達圖與方案排序長條圖
- [ ] 已執行至少一項準則之敏感度分析，檢驗排序結果穩健性
- [ ] 若同時執行 Fuzzy AHP，已誠實報告並討論與 AHP 結果之異同（含零權重現象，如有發生）
- [ ] 所有統計結果之文字敘述與表格數值一致，無謄寫錯誤

### 附錄 E：研究倫理提醒

本週研究設計涉及邀請專家填答成對比較問卷，除延續前三週已說明之知情同意、匿名性等基本倫理原則外，特別提醒：AHP 專家問卷之填答過程通常需要專家投入相對較長的思考時間（尤其準則數較多時，成對比較次數會以 $\binom{n}{2}$ 增加），應合理評估專家填答之時間成本，並考慮致贈適當之答謝（如車馬費或紀念品）；此外，若研究涉及企業內部供應商評選等敏感商業決策資訊，應與受訪企業明確約定研究資料之保密範圍與學術發表時之匿名化處理方式（如以代號取代真實供應商與企業名稱），避免研究發表對受訪企業之商業關係造成不必要之影響。

### 附錄 F：AHP、Fuzzy AHP、ANP、DEMATEL 方法定位總覽

隨著課程進入第二模組，學生將接觸一系列知識驅動型決策方法，以下表格預先總覽本模組（第 4–7 週）各方法之核心假設差異，幫助學生建立完整的方法論地圖：

| 方法 | 對應週次 | 核心假設 | 適用情境 |
|---|---|---|---|
| AHP | 第 4 週 | 準則彼此獨立，無交互影響 | 準則結構清晰、層級分明之決策問題 |
| Fuzzy AHP | 第 4 週 | 同 AHP，另允許專家判斷帶有語意模糊性 | 專家判斷存在猶豫、難以給出精確數值時 |
| TOPSIS | 第 4 週 | 方案評估可化約為與理想解之距離測度 | 已有明確準則權重，需對多方案綜合排序 |
| DEMATEL | 第 5 週 | 準則間可能存在因果影響關係，需辨識原因與結果 | 準則間交互影響為研究核心關注焦點 |
| DANP（DEMATEL-based ANP） | 第 6 週 | 突破準則獨立假設，以網路而非層級結構建模 | 準則間存在複雜回饋迴圈之決策系統 |
| BWM（最佳最差法） | 第 7 週 | 僅需 2n-3 次比較即可求解權重，大幅降低專家負荷 | 準則數較多、需降低專家填答負擔時 |

這也解釋了為何本課程將 AHP 安排在第二模組之首週：AHP 是整個知識驅動型決策科學方法家族中，理論最為簡潔、假設最為單純的入門方法，後續各週將逐步鬆綁「準則獨立」此一核心假設，帶領學生認識更貼近決策現實複雜度的進階技術。

### 附錄 G：第 1–4 週研究方法整合對照表

延續第 3 週附錄中之方法整合對照表概念，以下將第 4 週之知識驅動型方法一併納入，完整呈現本課程前四週方法論工具箱之全貌：

| 比較構面 | 第 1 週：EFA | 第 2 週：階層迴歸＋調節效應 | 第 3 週：PLS-SEM | 第 4 週：AHP／Fuzzy AHP／TOPSIS |
|---|---|---|---|---|
| 知識論基礎 | 資料驅動 | 資料驅動 | 資料驅動 | 知識驅動（專家判斷） |
| 典型樣本規模 | 大樣本（&gt; 200） | 大樣本（&gt; 200） | 中大樣本（&gt; 150） | 少數專家（3–15 位） |
| 核心產出 | 量表因素結構 | 前因效果與調節效果 | 完整因果路徑模型 | 準則權重與方案排序 |
| 品質檢驗工具 | Cronbach's α、KMO | R²、VIF | CR（組合信度）、HTMT | CR（一致性比率） |
| 適合回答的研究問題 | 「這個構念該怎麼測量？」 | 「什麼因素影響這個結果？」 | 「這些構念之間的完整因果機制是什麼？」 | 「在這些方案中，應該選哪一個？」 |

值得學生特別留意的是，**AHP 之一致性比率（CR）與第 1 週之 Cronbach's α（信度），雖然中文翻譯上都帶有「一致性」或「信度」字樣，但兩者的統計原理與適用資料型態截然不同**，這是初學者極容易混淆之處（見本週 Q8），務必掌握兩者的本質差異。

### 附錄 H：AHP 一致性比率手動計算之數值範例

為協助學生徹底理解理論篇 3.3–3.4 節之公式，以下以一個簡化的 3×3 成對比較矩陣為例，展示不依賴 `numpy` 之近似手動計算過程（列向量平均法，Row Average Method，為特徵向量法之常見近似替代解法，計算過程更適合紙筆演練）：

假設 3 個準則（品質、價格、服務）之成對比較矩陣為：

$$
A = \begin{bmatrix} 1 & 3 & 2 \\ 1/3 & 1 & 1/2 \\ 1/2 & 2 & 1 \end{bmatrix}
$$

**步驟一：將矩陣每一欄正規化（每個元素除以該欄總和）**

第一欄總和 $= 1 + 1/3 + 1/2 = 1.833$ ，正規化後第一欄為 $(0.545, 0.182, 0.273)$ ；依此類推處理第二、三欄。

**步驟二：將正規化後矩陣逐列取平均，即為近似權重向量**

經完整計算（過程從略），近似權重向量約為 $w \approx (0.539, 0.163, 0.298)$ ，與 `numpy.linalg.eig()` 之精確特徵向量解通常僅有小數點後第二、三位之些微差異。

**步驟三：計算 $\lambda_{max}$ **

$$
\lambda_{max} = \frac{1}{n}\sum_{i=1}^{n} \frac{(Aw)_i}{w_i}
$$

即將原矩陣 $A$ 與權重向量 $w$ 相乘後，逐一將結果除以對應的 $w_i$ ，再取平均，近似求得 $\lambda_{max} \approx 3.02$ 。

**步驟四：計算 CI 與 CR**

$$
CI = \frac{3.02 - 3}{3 - 1} \approx 0.01 \qquad CR = \frac{0.01}{0.58} \approx 0.017
$$

由於 $CR \approx 0.017 \ll 0.10$ ，此組判斷通過一致性檢驗。學生可將此簡化 3×3 範例之邏輯，對照本週 Step 2、3 之 `numpy` 精確運算輸出結果，作為驗證自己是否真正理解 AHP 權重求解與一致性檢驗背後數學原理的自我檢核練習，這也是論文口試中若被要求「請徒手示範一次 CR 怎麼算」時的最佳準備方式。

### 附錄 I：期刊審查意見對照檢核表

近年 AHP／Fuzzy AHP／TOPSIS 論文投稿常見之審查意見類型，以下彙整並對照本週教材對應可回應之章節，供學生於論文投稿或口試前自我演練：

| 常見審查意見 | 對應本週教材章節 | 回應要點 |
|---|---|---|
| 「作者未報告 AHP 一致性比率（CR），無法判斷專家判斷是否可信」 | 理論篇 3.4 節、Step 3 | 補充報告 CR 值與判斷依據（CR ≤ 0.10） |
| 「僅有單一組專家權重，未檢驗排序結果對權重變動之穩健性」 | 理論篇 3.7 節、Step 10 | 補充敏感度分析並誠實報告排序穩健程度 |
| 「Fuzzy AHP 部分準則權重為 0，是否為計算錯誤？」 | 理論篇 3.5 節、Q&A、參考文獻 Wang, Luo, & Hua (2008) | 說明此為 Chang 擴展分析法之已知方法論限制並引用文獻佐證 |
| 「TOPSIS 中效益型與成本型準則的處理方式為何未說明？」 | 理論篇 3.6 節、Q3 | 明確列出準則類型判斷表並說明計算邏輯 |
| 「僅 3 位專家參與判斷，樣本代表性是否足夠？」 | Q4 | 說明專家遴選標準與代表性依據，而非單純訴諸人數多寡 |
| 「AHP 假設準則獨立，是否與研究情境相符？」 | 4.7 節 | 說明準則獨立假設之限制，並視情況於研究限制中誠實揭露或改採 DEMATEL/ANP |

---

## 下週預告

第 5 週將延續本週之知識驅動型決策科學脈絡，進入「因果推論與圖形決策網絡：DEMATEL 與 INRM 演算法」。學生將學習如何突破本週 AHP 假設「準則彼此獨立」之限制，改以決策試驗與評估實驗室法（DEMATEL）處理準則之間錯綜複雜的因果影響關係，並透過直接／間接影響矩陣運算，辨識出真正驅動整個決策系統的「原因群」準則與被動接受影響的「結果群」準則。研究範例將以「企業推動智慧製造數位轉型關鍵成功因素之因果結構探討」為主題，並延伸至「半導體供應鏈脆弱性關鍵誘發因子之 DEMATEL 根本原因分析」之期末專題發想方向。
