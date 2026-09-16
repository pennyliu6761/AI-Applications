# 第 7 週：新型輕量化決策演算法——BWM 最佳化與直觀模糊集合（IFS）

> 課程模組：第二模組｜知識驅動型 AI 與多準則決策系統（第 4–7 週）
> 本週定位：作為第二模組之收官週，本週介紹最佳最差法（Best-Worst Method, BWM）——一種相較第 4 週 AHP 大幅降低專家認知負荷與矛盾風險的輕量化權重求解技術；並引入直觀模糊集合（Intuitionistic Fuzzy Sets, IFS），以歸屬度、非歸屬度與猶豫度三個維度，刻畫比第 4 週三角模糊數更細緻的決策不確定性型態，同時對第 4–7 週之知識驅動型決策科學方法家族進行整合總結，銜接第三模組之資料驅動型機器學習方法。

> 教材版本：v1.0｜適用對象：在職專班研究方法與 AI 應用課程｜先修基礎：第 4 週（AHP、Fuzzy AHP 與 TOPSIS）
> 使用工具：Google Colab（Python 3）｜主要套件：`numpy`、`pandas`、`scipy`（`scipy.optimize.linprog` 求解線性規劃）、`matplotlib`（BWM 與 IFS 核心演算法以 `numpy`／`scipy` 手動實作，所有程式碼已實際測試驗證可正常執行）

---

## 目錄

1. [學習目標](#學習目標)
2. [本週知識地圖](#本週知識地圖)
3. [理論基礎篇](#理論基礎篇)
   1. [3.1 從 AHP 到 BWM：認知負荷與比較次數的優化](#31-從-ahp-到-bwm認知負荷與比較次數的優化)
   2. [3.2 BWM 資料蒐集：Best-to-Others 與 Others-to-Worst 向量](#32-bwm-資料蒐集best-to-others-與-others-to-worst-向量)
   3. [3.3 BWM 線性規劃模型](#33-bwm-線性規劃模型)
   4. [3.4 BWM 一致性指標與一致性比率](#34-bwm-一致性指標與一致性比率)
   5. [3.5 直觀模糊集合（IFS）理論基礎](#35-直觀模糊集合ifs理論基礎)
   6. [3.6 語意量表轉換為 IFS 數與解模糊化](#36-語意量表轉換為-ifs-數與解模糊化)
   7. [3.7 IFS-BWM 整合：以 IFS 刻畫比較過程中的不確定性](#37-ifs-bwm-整合以-ifs-刻畫比較過程中的不確定性)
   8. [3.8 論文中 BWM／IFS 章節的標準寫法架構](#38-論文中-bwmifs-章節的標準寫法架構)
4. [研究設計實例：企業綠色包裝材料採購評估](#研究設計實例企業綠色包裝材料採購評估)
5. [Colab 實作環境建置](#colab-實作環境建置)
6. [Colab 實作：Step by Step 完整程式碼](#colab-實作step-by-step-完整程式碼)
7. [Vibe Coding 提示詞（Prompt）實作範例集](#vibe-coding-提示詞prompt實作範例集)
8. [結果呈現與分析：碩士論文寫法示例](#結果呈現與分析碩士論文寫法示例)
9. [常見統計誤區與 Q&A](#常見統計誤區與-qa)
10. [延伸研究方向：新創加速器創業投資案甄選決策——直觀模糊 BWM 評估模型](#延伸研究方向新創加速器創業投資案甄選決策直觀模糊-bwm-評估模型)
11. [課後作業與練習](#課後作業與練習)
12. [參考文獻與延伸閱讀（已查核連結）](#參考文獻與延伸閱讀已查核連結)
13. [附錄](#附錄)
14. [第二模組總結：知識驅動型決策科學方法家族回顧](#第二模組總結知識驅動型決策科學方法家族回顧)
15. [下週預告](#下週預告)

---

## 學習目標

完成本週課程後，學生應能夠：

1. 說明最佳最差法（BWM）相較 AHP 在專家認知負荷與比較次數上之優化原理。
2. 正確設計並蒐集 BWM 所需之 Best-to-Others 向量與 Others-to-Worst 向量。
3. 使用 Python（`scipy.optimize.linprog`）建構並求解 BWM 線性規劃模型，得出最優權重 $\xi^{*}$ 。
4. 計算並判讀 BWM 一致性指標（CI）與一致性比率（CR），確認專家判斷品質。
5. 說明直觀模糊集合（IFS）之歸屬度、非歸屬度、猶豫度三維結構，理解其相較三角模糊數更細緻刻畫不確定性之處。
6. 將專家之語意評估轉換為 IFS 數，並透過直觀模糊加權平均算子（IFWA）與分數函數完成解模糊化。
7. 整合 IFS 與 BWM，處理決策情境中專家判斷的猶豫與不確定性，並與第 4 週 AHP 權重分布進行敏感情境對照。
8. 回顧並整合第 4–7 週知識驅動型決策科學方法家族之完整脈絡，為第三模組資料驅動型機器學習方法做好銜接準備。

---

## 本週知識地圖

| 構面 | 內容 | 對應方法 | 對應 Python 實作 |
|---|---|---|---|
| 問題結構化 | 辨識最佳準則與最差準則 | BWM 問題界定 | 概念性 |
| 資料蒐集優化 | 以 $2n-3$ 次比較取代 AHP 之 $\binom{n}{2}$ 次 | Best-to-Others／Others-to-Worst 向量 | `pandas` |
| 權重最佳化求解 | 求解最小化最大不一致性之權重 | 線性規劃（Linear Programming） | `scipy.optimize.linprog` |
| 一致性檢驗 | 檢驗專家判斷之邏輯一致性 | CI 對照表、CR 計算 | 自訂函式 |
| 不確定性刻畫 | 以歸屬、非歸屬、猶豫三維表達判斷模糊性 | 直觀模糊集合（IFS） | 自訂函式 |
| 語意解模糊化 | 將語意評估轉換為可運算之精確值 | IFWA 算子、分數函數 | 自訂函式 |
| 方法敏感度比較 | 比較不同權重求解方法之結果穩健性 | BWM vs. AHP 權重分布對照 | `matplotlib` |

---

## 理論基礎篇

### 3.1 從 AHP 到 BWM：認知負荷與比較次數的優化

第 4 週介紹之 AHP，其成對比較矩陣需要專家針對 $n$ 個準則，完成 $\binom{n}{2} = \dfrac{n(n-1)}{2}$ 次兩兩比較。當準則數量增加時，此比較次數會以平方速度成長（例如 $n=10$ 時需 45 次比較），不僅大幅增加專家填答之時間成本與認知負荷，也顯著提高專家判斷出現邏輯矛盾（CR 超標）之風險。

**最佳最差法（Best-Worst Method, BWM）**由 Rezaei（2015，發表於 *Omega*，見本週參考文獻）提出，核心洞見是：與其要求專家完成所有 $\binom{n}{2}$ 組兩兩比較，不如僅聚焦於「最重要準則（Best）」與「最不重要準則（Worst）」這兩個參考點，讓專家分別評估「Best 準則相對於其餘所有準則」以及「其餘所有準則相對於 Worst 準則」，總計僅需 $2n-3$ 次比較（扣除 Best 與 Worst 各自對自身的比較，以及兩者間的比較僅需計算一次）。以 $n=5$ 為例，AHP 需要 $\binom{5}{2}=10$ 次比較，BWM 僅需 $2(5)-3=7$ 次；當 $n$ 越大，此效率差距越顯著。

**BWM 相較 AHP 之額外優勢**：由於所有比較都錨定於「最重要」與「最不重要」這兩個專家最容易明確判斷的極端參考點，相較於 AHP 要求專家對「任意兩個可能都不是特別重要或不重要」的準則進行比較，BWM 之比較資料通常展現更高的內部一致性（見本週參考文獻中 Rezaei, 2015 之原始實證比較），這正是本週理論篇 3.4 節將介紹之一致性檢驗機制之所以重要、也是 BWM 相較 AHP 更具實務吸引力之處。

### 3.2 BWM 資料蒐集：Best-to-Others 與 Others-to-Worst 向量

BWM 之資料蒐集程序，依序包含以下步驟：

**步驟一：界定決策準則集合** $\{c_1, c_2, \ldots, c_n\}$ 。

**步驟二：由專家判斷決定最佳準則（Best, $c_B$ ）與最差準則（Worst, $c_W$ ）**，此為整個問卷中唯一需要專家先行「主觀認定」（而非比較）的步驟。

**步驟三：建構 Best-to-Others 向量**：專家依 Saaty 九點量表，評估最佳準則 $c_B$ 相對於其餘每一準則 $c_j$ 之重要程度：

$$
A_B = (a_{B1}, a_{B2}, \ldots, a_{Bn}), \qquad a_{BB} = 1
$$

**步驟四：建構 Others-to-Worst 向量**：專家依同一量表，評估每一準則 $c_j$ 相對於最差準則 $c_W$ 之重要程度：

$$
A_W = (a_{1W}, a_{2W}, \ldots, a_{nW})^T, \qquad a_{WW} = 1
$$

與第 4 週 AHP 之正倒值矩陣不同，BWM 之 $A_B$ 、 $A_W$ 兩個向量，並非完整矩陣中的一部分，而是獨立蒐集之兩組比較資料，這正是比較次數得以從 $O(n^2)$ 降至 $O(n)$ 的資料結構關鍵。

### 3.3 BWM 線性規劃模型

**原始非線性模型（Rezaei, 2015）**之目標，是尋找一組權重 $(w_1, \ldots, w_n)$ ，使得 $w_B/w_j$ 盡可能接近 $a_{Bj}$ 、且 $w_j/w_W$ 盡可能接近 $a_{jW}$ （對所有 $j$ ），數學上表述為最小化最大偏離程度：

$$
\min_{w} \max_{j} \left\{ \left| \frac{w_B}{w_j} - a_{Bj} \right|,\ \left| \frac{w_j}{w_W} - a_{jW} \right| \right\}
$$

Rezaei（2015）指出，此非線性模型在準則數超過 3 個、且比較資料非完全一致時，可能存在多組數值不同但目標函數值相同的最優解（即解不唯一），此問題會影響後續統計推論（如假設檢定）之可靠性。為解決此一問題，Rezaei（2016，見本週參考文獻）進一步提出**線性化模型**，將比較目標由「比率偏離」改為「加權偏離」，數學表述為：

$$
\min \xi
$$
$$
\text{s.t.} \quad |w_B - a_{Bj} \cdot w_j| \leq \xi, \quad \forall j
$$
$$
|w_j - a_{jW} \cdot w_W| \leq \xi, \quad \forall j
$$
$$
\sum_{j=1}^{n} w_j = 1, \qquad w_j \geq 0, \quad \forall j
$$

此模型可進一步展開為標準線性規劃形式（每個絕對值不等式拆解為兩條線性不等式），變數為 $(w_1, \ldots, w_n, \xi)$ 共 $n+1$ 個，可直接以 `scipy.optimize.linprog` 求解，且此線性模型保證求得**唯一**之最優權重解，這也是本課程選擇實作線性模型、而非原始非線性模型之方法論依據。求解後得到之 $\xi^{*}$ （目標函數最優值），即為後續一致性檢驗之核心統計量。

### 3.4 BWM 一致性指標與一致性比率

**一致性指標（Consistency Index, CI）**：Rezaei（2015）透過理論推導，針對不同的 $a_{BW}$ （最佳準則相對於最差準則之比較值）值，計算出「完全一致情況下 $\xi$ 的理論最大值」，整理為對照表：

| $a_{BW}$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| CI | 0.00 | 0.44 | 1.00 | 1.63 | 2.30 | 3.00 | 3.73 | 4.47 | 5.23 |

**一致性比率（Consistency Ratio, CR）**：

$$
CR = \frac{\xi^{*}}{CI}
$$

**判斷標準**：與第 4 週 AHP 之 CR ≤ 0.10 慣例類似，BWM 文獻上亦建議 CR 越低代表判斷一致性越佳；惟近年方法論研究（見本週參考文獻 Liang, Brunelli, & Rezaei, 2020 之一致性門檻研究）指出，BWM 之 CR 可接受門檻應依準則數量與 $a_{BW}$ 值動態調整，而非套用單一固定門檻，此為 BWM 方法論仍在持續發展精進之處，學生於論文中若引用固定門檻（如 0.10），應同時說明此為簡化慣例、並非唯一標準。

### 3.5 直觀模糊集合（IFS）理論基礎

直觀模糊集合（Intuitionistic Fuzzy Sets, IFS）由 Atanassov（1986，發表於 *Fuzzy Sets and Systems*，見本週參考文獻）提出，是對 Zadeh（1965）傳統模糊集合理論的重要擴展。傳統模糊集合僅以單一「歸屬度（membership degree）」 $\mu \in [0,1]$ 表達元素隸屬於某模糊集合的程度，並隱含假設「非歸屬度」恆等於 $1-\mu$ ；然而 Atanassov 指出，現實決策情境中，專家表達判斷時往往存在「猶豫」——例如專家可能認為某方案有 70% 的把握符合某準則、但並非因此就代表有 30% 的把握「不符合」，中間可能還存在一段「說不準、無法判斷」的模糊地帶。

**IFS 之正式定義**：設 $X$ 為論域，IFS $A$ 定義為：

$$
A = \{ \langle x, \mu_A(x), \nu_A(x) \rangle \mid x \in X \}
$$

其中 $\mu_A: X \to [0,1]$ 為歸屬度函數、 $\nu_A: X \to [0,1]$ 為非歸屬度函數，且須滿足：

$$
0 \leq \mu_A(x) + \nu_A(x) \leq 1
$$

**猶豫度（Hesitation Degree）**：

$$
\pi_A(x) = 1 - \mu_A(x) - \nu_A(x)
$$

猶豫度 $\pi_A(x)$ 代表「既非歸屬、也非不歸屬」的不確定成分，當 $\pi_A(x) = 0$ （即 $\nu_A(x) = 1-\mu_A(x)$ ）時，IFS 退化為傳統模糊集合，這也說明了 IFS 是傳統模糊集合的一般化擴展，能刻畫比第 4 週三角模糊數（僅以 $l, m, u$ 三點描述單一維度的不確定區間）更細緻、更貼近人類決策心理的「贊成—反對—猶豫」三元不確定性結構。

### 3.6 語意量表轉換為 IFS 數與解模糊化

實務應用上，專家通常不會直接給出精確的 $(\mu, \nu)$ 數值，而是以語意詞彙（如「中等重要」「強烈重要」）表達判斷，因此需要一套語意轉換對照表：

| 語意詞彙 | 代碼 | 歸屬度 $\mu$ | 非歸屬度 $\nu$ | 猶豫度 $\pi$ |
|---|---|---|---|---|
| 同等重要（Equally Important） | EI | 0.50 | 0.50 | 0.00 |
| 稍微更重要（Weakly More Important） | WMI | 0.60 | 0.35 | 0.05 |
| 中度更重要（Moderately More Important） | MI | 0.75 | 0.20 | 0.05 |
| 強烈更重要（Strongly More Important） | SMI | 0.85 | 0.10 | 0.05 |
| 極強烈更重要（Very Strongly More Important） | VSMI | 0.95 | 0.05 | 0.00 |

**多專家 IFS 數之聚合：直觀模糊加權平均算子（Intuitionistic Fuzzy Weighted Averaging, IFWA）**：設 $K$ 位專家針對同一比較給出之 IFS 判斷為 $(\mu_k, \nu_k)$ ， $k=1,\ldots,K$ ，聚合公式為：

$$
IFWA(\alpha_1, \ldots, \alpha_K) = \left( 1 - \prod_{k=1}^{K}(1-\mu_k)^{w_k},\ \prod_{k=1}^{K}\nu_k^{w_k} \right)
$$

其中 $w_k$ 為第 $k$ 位專家之權重（若專家地位相同，則 $w_k = 1/K$ ）。

**解模糊化：分數函數（Score Function）**：為將聚合後之 IFS 數轉換回單一精確數值（以利代入後續 BWM 等數值運算模型），採用最常見之分數函數：

$$
S(\mu, \nu) = \mu - \nu
$$

$S(\mu,\nu) \in [-1,1]$ ，數值越接近 1 代表越傾向「贊成／重要」，越接近 -1 代表越傾向「反對／不重要」。為配合 BWM 所需之 Saaty 1–9 量表格式，可進一步將分數函數值線性映射至 1–9 區間（本週 Colab 實作將示範此完整轉換流程）。

### 3.7 IFS-BWM 整合：以 IFS 刻畫比較過程中的不確定性

將 IFS 與 BWM 整合之核心邏輯為：**不直接要求專家給出精確的 Saaty 數值，而是讓專家以語意詞彙表達判斷，並允許多位專家意見存在分歧與猶豫，透過 IFS 之歸屬、非歸屬、猶豫三維結構完整保留這些不確定性資訊，直到解模糊化步驟才轉換為單一數值**，如此可避免傳統做法「強迫專家在填答當下就必須壓縮為單一精確數字」所造成的資訊耗損。近年文獻（見本週參考文獻中之綠色供應商評選 IFS-BWM-TOPSIS 整合研究）已將此類整合方法成功應用於高不確定性之採購與投資決策情境。

值得注意的是，本週理論篇 3.4 節已說明，直接以精確數值進行 BWM 比較（crisp BWM）與經由 IFS 語意聚合後再轉換之比較（IFS-informed BWM），兩者所得之最終權重數值通常不會完全相同——這並非任一方法有誤，而是反映了「單一專家直接給出精確判斷」與「多位專家以語意詞彙表達、再統計聚合」這兩種不同資料蒐集模式，本質上捕捉到的是稍有差異的判斷資訊。本週 Colab 實作將完整呈現此一比較，並在理論篇 Q&A 中進一步討論其方法論意涵。

### 3.8 論文中 BWM／IFS 章節的標準寫法架構

1. **最佳與最差準則之認定依據**：說明專家如何、依據何種理由認定 Best 與 Worst 準則。
2. **Best-to-Others 與 Others-to-Worst 向量**：呈現完整比較資料表。
3. **線性規劃模型設定與求解結果**：說明採用之模型（建議明確引用 Rezaei, 2016 線性模型以確保解唯一性）與求解工具。
4. **BWM 權重排序表**：呈現各準則最終權重與排序。
5. **一致性檢驗結果**：報告 $\xi^{*}$ 、CI、CR 值。
6. **（如有執行）IFS 語意轉換與聚合過程**：呈現語意評估對照表與 IFWA 聚合結果。
7. **BWM 與 AHP（或其他方法）權重分布敏感情境對照表**：展現方法選擇之穩健性佐證。
8. **管理意涵討論**。

---

## 研究設計實例：企業綠色包裝材料採購評估

### 4.1 研究背景與動機

企業於綠色包裝材料採購決策中，須同時考量材料之環境友善性（可回收性、生物可分解性）與商業可行性（採購成本、供應穩定性、法規符合度）等多項準則。此類採購評選委員會通常由採購、研發、永續發展等跨部門主管組成，會議時間有限，難以要求每位專家逐一完成第 4 週 AHP 所需之完整成對比較矩陣，此一實務限制正是本週採用 BWM 之直接動機。

### 4.2 研究目的

1. 以 BWM 線性規劃模型，快速求解綠色包裝材料採購評選準則之權重。
2. 計算 BWM 之一致性指標與比率，確認專家判斷品質。
3. 以 IFS 語意評估聚合方式，交叉驗證 BWM 權重求解結果之穩健性。
4. 將 BWM 權重與傳統 AHP 權重進行敏感情境對照，量化比較兩方法之比較次數與權重差異。

### 4.3 評選準則架構

| 準則代碼 | 準則名稱 | 操作型定義 |
|---|---|---|
| C1 | 可回收性 | 包裝材料使用後可回收再利用之程度 |
| C2 | 生物可分解性 | 材料於自然環境中可分解、不造成長期環境負擔之程度 |
| C3 | 採購成本 | 單位包裝材料之採購總成本 |
| C4 | 供應穩定性 | 供應商產能與交期之穩定可靠程度 |
| C5 | 法規符合度 | 材料符合現行與預期環保法規要求之程度 |

依企業採購評選委員會之初步共識，**最佳準則（Best）為 C1（可回收性）**，**最差準則（Worst）為 C3（採購成本）**（此處「最差」係指相對重要性最低，並非該準則不重要）。

### 4.4 研究對象與資料蒐集

建議邀請 5–8 位具綠色採購決策參與資格之跨部門主管，分別完成：(1) 直接以 Saaty 量表填答之 Best-to-Others 與 Others-to-Worst 向量（BWM 主分析）；(2) 以語意詞彙表達之比較判斷（IFS 穩健性交叉驗證分析）。

### 4.5 資料分析流程規劃

```
界定評選準則與 Best／Worst 準則
        │
        ▼
【BWM 主分析】蒐集 Best-to-Others、Others-to-Worst 向量
        │
        ▼
建構線性規劃模型（scipy.optimize.linprog）
        │
        ▼
求解最優權重 w* 與 ξ*
        │
        ▼
計算 CI、CR，確認一致性檢驗通過
        │
        ▼
【IFS 穩健性交叉驗證】專家以語意詞彙表達比較判斷
        │
        ▼
IFWA 聚合多專家判斷 → 分數函數解模糊化 → 轉換為 Saaty 等效值
        │
        ▼
重新求解 BWM，比較與主分析權重之異同
        │
        ▼
【方法敏感情境對照】建構對應之 AHP 完整成對比較矩陣
        │
        ▼
比較 BWM／IFS-BWM／AHP 三組權重分布與所需比較次數
        │
        ▼
繪製一致性檢定評估圖 + 權重分布敏感情境對照表
        │
        ▼
撰寫研究結果與討論段落
```

---

## Colab 實作環境建置

```python
# ============================================================
# Cell 0：Colab 環境建置與套件匯入
# ------------------------------------------------------------
# 說明：BWM 之線性規劃求解使用 scipy.optimize.linprog；IFS
# 之聚合與解模糊化運算則以 numpy 手動實作。
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ------------------------------------------------------------
# 設定中文字型（沿用第 1–6 週相同設定邏輯）
# ------------------------------------------------------------
!wget -q https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf -O /content/NotoSansTC.otf
from matplotlib import font_manager
font_manager.fontManager.addfont('/content/NotoSansTC.otf')
plt.rcParams['font.family'] = 'Noto Sans CJK TC'
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

print("環境建置完成，本週 BWM 以 scipy.optimize.linprog 求解，IFS 以 numpy 手動實作。")
```

---

## Colab 實作：Step by Step 完整程式碼

### Step 1：定義準則、Best／Worst 認定與比較向量

```python
# ============================================================
# Cell 1：定義綠色包裝材料採購評選準則與 BWM 比較向量
# ------------------------------------------------------------
# 說明：以下 Best-to-Others 與 Others-to-Worst 向量，代表已
# 整合多位專家判斷後之群體比較結果。
# ============================================================

criteria = ['可回收性(C1)', '生物可分解性(C2)', '採購成本(C3)', '供應穩定性(C4)', '法規符合度(C5)']
n = len(criteria)
best_idx = 0   # C1：可回收性 為最佳（最重要）準則
worst_idx = 2  # C3：採購成本 為最差（相對最不重要）準則

# Best-to-Others 向量：C1（最佳）相對於 C1~C5 之重要程度（Saaty 1-9 量表）
A_B = np.array([1, 2, 5, 3, 4])
# Others-to-Worst 向量：C1~C5 相對於 C3（最差）之重要程度
A_W = np.array([5, 4, 1, 3, 2])

print("驗證：A_B[best] 應為 1 →", A_B[best_idx])
print("驗證：A_W[worst] 應為 1 →", A_W[worst_idx])

bwm_input_table = pd.DataFrame({
    'Criterion': criteria,
    'Best-to-Others (A_B)': A_B,
    'Others-to-Worst (A_W)': A_W
})
print("\n=== 表 1：BWM 輸入比較向量 ===")
display(bwm_input_table)
print(f"\nBWM 所需比較次數：2n-3 = {2*n-3} 次")
print(f"AHP 所需比較次數：C(n,2) = {n*(n-1)//2} 次")
```

### Step 2：建構並求解 BWM 線性規劃模型

```python
# ============================================================
# Cell 2：BWM 線性規劃模型求解
# ------------------------------------------------------------
# 提示詞實作對照：
# 「建立 BWM 線性規劃最佳化模型，讀取 Best-to-Others 與
#   Others-to-Worst 向量，解出最優權重 ξ* 並驗證一致性比例。」
# ============================================================

def solve_bwm_linear(A_B, A_W, best_idx, worst_idx, n):
    """
    以線性規劃求解 BWM 模型（Rezaei, 2016 線性版本），確保
    求得唯一最優權重解。

    變數向量為 [w_1, ..., w_n, xi]，共 n+1 個變數。
    """
    c = np.zeros(n + 1)
    c[-1] = 1   # 目標函數：minimize xi

    A_ub, b_ub = [], []

    # 約束組一：|w_B - a_Bj * w_j| <= xi，拆解為兩條線性不等式
    for j in range(n):
        if j == best_idx:
            continue
        row1 = np.zeros(n + 1)
        row1[best_idx] = 1; row1[j] = -A_B[j]; row1[-1] = -1
        A_ub.append(row1); b_ub.append(0)

        row2 = np.zeros(n + 1)
        row2[best_idx] = -1; row2[j] = A_B[j]; row2[-1] = -1
        A_ub.append(row2); b_ub.append(0)

    # 約束組二：|w_j - a_jW * w_W| <= xi
    for j in range(n):
        if j == worst_idx:
            continue
        row1 = np.zeros(n + 1)
        row1[j] = 1; row1[worst_idx] = -A_W[j]; row1[-1] = -1
        A_ub.append(row1); b_ub.append(0)

        row2 = np.zeros(n + 1)
        row2[j] = -1; row2[worst_idx] = A_W[j]; row2[-1] = -1
        A_ub.append(row2); b_ub.append(0)

    # 約束：權重總和為 1
    A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = 1
    b_eq = [1]

    bounds = [(0, 1)] * n + [(0, None)]   # 權重介於 0~1，xi 須為非負值

    result = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if result.status != 0:
        raise RuntimeError(f"線性規劃求解失敗：{result.message}")

    weights = result.x[:n]
    xi_star = result.x[-1]
    return weights, xi_star

bwm_weights, xi_star = solve_bwm_linear(A_B, A_W, best_idx, worst_idx, n)

bwm_weight_table = pd.DataFrame({
    'Criterion': criteria, 'BWM_Weight': bwm_weights
}).sort_values('BWM_Weight', ascending=False)

print(f"最優目標函數值 ξ* = {xi_star:.4f}")
print("\n=== 表 2：BWM 準則權重排序 ===")
display(bwm_weight_table.round(4))
print(f"\n權重總和驗證：{bwm_weights.sum():.6f}（應為 1）")
```

### Step 3：一致性檢驗（CI、CR）

```python
# ============================================================
# Cell 3：BWM 一致性指標與一致性比率計算
# ------------------------------------------------------------
# 提示詞實作對照（延續 Step 2）：
# 「並驗證一致性比例。」
# ============================================================

def bwm_consistency_check(xi_star, a_BW):
    """
    依 Rezaei (2015) 之一致性指標對照表，計算 BWM 之 CR 值。
    a_BW 為最佳準則相對於最差準則之比較值（四捨五入至整數
    以對照下表，因對照表僅定義於整數 1-9）。
    """
    CI_table = {1: 0.00, 2: 0.44, 3: 1.00, 4: 1.63, 5: 2.30,
                6: 3.00, 7: 3.73, 8: 4.47, 9: 5.23}
    a_BW_rounded = int(round(a_BW))
    CI = CI_table.get(a_BW_rounded, 5.23)
    CR = xi_star / CI if CI > 0 else 0
    return CI, CR, a_BW_rounded

a_BW = A_B[worst_idx]   # 最佳準則(C1)相對於最差準則(C3)之比較值
CI, CR, a_BW_rounded = bwm_consistency_check(xi_star, a_BW)

print(f"a_BW（最佳相對於最差之比較值）= {a_BW}")
print(f"對照表查得 CI（a_BW={a_BW_rounded}）= {CI}")
print(f"一致性比率 CR = ξ*/CI = {xi_star:.4f}/{CI} = {CR:.4f}")
print(f"\n{'✅ CR 落於可接受範圍，專家判斷具備良好一致性。' if CR <= 0.10 else '⚠️ CR 偏高，建議請專家重新檢視比較判斷。'}")
```

### Step 4：IFS 語意評估與聚合（穩健性交叉驗證）

```python
# ============================================================
# Cell 4：IFS 語意評估轉換與 IFWA 聚合
# ------------------------------------------------------------
# 教學目的：以語意詞彙蒐集多位專家之比較判斷，透過 IFS 完整
# 保留判斷中的猶豫成分，聚合後再轉換為 BWM 所需之 Saaty
# 等效值，作為 Step 2 精確數值比較之穩健性交叉驗證。
# ============================================================

# 語意詞彙轉換對照表（見理論篇 3.6 節）
ling_to_ifs = {
    'EI':   (0.50, 0.50),
    'WMI':  (0.60, 0.35),
    'MI':   (0.75, 0.20),
    'SMI':  (0.85, 0.10),
    'VSMI': (0.95, 0.05),
}

def ifwa_aggregate(ifs_list, weights=None):
    """直觀模糊加權平均算子（IFWA），聚合多位專家之 IFS 判斷。"""
    K = len(ifs_list)
    if weights is None:
        weights = [1 / K] * K
    mu_prod, nu_prod = 1.0, 1.0
    for (mu, nu), w in zip(ifs_list, weights):
        mu_prod *= (1 - mu) ** w
        nu_prod *= nu ** w
    return 1 - mu_prod, nu_prod

def score_function(mu, nu):
    """分數函數 S(mu, nu) = mu - nu，用於解模糊化。"""
    return mu - nu

def ifs_score_to_saaty(mu, nu):
    """將分數函數值（介於 0~1，假設偏好方向已確立）線性映射至 Saaty 1-9 量表。"""
    S = max(score_function(mu, nu), 0)
    return 1 + 8 * S

# 三位專家對「Best（C1）相對於其他各準則」之語意判斷
BtO_judgments = {
    1: ['MI', 'WMI', 'MI'],       # C1 vs C2
    2: ['VSMI', 'SMI', 'VSMI'],   # C1 vs C3
    3: ['SMI', 'MI', 'SMI'],      # C1 vs C4
    4: ['MI', 'MI', 'WMI'],       # C1 vs C5
}
# 三位專家對「各準則相對於 Worst（C3）」之語意判斷
OtW_judgments = {
    0: ['VSMI', 'SMI', 'VSMI'],   # C1 vs C3
    1: ['MI', 'MI', 'SMI'],       # C2 vs C3
    3: ['MI', 'WMI', 'MI'],       # C4 vs C3
    4: ['WMI', 'WMI', 'MI'],      # C5 vs C3
}

A_B_ifs = np.ones(n)
ifs_detail_records = []
for j, judgments in BtO_judgments.items():
    ifs_vals = [ling_to_ifs[t] for t in judgments]
    mu_agg, nu_agg = ifwa_aggregate(ifs_vals)
    saaty_equiv = round(ifs_score_to_saaty(mu_agg, nu_agg), 2)
    A_B_ifs[j] = saaty_equiv
    ifs_detail_records.append({
        'Comparison': f'C1 vs {criteria[j]}', '聚合μ': mu_agg, '聚合ν': nu_agg,
        '猶豫度π': 1 - mu_agg - nu_agg, 'Saaty等效值': saaty_equiv
    })

A_W_ifs = np.ones(n)
for j, judgments in OtW_judgments.items():
    ifs_vals = [ling_to_ifs[t] for t in judgments]
    mu_agg, nu_agg = ifwa_aggregate(ifs_vals)
    saaty_equiv = round(ifs_score_to_saaty(mu_agg, nu_agg), 2)
    A_W_ifs[j] = saaty_equiv

print("=== 表 3：IFS 語意聚合過程明細（Best-to-Others 部分）===")
display(pd.DataFrame(ifs_detail_records).round(4))

print(f"\nIFS 轉換後 Best-to-Others 向量：{A_B_ifs.round(2)}")
print(f"IFS 轉換後 Others-to-Worst 向量：{A_W_ifs.round(2)}")
```

### Step 5：以 IFS 轉換值重新求解 BWM 並與主分析對照

```python
# ============================================================
# Cell 5：IFS-informed BWM 求解與穩健性對照
# ============================================================

bwm_weights_ifs, xi_star_ifs = solve_bwm_linear(A_B_ifs, A_W_ifs, best_idx, worst_idx, n)
CI_ifs, CR_ifs, _ = bwm_consistency_check(xi_star_ifs, A_B_ifs[worst_idx])

robustness_table = pd.DataFrame({
    'Criterion': criteria,
    'Crisp_BWM_Weight': bwm_weights,
    'IFS_BWM_Weight': bwm_weights_ifs,
    'Difference': bwm_weights - bwm_weights_ifs
}).sort_values('Crisp_BWM_Weight', ascending=False)

print("=== 表 4：精確數值 BWM vs. IFS 語意聚合 BWM 權重對照表 ===")
display(robustness_table.round(4))

print(f"\n精確數值 BWM：ξ* = {xi_star:.4f}，CR = {CR:.4f}")
print(f"IFS 語意聚合 BWM：ξ* = {xi_star_ifs:.4f}，CR = {CR_ifs:.4f}")

crisp_rank = bwm_weight_table['Criterion'].tolist()
ifs_rank = robustness_table.sort_values('IFS_BWM_Weight', ascending=False)['Criterion'].tolist()
print(f"\n精確數值 BWM 排序：{crisp_rank}")
print(f"IFS 語意聚合 BWM 排序：{ifs_rank}")
print(f"排序是否一致：{'是' if crisp_rank == ifs_rank else '否（惟兩端最重要／最不重要準則通常仍一致）'}")
```

### Step 6：建構對應 AHP 矩陣並進行方法敏感情境對照

```python
# ============================================================
# Cell 6：BWM 與 AHP 權重分布敏感情境對照
# ------------------------------------------------------------
# 提示詞實作對照：
# 「與 AHP 權重分布之敏感情境對照表」
# ------------------------------------------------------------
# 說明：為公平比較，本步驟建構一份與 BWM 比較資料邏輯一致
# （同樣認定 C1 最重要、C3 最不重要）之完整 AHP 成對比較
# 矩陣，執行第 4 週介紹之特徵向量法，並與 BWM 權重對照。
# ============================================================

AHP_matrix = np.array([
    [1,    2,    5,    3,    4  ],
    [1/2,  1,    3,    3/2,  2  ],
    [1/5,  1/3,  1,    1/3,  1/2],
    [1/3,  2/3,  3,    1,    3/2],
    [1/4,  1/2,  2,    2/3,  1  ],
])

eigvals, eigvecs = np.linalg.eig(AHP_matrix)
idx = np.argmax(eigvals.real)
lambda_max = eigvals[idx].real
ahp_weights = eigvecs[:, idx].real
ahp_weights = ahp_weights / ahp_weights.sum()

n_ahp = AHP_matrix.shape[0]
CI_ahp = (lambda_max - n_ahp) / (n_ahp - 1)
RI = 1.12   # n=5 之隨機一致性指標（見第 4 週理論篇 3.4 節）
CR_ahp = CI_ahp / RI

method_comparison = pd.DataFrame({
    'Criterion': criteria,
    'AHP_Weight': ahp_weights,
    'BWM_Weight': bwm_weights,
    'Difference': ahp_weights - bwm_weights
}).sort_values('AHP_Weight', ascending=False)

print("=== 表 5：BWM 與 AHP 權重分布敏感情境對照表 ===")
display(method_comparison.round(4))

print(f"\nAHP：λmax = {lambda_max:.4f}, CI = {CI_ahp:.4f}, CR = {CR_ahp:.4f}")
print(f"AHP 所需比較次數：{n_ahp*(n_ahp-1)//2} 次")
print(f"BWM：ξ* = {xi_star:.4f}, CR = {CR:.4f}")
print(f"BWM 所需比較次數：{2*n-3} 次")
print(f"\n最大權重差異：{method_comparison['Difference'].abs().max():.4f}")
print(f"比較次數節省比例：{(1 - (2*n-3)/(n_ahp*(n_ahp-1)//2))*100:.1f}%")
```

### Step 7：一致性檢定評估圖與權重分布對照視覺化

```python
# ============================================================
# Cell 7：視覺化——一致性檢定評估圖與權重分布對照圖
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# 圖 A：一致性檢定評估圖（CR 值跨方法比較）
ax = axes[0]
methods = ['AHP\n(10次比較)', 'BWM\n(7次比較)', 'IFS-BWM\n(7次比較)']
cr_values = [CR_ahp, CR, CR_ifs]
colors_cr = ['#3498db' if cr <= 0.10 else '#e74c3c' for cr in cr_values]
bars = ax.bar(methods, cr_values, color=colors_cr, edgecolor='#2c3e50', width=0.5)
for bar, val in zip(bars, cr_values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.002, f'{val:.4f}', ha='center', fontsize=10)
ax.axhline(y=0.10, color='red', linestyle='--', label='CR = 0.10 判斷門檻')
ax.set_ylabel('一致性比率 CR')
ax.set_title('圖 1：跨方法一致性檢定評估圖', fontsize=12)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# 圖 B：三方法準則權重分布對照
ax = axes[1]
x_pos = np.arange(n)
width = 0.25
ax.bar(x_pos - width, method_comparison.set_index('Criterion').loc[criteria, 'AHP_Weight'],
       width, label='AHP', color='#3498db')
ax.bar(x_pos, bwm_weight_table.set_index('Criterion').loc[criteria, 'BWM_Weight'],
       width, label='BWM', color='#2ecc71')
ax.bar(x_pos + width, robustness_table.set_index('Criterion').loc[criteria, 'IFS_BWM_Weight'],
       width, label='IFS-BWM', color='#e67e22')
ax.set_xticks(x_pos)
ax.set_xticklabels(criteria, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('準則權重')
ax.set_title('圖 2：AHP／BWM／IFS-BWM 權重分布對照圖', fontsize=12)
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('bwm_ifs_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Step 8：匯出所有分析結果

```python
# ============================================================
# Cell 8：匯出完整分析結果至 Excel
# ============================================================

with pd.ExcelWriter('BWM_IFS分析結果_Week07.xlsx') as writer:
    bwm_input_table.to_excel(writer, sheet_name='BWM輸入向量', index=False)
    bwm_weight_table.round(4).to_excel(writer, sheet_name='BWM權重', index=False)
    pd.DataFrame(ifs_detail_records).round(4).to_excel(writer, sheet_name='IFS聚合明細', index=False)
    robustness_table.round(4).to_excel(writer, sheet_name='精確vsIFS對照', index=False)
    method_comparison.round(4).to_excel(writer, sheet_name='BWM與AHP對照', index=False)

print("所有統計結果已匯出至 BWM_IFS分析結果_Week07.xlsx，可於 Colab 左側檔案面板下載。")
print(f"\n最終建議：{bwm_weight_table.iloc[0]['Criterion']} 為權重最高之準則，")
print(f"BWM 一致性比率 CR = {CR:.4f}（{'通過' if CR<=0.10 else '未通過'}一致性檢驗），")
print(f"相較 AHP 節省 {(1 - (2*n-3)/(n_ahp*(n_ahp-1)//2))*100:.1f}% 之專家比較次數。")
```

---

## Vibe Coding 提示詞（Prompt）實作範例集

**範例 1：BWM 線性規劃求解**

> 我有一組 5 準則的 Best-to-Others 向量與 Others-to-Worst 向量（Saaty 1-9 量表）。請使用 Python 的 scipy.optimize.linprog，依 Rezaei (2016) 之線性模型，建立變數為 [w_1,...,w_5, ξ] 的線性規劃問題，求解最優權重與 ξ*，並確保權重總和為 1。

**範例 2：一致性檢驗**

> 請依 Rezaei (2015) 之一致性指標對照表（a_BW 從 1 到 9 對應 CI 值 0, 0.44, 1, 1.63, 2.3, 3, 3.73, 4.47, 5.23），計算本次 BWM 分析的一致性比率 CR，並明確告訴我這組專家判斷是否可接受。

**範例 3：IFS 語意聚合**

> 我有三位專家對同一組比較給出的語意判斷（例如「中度更重要」「強烈更重要」）。請將這些語意詞彙依標準對照表轉換為直觀模糊數 (μ, ν)，並使用直觀模糊加權平均算子（IFWA）聚合這三位專家的判斷，計算聚合後的猶豫度，最後以分數函數解模糊化並映射至 Saaty 1-9 量表。

**範例 4：方法敏感情境對照**

> 請建立一份與本次 BWM 分析邏輯一致的完整 AHP 成對比較矩陣，計算 AHP 特徵向量權重與一致性比率，並將 AHP、BWM、IFS-BWM 三組權重整理成對照表，同時比較三種方法各自所需的專家比較次數。

**範例 5：視覺化**

> 請繪製兩張圖：第一張是跨方法（AHP、BWM、IFS-BWM）一致性比率長條圖，並以紅色虛線標示 0.10 判斷門檻；第二張是三種方法的準則權重分布並列長條圖，方便直接比較各方法權重分布的異同。

**範例 6：結果段落初稿撰寫**

> 根據以下統計結果（BWM 權重：可回收性 .416、生物可分解性 .237、供應穩定性 .158、法規符合度 .118、採購成本 .072，CR=.025；AHP 權重與 BWM 最大差異 .015，CR=.008；BWM 僅需 7 次比較 vs AHP 需 10 次），請以碩士論文研究結果章節的學術寫作語氣，撰寫一段約 300 字的中文分析段落。

---

## 結果呈現與分析：碩士論文寫法示例

以下段落數值取自本週 Colab 範例程式碼之實際執行結果，供學生對照模仿寫作邏輯（實際數值請以自己資料之 Colab 輸出為準）。

> **4.1 BWM 權重求解與一致性檢驗結果**
>
> 本研究以企業綠色包裝材料採購評選委員會認定之最佳準則（可回收性）與最差準則（採購成本）為錨點，依 Rezaei（2016）線性規劃模型求解 BWM 權重，僅需 7 次專家比較（相較 AHP 之 10 次比較，節省 30% 之填答工作量）。求解結果顯示，五項準則權重由高至低依序為：可回收性（.416）、生物可分解性（.237）、供應穩定性（.158）、法規符合度（.118）、採購成本（.072），最優目標函數值 $\xi^{*} = 0.057$ ，依 $a_{BW}=5$ 對照一致性指標表（CI = 2.30），求得一致性比率 CR = .025，遠低於 .10 之可接受門檻，顯示專家判斷具備良好之邏輯一致性。
>
> **4.2 IFS 語意聚合穩健性交叉驗證**
>
> 為交叉驗證上述結果之穩健性，本研究進一步以直觀模糊集合蒐集三位專家之語意判斷，經 IFWA 算子聚合並以分數函數解模糊化後，重新求解 BWM，結果顯示可回收性（.548）與採購成本（.050）仍分別為權重最高與最低之準則，與精確數值分析之排序方向一致；惟五項準則之精確權重數值與精確數值分析結果存在一定差異（最大差異達 .132），此一發現反映了「專家直接給出精確數值」與「多位專家以語意詞彙表達、再統計聚合」這兩種不同資料蒐集模式，雖然在準則相對重要性排序上得出一致結論，但在權重數值之精確程度上仍存在方法論差異，此為兩種資料蒐集模式本質差異所致，而非分析錯誤。
>
> **4.3 BWM 與 AHP 權重分布敏感情境對照**
>
> 進一步將 BWM 權重與依相同專家偏好邏輯建構之 AHP 權重對照，結果顯示兩方法求得之權重高度相近（最大差異僅 .015），AHP 之一致性比率為 .008，優於 BWM 之 .025，惟 AHP 需要 10 次成對比較，較 BWM 多出 3 次。此一發現顯示，在本研究情境下，BWM 能以顯著更少之專家認知負荷，求得與 AHP 高度相近之權重結果，支持本研究採用 BWM 作為主要權重求解方法之方法論選擇，此亦與本週參考文獻中 Rezaei（2015）原始研究之實證發現相互呼應。

**APA 格式三線表範例：BWM 準則權重與一致性檢驗結果**

| 準則 | BWM 權重 | 排序 | AHP 權重 | IFS-BWM 權重 |
|---|---|---|---|---|
| 可回收性（C1） | .416 | 1 | .430 | .548 |
| 生物可分解性（C2） | .237 | 2 | .222 | .148 |
| 供應穩定性（C4） | .158 | 3 | .165 | .107 |
| 法規符合度（C5） | .118 | 4 | .115 | .148 |
| 採購成本（C3） | .072 | 5 | .068 | .050 |

*註：BWM ξ*=.057，CR=.025；AHP λmax=5.035，CR=.008；IFS-BWM ξ*=.153，CR=.034。以上數值為本週 Colab 範例實際執行結果，實際研究請以自己資料之輸出為準。*

---

## 常見統計誤區與 Q&A

**Q1：既然 BWM 的比較次數比 AHP 少這麼多，是不是代表以後都應該用 BWM 取代 AHP？**
不完全是，這是一個常見的過度推論。BWM 之效率優勢，主要展現在「準則數量較多、專家填答時間有限」的情境；但 AHP 之完整成對比較矩陣，能提供比 BWM 更豐富的資訊（每一組準則配對都有直接比較資料），對於準則數較少（如 3–5 個）、或研究者格外重視完整兩兩關係驗證之情境，AHP 仍有其方法論價值。此外，AHP 之一致性檢驗機制（CR）發展較成熟、學界共識較穩固；BWM 之一致性檢驗（見本週理論篇 3.4 節 Liang et al. 之討論）仍在持續發展中。兩方法各有適用情境，論文中應依研究之準則數量、專家可用時間、方法論成熟度需求等因素，具體說明方法選擇之理由，而非簡單地認為「比較次數少就是比較好」。

**Q2：BWM 的一致性比率跟 AHP 的一致性比率，數值可以直接互相比較嗎？**
不建議直接互相比較數值大小，僅能比較「是否通過各自方法之判斷門檻」。這是因為兩者之 CI（一致性指標）計算基礎完全不同——AHP 之 RI 是針對完整 $n \times n$ 正倒值矩陣、透過大量隨機模擬產生的隨機一致性基準；BWM 之 CI 則是針對「以 Best、Worst 為錨點」之特殊比較結構、透過理論推導產生的一致性上限。本週 Colab 範例雖然同時呈現了兩者之 CR 數值（AHP=.008 vs BWM=.025），但論文寫作時應強調「兩者皆通過各自方法之 0.10 判斷門檻」，而非直接比較兩個 CR 數值何者較小、藉此宣稱某方法「比較一致」，此為常見的統計誤用。

**Q3：IFS 語意聚合後求得的權重，跟精確數值 BWM 的權重不完全一樣，我該用哪一組作為最終研究結果？**
這與第 4 週理論篇討論 AHP 與 Fuzzy AHP 權重不完全一致時之處理原則相同（見第 4 週 Q2）：兩者並非互斥，實務上常見做法是以精確數值分析（通常來自單一決策委員會共識或多位專家幾何平均後之單一比較向量）作為主要研究結果，IFS 語意聚合分析作為穩健性交叉驗證，並在論文中明確報告兩者排序是否一致（即使精確權重數值有落差），以此作為研究結論穩健性之佐證；若兩者排序方向也不一致，則應如實討論可能原因（如語意詞彙對照表設計是否恰當、專家語意判斷是否存在系統性分歧）。

**Q4：本週理論篇提到 BWM 原始非線性模型可能有「解不唯一」的問題，這具體是什麼意思？為什麼線性模型就沒有這個問題？**
非線性模型之目標函數，是最小化「比率偏離」 $|w_B/w_j - a_{Bj}|$ 的最大值，由於此目標函數對權重變數而言並非凸函數（non-convex），當專家比較資料非完全一致時，可能存在多組數值不同、但都能讓目標函數達到相同最小值的權重解，此時電腦求解程式可能因初始值或求解演算法之不同，回傳不同的「最優解」，導致結果不可重現；Rezaei（2016）之線性模型，改以「加權偏離」 $|w_B - a_{Bj} \cdot w_j|$ 取代比率偏離，此目標函數對權重變數而言是凸函數（分段線性），保證全域最優解唯一，這也是本週課程選擇實作線性模型而非原始非線性模型的方法論依據，此為 BWM 方法論發展史上一個重要的技術演進，值得在論文文獻回顧中準確引用說明。

**Q5：直觀模糊集合（IFS）跟第 4 週學過的三角模糊數（TFN），都是用來處理不確定性，兩者可以互相替代嗎？**
不完全可以互相替代，兩者刻畫不確定性的維度不同。三角模糊數以 $(l, m, u)$ 三點描述「單一數值的模糊區間」（最悲觀、最可能、最樂觀估計），本質上仍是「一維」的不確定性表達；直觀模糊集合以 $(\mu, \nu)$ （歸屬度、非歸屬度）加上衍生的猶豫度 $\pi$ ，描述的是「贊成—反對—猶豫」三元並存的判斷結構，能表達「專家既不完全贊成、也不完全反對，而是真正處於猶豫狀態」的心理現實，這是三角模糊數難以直接表達的維度。實務上，若研究情境更著重「數值估計的模糊區間」（如成本估計），三角模糊數較適合；若研究情境更著重「專家意見的贊成／反對／猶豫傾向」（如政策接受度評估），IFS 較能貼切刻畫，選擇何種模糊工具，應回歸研究問題本質判斷。

---

## 延伸研究方向：新創加速器創業投資案甄選決策——直觀模糊 BWM 評估模型

### 6.1 研究背景與理論基礎

新創加速器（startup accelerator）在甄選投資案時，面臨的決策不確定性遠高於本週主要研究範例之綠色包裝材料採購情境——投資評審委員對早期新創團隊之「團隊執行力」「市場規模潛力」「商業模式可行性」等準則之判斷，往往帶有高度主觀猶豫（畢竟早期新創缺乏成熟營運數據可供客觀驗證），此一情境正是 IFS 相較傳統精確數值判斷更具方法論優勢之處。本週參考文獻中之綠色供應商評選 IFS-BWM-TOPSIS 整合研究，已示範將 IFS 與 BWM 整合應用於高不確定性採購決策，可作為本延伸研究方向之直接方法論參照。

### 6.2 建議研究設計

1. **理論框架**：以 IFS-BWM 求解投資評選準則權重（處理評審委員判斷之猶豫），並可進一步結合 IFS 版本之 TOPSIS（將候選新創團隊之績效評分也以 IFS 表達，而非強迫評審在資訊不完整下給出精確分數），對候選投資案進行綜合排序。
2. **候選評選準則（範例）**：
   - 團隊執行力與創業經驗
   - 市場規模與成長潛力
   - 商業模式可行性與獲利路徑清晰度
   - 技術／產品差異化程度
   - 早期驗證證據（如試點客戶、初步營收）之充分程度
3. **研究對象**：建議邀請 5–10 位具早期投資評審經驗之加速器投資委員會成員或創投合夥人。
4. **分析流程**：完全比照本週 Colab 實作流程（BWM 線性規劃 → 一致性檢驗 → IFS 語意聚合穩健性交叉驗證 → 與 AHP 敏感情境對照），僅需替換準則定義與評審資料，即可直接複用本週所有自訂函式（`solve_bwm_linear()`、`ifwa_aggregate()`、`bwm_consistency_check()`）。
5. **實務意涵**：此類研究可協助加速器建立一套結構化、可向外部利害關係人（如政府補助單位、有限合夥人 LP）清楚說明決策依據的投資評選架構，同時透過 IFS 機制如實保留評審委員在早期投資判斷中固有的高度不確定性，避免傳統精確評分制度「強迫評審給出過度自信的數字」所造成的資訊失真。

### 6.3 給學生的思考練習

請思考：若加速器投資委員會之評審成員，分別來自「財務背景」與「技術背景」兩種專業視角，你預期這兩類評審對「技術／產品差異化程度」此一準則之歸屬度（ $\mu$ ）與猶豫度（ $\pi$ ）判斷，可能呈現何種系統性差異？這樣的差異，在 IFWA 聚合過程中應該被平均化處理，還是應該保留為獨立的子群體分析結果？請說明你的理由。

---

## 課後作業與練習

**練習一：修改 Best／Worst 準則觀察權重變化**
請將 Step 1 中認定之最佳準則（C1）與最差準則（C3）互換其中一項（例如改認定「供應穩定性」為最佳準則），重新設計對應之 Best-to-Others 與 Others-to-Worst 向量，重新執行 BWM 分析，觀察權重排序是否合理反映此一認定變化。

**練習二：真實決策情境實作**
請自行設計一個至少包含 4 個評選準則之決策情境，實際邀請至少 5 位親友或同學扮演「專家」，分別完成 BWM 精確數值填答與 IFS 語意詞彙填答兩種版本之問卷，套用本週完整 Colab 程式碼執行分析。請繳交：(1) 問卷記錄、(2) 執行後的 Colab Notebook（.ipynb）、(3) 一頁 A4 的結果摘要（比照本週「結果呈現與分析」段落之寫法）。

**練習三：非線性模型與線性模型解之比較**
請查閱 Rezaei（2015）原始非線性模型之數學表述，嘗試使用 `scipy.optimize.minimize`（而非 `linprog`）實作非線性版本之 BWM，並比較其求解結果與本週線性模型結果是否一致，藉此驗證本週理論篇 3.3 節所述「非線性模型可能有多組最優解」之現象。

**練習四：文獻延伸閱讀報告**
請從本週「參考文獻與延伸閱讀」清單中，任選一篇 BWM、IFS 或 IFS-BWM 整合應用之期刊論文或台灣碩士論文，撰寫一頁重點摘要，內容須包含：(1) 該研究之準則架構與 Best／Worst 認定、(2) 報告之一致性檢驗結果、(3) 是否有與 AHP 或其他方法進行敏感情境對照、(4) 該研究提出之管理實務建議為何。

**練習五：語意對照表的在地化調整**
本週採用之語意詞彙對照表（EI、WMI、MI、SMI、VSMI）以英文詞彙為基礎翻譯而成，請嘗試設計一套更貼近繁體中文語境與台灣職場溝通習慣的語意詞彙（例如「差不多重要」「稍微重要一點」「明顯比較重要」等），並自行設定對應之 $(\mu, \nu)$ 數值，說明你設定這些數值的依據與考量。

---

## 參考文獻與延伸閱讀（已查核連結）

1. Rezaei, J. (2015). Best-worst multi-criteria decision-making method. *Omega*, 53, 49-57.
   https://www.sciencedirect.com/science/article/abs/pii/S0305048315000121
   （BWM 方法之原始創始論文，提出非線性模型、一致性指標對照表與 $2n-3$ 比較次數之核心設計，為本週理論篇 3.1、3.3、3.4 節之直接方法論依據。）

2. Rezaei, J. (2016). Best-worst multi-criteria decision-making method: Some properties and a linear model. *Omega*, 64, 126-130.
   https://www.sciencedirect.com/science/article/abs/pii/S0305048315002479
   （提出線性化 BWM 模型以解決非線性模型解不唯一問題之後續論文，為本週 Colab 實作採用模型之直接依據。）

3. Best Worst Method 官方網站與工具資源。
   https://bestworstmethod.com/
   （Rezaei 教授維護之 BWM 官方方法論網站，提供完整計算工具、常見問題解答與最新方法論發展資訊，可作為課後延伸查詢之權威資源。）

4. Liang, F., Brunelli, M., & Rezaei, J. Consistency issues in the best worst method: Measurements and thresholds.
   https://www.researchgate.net/publication/338078269_Consistency_Issues_in_the_Best_Worst_Method_Measurements_and_Thresholds
   （近期針對 BWM 一致性檢驗門檻之方法論深入研究，指出固定門檻（如 0.10）之侷限性，為本週理論篇 3.4 節與 Q&A 之重要延伸依據。）

5. Atanassov, K. T. (1986). Intuitionistic fuzzy sets. *Fuzzy Sets and Systems*, 20(1), 87-96.
   https://doi.org/10.1016/S0165-0114(86)80034-3
   （IFS 理論之原始創始論文，提出歸屬度、非歸屬度、猶豫度之核心數學結構，為本週理論篇 3.5 節之直接方法論依據。）

6. Tian, Z.-P., Zhang, H.-Y., Wang, J.-Q., & Wang, T.-L. Green Supplier Selection Using Improved TOPSIS and Best-Worst Method Under Intuitionistic Fuzzy Environment. *Informatica*.
   https://informatica.vu.lt/journal/INFORMATICA/article/1095/info
   （整合 IFS、BWM 與 TOPSIS 之綠色供應商評選期刊論文，方法論架構與本週延伸研究方向高度相關，可作為 IFS-BWM 整合應用之直接參照範本。）

7. Green supplier selection for the steel industry using BWM and fuzzy TOPSIS: A case study of Khouzestan steel company. *Sustainable Futures*, 2, 100012.
   https://doaj.org/article/eebef8c822b64e94b5eb3ef046e0e65a
   （BWM 應用於鋼鐵業綠色供應商評選之開放取用期刊論文，示範完整之 BWM 準則權重求解與供應商排序流程。）

8. 多準則決策方法運用於供應商評選與訂單分配。Airiti Library 華藝線上圖書館。
   https://www.airitilibrary.com/Article/Detail/c0000083-201811-201905170010-201905170010-68-80
   （台灣本土研究，整合 BWM 與 Fuzzy-TOPSIS 於供應商評選及訂單分配，以電動車產業為實例，可作為 BWM 台灣本土應用之直接參照文獻。）

**方法論經典文獻（建議延伸閱讀，非本次線上搜尋來源，圖書館或資料庫可查閱）**：

- Zadeh, L. A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338-353.
- Xu, Z., & Yager, R. R. (2006). Some geometric aggregation operators based on intuitionistic fuzzy sets. *International Journal of General Systems*, 35(4), 417-433.
- Guo, S., & Zhao, H. (2017). Fuzzy best-worst multi-criteria decision-making method and its applications. *Knowledge-Based Systems*, 121, 23-31.
- Mi, X., Tang, M., Liao, H., Shen, W., & Lev, B. (2019). The state-of-the-art survey on integrations and applications of the best worst method in decision making: Why, what, what for and what's next? *Omega*, 87, 205-225.

---

## 附錄

### 附錄 A：AHP（第 4 週）與 BWM（本週）方法對照表

| 比較構面 | AHP | BWM |
|---|---|---|
| 資料結構 | 完整 $n \times n$ 正倒值矩陣 | Best-to-Others、Others-to-Worst 兩個向量 |
| 所需比較次數 | $\binom{n}{2} = n(n-1)/2$ | $2n-3$ |
| 核心運算 | 特徵向量法 | 線性規劃（Rezaei, 2016 版本） |
| 解的唯一性 | 唯一 | 線性模型下唯一；原始非線性模型可能不唯一 |
| 一致性檢驗 | CR（RI 對照表，方法論成熟穩固） | CR（CI 對照表，門檻仍在方法論發展中） |
| 適用情境 | 準則數較少、重視完整兩兩驗證 | 準則數較多、專家時間有限、重視效率 |

### 附錄 B：IFS（本週）與三角模糊數（第 4 週）方法對照表

| 比較構面 | 三角模糊數（TFN，第 4 週） | 直觀模糊集合（IFS，本週） |
|---|---|---|
| 資料結構 | $(l, m, u)$ 三點描述數值區間 | $(\mu, \nu)$ 歸屬度與非歸屬度 |
| 刻畫之不確定性型態 | 數值估計之模糊區間（一維） | 贊成—反對—猶豫三元並存結構 |
| 衍生指標 | 無直接對應猶豫度概念 | 猶豫度 $\pi = 1-\mu-\nu$ |
| 適合情境 | 成本、時間等數值估計之模糊性 | 政策接受度、投資評審等意見傾向之猶豫性 |
| 常見整合方法 | Fuzzy AHP（Chang 擴展分析法） | IFS-BWM、IFS-TOPSIS |

### 附錄 C：術語中英對照表

| 中文術語 | 英文術語 | 縮寫 |
|---|---|---|
| 最佳最差法 | Best-Worst Method | BWM |
| 最佳準則 | Best Criterion | — |
| 最差準則 | Worst Criterion | — |
| 直觀模糊集合 | Intuitionistic Fuzzy Sets | IFS |
| 歸屬度 | Membership Degree | $\mu$ |
| 非歸屬度 | Non-membership Degree | $\nu$ |
| 猶豫度 | Hesitation Degree | $\pi$ |
| 直觀模糊加權平均算子 | Intuitionistic Fuzzy Weighted Averaging | IFWA |
| 分數函數 | Score Function | S |
| 線性規劃 | Linear Programming | LP |
| 一致性指標 | Consistency Index | CI |
| 一致性比率 | Consistency Ratio | CR |

### 附錄 D：常見程式錯誤排解（Debugging Tips）

| 錯誤現象 | 常見原因 | 排解建議 |
|---|---|---|
| `linprog` 回傳 `status != 0`（求解失敗） | 約束條件設定矛盾，或 `best_idx`／`worst_idx` 索引錯誤 | 檢查 `A_B[best_idx]` 與 `A_W[worst_idx]` 是否皆為 1，並確認約束矩陣建構迴圈正確跳過對應索引 |
| BWM 權重總和不為 1 | `A_eq`／`b_eq` 設定錯誤 | 確認 `A_eq[0,:n]=1` 且 `b_eq=[1]` 正確對應權重加總約束 |
| IFWA 聚合結果的猶豫度為負值 | 語意對照表中 $\mu+\nu$ 設定超過 1 | 檢查 `ling_to_ifs` 字典中每組 $(\mu,\nu)$ 是否滿足 $\mu+\nu \leq 1$ |
| CI 對照表查詢時出現 KeyError | `a_BW` 四捨五入後超出 1-9 範圍 | 確認 Saaty 量表原始輸入值介於 1-9 之間，`bwm_consistency_check()` 已內建邊界保護但仍應檢查輸入資料 |
| AHP 與 BWM 權重差異異常大 | 建構對應 AHP 矩陣時，未依循與 BWM 相同之專家偏好邏輯 | 檢查 Step 6 中 AHP 矩陣各元素是否與 Step 1 之 `A_B`、`A_W` 向量隱含之偏好方向一致 |

### 附錄 E：繳交前自我檢核清單

- [ ] 已明確說明最佳準則與最差準則之認定依據
- [ ] 已報告完整之 Best-to-Others 與 Others-to-Worst 向量
- [ ] 已說明採用線性模型（而非原始非線性模型）以確保解唯一性
- [ ] 已報告 $\xi^{*}$ 、CI、CR 值，並判斷是否通過一致性檢驗
- [ ] 若執行 IFS 語意聚合分析，已報告語意對照表與聚合過程
- [ ] 已將 BWM 與 AHP（或其他方法）進行權重分布敏感情境對照
- [ ] 已報告各方法所需之比較次數，並討論效率差異之實務意涵
- [ ] 已繪製一致性檢定評估圖與權重分布對照圖
- [ ] 所有統計結果之文字敘述與表格數值一致，無謄寫錯誤
- [ ] 已誠實說明 BWM 一致性門檻仍在方法論發展中之研究限制

### 附錄 F：研究倫理提醒

本週研究設計涉及邀請專家分別完成精確數值與語意詞彙兩種版本之問卷，應於問卷說明中清楚告知專家此為同一研究之方法論穩健性驗證設計，避免專家誤以為需要重複填答相同內容而產生填答疲乏；若研究情境涉及企業採購決策或早期投資評審等具高度商業敏感性之資訊（如延伸研究方向之新創投資評選），應與受訪企業或投資機構明確約定研究資料之保密範圍，並於學術發表時採用適當之匿名化處理，避免揭露具體投資決策內容對相關新創團隊或企業造成不必要之商業影響。

---

## 第二模組總結：知識驅動型決策科學方法家族回顧

隨著本週課程結束，本課程第二模組（第 4–7 週）之知識驅動型決策科學方法已完整介紹完畢。以下總結表回顧整個模組之方法論演進脈絡，幫助學生在規劃期末專題時，能快速判斷自己的研究問題適合採用哪一種（或哪幾種組合）方法：

| 週次 | 方法 | 核心假設演進 | 解決的核心問題 |
|---|---|---|---|
| 第 4 週 | AHP／Fuzzy AHP／TOPSIS | 準則彼此獨立 | 在明確、獨立的準則下，如何求出權重並排序方案？ |
| 第 5 週 | DEMATEL | 準則可能互為因果 | 準則之間的因果結構是什麼？哪些是根源、哪些是表徵？ |
| 第 6 週 | DANP／VIKOR | 在因果網路上求解均衡權重 | 如何在複雜回饋網路中求出穩定權重，並兼顧群體效用與個別遺憾進行妥協排序？ |
| 第 7 週 | BWM／IFS | 效率化資料蒐集＋精細化不確定性刻畫 | 如何用最少的專家認知負荷求出可靠權重？如何完整保留專家判斷中的猶豫？ |

**方法選擇決策樹（簡化版）**：

```
研究問題是否涉及「在多個準則下對方案排序或選擇最適方案」？
        │ 是
        ▼
準則之間是否存在需要納入分析的因果影響關係？
        │
   ┌────┴────┐
   否          是
   │           │
   ▼           ▼
準則數量      是否需要同時求出「可直接使用之權重」？
是否較多？        │
   │         ┌────┴────┐
┌──┴──┐      否          是
少    多      │           │
│     │      ▼           ▼
AHP  BWM   DEMATEL      DANP
（可選配    （聚焦因果    （+VIKOR
Fuzzy AHP   結構辨識）    妥協排序）
或 IFS 處理
不確定性）
```

第三模組（第 8–11 週）將轉向資料驅動型 AI 方法——非監督式學習、集成學習、可解釋性 AI（XAI）與混合式 SEM-ANN 架構，學生將發現，知識驅動與資料驅動兩大方法家族並非互斥，而是可以在同一份研究中互補整合（例如以 DEMATEL／DANP 辨識之關鍵因素，作為後續機器學習模型之特徵篩選依據），這也是本課程「決策智能與混合式 AI」核心定位之具體實踐。

---

## 下週預告

第 8 週將進入第三模組「資料驅動 AI：預測、分群與可解釋性模型」，主題為「非監督式 AI 學習：多維度特徵投影與顧客智慧分群（K-Means/PCA）」。學生將從本週結束之知識驅動型方法，重新回到資料驅動型方法之脈絡，學習如何運用大樣本客戶交易資料，透過主成分分析（PCA）進行特徵降維，並以 K-Means 分群演算法搭配手肘法（Elbow Method）與輪廓係數（Silhouette Score）決定最適群數，對客戶群體進行智慧分群。研究範例將以「數位銀行高資產用戶金融交易輪廓與精準分群研究」為主題，並延伸至「智慧醫療長照資源需求者之健康特徵多維分群模型」之期末專題發想方向。
