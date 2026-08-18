# 实验设计草案：统一 Backbone 下的 Taxonomy 等级受控对比实验

状态：草案（未实施）。用于支撑 `papers/survey.pdf`（*Agentic-SQL Taxonomy*）投稿 EACL 前补充的核心实验。

## 1. 背景与动机

`survey.pdf` 的 Table 2（`acl_paper.tex` 中 `tab:method_comparison`）把 DAIL-SQL（L1）、DIN-SQL / CodeS / Alpha-SQL（L2）、MAC-SQL / CHASE-SQL / XiYan-SQL / Scale-SQL（L3）的 BIRD EX 数字并排列出，用来论证"自主性等级越高，性能越好"。但这些数字来自不同论文、不同 backbone（GPT-4、GPT-4o、开源 32B 混杂），**taxonomy 等级和模型能力是混杂（confounded）的**：等级更高的方法往往也用了更强的模型，无法排除"其实是模型更强"这个替代解释。

本实验的目标是补一个受控对照：**固定 backbone LLM 不变，只改变推理阶段的编排结构（single-turn / iterative-refinement / multi-agent）**，直接测量 taxonomy 等级本身带来的边际增益，把 Table 2 的"跨论文横向比较"升级为"同一论文内的受控实验"。

## 2. 研究问题

- **RQ1**：固定 backbone 的前提下，L1→L2→L3 的 EX 是否单调提升？在 Spider 和 BIRD 上是否都成立？提升幅度是否显著（非随机噪声）？— BIRD 是 survey Table 2 本身的评测口径，两个数据集都跑是为了让这张受控实验表能直接回填/对照 Table 2，而不是另立一个不可比的基准；Spider 部分复用现成管线，成本低，可以先跑通验证 pipeline 再上 BIRD。
- **RQ2**：这一趋势是否只在某个难度区间成立（例如 L3 的增益主要来自 Hard/Extra Hard，而 Easy 上 L2/L3 甚至因过度修正而下降）？— 直接支撑 survey 6.4 节"over-thinking on Easy queries"的说法，目前那句话没有任何实测数据支持。
- **RQ3**：每提升一个等级，token 消耗 / 调用次数 / 延迟的边际成本是多少？— 为 6.2 节 efficiency-accuracy trade-off 提供实测曲线，而不是定性描述。
- **RQ4（稳健性，次优先）**：换一个 backbone，L1→L2→L3 的趋势是否依然成立？

## 3. 变量设计

| 变量 | 设定 |
|---|---|
| 自变量 | Taxonomy 等级：L1（单轮生成）/ L2（执行反馈迭代修正）/ L3（多 agent 协作） |
| 受控变量 | Backbone LLM 固定为同一个模型（见第 6 节）；schema 表示方式、few-shot 示例、温度等采样参数在三个等级间保持一致 |
| 因变量（主） | Execution Accuracy (EX)，Exact Match (EM) 作为辅助 |
| 因变量（效率） | 每样本 LLM 调用次数、输入+输出 token 数、wall-clock 延迟、估算 API 成本 |
| 分组变量 | SQL 难度（Spider: Easy/Medium/Hard/Extra Hard，复用现有分类逻辑；BIRD: 官方自带的 simple/moderate/challenging 标签） |
| 数据集（次自变量） | Spider test（跨库泛化，复用现有管线，成本低，先跑通）+ BIRD dev（与 survey Table 2 同口径，直接支撑论文核心表格） |

## 4. 三个等级的 Pipeline 设计

三个等级共享同一个 base prompt（schema + question + k-shot 示例），差异只在"生成之后发生什么"。

### L1 — Single-Turn Generation（复用现有代码，几乎不用新写）

直接对应 `case_study_spider_cot/scripts/07_baseline_api_eval.py` 的 3-shot baseline 逻辑：一次 LLM 调用，输出即为最终 SQL，不做任何执行反馈或多 agent 协作。这是三个等级共同的"第 0 次调用"，L2/L3 都从这里的 `Y_0` 出发。

### L2 — Iterative Refinement（execution-feedback self-correction）

对应 survey 5.1 节的公式 `Y_{t+1} = π(I, Q, S, Y_t, ε | θ)`：

```
Y_0 = generate(I, Q, S)              # 同 L1 的第一次调用
for t in range(MAX_ITER):            # MAX_ITER = 3
    result, err = execute(Y_t, db)
    if err is None:
        break                        # 执行成功，停止迭代
    Y_{t+1} = generate(I, Q, S, Y_t, err)   # 把报错信息喂回去要求修正
final_Y = Y_t
```

只处理"执行报错"这一类反馈（语法错误、未定义列、join 报错等），不引入语义判断——语义判断留给 L3 的 Reviewer 角色，避免 L2/L3 的边界模糊。

### L3 — Multi-Agent Collaboration（需要新写，最小可行版本）

对应 survey 6.2 节描述的 Planner–Coder–Reviewer 结构（MAC-SQL 风格），三个角色调用**同一个 backbone 模型**、只是 system prompt 和输入不同，这样才能保证"结构变了、模型没变"：

```
# Agent 1: Planner —— schema 过滤 + 问题分解
filtered_schema, subplan = planner(I, Q, S)

# Agent 2: Coder —— 基于过滤后的 schema 和分解结果生成 SQL
Y_0 = coder(I, Q, filtered_schema, subplan)

# 复用 L2 的执行反馈循环，但作用对象是 Coder（同一 agent 内部自愈）
for t in range(MAX_ITER):
    result, err = execute(Y_t, db)
    if err is None:
        break
    Y_{t+1} = coder(I, Q, filtered_schema, subplan, Y_t, err)

# Agent 3: Reviewer —— 检查执行结果是否在语义上回答了问题
#（捕捉 MIN/MAX 用反、条件写反这类不报执行错误的逻辑错误）
verdict, critique = reviewer(Q, filtered_schema, Y_t, sample_rows(result))
if verdict == "reject":
    Y_final = coder(I, Q, filtered_schema, subplan, Y_t, critique)   # 再修一轮，封顶 1 次
else:
    Y_final = Y_t
```

三个等级的调用次数上界：L1 = 1 次，L2 ≤ 4 次（1 + MAX_ITER），L3 ≤ 6 次（Planner 1 + Coder 最多 4 + Reviewer 1，再加最多 1 次 Reviewer 触发的修正）。这个调用次数差本身就是 RQ3 要报告的效率数据。

## 5. 数据集与采样方案

Spider 和 BIRD 并行跑，两者共用第 4 节同一套 L1/L2/L3 pipeline 代码（数据集只是 pipeline 的一个输入参数），分别产出两张结果表：Spider 表用于快速验证 pipeline 是否跑通、成本低；BIRD 表口径对齐 survey Table 2，是回填论文的主表。

### 5.1 Spider（复用现有管线）

- 数据源：`case_study_spider_cot/processed_data/test_for_cot.json`（2,147 条，已含 `question + schema + gold_sql`，40 个测试库，与训练库完全不重叠）。
- 难度标签：复用 `case_study_spider_cot/scripts/06_difficulty_analysis.py` 里的 `classify_difficulty()` 逻辑（规则式：Easy/Medium/Hard/Extra Hard），对全量 test 样本打标签。
- 数据库执行环境：复用 `case_study_spider_cot/spider_data/database`（SQLite，体积小，单库通常几十 MB 以内）。
- 无需额外下载或预处理，直接抽样即可，作为 pipeline 调试和第一轮结果的低成本数据集。

### 5.2 BIRD（新增，对齐 Table 2 口径）

- 数据源：BIRD 官方 dev 集（需从 [BIRD 官网](https://bird-bench.github.io/) 下载，含 `dev.json`、`dev_databases/`，SQLite 数据库；仓库目前还没有这份数据，需要新增下载步骤）。
- 难度标签：**直接使用 BIRD 官方自带的 `difficulty` 字段**（`simple` / `moderate` / `challenging`），不需要像 Spider 那样自己写规则分类器——这样也顺带避免了"用 Spider 的规则去分类 BIRD 的 SQL"这种口径不一致的问题。
- External knowledge（`evidence` 字段）：三个等级的 prompt 都统一附加 `evidence`（跟 survey 里 BIRD 的问题定义 $K$ 一致），保持"是否使用知识"这一维度在三个等级间不变量；是否需要单独测"L3 的 Planner 有没有更好利用 evidence"作为加分分析，留到主结果出来后再看是否值得做。
- 数据库体积：BIRD 的 SQLite 库比 Spider 大得多（部分库到几百 MB～GB 级别），需要检查磁盘空间，并给执行超时设置留够余量（比如从 Spider 用的默认超时适当调大），避免把"执行慢"误判成"执行错误"。
- 需要新的数据预处理步骤（结构上镜像 `case_study_spider_cot` 的 01/02 步，但产出格式要和 Spider 一致，这样才能复用同一套 pipeline 代码）：下载 → schema 提取（含 BIRD 的列描述文件 `column_meaning.csv`/`database_description`，比 Spider 的 schema 更丰富）→ 转成 `question + schema + evidence + gold_sql + difficulty` 统一格式。

### 5.3 采样方案（两个数据集共用同一套抽样逻辑）

**分层抽样而非全量跑**：L2/L3 每条样本要打 4~6 次 API，两个数据集 × 三个等级全量跑成本过高。分别在 Spider（按 Easy/Medium/Hard/Extra Hard 四档）和 BIRD（按 simple/moderate/challenging 三档）上等比例分层抽样，每档抽 80~100 条，Spider 侧 **N ≈ 320~400**，BIRD 侧 **N ≈ 240~300**。两个数据集内部都用固定随机种子抽样，保证三个等级（以及做稳健性检验时的两个 backbone）用的是完全相同的样本子集（这是配对检验的前提，见第 7 节）。

## 6. Backbone 选择

**更新（2026-08）**：DeepSeek 已于 2026-04-24 发布 V4，旧模型名 `deepseek-chat`（对应 V3）已在 2026-07-24 完全下线，现由 `deepseek-v4-flash` / `deepseek-v4-pro` 取代。API key 是账号级别的凭证，不随模型版本变化，可以直接用；但 `case_study_taxonomy_ablation/scripts/common.py` 里 `PROVIDER_CONFIGS["deepseek"]["model"]` 已经改成 `deepseek-v4-flash` 作为默认值，各 run 脚本也加了 `--model`/`--base_url` 覆盖参数，方便切到 `deepseek-v4-pro` 或未来的模型名变化。

主 backbone：**DeepSeek V4**（`deepseek-v4-flash`，`deepseek-chat` 的直接继任者）。原计划用旧的 3-shot 基线（EX 51.47%，`case_study_spider_cot/outcome/eval_results_deepseek_3shot.json`，跑在已下线的 `deepseek-chat`=V3 上）做口径核对，**现在已经不适用**——那是不同的模型版本，数字不再具有校准意义，只能作为历史参考，不能拿来验证新 L1 结果是否"跑对了"。好在这不影响受控实验本身的有效性：L1/L2/L3 三个等级依然是在同一个 V4 backbone 上跑的，"固定 backbone、只变编排结构"这个核心设计没有被破坏，只是不再有一个现成的旧基线可以拿来做健全性检查（sanity check）——第一次跑 V4 的 L1 时，除了看数字是否合理（不应该是 0% 或接近 100%），没有其他外部参照了，这点在正式实验前需要注意。

次 backbone（RQ4 稳健性检验，可选）：**GLM-4**，同样已在 `PROVIDER_CONFIGS` 中配置好，且已有 L1 基线（EX 66.28%）。如果时间/预算有限，可以先只做 DeepSeek V4 一个 backbone，把 GLM-4 作为 rebuttal 阶段的补充实验储备。

不建议用仓库里已微调的 Qwen3-8B/LLaMA3.1-8B 作为 L3 的 backbone——多 agent 角色扮演对模型的指令遵循和长上下文能力要求较高，8B 微调模型大概率无法稳定跑通 Planner/Reviewer 的结构化输出，会引入"模型能力不足"而非"taxonomy 结构无效"的混杂，违背受控实验的初衷。

## 7. 评测与统计检验

- 主指标：EX（复用现有 SQL 执行比对逻辑）；EM 作为辅助。
- 难度分组：直接调用 `06_difficulty_analysis.py` 的分类函数对结果分组。
- 效率指标：每条样本记录 `num_llm_calls`、`total_input_tokens`、`total_output_tokens`、`wall_clock_seconds`；成本按各 provider 当前定价折算（实施前需核对最新价格，不要用旧数字硬编码）。
- **显著性检验**：三个等级在同一个样本子集上跑（配对设计），用 **McNemar's test** 对 L1-vs-L2、L2-vs-L3、L1-vs-L3 三组做两两配对比较（EX 是二元正确/错误），而不是只看均值差——这是当前 survey 里完全缺失、但审稿人很可能会问的一环。

## 8. 需要新增的代码 / 与现有代码的映射

建议新建同级目录 `case_study_taxonomy_ablation/`（与 `case_study_spider_cot/` 平级）。Pipeline 和分析脚本对 Spider/BIRD 通用，统一加一个 `--dataset {spider,bird}` 参数；BIRD 需要额外一组数据准备脚本（镜像 `case_study_spider_cot` 的 01/02 步）：

| 脚本 | 状态 | 说明 |
|---|---|---|
| `00a_download_bird.py` | 新写 | 下载/校验 BIRD dev 集（`dev.json` + `dev_databases/`），仓库里目前没有这份数据 |
| `00b_prepare_bird_dataset.py` | 新写，参考 `case_study_spider_cot/scripts/01_extract_schema.py` + `02_prepare_dataset.py` | 提取 BIRD schema（含列描述）、拼 `evidence`，统一转成与 Spider 一致的 `question + schema + evidence + gold_sql + difficulty` 格式 |
| `01_sample_stratified_subset.py` | 新写 | 按难度分层抽样，`--dataset spider` 时用 Easy/Medium/Hard/Extra Hard、N≈320-400；`--dataset bird` 时用官方 simple/moderate/challenging、N≈240-300；固定 seed，输出样本 id 列表供三个等级共用 |
| `02_run_l1_single_turn.py` | 可基于 `07_baseline_api_eval.py` 改造 | 在抽样子集上跑，同时落盘 token/延迟；BIRD 模式下 prompt 里带上 `evidence` |
| `03_run_l2_iterative_refine.py` | 新写 | 第 4 节 L2 伪代码的实现，MAX_ITER=3，两个数据集共用同一份代码 |
| `04_run_l3_multi_agent.py` | 新写 | Planner/Coder/Reviewer 三角色实现，两个数据集共用同一份代码 |
| `05_evaluate_taxonomy_levels.py` | 可复用 `05_evaluate*.py` 的 EX/EM 计算逻辑 | 对三个等级、两个数据集的输出统一评测 |
| `06_difficulty_breakdown.py` | Spider 侧直接复用 `06_difficulty_analysis.py`；BIRD 侧直接按官方 `difficulty` 字段分组 | 两个数据集分别产出难度分组表，不混在一起比较（难度定义口径不同，见第 11 节） |
| `07_significance_test.py` | 新写 | McNemar's test，两个数据集分别做三组两两比较 |
| `08_efficiency_report.py` | 新写 | 汇总 token/调用次数/延迟/成本，Spider 和 BIRD 分别产出 accuracy-vs-cost 图表数据（BIRD 单条样本 schema 更大，token 成本会明显更高，两者不应合并平均） |
| `09_export_error_cases.py` | 新写（可选） | 分别从 Spider 和 BIRD 里导出 L1 失败但 L2/L3 修正成功的 3-5 个案例，供 survey 定性分析用 |

## 9. 产出如何回填到 survey 论文

- 新增两张表：固定 backbone（DeepSeek V4，`deepseek-v4-flash`）下 L1/L2/L3 在 **Spider** 和 **BIRD** 上各自的 EX/EM + 难度分组结果。BIRD 表是主表，直接放在 Table 2 附近，配一句"与 Table 2 的跨论文结果一致，但排除了 backbone 混杂，且同一批样本上可比"；Spider 表作为跨 benchmark 一致性的补充证据。
- 6.2 节 efficiency-accuracy trade-off 从纯定性描述改为引用 `08_efficiency_report.py` 产出的 accuracy-vs-cost 曲线/表格（Spider、BIRD 分开画，因为 BIRD 单样本 token 成本更高，合并画会误导）。
- 6.4 节"Easy 查询上 agentic overhead 可能带来负收益"补一句实测数据（RQ2），如果 Spider 和 BIRD 的 Easy/simple 结果趋势一致，这句话的说服力会明显增强。
- 若做了 GLM-4 第二 backbone，可在附录加一张稳健性表（同样 Spider + BIRD 两份）。

## 10. 成本与规模估算（实施前需重新核对）

Spider 侧：N=360（4 档 × 90）、DeepSeek V4（`deepseek-v4-flash`）单 backbone，L1 需 360 次调用，L2 最多 360×4=1,440 次，L3 最多 360×6=2,160 次，合计上限约 4,000 次 API 调用。

BIRD 侧：N=270（3 档 × 90），调用次数上限比例相同（L1=270，L2≤1,080，L3≤1,620，合计约 3,000 次），但单次调用的输入 token 明显更高（BIRD schema 更大、外加 evidence 字段），且部分数据库执行更慢，实际 wall-clock 时间和成本都要按 token 数重新估算，不能直接套用 Spider 的单价估算。

两个数据集合计上限约 7,000 次 API 调用。按 DeepSeek V4 当前定价（V4 上线后计价方式可能与 V3 不同，实施前务必在 [api-docs.deepseek.com](https://api-docs.deepseek.com/) 核实最新价格，此处不写死数字）估算总成本，Spider 部分量级应在几美元到十几美元，BIRD 部分因 token 更多可能翻倍，合计建议预留几十美元的预算余量；主要瓶颈是 rate limit、串行执行 SQL 的耗时，以及 BIRD 大数据库的下载/加载时间，建议沿用 `07_baseline_api_eval.py` 已有的多线程 `--workers` 机制，并先跑通 Spider 验证 pipeline 无误后再上 BIRD，避免在贵的数据集上反复调试。

## 11. 局限与风险

- L3 的 Planner/Coder/Reviewer 是本仓库自行设计的最小实现，不是 MAC-SQL/CHASE-SQL 的原版复现，论文中应明确标注为"a minimal representative L3 pipeline in the spirit of MAC-SQL"，避免声称复现了具体论文。
- Spider N≈360、BIRD N≈270 的抽样规模足以做 McNemar 检验，但可能不足以支撑非常细粒度的难度×等级交互分析；如果结果边界不清晰，可考虑扩大抽样或聚焦在 Hard/Extra Hard（Spider）、challenging（BIRD）子集加样本。
- **Spider 与 BIRD 的难度标签口径不同**（Spider 是本仓库的规则式分类器，BIRD 是官方自评的 simple/moderate/challenging），两边的难度分组结果不能直接跨数据集比较数值，只能分别看"L1→L3 在各自数据集内部的难度趋势是否一致"，论文里要把这一点说清楚，避免读者误以为两个数据集的"Hard"是同一个标准。
- BIRD 的 evidence 字段是否/如何被三个等级利用，目前设计里只做了"三个等级都统一喂 evidence"这一种处理，没有单独做"有无 evidence"的消融；如果时间允许，这是一个值得补充的后续分析,但本轮不作为必需项。
- Reviewer agent 用同一个 backbone 自己审自己生成的 SQL，存在"自我一致性偏差"（self-consistency bias，模型倾向于认可自己的输出）——这是 L3 设计已知的局限，论文里应提及，而非回避。

## 12. 待办 Checklist

- [ ] 确认是否先只做 DeepSeek V4（`deepseek-v4-flash`）单 backbone（推荐），GLM-4 留作 rebuttal 储备
- [ ] 下载并校验 BIRD dev 集，跑通 `00a_download_bird.py` / `00b_prepare_bird_dataset.py`
- [ ] 实现 `01_sample_stratified_subset.py`（Spider + BIRD 两套难度口径），固定 seed，产出可复现的样本 id 列表
- [ ] 实现 L2/L3 pipeline 脚本，先在 Spider 的 10-20 条样本上跑通，再切到 BIRD 的 10-20 条样本确认 evidence/大 schema/执行超时都处理正确，最后放量
- [ ] 核实 DeepSeek V4 / GLM-4 当前 API 定价，写入 `08_efficiency_report.py` 的成本换算表（Spider、BIRD 分别算）
- [ ] 跑完后用 McNemar's test 检验三组两两差异是否显著（Spider、BIRD 分别检验）
- [ ] 把 Spider 表、BIRD 主表和 accuracy-vs-cost 图整理进 `acl_paper.tex`，补充到 6.2/6.4 节并更新 Table 2 附近的论述
