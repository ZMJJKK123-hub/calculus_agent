# calculus_agent

## 开发维护指南

本指南面向项目维护者，目标是让你在不阅读全部源码的情况下，快速理解系统结构、可调参数、扩展入口和常见故障处理。

## 1. 项目定位

这是一个微积分题目求解代理，核心行为是：

1. 输入题目文本。
2. 根据题目特征选择求解策略。
3. 输出统一 JSON（`reasoning_process` + `answer`）。

主工程目录：`ourwork/`

## 2. 仓库结构与职责

### 2.1 关键目录

| 路径 | 作用 |
| --- | --- |
| `ourwork/` | 主工程目录（实际提交内容）。 |
| `ourwork/agent/` | 求解器实现与策略逻辑。 |
| `ourwork/data/` | 样例题与知识库数据。 |

### 2.2 关键文件

| 路径 | 作用 | 维护要点 |
| --- | --- | --- |
| `ourwork/main.py` | 命令行入口；读取题目，调用 Agent，输出 JSON。 | 入口行为修改优先在这里做，不要在策略层混入 CLI 逻辑。 |
| `ourwork/agent/__init__.py` | 导出 `KimiCalculusAgent`。 | 保持导出名称稳定，避免提交配置失效。 |
| `ourwork/agent/calculus_agent.py` | 核心类与全部策略、知识检索、API 调用、输出清洗。 | 核心维护文件。改动需做回归验证。 |
| `ourwork/data/train.json` | few-shot 与训练样例数据。 | 字段结构变更会影响 `_load_examples`。 |
| `ourwork/data/theory.json` | 结构化理论知识库。 | 用于构建知识点索引。 |
| `ourwork/data/knowledge_points.json` | 知识点缓存索引（可由 `theory.json` 自动重建）。 | 不一致时会自动重建。 |
| `ourwork/requirements.txt` | 依赖声明（当前 `requests`、`sympy`）。 | 新依赖必须同步更新。 |
| `ourwork/submission.json` | 提交入口配置。 | `class`、`method` 必须与导出类一致。 |

## 3. 运行与环境

### 3.1 本地运行

```bash
cd ourwork
pip install -r requirements.txt
python main.py "求极限 lim_{x->0} sinx/x"
```

### 3.2 输出约定

输出固定 JSON 字段：

- `reasoning_process`
- `answer`

### 3.3 环境变量

| 变量名 | 默认值 | 作用 |
| --- | --- | --- |
| `KIMI_API_KEY` | 无 | API Key（核心必需）。 |
| `AGENT_STRATEGY` | `auto` | 强制指定策略（如 `pot`、`tot`、`mcts`）。 |
| `AGENT_FAST_MODE` | `0` | 快速模式，牺牲部分稳健性以换取速度。 |

> 安全说明：`ourwork/main.py` 当前构造 `KimiCalculusAgent` 时存在硬编码 `api_key`。维护时建议改为仅依赖环境变量，避免密钥泄露风险。

## 4. 执行链路（从入口到答案）

主链路如下：

1. `main.py:main()` 读取题目。
2. `main.py:run()` 实例化 `KimiCalculusAgent`。
3. `KimiCalculusAgent.solve()` 选择策略并求解。
4. 返回统一 JSON，并在 CLI 打印。

类初始化时会做三件事：

1. 加载 few-shot 示例（`train.json`）。
2. 构建检索索引（TF-IDF + BM25 特征）。
3. 加载或重建知识点缓存（`knowledge_points.json`）。

## 5. 策略路由系统

### 5.1 `solve()` 支持的策略名与行为

| 策略名 | 执行方法 | 回退逻辑 |
| --- | --- | --- |
| `symbolic-limit` / `limit-symbolic` / `limit` | `_solve_default` | 无 |
| `kb` / `knowledge` / `knowledge-first` / `kb-default` | `_solve_default` | 无 |
| `pot` | `_solve_with_pot` | 失败后 `_solve_default` |
| `step_back` / `step-back` / `stepback` | `_solve_with_step_back` | 异常后 `_solve_default` |
| `prm` / `process_reward` / `process-reward` | `_solve_with_prm` | 异常后 `_solve_default` |
| `constraints` / `constraint` / `system2` / `system-2` / `s2` | `_solve_with_constraints` | 异常后 `_solve_default` |
| `ltm` / `least_to_most` / `least-to-most` | `_solve_with_ltm` | 无 |
| `self_consistency` | `_self_consistency` | 样本失效后 `_solve_default` |
| `mcts` | `_solve_with_mcts` | 异常后 `_solve_with_tot` |
| `tot` | `_solve_with_tot` | 异常后 `_self_consistency` |
| `debate` | `_solve_with_debate` | 异常后 `_solve_default` |
| `pot-first` / `auto` | `_solve_with_pot` | PoT 失败后 `_solve_default` |

### 5.2 `auto` 模式判定顺序（`_resolve_strategy`）

按以下优先级依次命中：

1. 知识库直接命中 -> `kb-default`
2. 高阶极限特征 -> `pot`
3. 极限特征 -> `symbolic-limit`
4. 证明 + 多阶段 -> `mcts`
5. 多阶段 -> `ltm`
6. 抽象理论型 -> `step_back`
7. 约束密集型 -> `constraints`
8. 过程奖励需求 -> `prm`
9. 证明题 -> `tot`
10. 数值题 -> `pot-first`
11. 默认 -> `pot-first`

## 6. 核心策略实现速查

### 6.1 默认策略 `_solve_default`

执行顺序：

1. 本地算术表达式快速求值（`_extract_math_expression` + `_safe_eval`）。
2. 尝试符号极限（`_try_symbolic_limit`）。
3. 构建消息（系统提示 + few-shot + 知识上下文）。
4. 调用模型 `_chat_completion`。
5. 输出结构化 `_ensure_schema`。
6. 非 FAST 模式下做二次修正 `_refine_answer`。

### 6.2 PoT `_solve_with_pot`

1. 让模型只生成 Python 代码。
2. 代码安全校验：关键字拦截 + AST 白名单。
3. 子进程隔离执行（`python -I`，临时文件）。
4. 超时/输出上限控制。
5. 失败时自动让模型修复代码（重试次数由 `POT_RETRY` 控制）。

### 6.3 Self-Consistency `_self_consistency`

1. 并行采样多个结果。
2. 归一化答案后多数投票。
3. 返回投票明细。

### 6.4 ToT `_solve_with_tot`

1. 逐层生成分支（`_generate_branches`）。
2. 每分支打分（`_evaluate_branch`）。
3. Beam 保留高分路径。
4. 用最佳路径生成最终答案。

### 6.5 MCTS `_solve_with_mcts`

1. UCB 选择节点（`_mcts_select` + `_mcts_ucb`）。
2. 分支扩展。
3. rollout 快速求解并给置信分。
4. 反向传播评分。
5. 选最优子树做最终总结。

### 6.6 Debate `_solve_with_debate`

1. 求解者先出答案。
2. 批评者指出漏洞。
3. 循环修订若干轮。

## 7. 知识库系统与数据流

### 7.1 数据来源

- 首选 `theory.json` 构建知识点。
- 回退 `train.json` 生成知识点。

### 7.2 缓存机制

`_load_or_build_knowledge_points` 会检查：

1. 缓存文件是否存在（`knowledge_points.json`）。
2. 缓存元信息中的源文件与 mtime 是否匹配。

不匹配则自动重建并覆盖缓存。

### 7.3 检索与匹配

核心流程：

1. `_kb_lookup` 计算条目得分（名称、别名、关键词、公式、规则映射信号）。
2. `_direct_theory_matches` 从规则表直接命中目标理论。
3. `decompose_with_knowledge` 融合检索命中与强制命中。
4. `_build_kb_context` 生成模型可读的知识提示。

## 8. 关键参数与调优建议

### 8.1 全局生成参数

| 常量 | 默认值 | 说明 |
| --- | --- | --- |
| `MAX_TOKENS` | 4096 | 常规输出上限。 |
| `TEMPERATURE` | 0.15 | 全局温度。 |
| `TOP_P` | 0.3 | 全局 nucleus 采样。 |
| `RETRY_COUNT` | 1 | API 重试次数。 |

### 8.2 检索与知识参数

| 常量 | 默认值 | 说明 |
| --- | --- | --- |
| `FEWSHOT_TOP_K` | 3 | few-shot 样例数。 |
| `FEWSHOT_MAX_REASONING_CHARS` | 2000 | 示例推理截断长度。 |
| `KB_TOP_K` | 3 | 知识点默认返回数。 |
| `KB_MERGED_TOP_K` | 8 | 合并命中后上限。 |

### 8.3 PoT 参数

| 常量 | 默认值 | 说明 |
| --- | --- | --- |
| `POT_TIMEOUT` | 45 | 子进程超时秒数。 |
| `POT_MAX_TOKENS` | 8192 | 代码生成 token 上限。 |
| `POT_MAX_CODE_CHARS` | 120000 | 代码长度保护。 |
| `POT_ALLOWED_IMPORTS` | `math/sympy/mpmath/numpy` | 导入白名单。 |
| `POT_MAX_OUTPUT_CHARS` | 120000 | 输出长度保护。 |
| `POT_MAX_OUTPUT_LINES` | 4000 | 输出行数保护。 |

### 8.4 其他策略参数

| 策略 | 关键常量 |
| --- | --- |
| Self-Consistency | `SELF_CONSISTENCY_SAMPLES`、`SELF_CONSISTENCY_TEMP`、`SELF_CONSISTENCY_TOP_P` |
| ToT | `TOT_BRANCHING`、`TOT_DEPTH`、`TOT_BEAM_WIDTH` |
| Debate | `DEBATE_ROUNDS` |
| LTM | `LTM_MAX_STEPS`、`LTM_STEP_MAX_TOKENS` |
| PRM | `PRM_MAX_STEPS`、`PRM_GENERATE_MAX_TOKENS`、`PRM_VERIFY_MAX_TOKENS`、`PRM_MAX_ROUNDS` |
| Constraints | `CONSTRAINT_MAX_TOKENS`、`CONSTRAINT_CONTEXT_MAX_CHARS` |
| MCTS | `MCTS_SIMULATIONS`、`MCTS_MAX_BRANCH`、`MCTS_ROLLOUT_TEMP`、`MCTS_ROLLOUT_TOP_P`、`MCTS_UCB_C` |

### 8.5 调优经验（维护建议）

- 追求速度：开启 `AGENT_FAST_MODE=1`，并降低 `SELF_CONSISTENCY_SAMPLES`、`MCTS_SIMULATIONS`。
- 追求稳定：提高 `PRM_MAX_ROUNDS` 或 `TOT_DEPTH`，但注意延迟上升。
- API 超时频繁：降低 `MAX_TOKENS`，减少复杂策略链长度。
- 知识命中偏差：调整规则映射表和 `_kb_entry_score` 权重。

## 9. 维护改造任务模板

### 9.1 新增一种策略

步骤：

1. 在 `KimiCalculusAgent` 内新增 `_solve_with_xxx`。
2. 在 `solve()` 增加策略名分支及回退逻辑。
3. 在 `_resolve_strategy()` 增加 auto 触发条件（如需要）。
4. 新增常量参数并给默认值。
5. 在本 README 的策略表中补充文档。

### 9.2 扩展理论映射

1. 修改 `THEORY_DIRECT_MAP_RULES`。
2. 补充 `signals/targets/reminder`。
3. 用代表性题目验证 `_direct_theory_matches` 命中。

### 9.3 更新知识库

1. 更新 `ourwork/data/theory.json`。
2. 删除或保留 `knowledge_points.json` 均可；运行时会按 mtime 自动重建。
3. 抽样检查 `_build_kb_context` 输出是否合理。

## 10. 常见故障与排查

### 10.1 API 调用失败

现象：返回“调用接口出现错误”。

排查：

1. `KIMI_API_KEY` 是否有效。
2. 网络与代理是否可访问 `base_url`。
3. `model` 是否可用。

### 10.2 PoT 经常失败

可能原因：

1. 生成代码触发安全拦截。
2. 运行超时或输出超限。
3. 代码修复轮数不够。

处理建议：

1. 适当增大 `POT_TIMEOUT`。
2. 优化提示词，减少不必要打印。
3. 必要时回退到 `default/tot`。

### 10.3 策略选错

处理建议：

1. 调用时显式设置 `AGENT_STRATEGY`。
2. 调整 `_resolve_strategy` 的规则顺序。
3. 对难题可使用 `evaluate_and_solve` 的元信息评估路径。


```