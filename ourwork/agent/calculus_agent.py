"""Kimi:K2.5 based calculus solving agent."""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import math
import operator
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
    )
except Exception:  # noqa: BLE001
    sp = None
    parse_expr = None
    standard_transformations = ()
    implicit_multiplication_application = None

logger = logging.getLogger(__name__)

FAST_MODE = False
DEFAULT_STRATEGY = "auto"
SECOND_PASS_SCHEMA_FIX = True

DEFAULT_MODEL = os.getenv("MODEL_NAME", "deepseek-chat")
DEFAULT_BASE_URL = os.getenv("API_BASE_URL", "https://api.deepseek.com/v1/chat/completions")
DEFAULT_TIMEOUT = 60

MAX_TOKENS = 4096
TEMPERATURE = 0.2
TOP_P = 0.4
RETRY_COUNT = 2

FEWSHOT_TOP_K = 3
FEWSHOT_MAX_REASONING_CHARS = 320

THEORY_FILE_NAME = "theory.json"
KB_FILE_NAME = "knowledge_points.json"
KB_TOP_K = 4
KB_MERGED_TOP_K = 8


def _log_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


SYSTEM_PROMPT = (
    "# 身份与使命\n"
    "你是一位世界级的数学分析专家，拥有数十年微积分教学与研究经验。"
    "你的唯一目标是：对每一道微积分问题，给出绝对正确、严谨、完整的解答。宁慢勿错。\n\n"
    "# 核心工作流程（必须严格遵守，不可跳过任何一步）\n\n"
    "## 第一步：问题诊断（内部思考，用【】标出）\n"
    "【问题类型】识别题目属于：极限/导数/不定积分/定积分/反常积分/数值级数/函数项级数/幂级数/含参变量积分/微分方程/证明题\n"
    "【核心难点】指出本题最关键的技术难点\n"
    "【易错陷阱】列出本题最容易出错的2-3个地方\n"
    "【收敛性检查】如果涉及积分或级数，先判断敛散性\n\n"
    "## 第二步：解题方案设计\n"
    "列出至少两种可能的解法路径：\n"
    "- 方法A：[描述] - 可行性：[高/中/低]，风险：[说明]\n"
    "- 方法B：[描述] - 可行性：[高/中/低]，风险：[说明]\n"
    "选定最优方案：【方案X】，理由：[说明]\n\n"
    "## 第三步：严谨推导（零跳步要求）\n"
    "逐行写出推导过程，每行编号。每个等号必须注明依据：\n"
    "依据类型包括：\n"
    "- [代换]：写出完整的 u=..., du=..., 积分限变换过程\n"
    "- [分部积分]：明确写出 u=..., dv=..., du=..., v=...，再代入公式 ∫udv = uv - ∫vdu\n"
    "- [定理]：如牛顿-莱布尼茨公式、积分中值定理、控制收敛定理、Weierstrass M-判别法\n"
    "- [已知结果]：如 ∑1/n²=π²/6、∫₀^∞ sinx/x dx=π/2 等，必须明确写出\n"
    "- [代数恒等]：因式分解、部分分式、三角恒等式\n"
    "- [极限运算]：洛必达法则需验证条件（0/0或∞/∞型）；夹逼定理需给出上下界\n"
    "- [不等式]：注明不等号方向，等号成立条件\n\n"
    "## 第四步：交叉验证\n"
    "- [特殊值检验]：代入边界值或特殊点验证\n"
    "- [量纲/符号检查]：结果的正负号、量级是否合理\n"
    "- [数值验证]：给出解析结果的数值近似（保留6位小数），与估算值对比\n"
    "- [极限情况]：检查参数趋于边界时结果是否退化到已知简单情况\n\n"
    "## 第五步：最终答案\n"
    "用清晰格式给出：\n"
    "- 解析表达式：（最简形式）\n"
    "- 数值近似：（保留6位小数）\n"
    "- 收敛域/约束条件：（如有点明）\n\n"
    "# 关键定理与常见结果速查（解题时优先调用）\n"
    "- ∑_{n=1}^∞ 1/n² = π²/6\n"
    "- ∑_{n=1}^∞ 1/n⁴ = π⁴/90\n"
    "- ∑_{n=1}^∞ (-1)^{n+1}/n = ln2\n"
    "- ∫₀^∞ e^{-x²} dx = √π/2\n"
    "- ∫₀^∞ sinx/x dx = π/2\n"
    "- Γ(1/2)=√π, Γ(n+1)=n!\n"
    "- B(p,q)=Γ(p)Γ(q)/Γ(p+q)\n"
    "- ψ(z)=Γ'(z)/Γ(z)，ψ(1)=-γ，其中γ≈0.577216为欧拉常数\n\n"
    "# 必须避免的常见错误（每次推导后自查）\n"
    "1. 变量代换时忘记更新积分上下限\n"
    "2. 分部积分时符号搞反：∫udv=uv-∫vdu，不是uv+∫vdu\n"
    "3. 幂级数逐项积分/求导时忽略收敛域端点的变化\n"
    "4. 反常积分没有检查所有瑕点（包括区间内部可去奇点）\n"
    "5. 把条件收敛当绝对收敛使用（重排、交换次序需要绝对收敛）\n"
    "6. 含参变量积分求导时没有验证Leibniz法则的条件（被积函数和偏导数需一致连续或有控制函数）\n"
    "7. 用等价无穷小代换时忽略了加减法中不能随意代换的限制\n"
    "8. 写成 ∫f(x)dx 时遗漏常数项C或定积分代入时计算错误\n"
    "9. 级数求和时索引偏移处理错误，尤其是从n=0和n=1开始的转换\n"
    "10. 分母为零、取对数时定义域等隐藏约束未检查\n\n"
    "# 特殊情况的强制检查清单\n"
    "- 遇到对称区间[-a,a]：检查被积函数奇偶性\n"
    "- 遇到含参变量积分：检查可否求导，检查参变量范围\n"
    "- 遇到无穷级数：先做比值/根值判别，再求和\n"
    "- 遇到幂级数：单独检查收敛区间端点\n"
    "- 遇到瑕积分：逐个找出所有瑕点，分段处理\n"
    "- 遇到极限与积分/求和交换顺序：必须有控制收敛定理或一致收敛定理支撑\n"
    "- 遇到绝对值/取整/符号函数：分段讨论\n\n"
    "# 输出格式要求\n"
    "不得引入题目未给出的条件；若需假设，明确写出并说明依据。"
    "答案必须给出最终结论，表达清晰一致。"
    "输出必须是 JSON，对象包含 reasoning_process 与 answer 两个字段。"
)

POT_RETRY = 2
POT_TIMEOUT = 15
POT_MAX_TOKENS = 8192
POT_MAX_CODE_CHARS = 120000
POT_ALLOWED_IMPORTS = {"math", "sympy", "mpmath", "numpy"}
POT_MAX_OUTPUT_CHARS = 120000
POT_MAX_OUTPUT_LINES = 4000
SELF_CONSISTENCY_SAMPLES = 1 if FAST_MODE else 3
SELF_CONSISTENCY_TEMP = 0.5
SELF_CONSISTENCY_TOP_P = 0.8
TOT_BRANCHING = 4
TOT_DEPTH = 3
TOT_BEAM_WIDTH = 4
DEBATE_ROUNDS = 3
LTM_MAX_STEPS = 6
LTM_STEP_MAX_TOKENS = 420
STEP_BACK_PRINCIPLE_MAX_TOKENS = 420
STEP_BACK_CONTEXT_MAX_CHARS = 1400
PRM_MAX_STEPS = 10
PRM_GENERATE_MAX_TOKENS = 1200
PRM_VERIFY_MAX_TOKENS = 200
PRM_MAX_ROUNDS = 2
CONSTRAINT_MAX_TOKENS = 360
CONSTRAINT_CONTEXT_MAX_CHARS = 1400
MCTS_SIMULATIONS = 18
MCTS_ROLLOUT_TEMP = 0.35
MCTS_ROLLOUT_TOP_P = 0.7
MCTS_MAX_BRANCH = 3
MCTS_UCB_C = 1.4

THEORY_DIRECT_MAP_RULES: List[Dict[str, Any]] = [
    {
        "signals": ["lim", "limit", "极限", "趋于", r"\to", "→"],
        "targets": [
            "带Peano余项的Taylor公式",
            "带Lagrange余项的Taylor公式",
            "二重积分的变量代换公式",
            "极坐标变换公式",
            "三重积分的累次积分法",
            "柱坐标变换",
            "球坐标变换",
        ],
        "reminder": "重积分题先选坐标系和积分次序，再做Jacobi变换。",
    },
    {
        "signals": ["曲线积分", "line integral", "曲面积分", "surface integral", "通量", "做功"],
        "targets": ["第一类曲线积分的计算公式", "第一类曲面积分的计算公式", "第二类曲线积分的定义"],
        "reminder": "曲线/曲面积分按参数化写出微元后代入积分公式。",
    },
    {
        "signals": ["面积", "surface area", "质心", "惯量", "引力"],
        "targets": ["曲面面积公式（显式形式）", "曲面面积公式（参数形式）", "质心公式", "转动惯量公式", "引力公式"],
        "reminder": "应用题优先匹配几何/物理量定义积分式。",
    },
]


@dataclass
class _MCTSNode:
    path: List[str]
    parent: Optional["_MCTSNode"] = None
    children: List["_MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    action: Optional[str] = None


class KimiCalculusAgent:
    @_log_call
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("KIMI_API_KEY") or ""
        self.model = model or DEFAULT_MODEL
        self.base_url = base_url or DEFAULT_BASE_URL
        self.timeout = int(timeout or DEFAULT_TIMEOUT)

        self._few_shot_examples = self._load_examples()
        (
            self._idf_map,
            self._tfidf_vectors,
            self._doc_tokens,
            self._doc_lengths,
            self._avg_doc_len,
        ) = self._build_tfidf_index(self._few_shot_examples)
        self._kb_entries = self._load_or_build_knowledge_points()
        self._kb_name_index = self._build_kb_name_index(self._kb_entries)

    @_log_call
    def solve(self, item: Dict[str, Any] | str, strategy: Optional[str] = None) -> Dict[str, str]:
        if isinstance(item, dict):
            question_id = str(item.get("question_id", "")).strip()
            question = str(item.get("question", "")).strip()
            difficulty = str(item.get("difficulty", "")).strip()
        else:
            question_id = ""
            question = str(item).strip()
            difficulty = ""

        if not question:
            result = {"reasoning_process": "题目为空，无法作答。", "answer": ""}
        else:
            result = self.evaluate_and_solve(
                question,
                problem_id=question_id or None,
                difficulty=difficulty or None,
                strategy=strategy,
            )
            result = self._ensure_reasoning_detail(question, result)
            result = self._ensure_no_unstated_assumptions(question, result)
            result = self._ensure_proof_rigor(question, result)
            result = self._ensure_operation_consistency(question, result)
            result = self._ensure_answer_validity(question, result)
            result = self._verify_solution(question, result)

        return {
            "question_id": question_id,
            "reasoning_process": str(result.get("reasoning_process", "")).strip(),
            "answer": str(result.get("answer", "")).strip(),
        }

    @_log_call
    def evaluate_and_solve(
        self,
        question: str,
        problem_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        question = str(question).strip()
        if not question:
            return {"reasoning_process": "题目为空，无法作答。", "answer": ""}

        if strategy:
            mode = self._resolve_strategy(question, strategy)
            reason = f"指定策略: {strategy}"
        elif difficulty:
            mode, reason = self._pick_strategy_with_metadata(question, difficulty)
        else:
            mode = self._resolve_strategy(question, None)
            reason = "规则匹配"

        if mode == "symbolic-limit":
            limit_result = self._try_symbolic_limit(question)
            if limit_result is not None:
                return limit_result
            mode = "pot-first"

        if mode in {"kb", "knowledge", "knowledge-first", "kb-default"}:
            result = self._solve_default(question)
        elif mode == "pot":
            pot_result = self._solve_with_pot(question)
            result = pot_result or self._solve_default(question)
        elif mode in {"step_back", "step-back", "stepback"}:
            try:
                result = self._solve_with_step_back(question)
            except Exception:
                logger.warning("Step-Back failed, falling back to default")
                result = self._solve_default(question)
        elif mode in {"prm", "process_reward", "process-reward"}:
            try:
                result = self._solve_with_prm(question)
            except Exception:
                logger.warning("PRM simulation failed, falling back to default")
                result = self._solve_default(question)
        elif mode in {"constraints", "constraint", "system2", "system-2", "s2"}:
            try:
                result = self._solve_with_constraints(question)
            except Exception:
                logger.warning("Constraint extraction failed, falling back to default")
                result = self._solve_default(question)
        elif mode in {"ltm", "least_to_most", "least-to-most"}:
            result = self._solve_with_ltm(question)
        elif mode == "self_consistency":
            result = self._self_consistency(question)
        elif mode == "mcts":
            try:
                result = self._solve_with_mcts(question)
            except Exception:
                logger.warning("MCTS failed, falling back to Tree-of-Thought")
                result = self._solve_with_tot(question)
        elif mode == "tot":
            try:
                result = self._solve_with_tot(question)
            except Exception:
                logger.warning("ToT failed, falling back to self-consistency")
                result = self._self_consistency(question)
        elif mode == "debate":
            try:
                result = self._solve_with_debate(question)
            except Exception:
                logger.warning("Debate failed, falling back to default")
                result = self._solve_default(question)
        elif mode in {"pot-first", "auto"}:
            pot_result = self._solve_with_pot(question)
            result = pot_result or self._solve_default(question)
        else:
            result = self._solve_default(question)

        if problem_id or difficulty:
            prefix_parts = [
                f"题号: {problem_id or '未提供'}",
                f"难度: {difficulty or '未提供'}",
                f"选择策略: {mode}",
                f"选择依据: {reason}",
            ]
            prefix = " | ".join(prefix_parts) + "\n"
            result["reasoning_process"] = prefix + result.get("reasoning_process", "")
            result["problem_id"] = problem_id or ""
            result["chosen_strategy"] = mode

        return result

    @_log_call
    def _pick_strategy_with_metadata(
        self, question: str, difficulty: Optional[str]
    ) -> Tuple[str, str]:
        """Score strategies using题面与难度信号，输出(策略, 理由)。"""
        difficulty_level = (difficulty or "").lower()
        looks_constraint = self._looks_constraint_heavy(question)
        looks_multi = self._looks_multi_stage(question)
        looks_prm = self._looks_prm_needed(question)
        looks_proof = self._looks_like_proof(question)
        looks_numeric = self._looks_numeric(question)
        looks_abstract = self._looks_abstract_needed(question)

        score: Dict[str, int] = {
            "constraints": 0,
            "prm": 0,
            "ltm": 0,
            "step_back": 0,
            "tot": 0,
            "pot-first": 0,
            "self_consistency": 0,
            "mcts": 0,
        }

        if difficulty_level:
            logger.info("[%s] if-branch@L326", "_pick_strategy_with_metadata")
            if any(tag in difficulty_level for tag in ["hard", "困难", "挑战", "竞赛", "proof"]):
                logger.info("[%s] if-branch@L327", "_pick_strategy_with_metadata")
                score["prm"] += 3
                score["tot"] += 2
                score["constraints"] += 2

        if looks_constraint:
            score["constraints"] += 5
        if looks_multi:
            score["ltm"] += 4
        if looks_prm or looks_proof:
            score["prm"] += 3
        if looks_numeric:
            score["pot-first"] += 4
        if looks_abstract:
            score["step_back"] += 3

        # Fallback: ensure at least one strategy has a positive score
        if all(v <= 0 for v in score.values()):
            score["pot-first"] = 1

        best = max(score, key=score.get)
        reasons: Dict[str, str] = {
            "constraints": "题面含明显约束条件，适合约束驱动推导",
            "prm": "题面需要多步严格验证，适合过程奖励模型",
            "ltm": "题面可拆分为多步子任务",
            "step_back": "适合先抽象原理再求解",
            "tot": "难度高，适合树状搜索全局最优路径",
            "pot-first": "优先使用程序化计算验证",
            "self_consistency": "适合多样化采样投票",
            "mcts": "非常适合全局蒙特卡洛树搜索",
        }
        reason = reasons.get(best, "自动规则匹配")
        return best, reason

    @_log_call
    def evaluate_batch(
        self,
        items: List[Dict[str, Any]],
        difficulty: Optional[str] = None,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """批量评估：返回每题结果与综合分（兼顾准确率与平均耗时）。"""
        results: List[Dict[str, Any]] = []
        correct = 0
        total_time = 0.0

        for item in items:
            q = item.get("question", "")
            pid = item.get("problem_id") or item.get("id") or ""
            gold = item.get("gold_answer") or item.get("answer") or ""

            start = time.monotonic()
            solved = self.evaluate_and_solve(q, problem_id=pid, difficulty=difficulty)
            elapsed = time.monotonic() - start
            total_time += elapsed

            pred_norm = self._normalize_fraction(solved.get("answer", ""))
            gold_norm = self._normalize_fraction(str(gold)) if gold else ""
            is_correct = bool(gold_norm) and pred_norm == gold_norm
            correct += int(is_correct)

            record: Dict[str, Any] = {
                "problem_id": pid,
                "reasoning_process": solved.get("reasoning_process", ""),
                "answer": solved.get("answer", ""),
                "elapsed_sec": round(elapsed, 3),
                "correct": is_correct,
            }
            if include_metadata:
                logger.info("[%s] if-branch@L431", "evaluate_batch")
                record["chosen_strategy"] = solved.get("chosen_strategy", "")
            results.append(record)

        n = len(items) or 1
        accuracy = correct / n
        avg_time = total_time / n
        time_score = 1.0 / (1.0 + avg_time)
        combined_score = 0.8 * accuracy + 0.2 * time_score

        summary = {
            "count": len(items),
            "accuracy": accuracy,
            "avg_time_sec": avg_time,
            "combined_score": combined_score,
        }

        return {"results": results, "metrics": summary}

    @_log_call
    def _resolve_strategy(self, question: str, strategy: Optional[str]) -> str:
        mode = (strategy or DEFAULT_STRATEGY or "auto").lower()
        if mode != "auto":
            logger.info("[%s] if-branch@L453", "_resolve_strategy")
            return mode
        if self._kb_lookup(question):
            logger.info("[%s] if-branch@L455", "_resolve_strategy")
            return "kb-default"
        if self._looks_high_order_limit(question):
            logger.info("[%s] if-branch@L457", "_resolve_strategy")
            return "pot"
        if self._looks_like_limit_question(question):
            logger.info("[%s] if-branch@L459", "_resolve_strategy")
            return "symbolic-limit"
        if self._looks_like_proof(question) and self._looks_multi_stage(question):
            logger.info("[%s] if-branch@L461", "_resolve_strategy")
            return "mcts"
        if self._looks_multi_stage(question):
            logger.info("[%s] if-branch@L463", "_resolve_strategy")
            return "ltm"
        if self._looks_abstract_needed(question):
            logger.info("[%s] if-branch@L465", "_resolve_strategy")
            return "step_back"
        if self._looks_constraint_heavy(question):
            logger.info("[%s] if-branch@L467", "_resolve_strategy")
            return "constraints"
        if self._looks_prm_needed(question):
            logger.info("[%s] if-branch@L469", "_resolve_strategy")
            return "prm"
        if self._looks_like_proof(question):
            logger.info("[%s] if-branch@L471", "_resolve_strategy")
            return "tot"
        if self._looks_numeric(question):
            logger.info("[%s] if-branch@L473", "_resolve_strategy")
            return "pot-first"
        # 默认优先尝试 PoT，再退到自洽
        return "pot-first"

    @staticmethod
    @_log_call
    def _looks_like_proof(text: str) -> bool:
        lowered = text.lower()
        keywords = ["证明", "推导", "充分必要", "show that", "prove", "why", "理由", "证明题", "证明其"]
        return any(k in lowered for k in keywords)

    @staticmethod
    @_log_call
    def _looks_numeric(text: str) -> bool:
        if re.search(r"[\d\+\-\*/=]", text):
            logger.info("[%s] if-branch@L488", "_looks_numeric")
            return True
        keywords = ["计算", "求值", "极限", "积分", "导数", "曲线", "面积", "体积", "微分", "求"]
        return any(k in text for k in keywords)

    @staticmethod
    @_log_call
    def _looks_like_limit_question(text: str) -> bool:
        lowered = text.lower()
        signals = ["lim", "limit", "极限", "趋于", "→", "->", "\\to"]
        return any(sig in text for sig in signals) or any(sig in lowered for sig in signals)

    @staticmethod
    @_log_call
    def _looks_high_order_limit(text: str) -> bool:
        lowered = text.lower()
        limit_signals = ["lim", "limit", "极限", "→", "->", "趋于", "\to"]
        series_signals = ["泰勒", "taylor", "展开", "高阶", "级数", "展开式", "x^", "x**", "小o", "大o"]
        if any(sig in text for sig in limit_signals) or any(sig in lowered for sig in limit_signals):
            logger.info("[%s] if-branch@L506", "_looks_high_order_limit")
            if any(sig in text for sig in series_signals) or any(sig in lowered for sig in series_signals):
                logger.info("[%s] if-branch@L507", "_looks_high_order_limit")
                return True
        return False

    @staticmethod
    @_log_call
    def _looks_multi_stage(text: str) -> bool:
        connectors = ["然后", "之后", "接着", "最后", "并且", "同时", "分别", "再求", "再计算", "再求出", "再证明", "按顺序", "步骤"]
        if any(k in text for k in connectors):
            logger.info("[%s] if-branch@L515", "_looks_multi_stage")
            return True
        if text.count("?") + text.count("？") >= 2:
            logger.info("[%s] if-branch@L517", "_looks_multi_stage")
            return True
        return len(text) > 220

    @staticmethod
    @_log_call
    def _looks_abstract_needed(text: str) -> bool:
        keywords = ["定理", "判别", "展开", "敛散", "级数", "波动", "惯量", "渐近", "逼近", "泰勒", "taylor", "convergen", "theorem"]
        lowered = text.lower()
        return any(k in text for k in keywords) or any(k in lowered for k in keywords)

    @staticmethod
    @_log_call
    def _looks_constraint_heavy(text: str) -> bool:
        if len(text) < 140:
            logger.info("[%s] if-branch@L531", "_looks_constraint_heavy")
            return False
        keywords = ["约束", "限制", "边界", "条件", "区间", "上界", "下界", "必须", "满足", "约束条件", "boundary", "constraint", "inequality", "范围"]
        lowered = text.lower()
        return any(k in text for k in keywords) or any(k in lowered for k in keywords)

    @staticmethod
    @_log_call
    def _looks_prm_needed(text: str) -> bool:
        if len(text) < 180:
            logger.info("[%s] if-branch@L540", "_looks_prm_needed")
            return False
        keywords = ["证明", "推导", "步骤", "变形", "严谨", "长", "长证明", "step"]
        lowered = text.lower()
        return any(k in text for k in keywords) or any(k in lowered for k in keywords)

    @_log_call
    def _solve_default(self, question: str) -> Dict[str, str]:
        expr = self._extract_math_expression(question)
        if expr:
            logger.info("[%s] if-branch@L549", "_solve_default")
            try:
                value = self._safe_eval(expr)
                return {
                    "reasoning_process": f"本地识别到可计算表达式 {expr}，使用安全求值得到结果。",
                    "answer": str(value),
                }
            except Exception:
                logger.warning("[%s] except-branch@L556", "_solve_default", exc_info=True)
                pass

        if not re.search(r"证明|prove", question, re.IGNORECASE):
            limit_result = self._try_symbolic_limit(question)
            if limit_result is not None:
                logger.info("[%s] if-branch@L560", "_solve_default")
                return limit_result

        # 可选：跳过 KB 直接走模型，避免题型误匹配导致的无答案

        messages = self._build_messages(question)
        response_text = self._chat_completion(messages)
        result = self._ensure_schema(response_text)
        if FAST_MODE:
            logger.info("[%s] if-branch@L568", "_solve_default")
            return result
        refined = self._refine_answer(question, result)
        return self._ensure_reasoning_detail(question, refined)

    @_log_call
    def _build_messages(self, question: str) -> List[Dict[str, str]]:
        few_shot_context = self._build_few_shot_context(question)
        kb_context = self._build_kb_context(question)
        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if few_shot_context:
            logger.info("[%s] if-branch@L578", "_build_messages")
            messages.append({"role": "system", "content": few_shot_context})
        if kb_context:
            logger.info("[%s] if-branch@L580", "_build_messages")
            messages.append({"role": "system", "content": kb_context})
        messages.append({"role": "user", "content": self._format_question(question)})
        return messages

    @staticmethod
    @_log_call
    def _format_question(question: str) -> str:
        return (
            "请严格输出 JSON，对象包含 reasoning_process 与 answer 两个字符串字段。"
            "reasoning_process 必须包含分步骤推导（至少 3 步），不得只给评价或一句话。"
            "不得引入题目未给出的条件（如单调、凸性、有界等）。"
            "若为计算题，请加入简短数值/量级自检。"
            "不要添加 Markdown 代码块或额外文本，只输出 JSON。题目如下：\n" + question.strip()
        )

    @_log_call
    def _build_few_shot_context(self, question: str) -> str:
        if not self._few_shot_examples:
            logger.info("[%s] if-branch@L595", "_build_few_shot_context")
            return ""

        scored = [
            (self._example_score(question, ex, idx), ex)
            for idx, ex in enumerate(self._few_shot_examples)
        ]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        top_examples = [ex for _, ex in scored[:FEWSHOT_TOP_K]]
        if not top_examples:
            logger.info("[%s] if-branch@L605", "_build_few_shot_context")
            return ""

        chunks: List[str] = ["以下是相似题的示例，请学习表达与格式，但不要照搬答案，仅用于风格参考。"]
        for idx, ex in enumerate(top_examples, 1):
            question_text = ex["content"]["question_text"].strip()
            reasoning = self._truncate(str(ex["solution"]["reasoning_process"]), FEWSHOT_MAX_REASONING_CHARS)
            answer = str(ex["solution"].get("final_answer") or ex["solution"].get("latex_answer") or "")
            chunks.append(
                f"示例{idx}: 问题：{question_text}\n"
                f"参考输出：{{\"reasoning_process\": \"{reasoning}\", \"answer\": \"{answer}\"}}"
            )

        return "\n".join(chunks)

    @_log_call
    def _example_score(self, question: str, example: Dict[str, Any], index: int) -> float:
        q_text = example.get("content", {}).get("question_text", "")
        primary = example.get("classification", {}).get("primary_type", "").lower()
        secondary = example.get("classification", {}).get("secondary_type", "").lower()
        question_lower = question.lower()
        base_score = float(self._similarity_score(question, q_text))
        tfidf_score = self._tfidf_similarity(question, index)
        bm25_score = self._bm25_similarity(question, index)
        char_overlap = len(set(self._char_ngrams(question, 2)) & set(self._char_ngrams(q_text, 2)))
        score = base_score + tfidf_score * 5.0 + bm25_score * 1.6 + char_overlap * 0.2
        if primary and primary in question_lower:
            logger.info("[%s] if-branch@L631", "_example_score")
            score *= 1.2
        if secondary and secondary in question_lower:
            logger.info("[%s] if-branch@L633", "_example_score")
            score *= 1.1
        return score

    @_log_call
    def _load_examples(self) -> List[Dict[str, Any]]:
        try:
            data_path = Path(__file__).resolve().parent.parent / "data" / "train.json"
            if not data_path.exists():
                logger.info("[%s] if-branch@L641", "_load_examples")
                return []
            with data_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("questions", [])
        except Exception:
            logger.warning("[%s] except-branch@L646", "_load_examples", exc_info=True)
            return []

    @_log_call
    def _load_or_build_knowledge_points(self) -> List[Dict[str, Any]]:
        data_dir = Path(__file__).resolve().parent.parent / "data"
        theory_path = data_dir / THEORY_FILE_NAME
        source_path = theory_path if theory_path.exists() else data_dir / "train.json"
        kb_path = data_dir / KB_FILE_NAME
        source_mtime = source_path.stat().st_mtime if source_path.exists() else 0.0

        if kb_path.exists():
            logger.info("[%s] if-branch@L657", "_load_or_build_knowledge_points")
            try:
                with kb_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, list):
                    logger.info("[%s] if-branch@L661", "_load_or_build_knowledge_points")
                    points = payload
                else:
                    logger.info("[%s] else-branch@L663", "_load_or_build_knowledge_points")
                    points = payload.get("points", [])
                meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
                if (
                    isinstance(points, list)
                    and points
                    and meta.get("source") == source_path.name
                    and float(meta.get("source_mtime", -1)) == source_mtime
                ):
                    return points
            except Exception:
                logger.warning("Failed to load cached knowledge points, will rebuild.")

        if source_path.name == THEORY_FILE_NAME:
            logger.info("[%s] if-branch@L676", "_load_or_build_knowledge_points")
            try:
                with source_path.open("r", encoding="utf-8") as f:
                    theory_payload = json.load(f)
                points = self._build_knowledge_points_from_theory(theory_payload)
                payload = {
                    "meta": {
                        "source": source_path.name,
                        "source_mtime": source_mtime,
                        "updated_at": time.time(),
                        "count": len(points),
                        "schema": "knowledge-point-v1",
                    },
                    "points": points,
                }
                try:
                    with kb_path.open("w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                except Exception:
                    logger.warning("Failed to persist knowledge points cache.", exc_info=True)
                return points
            except Exception:
                logger.warning("Failed to build knowledge points from theory.json, falling back to train.json.", exc_info=True)

        points_map: Dict[str, Dict[str, Any]] = {}
        for ex in self._few_shot_examples:
            classification = ex.get("classification", {})
            kps = classification.get("knowledge_points") or []
            if not isinstance(kps, list):
                logger.info("[%s] if-branch@L704", "_load_or_build_knowledge_points")
                continue

            content = ex.get("content", {})
            solution = ex.get("solution", {})
            question_text = str(content.get("question_text", "")).strip()
            primary = str(classification.get("primary_type", "")).strip()
            secondary = str(classification.get("secondary_type", "")).strip()

            formula_pool = []
            latex_question = str(content.get("latex_question", "")).strip()
            if latex_question:
                logger.info("[%s] if-branch@L715", "_load_or_build_knowledge_points")
                formula_pool.append(latex_question)
            latex_answer = str(solution.get("latex_answer", "")).strip()
            if latex_answer:
                logger.info("[%s] if-branch@L718", "_load_or_build_knowledge_points")
                formula_pool.append(latex_answer)
            final_answer = str(solution.get("final_answer", "")).strip()
            if final_answer:
                logger.info("[%s] if-branch@L721", "_load_or_build_knowledge_points")
                formula_pool.append(final_answer)

            steps = solution.get("step_by_step") or []
            if isinstance(steps, list):
                logger.info("[%s] if-branch@L725", "_load_or_build_knowledge_points")
                for step in steps:
                    formula_pool.extend(self._extract_formula_snippets(str(step)))

            reasoning = str(solution.get("reasoning_process", "")).strip()
            formula_pool.extend(self._extract_formula_snippets(reasoning))

            for kp_raw in kps:
                kp = str(kp_raw).strip()
                if not kp:
                    logger.info("[%s] if-branch@L734", "_load_or_build_knowledge_points")
                    continue
                key = self._normalize_lookup_text(kp)
                entry = points_map.setdefault(
                    key,
                    {
                        "name": kp,
                        "aliases": [],
                        "keywords": [],
                        "formulas": [],
                        "principles": [],
                        "related_types": [],
                        "examples": [],
                    },
                )

                if primary and primary not in entry["related_types"]:
                    logger.info("[%s] if-branch@L750", "_load_or_build_knowledge_points")
                    entry["related_types"].append(primary)
                if secondary and secondary not in entry["related_types"]:
                    logger.info("[%s] if-branch@L752", "_load_or_build_knowledge_points")
                    entry["related_types"].append(secondary)

                if question_text and len(entry["examples"]) < 4 and question_text not in entry["examples"]:
                    logger.info("[%s] if-branch@L755", "_load_or_build_knowledge_points")
                    entry["examples"].append(question_text)

                for formula in formula_pool:
                    candidate = formula.strip()
                    if candidate and candidate not in entry["formulas"]:
                        logger.info("[%s] if-branch@L760", "_load_or_build_knowledge_points")
                        entry["formulas"].append(candidate)

                if reasoning and reasoning not in entry["principles"] and len(entry["principles"]) < 3:
                    logger.info("[%s] if-branch@L763", "_load_or_build_knowledge_points")
                    entry["principles"].append(self._truncate(reasoning, 240))

                for kw in [primary, secondary]:
                    kw = kw.strip()
                    if kw and kw not in entry["keywords"]:
                        logger.info("[%s] if-branch@L768", "_load_or_build_knowledge_points")
                        entry["keywords"].append(kw)

        points = sorted(points_map.values(), key=lambda x: x.get("name", ""))

        payload = {
            "meta": {
                "source": "train.json",
                "source_mtime": source_mtime,
                "updated_at": time.time(),
                "count": len(points),
                "schema": "knowledge-point-v1",
            },
            "points": points,
        }
        try:
            with kb_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            logger.warning("Failed to persist knowledge points cache.", exc_info=True)

        return points

    @_log_call
    def _build_knowledge_points_from_theory(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        points_map: Dict[str, Dict[str, Any]] = {}
        knowledge_base = payload.get("knowledge_base", [])
        if not isinstance(knowledge_base, list):
            logger.info("[%s] if-branch@L795", "_build_knowledge_points_from_theory")
            return []

        for section_entry in knowledge_base:
            if not isinstance(section_entry, dict):
                logger.info("[%s] if-branch@L799", "_build_knowledge_points_from_theory")
                continue

            section = str(section_entry.get("section", "")).strip()
            topics = section_entry.get("topics", [])
            if not isinstance(topics, list):
                logger.info("[%s] if-branch@L804", "_build_knowledge_points_from_theory")
                continue

            for topic in topics:
                if not isinstance(topic, dict):
                    logger.info("[%s] if-branch@L808", "_build_knowledge_points_from_theory")
                    continue

                name = str(topic.get("name", "")).strip()
                content = str(topic.get("content", "")).strip()
                proof = str(topic.get("proof", "")).strip()
                if not name:
                    logger.info("[%s] if-branch@L814", "_build_knowledge_points_from_theory")
                    continue

                key = self._normalize_lookup_text(f"{section} {name}")
                entry = points_map.setdefault(
                    key,
                    {
                        "name": name,
                        "aliases": [],
                        "keywords": [],
                        "formulas": [],
                        "principles": [],
                        "related_types": [],
                        "examples": [],
                        "section": section,
                    },
                )

                if section and section not in entry["related_types"]:
                    logger.info("[%s] if-branch@L832", "_build_knowledge_points_from_theory")
                    entry["related_types"].append(section)
                if section and section not in entry["aliases"]:
                    logger.info("[%s] if-branch@L834", "_build_knowledge_points_from_theory")
                    entry["aliases"].append(section)

                for text in (name, content, proof, section):
                    for formula in self._extract_formula_snippets(text):
                        if formula not in entry["formulas"]:
                            logger.info("[%s] if-branch@L839", "_build_knowledge_points_from_theory")
                            entry["formulas"].append(formula)

                if content and content not in entry["principles"] and len(entry["principles"]) < 3:
                    logger.info("[%s] if-branch@L842", "_build_knowledge_points_from_theory")
                    entry["principles"].append(self._truncate(content, 260))
                if proof and proof not in entry["principles"] and len(entry["principles"]) < 5:
                    logger.info("[%s] if-branch@L844", "_build_knowledge_points_from_theory")
                    entry["principles"].append(self._truncate(proof, 260))

                for keyword in self._extract_theory_keywords(section, name, content, proof):
                    if keyword not in entry["keywords"]:
                        logger.info("[%s] if-branch@L848", "_build_knowledge_points_from_theory")
                        entry["keywords"].append(keyword)

        return sorted(points_map.values(), key=lambda x: x.get("name", ""))

    @staticmethod
    @_log_call
    def _extract_formula_snippets(text: str) -> List[str]:
        if not text:
            logger.info("[%s] if-branch@L856", "_extract_formula_snippets")
            return []
        cleaned = re.sub(r"\[cite:\s*\d+\]", "", text)
        patterns = [
            r"[A-Za-z][A-Za-z0-9_]*\s*=\s*[^，。；;\n]{1,80}",
            r"∫[^，。；;\n]{1,80}",
            r"lim[^，。；;\n]{1,80}",
            r"\\int_[^\s]{1,30}",
            r"\\lim_[^\s]{1,30}",
        ]
        seen: Dict[str, None] = {}
        for pattern in patterns:
            for match in re.findall(pattern, cleaned, flags=re.IGNORECASE):
                candidate = str(match).strip()
                if candidate:
                    logger.info("[%s] if-branch@L870", "_extract_formula_snippets")
                    seen[candidate] = None
        return list(seen.keys())

    @staticmethod
    @_log_call
    def _extract_theory_keywords(*texts: str) -> List[str]:
        keywords: List[str] = []
        patterns = [
            r"Taylor公式|泰勒公式|Peano余项|Lagrange余项|全微分|偏导数|方向导数|梯度|隐函数定理|反函数定理|极值|鞍点|Hesse矩阵|含参积分|重积分|曲线积分|曲面积分|变量代换|极坐标|柱坐标|球坐标",
            r"[A-Za-z]+(?:\s*[A-Za-z]+)*公式",
        ]
        merged = " \n".join(texts)
        for pattern in patterns:
            for match in re.findall(pattern, merged, flags=re.IGNORECASE):
                candidate = str(match).strip()
                if candidate and candidate not in keywords:
                    logger.info("[%s] if-branch@L886", "_extract_theory_keywords")
                    keywords.append(candidate)

        return keywords

    @staticmethod
    @_log_call
    def _build_kb_name_index(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            name = str(entry.get("name", "")).strip()
            if name:
                logger.info("[%s] if-branch@L897", "_build_kb_name_index")
                index[KimiCalculusAgent._normalize_lookup_text(name)] = entry
            for alias in entry.get("aliases") or []:
                alias_text = str(alias).strip()
                if alias_text:
                    logger.info("[%s] if-branch@L901", "_build_kb_name_index")
                    index[KimiCalculusAgent._normalize_lookup_text(alias_text)] = entry
        return index

    @staticmethod
    @_log_call
    def _entry_to_kb_hit(entry: Dict[str, Any], score: float) -> Dict[str, Any]:
        return {
            "score": float(score),
            "name": str(entry.get("name", "")),
            "formulas": list(entry.get("formulas", []))[:6],
            "principles": list(entry.get("principles", []))[:3],
            "related_types": list(entry.get("related_types", []))[:3],
        }

    @staticmethod
    @_log_call
    def _merge_kb_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for hit in hits:
            name = str(hit.get("name", "")).strip()
            if not name:
                logger.info("[%s] if-branch@L922", "_merge_kb_hits")
                continue
            current = merged.get(name)
            if current is None:
                logger.info("[%s] if-branch@L925", "_merge_kb_hits")
                merged[name] = {
                    "score": float(hit.get("score", 0.0)),
                    "name": name,
                    "formulas": list(hit.get("formulas") or []),
                    "principles": list(hit.get("principles") or []),
                    "related_types": list(hit.get("related_types") or []),
                }
                continue

            current["score"] = max(float(current.get("score", 0.0)), float(hit.get("score", 0.0)))
            for key in ("formulas", "principles", "related_types"):
                existing = current.get(key) or []
                for item in hit.get(key) or []:
                    if item not in existing:
                        logger.info("[%s] if-branch@L939", "_merge_kb_hits")
                        existing.append(item)
                current[key] = existing

        ordered = sorted(merged.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return ordered

    @_log_call
    def _build_tfidf_index(
        self, examples: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], List[Dict[str, float]], List[List[str]], List[int], float]:
        if not examples:
            logger.info("[%s] if-branch@L950", "_build_tfidf_index")
            return {}, [], [], [], 0.0

        df: Counter[str] = Counter()
        doc_tokens: List[List[str]] = []
        doc_lengths: List[int] = []
        for ex in examples:
            q_text = ex.get("content", {}).get("question_text", "")
            tokens = self._tokenize(q_text)
            doc_tokens.append(tokens)
            doc_lengths.append(len(tokens))
            for token in set(tokens):
                df[token] += 1

        n_docs = max(len(examples), 1)
        avg_len = sum(doc_lengths) / n_docs if n_docs else 0.0
        idf_map: Dict[str, float] = {token: math.log((1 + n_docs) / (1 + freq)) + 1.0 for token, freq in df.items()}
        vectors: List[Dict[str, float]] = []
        for tokens in doc_tokens:
            if not tokens:
                logger.info("[%s] if-branch@L969", "_build_tfidf_index")
                vectors.append({})
                continue
            tf = Counter(tokens)
            length = len(tokens)
            vec = {token: (tf[token] / length) * idf_map.get(token, 0.0) for token in tf}
            vectors.append(vec)

        return idf_map, vectors, doc_tokens, doc_lengths, avg_len

    @staticmethod
    @_log_call
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            logger.info("[%s] if-branch@L982", "_truncate")
            return text
        return text[:limit].rstrip() + "..."

    @staticmethod
    @_log_call
    def _tokenize(text: str) -> List[str]:
        normalized = re.sub(r"[^\w]+", " ", text.lower())
        return [t for t in normalized.split() if t]

    @staticmethod
    @_log_call
    def _normalize_lookup_text(text: str) -> str:
        return re.sub(r"\s+", "", text).strip().lower()

    @_log_call
    def _build_query_tfidf(self, text: str) -> Dict[str, float]:
        tokens = self._tokenize(text)
        if not tokens or not self._idf_map:
            logger.info("[%s] if-branch@L1000", "_build_query_tfidf")
            return {}
        tf = Counter(tokens)
        length = len(tokens)
        return {token: (tf[token] / length) * self._idf_map.get(token, 0.0) for token in tf}

    @_log_call
    def _bm25_similarity(self, text: str, doc_index: int, k1: float = 1.5, b: float = 0.75) -> float:
        if not self._idf_map or doc_index >= len(self._doc_tokens):
            logger.info("[%s] if-branch@L1008", "_bm25_similarity")
            return 0.0
        query_tokens = self._tokenize(text)
        if not query_tokens:
            logger.info("[%s] if-branch@L1011", "_bm25_similarity")
            return 0.0
        doc_tokens = self._doc_tokens[doc_index]
        if not doc_tokens:
            logger.info("[%s] if-branch@L1014", "_bm25_similarity")
            return 0.0
        doc_tf = Counter(doc_tokens)
        doc_len = self._doc_lengths[doc_index] if self._doc_lengths else len(doc_tokens)
        avg_len = self._avg_doc_len or 1.0
        score = 0.0
        for token in query_tokens:
            idf = self._idf_map.get(token, 0.0)
            tf = doc_tf.get(token, 0)
            if tf == 0:
                logger.info("[%s] if-branch@L1023", "_bm25_similarity")
                continue
            numer = tf * (k1 + 1)
            denom = tf + k1 * (1 - b + b * doc_len / avg_len)
            score += idf * numer / denom
        return score

    @_log_call
    def _tfidf_similarity(self, text: str, doc_index: int) -> float:
        if not self._idf_map or doc_index >= len(self._tfidf_vectors):
            logger.info("[%s] if-branch@L1032", "_tfidf_similarity")
            return 0.0
        query_vec = self._build_query_tfidf(text)
        doc_vec = self._tfidf_vectors[doc_index]
        if not query_vec or not doc_vec:
            logger.info("[%s] if-branch@L1036", "_tfidf_similarity")
            return 0.0
        return sum(weight * doc_vec.get(token, 0.0) for token, weight in query_vec.items())

    @_log_call
    def _similarity_score(self, a: str, b: str) -> int:
        tokens_a = set(self._tokenize(a))
        tokens_b = set(self._tokenize(b))
        if not tokens_a or not tokens_b:
            logger.info("[%s] if-branch@L1044", "_similarity_score")
            return 0
        return len(tokens_a & tokens_b)

    @staticmethod
    @_log_call
    def _char_ngrams(text: str, n: int = 2) -> List[str]:
        clean = re.sub(r"\s+", "", text)
        return [clean[i : i + n] for i in range(max(len(clean) - n + 1, 0))]

    @_log_call
    def _ensure_schema(self, raw_text: str) -> Dict[str, str]:
        parsed = self._extract_json(raw_text)
        if parsed is None and SECOND_PASS_SCHEMA_FIX:
            logger.info("[%s] if-branch@L1057", "_ensure_schema")
            parsed = self._schema_fix(raw_text)
        if parsed is None:
            logger.info("[%s] if-branch@L1059", "_ensure_schema")
            return {"reasoning_process": raw_text.strip(), "answer": ""}

        reasoning = str(parsed.get("reasoning_process", "")).strip()
        answer = str(parsed.get("answer", "")).strip()

        if not reasoning:
            logger.info("[%s] if-branch@L1065", "_ensure_schema")
            reasoning = raw_text.strip()

        return {"reasoning_process": reasoning, "answer": answer}

    @staticmethod
    @_log_call
    def _should_skip_symbolic_limit(question: str, expr_text: str) -> bool:
        combined = f"{question} {expr_text}"
        if re.search(r"[∑Σ∫∏]", combined):
            return True
        if re.search(r"\b(sum|sigma|prod|product|series|integral|int)\b", combined, re.IGNORECASE):
            return True
        if re.search(r"级数|收敛域|无穷级数", question):
            return True
        func_matches = re.findall(r"([a-zA-Z][a-zA-Z0-9_]*)\s*\(", expr_text)
        if func_matches:
            allowed = {"sin", "cos", "tan", "log", "ln", "exp", "sqrt"}
            for name in func_matches:
                if name not in allowed:
                    return True
        return False

    @_log_call
    def _try_symbolic_limit(self, question: str) -> Optional[Dict[str, str]]:
        if not sp:
            logger.info("[%s] if-branch@L1074", "_try_symbolic_limit")
            return None
        if re.search(r"证明|prove", question, re.IGNORECASE):
            logger.info("[%s] if-branch@L1076", "_try_symbolic_limit")
            return None

        normalized = question.replace("\\to", "->").replace("\u2192", "->")
        match = re.search(
            r"(?:\\lim|lim)\s*(?:_\{(?P<var1>[a-zA-Z])\s*->\s*(?P<point1>[^}]*)\}"
            r"|_\((?P<var2>[a-zA-Z])\s*->\s*(?P<point2>[^)]*)\)"
            r"|_(?P<var3>[a-zA-Z])\s*->\s*(?P<point3>[^\s]+)"
            r"|(?P<var4>[a-zA-Z])\s*->\s*(?P<point4>[^\s]+))?",
            normalized,
            re.IGNORECASE,
        )
        if not match:
            logger.info("[%s] if-branch@L1082", "_try_symbolic_limit")
            return None

        var_name = next(
            (match.group(name) for name in ("var1", "var2", "var3", "var4") if match.group(name)),
            "x",
        )
        point_text = next(
            (match.group(name) for name in ("point1", "point2", "point3", "point4") if match.group(name)),
            "0",
        )
        point_text = re.sub(r"\^\s*[-+]", "", point_text)
        point_text = re.sub(r"[-+]\s*$", "", point_text)
        tail = normalized[match.end():].strip()
        if not tail:
            logger.info("[%s] if-branch@L1088", "_try_symbolic_limit")
            return None

        tail = tail.lstrip("：:，, ")
        expr_text = self._strip_leading_delimiters(tail)
        expr_text = re.sub(r"^_\s*(\{[^}]*\}|\([^)]*\)|[^\s]+)\s*", "", expr_text)
        expr_text = re.split(r"[，,。;；]|\b其中\b|\bwhere\b|=|\?|？", expr_text, maxsplit=1)[0].strip()
        if re.search(r"\.{2,}|…|⋯|\\cdots", expr_text):
            logger.info("[%s] if-branch@L1094", "_try_symbolic_limit")
            return None
        if self._should_skip_symbolic_limit(question, expr_text):
            logger.info("[%s] if-branch@L1094", "_try_symbolic_limit")
            return None
        expr_text = self._tex_to_sympy_expr(expr_text)
        if not expr_text:
            logger.info("[%s] if-branch@L1094", "_try_symbolic_limit")
            return None

        try:
            symbol = sp.Symbol(var_name)
            locals_map = {
                "E": sp.E,
                "e": sp.E,
                "pi": sp.pi,
                "oo": sp.oo,
                "ln": sp.log,
                "sin": sp.sin,
                "cos": sp.cos,
                "tan": sp.tan,
                "log": sp.log,
                "exp": sp.exp,
                "sqrt": sp.sqrt,
                var_name: symbol,
                "x": sp.Symbol("x"),
            }
            point_expr = self._tex_to_sympy_expr(point_text)
            if parse_expr:
                transformations = standard_transformations + (implicit_multiplication_application,)
                point = parse_expr(point_expr, local_dict=locals_map, transformations=transformations)
                expr = parse_expr(expr_text, local_dict=locals_map, transformations=transformations)
            else:
                point = sp.sympify(point_expr, locals=locals_map)
                expr = sp.sympify(expr_text, locals=locals_map)
            if not isinstance(expr, sp.Basic) or not isinstance(point, sp.Basic):
                logger.info("[%s] if-branch@L1101", "_try_symbolic_limit")
                return None
            if isinstance(expr, (list, tuple)):
                logger.info("[%s] if-branch@L1102", "_try_symbolic_limit")
                return None
            if expr.has(sp.Sum, sp.Integral, sp.Product):
                logger.info("[%s] if-branch@L1103", "_try_symbolic_limit")
                return None
            value = sp.limit(expr, symbol, point)
        except Exception:
            logger.warning("[%s] except-branch@L1103", "_try_symbolic_limit", exc_info=True)
            return None

        if value is None:
            logger.info("[%s] if-branch@L1106", "_try_symbolic_limit")
            return None

        if getattr(value, "is_number", False):
            logger.info("[%s] if-branch@L1109", "_try_symbolic_limit")
            answer = str(sp.simplify(value))
        else:
            logger.info("[%s] else-branch@L1111", "_try_symbolic_limit")
            answer = str(value)

        reasoning = (
            f"本地使用 SymPy 解析极限：变量 {var_name} 趋于 {point_text}，"
            f"表达式 {expr_text}，直接求得结果。"
        )
        return {"reasoning_process": reasoning, "answer": answer}

    @staticmethod
    @_log_call
    def _strip_leading_delimiters(text: str) -> str:
        stripped = text.strip()
        pairs = {"(": ")", "[": "]", "{": "}"}
        while stripped and stripped[0] in pairs:
            left = stripped[0]
            right = pairs[left]
            depth = 0
            match_index = -1
            for idx, ch in enumerate(stripped):
                if ch == left:
                    depth += 1
                elif ch == right:
                    depth -= 1
                    if depth == 0:
                        match_index = idx
                        break
            if match_index == len(stripped) - 1:
                stripped = stripped[1:-1].strip()
                continue
            break
        return stripped

    @_log_call
    def _tex_to_sympy_expr(self, text: str) -> str:
        expr = text.strip()
        if not expr:
            logger.info("[%s] if-branch@L1131", "_tex_to_sympy_expr")
            return expr

        expr = expr.replace("\\left", "").replace("\\right", "")
        expr = expr.replace("\\cdot", "*").replace("\\times", "*")
        expr = expr.replace("\\,", "")
        expr = expr.replace("^", "**")
        expr = expr.replace("\\pi", "pi")
        expr = expr.replace("\\infty", "oo")
        expr = expr.replace("\\sin", "sin").replace("\\cos", "cos").replace("\\tan", "tan")
        expr = expr.replace("\\log", "log").replace("\\ln", "log").replace("\\exp", "exp")
        expr = expr.replace("\\sqrt", "sqrt")
        expr = self._replace_tex_fractions(expr)
        expr = expr.replace("{", "(").replace("}", ")")
        expr = expr.replace("\\", "")
        return expr

    @_log_call
    def _replace_tex_fractions(self, text: str) -> str:
        result = text
        while "\\frac" in result:
            start = result.find("\\frac")
            prefix = result[:start]
            remainder = result[start + 5 :]
            numerator, after_numerator = self._extract_braced_group(remainder)
            if numerator is None:
                logger.info("[%s] if-branch@L1156", "_replace_tex_fractions")
                break
            denominator, after_denominator = self._extract_braced_group(after_numerator)
            if denominator is None:
                logger.info("[%s] if-branch@L1159", "_replace_tex_fractions")
                break
            replacement = f"(({self._tex_to_sympy_expr(numerator)})/({self._tex_to_sympy_expr(denominator)}))"
            result = prefix + replacement + after_denominator
        return result

    @staticmethod
    @_log_call
    def _extract_braced_group(text: str) -> Tuple[Optional[str], str]:
        stripped = text.lstrip()
        offset = len(text) - len(stripped)
        if not stripped.startswith("{"):
            logger.info("[%s] if-branch@L1170", "_extract_braced_group")
            return None, text

        depth = 0
        start_idx = -1
        for idx, char in enumerate(stripped):
            if char == "{":
                logger.info("[%s] if-branch@L1176", "_extract_braced_group")
                depth += 1
                if depth == 1:
                    logger.info("[%s] if-branch@L1178", "_extract_braced_group")
                    start_idx = idx + 1
            elif char == "}":
                logger.info("[%s] if-branch@L1180", "_extract_braced_group")
                depth -= 1
                if depth == 0 and start_idx >= 0:
                    logger.info("[%s] if-branch@L1182", "_extract_braced_group")
                    content = stripped[start_idx:idx]
                    remainder = stripped[idx + 1 :]
                    return content, remainder

        return None, text

    @staticmethod
    def _repair_truncated_json(raw: str) -> Optional[str]:
        """Repair JSON truncated by max_tokens: close strings, braces, brackets.

        Walks character-by-character to track string/escape state, then
        appends closing quotes/braces/brackets as needed.
        """
        if not raw:
            return None
        s = raw.strip()
        if not s.startswith("{"):
            return None

        # Track brace/bracket depth and string state
        brace_depth = 0
        bracket_depth = 0
        in_string = False
        escape_next = False
        for ch in s:
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
            elif ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth -= 1

        # Close unterminated string
        if in_string:
            s = s + '"'

        # Close unclosed braces & brackets
        s = s + "}" * max(brace_depth, 0)
        s = s + "]" * max(bracket_depth, 0)

        try:
            json.loads(s)
            return s
        except json.JSONDecodeError:
            pass

        # Fallback: truncate the last broken key-value pair and retry
        last_comma = -1
        for m in re.finditer(r',\s*"', s):
            last_comma = m.start()
        if last_comma > 0:
            truncated = s[:last_comma].rstrip().rstrip(",")
            # Re-count depths on truncated version
            bd = 0
            brd = 0
            in_s = False
            esc = False
            for ch in truncated:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_s = not in_s
                    continue
                if in_s:
                    continue
                if ch == "{":
                    bd += 1
                elif ch == "}":
                    bd -= 1
                elif ch == "[":
                    brd += 1
                elif ch == "]":
                    brd -= 1
            if in_s:
                truncated = truncated + '"'
            truncated = truncated + "}" * max(bd, 0) + "]" * max(brd, 0)
            try:
                json.loads(truncated)
                return truncated
            except json.JSONDecodeError:
                pass

        return None

    @_log_call
    def _extract_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        text = raw_text.strip()
        text = self._strip_code_fences(text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # --- stack-based extraction: find outermost balanced { … } ---
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        end = -1
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end == -1:
            # No closing brace → truncated
            candidate = text[start:]
        else:
            candidate = text[start : end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # --- attempt auto-repair ---
        repaired = self._repair_truncated_json(candidate)
        if repaired is not None:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

        return None

    @_log_call
    def _schema_fix(self, raw_text: str) -> Optional[Dict[str, Any]]:
        messages = [
            {"role": "system", "content": "你是输出修复器，只返回 JSON 对象，包含 reasoning_process 与 answer 两个字符串字段，不要添加任何其他文本或代码块。"},
            {"role": "user", "content": f"请将以下模型输出修复为 JSON：{raw_text}"},
        ]
        fixed = self._chat_completion(messages)
        try:
            return json.loads(self._strip_code_fences(fixed))
        except Exception:
            logger.warning("[%s] except-branch@L1216", "_schema_fix", exc_info=True)
            return None

    @_log_call
    def _refine_answer(self, question: str, draft: Dict[str, str]) -> Dict[str, str]:
        if not draft:
            logger.info("[%s] if-branch@L1221", "_refine_answer")
            return draft
        messages = [
            {
                "role": "system",
                "content": (
                    "你是答案校对器，请严格输出 JSON，字段 reasoning_process 与 answer。"
                    "reasoning_process 必须保留完整推导步骤（至少 3 步），不要缩写或省略。"
                    "严禁只给评价或一句话结论，不要添加代码块。"
                ),
            },
            {
                "role": "user",
                "content": "题目：" + question.strip() + "\n" + "初稿：" + json.dumps(draft, ensure_ascii=False),
            },
        ]
        text = self._chat_completion(messages, max_tokens=1600)
        parsed = self._extract_json(text)
        if parsed is None:
            logger.info("[%s] if-branch@L1232", "_refine_answer")
            return draft
        reasoning = str(parsed.get("reasoning_process", "")).strip() or draft.get("reasoning_process", "")
        answer = str(parsed.get("answer", "")).strip() or draft.get("answer", "")
        if self._needs_reasoning_expansion(reasoning):
            return {"reasoning_process": draft.get("reasoning_process", reasoning), "answer": answer}
        return {"reasoning_process": reasoning, "answer": answer}

    @staticmethod
    @_log_call
    def _needs_reasoning_expansion(reasoning: str) -> bool:
        cleaned = (reasoning or "").strip()
        if not cleaned:
            return True
        if len(cleaned) < 80:
            return True
        eval_phrases = [
            "初稿",
            "答案无误",
            "结果准确",
            "步骤完整",
            "正确",
            "无误",
            "略",
            "省略",
        ]
        if any(phrase in cleaned for phrase in eval_phrases):
            return True
        return False

    @_log_call
    def _ensure_reasoning_detail(self, question: str, result: Dict[str, str]) -> Dict[str, str]:
        reasoning = str(result.get("reasoning_process", "")).strip()
        if not self._needs_reasoning_expansion(reasoning):
            return result
        answer = str(result.get("answer", "")).strip()
        messages = [
            {
                "role": "system",
                "content": (
                    "你是解题过程补全器，只输出 JSON，对象包含 reasoning_process 与 answer 两个字符串字段。"
                    "reasoning_process 必须分步骤（至少 3 步），写清关键公式、代入、化简与结论。"
                    "不要评价，不要复述题目，不要添加代码块。"
                ),
            },
            {
                "role": "user",
                "content": f"题目：{question.strip()}\n已知最终答案：{answer}\n请补全完整推导过程。",
            },
        ]
        text = self._chat_completion(messages, max_tokens=2048, temperature=0.2, top_p=0.4)
        parsed = self._extract_json(text)
        if parsed is None:
            return result
        expanded_reasoning = str(parsed.get("reasoning_process", "")).strip()
        if not expanded_reasoning:
            return result
        return {"reasoning_process": expanded_reasoning, "answer": answer}

    @staticmethod
    @_log_call
    def _contains_unstated_monotonicity(question: str, reasoning: str) -> bool:
        q_text = (question or "").strip()
        r_text = (reasoning or "").strip()
        if not r_text:
            return False
        terms = [
            "单调",
            "严格单调",
            "单调性",
            "单调递增",
            "单调递减",
            "递增",
            "递减",
            "增函数",
            "减函数",
        ]
        if any(term in r_text for term in terms) and not any(term in q_text for term in terms):
            return True
        if re.search(r"f'\s*[>≥]\s*0", r_text) and not re.search(r"f'\s*[>≥]\s*0", q_text):
            return True
        if re.search(r"f'\s*[<≤]\s*0", r_text) and not re.search(r"f'\s*[<≤]\s*0", q_text):
            return True
        return False

    @_log_call
    def _ensure_no_unstated_assumptions(self, question: str, result: Dict[str, str]) -> Dict[str, str]:
        reasoning = str(result.get("reasoning_process", "")).strip()
        if not self._contains_unstated_monotonicity(question, reasoning):
            return result
        answer = str(result.get("answer", "")).strip()
        messages = [
            {
                "role": "system",
                "content": (
                    "你是推导修正器，只输出 JSON，对象包含 reasoning_process 与 answer 两个字符串字段。"
                    "不得引入题目未给出的条件，尤其禁止假设单调/凸性/有界。"
                    "请在不增加新假设的前提下改写推导，保持答案不变，步骤至少 3 步。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"题目：{question.strip()}\n"
                    f"当前推导：{reasoning}\n"
                    f"已知答案：{answer}\n"
                    "请改写推导，去除未给出的假设。"
                ),
            },
        ]
        text = self._chat_completion(messages, max_tokens=900, temperature=0.2, top_p=0.4)
        parsed = self._extract_json(text)
        if parsed is None:
            return result
        fixed_reasoning = str(parsed.get("reasoning_process", "")).strip()
        if not fixed_reasoning:
            return result
        return {"reasoning_process": fixed_reasoning, "answer": answer}

    @staticmethod
    @_log_call
    def _needs_proof_rigor(question: str, reasoning: str) -> bool:
        q_text = (question or "").strip()
        r_text = (reasoning or "").strip()
        if not q_text or not r_text:
            return False
        if not re.search(r"证明|prove", q_text, re.IGNORECASE):
            return False
        vague_markers = ["显然", "不难", "容易", "略", "省略", "可得", "可知", "可选取", "可以选取"]
        if any(marker in r_text for marker in vague_markers):
            return True
        if "存在" in r_text and "使得" in r_text:
            construction_markers = ["定义", "构造", "递归", "归纳", "下确界", "inf", "最小", "最小点"]
            if not any(marker in r_text for marker in construction_markers):
                return True
        return False

    @_log_call
    def _ensure_proof_rigor(self, question: str, result: Dict[str, str]) -> Dict[str, str]:
        reasoning = str(result.get("reasoning_process", "")).strip()
        if not self._needs_proof_rigor(question, reasoning):
            return result
        answer = str(result.get("answer", "")).strip()
        messages = [
            {
                "role": "system",
                "content": (
                    "你是证明严谨性修正器，只输出 JSON，对象包含 reasoning_process 与 answer 两个字符串字段。"
                    "请补全存在性/选点/排序等步骤中的严格论证，给出明确构造（如递归取点或下确界定义）。"
                    "避免使用‘显然/容易/略/可选取’等含糊表述，步骤至少 3 步。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"题目：{question.strip()}\n"
                    f"当前推导：{reasoning}\n"
                    f"已知答案：{answer}\n"
                    "请改写为严谨推导。"
                ),
            },
        ]
        text = self._chat_completion(messages, max_tokens=1100, temperature=0.2, top_p=0.4)
        parsed = self._extract_json(text)
        if parsed is None:
            return result
        fixed_reasoning = str(parsed.get("reasoning_process", "")).strip()
        if not fixed_reasoning:
            return result
        return {"reasoning_process": fixed_reasoning, "answer": answer}

    @staticmethod
    @_log_call
    def _needs_operation_consistency_fix(reasoning: str) -> bool:
        r_text = (reasoning or "").strip()
        if not r_text:
            return False
        if re.search(r"相[加减].{0,6}(消去|消元|抵消)", r_text):
            return True
        if "相减" in r_text and "消去" in r_text:
            return True
        if "相加" in r_text and "消去" in r_text:
            return True
        return False

    @_log_call
    def _ensure_operation_consistency(self, question: str, result: Dict[str, str]) -> Dict[str, str]:
        reasoning = str(result.get("reasoning_process", "")).strip()
        if not self._needs_operation_consistency_fix(reasoning):
            return result
        answer = str(result.get("answer", "")).strip()
        messages = [
            {
                "role": "system",
                "content": (
                    "你是表述一致性修正器，只输出 JSON，对象包含 reasoning_process 与 answer 两个字符串字段。"
                    "请检查相加/相减/线性组合等表述是否与后续公式一致，必要时改为正确操作或改成明确的线性组合表述。"
                    "不要改变公式结果与答案，步骤至少 3 步，不要添加代码块。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"题目：{question.strip()}\n"
                    f"当前推导：{reasoning}\n"
                    f"已知答案：{answer}\n"
                    "请修正表述不一致的问题。"
                ),
            },
        ]
        text = self._chat_completion(messages, max_tokens=900, temperature=0.2, top_p=0.4)
        parsed = self._extract_json(text)
        if parsed is None:
            return result
        fixed_reasoning = str(parsed.get("reasoning_process", "")).strip()
        if not fixed_reasoning:
            return result
        return {"reasoning_process": fixed_reasoning, "answer": answer}

    @staticmethod
    @_log_call
    def _needs_answer_repair(question: str, answer: str) -> bool:
        q_text = (question or "").strip()
        a_text = (answer or "").strip()
        if not a_text:
            return True
        if re.search(r"证明|prove", q_text, re.IGNORECASE):
            proof_placeholders = {"证毕", "证明完毕", "QED", "成立", "已证"}
            if a_text in proof_placeholders:
                return True
            if any(token in a_text for token in proof_placeholders):
                return True
            if len(a_text) < 6 and not re.search(r"=|lim|->|→|∞|\\", a_text):
                return True
        if any(token in a_text for token in ["lim", "->", "→"]):
            return True
        if any(token in a_text for token in ["。", "，", ",", "=", ";", "；"]) and len(a_text) < 6:
            return True
        if "∞" in a_text and a_text not in {"∞", "+∞", "-∞", "无穷大", "发散", "不存在"}:
            return True
        normalized = q_text.replace("\\to", "->").replace("\u2192", "->")
        match = re.search(r"(?:\\lim|lim)\s*_\{?\s*(?P<var>[a-zA-Z])\s*->", normalized)
        if match:
            var = match.group("var")
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(var)}(?![A-Za-z0-9_])", a_text):
                return True
            if re.search(rf"{re.escape(var)}\s*\\pi|{re.escape(var)}\s*π", a_text):
                return True
            if re.search(rf"{re.escape(var)}\s*\+|{re.escape(var)}\s*-", a_text):
                return True
        if re.search(r"a_n|a_n|a_\{n\}", a_text):
            return True
        return False

    @_log_call
    def _ensure_answer_validity(self, question: str, result: Dict[str, str]) -> Dict[str, str]:
        answer = str(result.get("answer", "")).strip()
        if not self._needs_answer_repair(question, answer):
            return result
        reasoning = str(result.get("reasoning_process", "")).strip()
        conclusion = self._extract_conclusion_from_reasoning(reasoning)
        if conclusion:
            return {"reasoning_process": reasoning, "answer": conclusion}
        cleaned = re.sub(r"(证毕|证明完毕|QED|已证|成立)[。．\.]*", "", answer).strip()
        if cleaned and cleaned != answer:
            return {"reasoning_process": reasoning, "answer": cleaned}
        messages = [
            {
                "role": "system",
                "content": (
                    "你是答案校验修正器，只输出 JSON，对象包含 reasoning_process 与 answer 两个字符串字段。"
                    "请检查答案是否仍含极限变量或不完整表达；若有，给出正确极限结果。"
                    "必要时同步修正推导，使结论与步骤一致，步骤至少 3 步。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"题目：{question.strip()}\n"
                    f"当前推导：{reasoning}\n"
                    f"当前答案：{answer}\n"
                    "请修正答案（必要时修正推导）。"
                ),
            },
        ]
        text = self._chat_completion(messages, max_tokens=1000, temperature=0.2, top_p=0.4)
        parsed = self._ensure_schema(text)
        fixed_reasoning = str(parsed.get("reasoning_process", "")).strip() or reasoning
        fixed_answer = str(parsed.get("answer", "")).strip() or answer
        return {"reasoning_process": fixed_reasoning, "answer": fixed_answer}

    @_log_call
    def _verify_solution(self, question: str, result: Dict[str, str]) -> Dict[str, str]:
        if FAST_MODE:
            return result
        if re.search(r"证明|prove", question, re.IGNORECASE):
            return result
        if not re.search(r"计算|求|极限|积分|级数|求和|求导", question):
            return result

        reasoning = str(result.get("reasoning_process", "")).strip()
        answer = str(result.get("answer", "")).strip()
        messages = [
            {
                "role": "system",
                "content": (
                    "你是解答校验器，请独立复算并核对答案。"
                    "如发现错误或不严谨，请给出修正后的 reasoning_process 与 answer。"
                    "输出必须是 JSON，只包含 reasoning_process 与 answer 两个字段。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"题目：{question.strip()}\n"
                    f"当前推导：{reasoning}\n"
                    f"当前答案：{answer}\n"
                    "请核对并修正（如有必要）。"
                ),
            },
        ]
        text = self._chat_completion(messages, max_tokens=1200, temperature=0.2, top_p=0.4)
        parsed = self._ensure_schema(text)
        fixed_reasoning = str(parsed.get("reasoning_process", "")).strip() or reasoning
        fixed_answer = str(parsed.get("answer", "")).strip() or answer
        return {"reasoning_process": fixed_reasoning, "answer": fixed_answer}

    @staticmethod
    @_log_call
    def _extract_conclusion_from_reasoning(reasoning: str) -> str:
        if not reasoning:
            return ""
        patterns = [
            r"(lim_\{[^}]+\}[^。\n]*=\s*[^。\n]+)",
            r"(lim_{[^}]+}[^。\n]*=\s*[^。\n]+)",
            r"(lim_{[^}]+}[^。\n]*)",
            r"极限为([^。\n]+)",
            r"结果为([^。\n]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, reasoning)
            if match:
                return match.group(1).strip()
        return ""

    @_log_call
    def _solve_with_ltm(self, question: str) -> Dict[str, str]:
        sub_questions = self._decompose_question(question)
        if not sub_questions:
            logger.info("[%s] if-branch@L1241", "_solve_with_ltm")
            return self._solve_default(question)

        context_lines: List[str] = []
        for idx, sub in enumerate(sub_questions[:LTM_MAX_STEPS], 1):
            background = self._truncate("\n".join(context_lines), 1200)
            messages = [
                {"role": "system", "content": "你是微积分子问题求解器，请简洁回答当前子问题，避免 JSON 和代码块。"},
                {"role": "user", "content": f"已知背景信息：{background or '无'}\n当前子问题：{sub}"},
            ]
            reply = self._chat_completion(messages, max_tokens=LTM_STEP_MAX_TOKENS, temperature=0.25, top_p=0.6)
            context_lines.append(f"子问题{idx}: {sub}\n解答{idx}: {reply.strip()}")

        stitched_context = "\n".join(context_lines)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": "请基于拆解步骤与中间解答整合最终 JSON，保持步骤引用，answer 给出最终结果。"},
            {"role": "user", "content": f"原始题目：{question.strip()}\n拆解与中间解：\n{stitched_context}"},
        ]
        final_text = self._chat_completion(messages, max_tokens=MAX_TOKENS, temperature=0.2, top_p=0.4)
        return self._ensure_schema(final_text)

    @_log_call
    def _solve_with_step_back(self, question: str) -> Dict[str, str]:
        abstraction_messages = [
            {
                "role": "system",
                "content": "这是一个具体的科学/工程问题，请不要直接解答。请指出解决该问题需要用到哪些核心原理、公式或定理，并给出其严谨定义。",
            },
            {"role": "user", "content": "问题：" + question.strip()},
        ]
        principles = self._chat_completion(
            abstraction_messages, max_tokens=STEP_BACK_PRINCIPLE_MAX_TOKENS, temperature=0.2, top_p=0.4
        ).strip()
        if not principles:
            logger.info("[%s] if-branch@L1275", "_solve_with_step_back")
            return self._solve_default(question)

        principles_ctx = self._truncate(principles, STEP_BACK_CONTEXT_MAX_CHARS)
        solve_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": "基于以下核心原理，逐步解答并输出 JSON：\n" + principles_ctx},
            {"role": "user", "content": self._format_question(question)},
        ]
        final_text = self._chat_completion(solve_messages, max_tokens=MAX_TOKENS, temperature=0.2, top_p=0.35)
        parsed = self._ensure_schema(final_text)
        prefix = "Step-Back 抽象原理：\n" + principles_ctx + "\n"
        parsed["reasoning_process"] = prefix + parsed.get("reasoning_process", "")
        return parsed

    @_log_call
    def _solve_with_prm(self, question: str) -> Dict[str, str]:
        steps = self._prm_generate_steps(question, [], restart_from=1, error_reason="")
        steps = self._prm_extract_steps(steps)[:PRM_MAX_STEPS]
        if not steps:
            logger.info("[%s] if-branch@L1294", "_solve_with_prm")
            return self._solve_default(question)

        verdicts: List[Tuple[bool, str]] = []
        for round_idx in range(PRM_MAX_ROUNDS):
            invalid_idx = None
            invalid_reason = ""
            verdicts = []
            for idx, step in enumerate(steps):
                valid, reason = self._prm_verify_step(question, steps[:idx], step, idx + 1)
                verdicts.append((valid, reason))
                if not valid:
                    logger.info("[%s] if-branch@L1305", "_solve_with_prm")
                    invalid_idx = idx
                    invalid_reason = reason
                    break

            if invalid_idx is None:
                logger.info("[%s] if-branch@L1310", "_solve_with_prm")
                return self._prm_finalize_answer(question, steps, verdicts)

            keep = steps[:invalid_idx]
            regenerated = self._prm_generate_steps(
                question, keep, restart_from=invalid_idx + 1, error_reason=invalid_reason
            )
            new_steps = self._prm_extract_steps(regenerated)
            if not new_steps:
                logger.info("[%s] if-branch@L1318", "_solve_with_prm")
                break
            steps = (keep + new_steps)[:PRM_MAX_STEPS]

        return self._prm_finalize_answer(question, steps, verdicts)

    @_log_call
    def _solve_with_constraints(self, question: str) -> Dict[str, str]:
        constraint_text = self._extract_constraints(question)
        constraint_ctx = self._truncate(constraint_text, CONSTRAINT_CONTEXT_MAX_CHARS) if constraint_text else ""

        few_shot_context = self._build_few_shot_context(question)
        kb_context = self._build_kb_context(question)
        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if constraint_ctx:
            logger.info("[%s] if-branch@L1332", "_solve_with_constraints")
            messages.append({"role": "system", "content": "在解答过程中，你必须严格遵守以下条件：" + constraint_ctx})
        if few_shot_context:
            logger.info("[%s] if-branch@L1334", "_solve_with_constraints")
            messages.append({"role": "system", "content": few_shot_context})
        if kb_context:
            logger.info("[%s] if-branch@L1336", "_solve_with_constraints")
            messages.append({"role": "system", "content": kb_context})
        messages.append({"role": "user", "content": self._format_question(question)})

        final_text = self._chat_completion(messages, max_tokens=MAX_TOKENS, temperature=0.18, top_p=0.35)
        parsed = self._ensure_schema(final_text)
        if constraint_ctx:
            logger.info("[%s] if-branch@L1342", "_solve_with_constraints")
            prefix = "System-2 约束清单：\n" + constraint_ctx + "\n"
            parsed["reasoning_process"] = prefix + parsed.get("reasoning_process", "")
        return parsed

    @_log_call
    def _decompose_question(self, question: str) -> List[str]:
        messages = [
            {
                "role": "system",
                "content": "将以下复杂问题拆解为需要依次解决的子问题列表。不要解答，只输出格式化列表：1. [子问题A] 2. [子问题B]...",
            },
            {"role": "user", "content": "问题：" + question.strip()},
        ]
        text = self._chat_completion(messages, max_tokens=360, temperature=0.2, top_p=0.4)
        steps = self._parse_numbered_list(text)
        return steps[:LTM_MAX_STEPS]

    @staticmethod
    @_log_call
    def _parse_numbered_list(text: str) -> List[str]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        items: List[str] = []
        for line in lines:
            match = re.match(r"^\s*\d+[).:\-]*\s*(.+)", line)
            if match:
                logger.info("[%s] if-branch@L1367", "_parse_numbered_list")
                candidate = match.group(1).strip().strip(";").strip("。")
                if candidate:
                    logger.info("[%s] if-branch@L1369", "_parse_numbered_list")
                    items.append(candidate)
        if not items and text.strip():
            logger.info("[%s] if-branch@L1371", "_parse_numbered_list")
            items.append(text.strip())
        return items

    @_log_call
    def _prm_generate_steps(
        self,
        question: str,
        confirmed_steps: List[str],
        restart_from: int,
        error_reason: str,
    ) -> str:
        confirmed_text = "\n".join(f"<step {i + 1}>{s}</step>" for i, s in enumerate(confirmed_steps))
        user_content = ["题目：" + question.strip()]
        if confirmed_steps:
            logger.info("[%s] if-branch@L1385", "_prm_generate_steps")
            user_content.append("已确认无误的前置步骤：")
            user_content.append(confirmed_text)
        if error_reason:
            logger.info("[%s] if-branch@L1388", "_prm_generate_steps")
            user_content.append(f"上一轮在第 {restart_from} 步出现问题：{error_reason}。请从该步重写后续推导。")
        user_content.append(f"请从第 {restart_from} 步开始继续推导，总步数不超过 {PRM_MAX_STEPS}。")
        user_block = "\n".join(user_content)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是步骤分解与推导助手。严格按 <step n>...</step> 输出每一步，编号递增，避免 JSON 和代码块。"
                    "每步保持简洁且自洽，不要直接给最终答案，先完成推导步骤。"
                ),
            },
            {"role": "user", "content": user_block},
        ]
        return self._chat_completion(
            messages, max_tokens=PRM_GENERATE_MAX_TOKENS, temperature=0.35, top_p=0.7
        )

    @_log_call
    def _prm_extract_steps(self, text: str) -> List[str]:
        pattern = re.compile(r"<step[^>]*>(.*?)</step>", re.DOTALL | re.IGNORECASE)
        steps = [m.group(1).strip() for m in pattern.finditer(text)]
        return [s for s in steps if s]

    @_log_call
    def _prm_verify_step(
        self, question: str, prior_steps: List[str], step: str, idx: int
    ) -> Tuple[bool, str]:
        prior_block = "\n".join(f"步骤{i + 1}: {s}" for i, s in enumerate(prior_steps)) or "(无)"
        messages = [
            {
                "role": "system",
                "content": "你是步骤验证器，只输出 'Valid' 或 "
                "'Invalid: <原因>'，检查逻辑严谨性与计算正确性。",
            },
            {
                "role": "user",
                "content": (
                    "题目：" + question.strip() + "\n"
                    "已有步骤：\n" + prior_block + "\n"
                    f"待检验步骤{idx}：{step}"
                ),
            },
        ]
        verdict = self._chat_completion(
            messages, max_tokens=PRM_VERIFY_MAX_TOKENS, temperature=0.0, top_p=0.1
        ).strip()
        if verdict.lower().startswith("valid"):
            logger.info("[%s] if-branch@L1435", "_prm_verify_step")
            return True, ""
        match = re.match(r"invalid\s*:?\s*(.*)", verdict, re.IGNORECASE)
        reason = match.group(1).strip() if match else verdict
        return False, reason or "该步未通过验证"

    @_log_call
    def _prm_finalize_answer(
        self, question: str, steps: List[str], verdicts: List[Tuple[bool, str]]
    ) -> Dict[str, str]:
        step_lines = []
        for i, step in enumerate(steps, 1):
            verdict_note = ""
            if verdicts and i <= len(verdicts) and verdicts[i - 1][0] is False:
                logger.info("[%s] if-branch@L1448", "_prm_finalize_answer")
                verdict_note = f" (验证提示: {verdicts[i - 1][1]})"
            step_lines.append(f"步骤{i}: {step}{verdict_note}")
        stitched = "\n".join(step_lines)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": "以下是经步骤验证的推导，请据此输出最终 JSON，确保结果与步骤一致：\n" + stitched,
            },
            {"role": "user", "content": self._format_question(question)},
        ]
        final_text = self._chat_completion(messages, max_tokens=MAX_TOKENS, temperature=0.18, top_p=0.35)
        parsed = self._ensure_schema(final_text)
        prefix = "Process Reward Model 模拟：步骤经验证。\n" + stitched + "\n"
        parsed["reasoning_process"] = prefix + parsed.get("reasoning_process", "")
        return parsed

    @_log_call
    def _extract_constraints(self, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "阅读题目，提取所有已知数值、边界条件和题目限制。仅输出 JSON，键名为 constraints。",
            },
            {"role": "user", "content": "题目：" + question.strip()},
        ]
        text = self._chat_completion(
            messages, max_tokens=CONSTRAINT_MAX_TOKENS, temperature=0.15, top_p=0.35
        )
        parsed = self._extract_json(text)
        if parsed is None:
            logger.info("[%s] if-branch@L1479", "_extract_constraints")
            return text.strip()
        constraints = parsed.get("constraints")
        if isinstance(constraints, list):
            logger.info("[%s] if-branch@L1482", "_extract_constraints")
            return "\n".join(str(item).strip() for item in constraints if str(item).strip())
        if constraints:
            logger.info("[%s] if-branch@L1484", "_extract_constraints")
            return str(constraints).strip()
        return text.strip()

    @_log_call
    def _solve_with_pot(self, question: str) -> Optional[Dict[str, str]]:
        messages = [
            {"role": "system", "content": "你是编写可执行 Python 代码的助手，只输出代码，不要解释。避免网络与文件写入，只使用标准库和 math。"},
            {
                "role": "user",
                "content": (
                    "你是一个解题专家。请仅输出解决以下问题的 Python 代码，不要输出其他文本。"
                    "务必将最终答案打印到 stdout。题目：" + question.strip()
                ),
            },
        ]
        code = self._extract_code_only(
            self._chat_completion(messages, max_tokens=POT_MAX_TOKENS, temperature=0.2, top_p=0.3)
        )
        if not code:
            logger.info("[%s] if-branch@L1503", "_solve_with_pot")
            return None

        if len(code) > POT_MAX_CODE_CHARS or not self._is_pot_code_safe(code):
            logger.warning("PoT code rejected for safety/size", extra={"length": len(code)})
            return None

        last_error = ""
        for attempt in range(POT_RETRY):
            stdout, stderr, returncode = self._run_python_code(code)
            if returncode == 0 and stdout.strip():
                logger.info("[%s] if-branch@L1513", "_solve_with_pot")
                return {
                    "reasoning_process": f"Program-of-Thought: 生成代码并在本地执行，第 {attempt + 1} 次尝试成功。",
                    "answer": stdout.strip(),
                }

            last_error = stderr.strip() or f"执行失败，返回码 {returncode}"
            fix_messages = [
                {"role": "system", "content": "修复以下 Python 代码以解决题目，只输出修正后的代码。"},
                {
                    "role": "user",
                    "content": "题目：" + question.strip() + "\n" + "当前代码：\n" + code + "\n" + "错误信息：" + last_error,
                },
            ]
            code = self._extract_code_only(
                self._chat_completion(fix_messages, max_tokens=POT_MAX_TOKENS, temperature=0.25, top_p=0.35)
            )
            if not code:
                logger.info("[%s] if-branch@L1530", "_solve_with_pot")
                break

        logger.warning("PoT 执行失败，将回退其他策略", extra={"error": last_error})
        return None

    @_log_call
    def _run_python_code(self, code: str) -> Tuple[str, str, int]:
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            result = subprocess.run(
                [sys.executable, "-I", tmp_path],
                capture_output=True,
                text=True,
                timeout=POT_TIMEOUT,
            )
            stdout, stderr = result.stdout, result.stderr
            if len(stdout) > POT_MAX_OUTPUT_CHARS or stdout.count("\n") > POT_MAX_OUTPUT_LINES:
                logger.info("[%s] if-branch@L1550", "_run_python_code")
                stderr = stderr + f"\n输出超限: {len(stdout)} chars, {stdout.count('\\n')} lines"
                return stdout[:POT_MAX_OUTPUT_CHARS], stderr.strip(), 1
            return stdout, stderr, result.returncode
        except subprocess.TimeoutExpired as exc:
            logger.warning("[%s] except-branch@L1554", "_run_python_code", exc_info=True)
            return "", f"执行超时: {exc}", 1
        except Exception as exc:
            logger.warning("[%s] except-branch@L1556", "_run_python_code", exc_info=True)
            return "", str(exc), 1
        finally:
            if tmp_path:
                logger.info("[%s] if-branch@L1559", "_run_python_code")
                try:
                    os.remove(tmp_path)
                except Exception:
                    logger.warning("[%s] except-branch@L1562", "_run_python_code", exc_info=True)
                    pass

    @_log_call
    def _extract_code_only(self, text: str) -> str:
        cleaned = self._strip_code_fences(text.strip())
        if cleaned.startswith("python"):
            logger.info("[%s] if-branch@L1568", "_extract_code_only")
            cleaned = cleaned[len("python"):].strip()
        return cleaned

    @staticmethod
    @_log_call
    def _is_pot_code_safe(code: str) -> bool:
        lower = code.lower()
        blocked_keywords = [
            "import os",
            "import sys",
            "import subprocess",
            "import socket",
            "import requests",
            "import shutil",
            "open(",
            "__import__",
            "eval(",
            "exec(",
            "globals()",
            "locals()",
        ]
        if any(bad in lower for bad in blocked_keywords):
            logger.info("[%s] if-branch@L1590", "_is_pot_code_safe")
            return False
        return KimiCalculusAgent._is_pot_code_safe_ast(code)

    @staticmethod
    @_log_call
    def _is_pot_code_safe_ast(code: str) -> bool:
        allowed_imports = POT_ALLOWED_IMPORTS
        blocked_calls = {"open", "exec", "eval", "__import__"}
        blocked_modules = {"os", "sys", "subprocess", "socket", "requests", "shutil"}
        allowed_builtins = {
            "abs",
            "max",
            "min",
            "sum",
            "len",
            "range",
            "enumerate",
            "float",
            "int",
            "print",
            "round",
            "pow",
            "map",
            "zip",
        }

        class Guard(ast.NodeVisitor):
            @_log_call
            def __init__(self) -> None:
                self.ok = True

            @_log_call
            def visit_Import(self, node: ast.Import) -> None:  # noqa: D401
                for alias in node.names:
                    if alias.name not in allowed_imports:
                        logger.info("[%s] if-branch@L1625", "visit_Import")
                        self.ok = False
                self.generic_visit(node)

            @_log_call
            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: D401
                if node.module is None or node.module not in allowed_imports:
                    logger.info("[%s] if-branch@L1631", "visit_ImportFrom")
                    self.ok = False
                self.generic_visit(node)

            @_log_call
            def visit_Call(self, node: ast.Call) -> None:  # noqa: D401
                if isinstance(node.func, ast.Name):
                    logger.info("[%s] if-branch@L1637", "visit_Call")
                    if node.func.id in blocked_calls:
                        logger.info("[%s] if-branch@L1638", "visit_Call")
                        self.ok = False
                if isinstance(node.func, ast.Attribute):
                    logger.info("[%s] if-branch@L1640", "visit_Call")
                    root = node.func.value
                    if isinstance(root, ast.Name) and root.id in blocked_modules:
                        logger.info("[%s] if-branch@L1642", "visit_Call")
                        self.ok = False
                self.generic_visit(node)

            @_log_call
            def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: D401
                if isinstance(node.value, ast.Name) and node.value.id in blocked_modules:
                    logger.info("[%s] if-branch@L1648", "visit_Attribute")
                    self.ok = False
                self.generic_visit(node)

        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError:
            logger.warning("[%s] except-branch@L1654", "_is_pot_code_safe_ast", exc_info=True)
            return False
        guard = Guard()
        guard.visit(tree)
        return guard.ok

    @_log_call
    def _self_consistency(self, question: str, samples: int = SELF_CONSISTENCY_SAMPLES) -> Dict[str, str]:
        async def _gather() -> List[Dict[str, str]]:
            loop = asyncio.get_running_loop()
            tasks = [loop.run_in_executor(None, self._single_consistency_sample, question) for _ in range(samples)]
            return await asyncio.gather(*tasks, return_exceptions=True)

        try:
            results = asyncio.run(_gather())
        except RuntimeError:
            logger.warning("[%s] except-branch@L1669", "_self_consistency", exc_info=True)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(_gather())
            loop.close()
            asyncio.set_event_loop(None)

        clean_results: List[Dict[str, str]] = []
        for res in results:
            if isinstance(res, Exception):
                logger.warning("Self-consistency sample failed", exc_info=res)
                continue
            clean_results.append(res)

        if not clean_results:
            logger.info("[%s] if-branch@L1683", "_self_consistency")
            return self._solve_default(question)

        answers = [self._normalize_fraction(item.get("answer", "")) for item in clean_results]
        counter = Counter(answers)
        if not counter:
            logger.info("[%s] if-branch@L1688", "_self_consistency")
            return self._solve_default(question)
        best_answer, freq = counter.most_common(1)[0]
        vote_lines = [f"样本{i + 1}: {clean_results[i].get('answer', '').strip()}" for i in range(len(clean_results))]
        reasoning = "Self-Consistency 多样化采样投票，最高票 {} 次。\n".format(freq) + "\n".join(vote_lines)
        return {"reasoning_process": reasoning, "answer": best_answer or clean_results[0].get("answer", "")}

    @_log_call
    def _single_consistency_sample(self, question: str) -> Dict[str, str]:
        messages = self._build_messages(question)
        text = self._chat_completion(
            messages,
            temperature=SELF_CONSISTENCY_TEMP,
            top_p=SELF_CONSISTENCY_TOP_P,
        )
        return self._ensure_schema(text)

    @_log_call
    def _solve_with_tot(self, question: str) -> Dict[str, str]:
        root = {"path": [], "score": 0.0}
        frontier = [root]
        for _ in range(TOT_DEPTH):
            new_frontier: List[Dict[str, Any]] = []
            for node in frontier:
                branches = self._generate_branches(question, node["path"], TOT_BRANCHING)
                for branch in branches:
                    score = node["score"] + self._evaluate_branch(question, node["path"], branch)
                    new_frontier.append({"path": node["path"] + [branch], "score": score})

            if not new_frontier:
                logger.info("[%s] if-branch@L1717", "_solve_with_tot")
                break
            frontier = sorted(new_frontier, key=lambda x: x["score"], reverse=True)[:TOT_BEAM_WIDTH]

        if not frontier:
            logger.info("[%s] if-branch@L1721", "_solve_with_tot")
            return self._solve_default(question)

        best = frontier[0]
        return self._finalize_tot_answer(question, best)

    @_log_call
    def _generate_branches(self, question: str, path: List[str], branching: int) -> List[str]:
        path_text = "\n".join(f"步骤{i + 1}: {step}" for i, step in enumerate(path)) or "(尚未展开)"
        messages = [
            {"role": "system", "content": "你是推理生成器，请提出多个可能的下一步推导，使用编号列出，每条简洁。不要给最终答案。"},
            {
                "role": "user",
                "content": (
                    "题目：" + question.strip() + "\n"
                    "已选路径：\n" + path_text + "\n"
                    f"请给出 {branching} 条下一步推导，每条一行，前缀用序号。"
                ),
            },
        ]
        text = self._chat_completion(messages, max_tokens=420, temperature=0.55, top_p=0.85)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        branches: List[str] = []
        for line in lines:
            clean = re.sub(r"^\s*\d+[\).:\-]*\s*", "", line)
            branches.append(clean)
            if len(branches) >= branching:
                logger.info("[%s] if-branch@L1747", "_generate_branches")
                break
        return branches

    @_log_call
    def _evaluate_branch(self, question: str, path: List[str], branch: str) -> float:
        messages = [
            {"role": "system", "content": "你是评估器，只返回 1-10 的整数评分，数字之外不要输出。"},
            {
                "role": "user",
                "content": (
                    "题目：" + question.strip() + "\n"
                    "已有路径：" + " | ".join(path) + "\n"
                    "候选下一步：" + branch + "\n"
                    "请给出该候选对解题帮助的评分（1-10）。"
                ),
            },
        ]
        text = self._chat_completion(messages, max_tokens=16, temperature=0.0, top_p=0.1)
        match = re.search(r"10|[1-9]", text)
        if not match:
            logger.info("[%s] if-branch@L1767", "_evaluate_branch")
            return 5.0
        return float(match.group(0))

    @_log_call
    def _finalize_tot_answer(self, question: str, node: Dict[str, Any]) -> Dict[str, str]:
        path_lines = "\n".join(f"步骤{i + 1}: {step}" for i, step in enumerate(node.get("path", [])))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "基于以下 Tree-of-Thought 推导路径，整理严谨推理并输出 JSON：\n"
                    + path_lines
                    + "\n题目：" + question.strip()
                ),
            },
        ]
        text = self._chat_completion(messages, max_tokens=MAX_TOKENS, temperature=0.2, top_p=0.4)
        parsed = self._ensure_schema(text)
        reasoning_prefix = "Tree-of-Thought 搜索路径得分 {:.1f}\n".format(node.get("score", 0.0))
        parsed["reasoning_process"] = reasoning_prefix + parsed.get("reasoning_process", "")
        return parsed

    @_log_call
    def _solve_with_mcts(self, question: str) -> Dict[str, str]:
        root = _MCTSNode(path=[])
        last_score = 0.0
        for _ in range(MCTS_SIMULATIONS):
            leaf = self._mcts_select(root)
            if leaf.visits == 0 and leaf.path:
                logger.info("[%s] if-branch@L1797", "_solve_with_mcts")
                rollout = self._mcts_rollout(question, leaf.path)
                last_score = rollout.get("score", 0.0)
                self._mcts_backpropagate(leaf, last_score)
                leaf.rollout_answer = rollout.get("answer", "")
                leaf.rollout_reasoning = rollout.get("reasoning", "")
                leaf.rollout_score = last_score
                continue

            branches = self._generate_branches(question, leaf.path, MCTS_MAX_BRANCH)
            if not branches:
                logger.info("[%s] if-branch@L1807", "_solve_with_mcts")
                rollout = self._mcts_rollout(question, leaf.path)
                last_score = rollout.get("score", 0.0)
                self._mcts_backpropagate(leaf, last_score)
                leaf.rollout_answer = rollout.get("answer", "")
                leaf.rollout_reasoning = rollout.get("reasoning", "")
                leaf.rollout_score = last_score
                continue

            if not leaf.children:
                logger.info("[%s] if-branch@L1816", "_solve_with_mcts")
                for step in branches:
                    child = _MCTSNode(path=leaf.path + [step], parent=leaf, action=step)
                    leaf.children.append(child)

            child = max(leaf.children, key=lambda n: -n.visits)
            rollout = self._mcts_rollout(question, child.path)
            last_score = rollout.get("score", 0.0)
            self._mcts_backpropagate(child, last_score)
            child.rollout_answer = rollout.get("answer", "")
            child.rollout_reasoning = rollout.get("reasoning", "")
            child.rollout_score = last_score

        if not root.children:
            logger.info("[%s] if-branch@L1829", "_solve_with_mcts")
            return self._solve_with_tot(question)

        best = max(root.children, key=lambda n: (n.value / max(n.visits, 1)))
        return self._mcts_finalize(question, best)

    @_log_call
    def _mcts_select(self, node: _MCTSNode) -> _MCTSNode:
        current = node
        while current.children:
            unvisited = [c for c in current.children if c.visits == 0]
            if unvisited:
                logger.info("[%s] if-branch@L1840", "_mcts_select")
                return unvisited[0]
            current = max(current.children, key=lambda c: self._mcts_ucb(c))
        return current

    @_log_call
    def _mcts_ucb(self, node: _MCTSNode) -> float:
        if node.visits == 0 or node.parent is None:
            logger.info("[%s] if-branch@L1847", "_mcts_ucb")
            return float("inf")
        exploit = node.value / node.visits
        explore = MCTS_UCB_C * math.sqrt(math.log(node.parent.visits + 1) / (node.visits + 1))
        return exploit + explore

    @_log_call
    def _mcts_rollout(self, question: str, path: List[str]) -> Dict[str, Any]:
        path_lines = "\n".join(f"步骤{i + 1}: {step}" for i, step in enumerate(path)) or "(当前尚无步骤)"
        messages = [
            {
                "role": "system",
                "content": (
                    "你是快速蒙特卡洛推演器，请基于已有步骤迅速完成解题，输出 JSON:"
                    " {reasoning_process, answer, confidence}，confidence 取 0-1。"
                ),
            },
            {
                "role": "user",
                "content": f"题目：{question.strip()}\n当前步骤：\n{path_lines}",
            },
        ]
        text = self._chat_completion(
            messages,
            max_tokens=MAX_TOKENS,
            temperature=MCTS_ROLLOUT_TEMP,
            top_p=MCTS_ROLLOUT_TOP_P,
        )
        parsed = self._extract_json(text) or {}
        reasoning = str(parsed.get("reasoning_process") or parsed.get("reasoning") or "").strip()
        answer = str(parsed.get("answer") or "").strip()
        confidence_raw = parsed.get("confidence")
        score = 0.0
        try:
            score = float(confidence_raw)
        except Exception:
            logger.warning("[%s] except-branch@L1882", "_mcts_rollout", exc_info=True)
            match = re.search(r"0\.\d+|1\.0|1", str(confidence_raw))
            score = float(match.group(0)) if match else 0.0
        score = max(0.0, min(score, 1.0))
        return {"reasoning": reasoning, "answer": answer, "score": score}

    @_log_call
    def _mcts_backpropagate(self, node: _MCTSNode, score: float) -> None:
        current: Optional[_MCTSNode] = node
        while current is not None:
            current.visits += 1
            current.value += score
            current = current.parent

    @_log_call
    def _mcts_finalize(self, question: str, node: _MCTSNode) -> Dict[str, str]:
        path_lines = "\n".join(f"步骤{i + 1}: {step}" for i, step in enumerate(node.path))
        rollout_hint = node.rollout_reasoning or "(无)"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": (
                    "以下是 MCTS 搜索得分最高的路径与快速推演提示，请据此整理最终 JSON：\n"
                    + path_lines
                    + "\n推演提示："
                    + rollout_hint
                ),
            },
            {"role": "user", "content": self._format_question(question)},
        ]
        text = self._chat_completion(messages, max_tokens=MAX_TOKENS, temperature=0.2, top_p=0.4)
        parsed = self._ensure_schema(text)
        prefix = "MCTS 全局搜索：路径访问 {} 次，平均得分 {:.3f}\n".format(
            node.visits, node.value / max(node.visits, 1)
        )
        parsed["reasoning_process"] = prefix + parsed.get("reasoning_process", "")
        return parsed

    @_log_call
    def _solve_with_debate(self, question: str) -> Dict[str, str]:
        solver_output = self._chat_completion(
            [
                {"role": "system", "content": "你是求解者，请给出初始解答，输出 JSON，字段 reasoning_process 与 answer。"},
                {"role": "user", "content": question.strip()},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.25,
            top_p=0.4,
        )

        for _ in range(DEBATE_ROUNDS):
            critic_feedback = self._chat_completion(
                [
                    {"role": "system", "content": "你是严格的批评者，指出解答中的漏洞与改进意见，仅输出批评要点。"},
                    {"role": "user", "content": solver_output},
                ],
                max_tokens=420,
                temperature=0.35,
                top_p=0.6,
            )
            if re.search(r"无(明显)?问题|通过|正确", critic_feedback):
                logger.info("[%s] if-branch@L1943", "_solve_with_debate")
                break
            solver_output = self._chat_completion(
                [
                    {"role": "system", "content": "你是求解者，请结合批评意见修正答案，输出 JSON，字段 reasoning_process 与 answer。"},
                    {"role": "user", "content": f"题目：{question.strip()}\n批评：{critic_feedback}"},
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.25,
                top_p=0.45,
            )

        return self._ensure_schema(solver_output)

    @staticmethod
    @_log_call
    def _normalize_answer(ans: str) -> str:
        clean = re.sub(r"\s+", "", ans).strip().lower()
        clean = clean.replace("。", "")
        return clean

    @_log_call
    def _normalize_fraction(self, ans: str) -> str:
        base = self._normalize_answer(ans)
        if not sp:
            logger.info("[%s] if-branch@L1967", "_normalize_fraction")
            return base
        try:
            expr = sp.nsimplify(ans)
            if expr.is_rational:
                logger.info("[%s] if-branch@L1971", "_normalize_fraction")
                return str(sp.Rational(expr))
            return str(expr)
        except Exception:
            logger.warning("[%s] except-branch@L1974", "_normalize_fraction", exc_info=True)
            return base

    @_log_call
    def _chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else TEMPERATURE,
            "top_p": top_p if top_p is not None else TOP_P,
            "max_tokens": max_tokens or MAX_TOKENS,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Optional[Exception] = None
        for attempt in range(RETRY_COUNT + 1):
            try:
                logger.debug("Sending request to Kimi", extra={"payload": payload, "attempt": attempt})
                response = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return self._sanitize_output(content)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] except-branch@L2006", "_chat_completion", exc_info=True)
                last_error = exc
                logger.warning("Kimi API call failed, will retry if allowed", exc_info=exc)
                continue

        logger.error("Kimi API call failed after retries", exc_info=last_error)
        return '{"reasoning_process": "解题失败：调用 Kimi 接口出现错误。请检查 API Key、网络或模型配置。", "answer": "无法生成答案"}'

    @staticmethod
    @_log_call
    def _sanitize_output(text: str) -> str:
        text = text.strip()
        text = KimiCalculusAgent._strip_code_fences(text)
        return text

    @staticmethod
    @_log_call
    def _strip_code_fences(text: str) -> str:
        if text.startswith("```") and text.endswith("```"):
            logger.info("[%s] if-branch@L2024", "_strip_code_fences")
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]
        return text.strip()

    @_log_call
    def _kb_lookup(self, question: str) -> List[Dict[str, Any]]:
        if not self._kb_entries:
            logger.info("[%s] if-branch@L2031", "_kb_lookup")
            return []
        direct_matches = self._direct_theory_matches(question)
        candidates: List[Dict[str, Any]] = []
        for entry in self._kb_entries:
            score = self._kb_entry_score(question, entry, direct_matches=direct_matches)
            if score <= 0:
                logger.info("[%s] if-branch@L2037", "_kb_lookup")
                continue
            candidates.append(self._entry_to_kb_hit(entry, score))
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:KB_TOP_K]

    @_log_call
    def decompose_with_knowledge(self, question: str) -> List[Dict[str, Any]]:
        """将题目拆解为知识点命中列表，供外部 API 或调试直接调用。"""
        hits = self._kb_lookup(question)
        direct_targets = self._direct_theory_matches(question)
        if not direct_targets:
            logger.info("[%s] if-branch@L2048", "decompose_with_knowledge")
            return hits

        forced_hits = self._lookup_targets_from_kb(direct_targets)
        merged_hits = self._merge_kb_hits(hits + forced_hits)
        top_k = max(KB_TOP_K, min(KB_MERGED_TOP_K, len(direct_targets) + 2))
        return merged_hits[:top_k]

    @_log_call
    def _lookup_targets_from_kb(self, targets: List[str]) -> List[Dict[str, Any]]:
        if not self._kb_entries:
            logger.info("[%s] if-branch@L2058", "_lookup_targets_from_kb")
            return []

        forced_hits: List[Dict[str, Any]] = []
        for target in targets:
            target_norm = self._normalize_lookup_text(target)
            if not target_norm:
                logger.info("[%s] if-branch@L2064", "_lookup_targets_from_kb")
                continue

            entry = self._kb_name_index.get(target_norm)
            if entry is not None:
                logger.info("[%s] if-branch@L2068", "_lookup_targets_from_kb")
                forced_hits.append(self._entry_to_kb_hit(entry, 120.0))
                continue

            best_entry: Optional[Dict[str, Any]] = None
            best_score = 0.0
            for candidate in self._kb_entries:
                name = str(candidate.get("name", "")).strip()
                aliases = [str(x).strip() for x in (candidate.get("aliases") or []) if str(x).strip()]
                if not name and not aliases:
                    logger.info("[%s] if-branch@L2077", "_lookup_targets_from_kb")
                    continue

                name_norm = self._normalize_lookup_text(name)
                alias_norms = [self._normalize_lookup_text(a) for a in aliases]
                entry_text = self._normalize_lookup_text(
                    " ".join(
                        [
                            name,
                            " ".join(aliases),
                            " ".join(str(x) for x in (candidate.get("keywords") or [])),
                            " ".join(str(x) for x in (candidate.get("formulas") or [])),
                            " ".join(str(x) for x in (candidate.get("principles") or [])),
                            str(candidate.get("section", "")),
                        ]
                    )
                )

                score = 0.0
                if name_norm and (target_norm == name_norm or target_norm in name_norm or name_norm in target_norm):
                    logger.info("[%s] if-branch@L2096", "_lookup_targets_from_kb")
                    score = max(score, 95.0)
                for alias_norm in alias_norms:
                    if alias_norm and (target_norm == alias_norm or target_norm in alias_norm or alias_norm in target_norm):
                        logger.info("[%s] if-branch@L2099", "_lookup_targets_from_kb")
                        score = max(score, 90.0)
                if entry_text and target_norm in entry_text:
                    logger.info("[%s] if-branch@L2101", "_lookup_targets_from_kb")
                    score = max(score, 82.0)

                if score > best_score:
                    logger.info("[%s] if-branch@L2104", "_lookup_targets_from_kb")
                    best_score = score
                    best_entry = candidate

            if best_entry is not None:
                logger.info("[%s] if-branch@L2108", "_lookup_targets_from_kb")
                forced_hits.append(self._entry_to_kb_hit(best_entry, best_score))
            else:
                logger.info("[%s] else-branch@L2110", "_lookup_targets_from_kb")
                forced_hits.append(
                    {
                        "score": 70.0,
                        "name": target,
                        "formulas": [],
                        "principles": ["规则映射命中该知识点，请优先按该定理或公式组织推导。"],
                        "related_types": ["direct-map"],
                    }
                )

        return forced_hits

    @_log_call
    def _kb_entry_score(
        self,
        question: str,
        entry: Dict[str, Any],
        direct_matches: Optional[List[str]] = None,
    ) -> float:
        question_norm = self._normalize_lookup_text(question)
        question_lower = question.lower()
        score = 0.0

        name = str(entry.get("name", "")).strip()
        if not name:
            logger.info("[%s] if-branch@L2135", "_kb_entry_score")
            return 0.0

        name_norm = self._normalize_lookup_text(name)
        if name_norm and name_norm in question_norm:
            logger.info("[%s] if-branch@L2139", "_kb_entry_score")
            score += 10.0

        aliases = entry.get("aliases") or []
        for alias in aliases:
            alias_norm = self._normalize_lookup_text(str(alias))
            if alias_norm and alias_norm in question_norm:
                logger.info("[%s] if-branch@L2145", "_kb_entry_score")
                score += 6.0

        keywords = entry.get("keywords") or []
        for kw in keywords:
            kw_text = str(kw).strip()
            if not kw_text:
                logger.info("[%s] if-branch@L2151", "_kb_entry_score")
                continue
            if kw_text in question or kw_text.lower() in question_lower:
                logger.info("[%s] if-branch@L2153", "_kb_entry_score")
                score += 3.0

        related_types = entry.get("related_types") or []
        for tp in related_types:
            tp_text = str(tp).strip()
            if not tp_text:
                logger.info("[%s] if-branch@L2159", "_kb_entry_score")
                continue
            if tp_text in question or tp_text.lower() in question_lower:
                logger.info("[%s] if-branch@L2161", "_kb_entry_score")
                score += 2.0

        q_tokens = set(self._tokenize(question))
        name_tokens = set(self._tokenize(name + " " + " ".join(str(a) for a in aliases)))
        score += float(len(q_tokens & name_tokens))

        formulas = entry.get("formulas") or []
        formula_text = " ".join(str(f) for f in formulas)
        if formula_text:
            logger.info("[%s] if-branch@L2170", "_kb_entry_score")
            score += float(self._similarity_score(question, formula_text)) * 0.4

        direct_matches = direct_matches if direct_matches is not None else self._direct_theory_matches(question)
        entry_text = self._normalize_lookup_text(
            " ".join(
                [
                    name,
                    " ".join(str(a) for a in aliases),
                    " ".join(str(k) for k in keywords),
                    " ".join(str(f) for f in formulas),
                    " ".join(str(p) for p in entry.get("principles") or []),
                    " ".join(str(t) for t in related_types),
                    str(entry.get("section", "")),
                ]
            )
        )
        for hint in direct_matches:
            hint_norm = self._normalize_lookup_text(hint)
            if not hint_norm:
                logger.info("[%s] if-branch@L2189", "_kb_entry_score")
                continue
            if hint_norm == self._normalize_lookup_text(name):
                logger.info("[%s] if-branch@L2191", "_kb_entry_score")
                score += 12.0
            elif any(hint_norm == self._normalize_lookup_text(str(alias)) for alias in aliases):
                logger.info("[%s] if-branch@L2193", "_kb_entry_score")
                score += 10.0
            elif hint_norm in entry_text:
                logger.info("[%s] if-branch@L2195", "_kb_entry_score")
                score += 4.0

        return score

    @_log_call
    def _matched_theory_rules(self, question: str) -> List[Dict[str, Any]]:
        lowered = question.lower()
        matched: List[Dict[str, Any]] = []
        for rule in THEORY_DIRECT_MAP_RULES:
            signals = [str(x) for x in rule.get("signals", [])]
            if not signals:
                logger.info("[%s] if-branch@L2206", "_matched_theory_rules")
                continue
            if any(sig in question or sig.lower() in lowered for sig in signals):
                logger.info("[%s] if-branch@L2208", "_matched_theory_rules")
                matched.append(rule)
        return matched

    @_log_call
    def _direct_theory_matches(self, question: str) -> List[str]:
        matches: List[str] = []
        for rule in self._matched_theory_rules(question):
            for target in rule.get("targets", []):
                target_text = str(target).strip()
                if target_text and target_text not in matches:
                    logger.info("[%s] if-branch@L2218", "_direct_theory_matches")
                    matches.append(target_text)
        return matches

    @_log_call
    def _build_mapping_reminders(self, question: str) -> List[str]:
        reminders: List[str] = []
        for rule in self._matched_theory_rules(question):
            reminder = str(rule.get("reminder", "")).strip()
            if reminder and reminder not in reminders:
                logger.info("[%s] if-branch@L2227", "_build_mapping_reminders")
                reminders.append(reminder)
        return reminders

    @_log_call
    def _extract_theory_hints(self, question: str) -> List[str]:
        direct_matches = self._direct_theory_matches(question)
        if direct_matches:
            logger.info("[%s] if-branch@L2234", "_extract_theory_hints")
            return direct_matches
        fallback_hits = self._kb_lookup(question)
        return [str(hit.get("name", "")).strip() for hit in fallback_hits if str(hit.get("name", "")).strip()]

    @_log_call
    def _build_kb_context(self, question: str) -> str:
        hits = self.decompose_with_knowledge(question)
        if not hits:
            logger.info("[%s] if-branch@L2242", "_build_kb_context")
            return ""
        direct_matches = self._direct_theory_matches(question)
        hints = self._extract_theory_hints(question)
        reminders = self._build_mapping_reminders(question)
        lines = ["以下是知识点库检索结果：请先按知识点拆解问题，再调用对应公式与原则进行推导。"]
        if direct_matches:
            logger.info("[%s] if-branch@L2248", "_build_kb_context")
            lines.append("直接映射: " + " / ".join(direct_matches[:8]))
        if hints:
            logger.info("[%s] if-branch@L2250", "_build_kb_context")
            lines.append("题面信号提示: " + " / ".join(hints[:6]))
        for reminder in reminders[:3]:
            lines.append("映射提醒: " + reminder)
        for idx, hit in enumerate(hits, 1):
            formulas = hit.get("formulas") or []
            principles = hit.get("principles") or []
            related_types = hit.get("related_types") or []

            lines.append(f"知识点{idx}: {hit.get('name', '')} (score={hit.get('score', 0):.2f})")
            if related_types:
                logger.info("[%s] if-branch@L2260", "_build_kb_context")
                lines.append("相关题型: " + " / ".join(str(x) for x in related_types[:3]))
            if formulas:
                logger.info("[%s] if-branch@L2262", "_build_kb_context")
                lines.append("可调用公式: " + " ; ".join(str(x) for x in formulas[:4]))
            if principles:
                logger.info("[%s] if-branch@L2264", "_build_kb_context")
                lines.append("使用原则: " + " ; ".join(str(x) for x in principles[:2]))

        return "\n".join(lines)

    @_log_call
    def _extract_math_expression(self, text: str) -> Optional[str]:
        normalized = text.replace("×", "*").replace("÷", "/").replace("（", "(").replace("）", ")").strip()
        # Only treat input as local arithmetic when the whole prompt is an arithmetic expression.
        if re.search(r"[A-Za-z]", normalized):
            logger.info("[%s] if-branch@L2273", "_extract_math_expression")
            return None
        if not re.fullmatch(r"[\d\.\s\+\-\*/\(\)\%\^=]+", normalized):
            logger.info("[%s] if-branch@L2275", "_extract_math_expression")
            return None

        expr = normalized.replace("=", " ").strip()
        if not expr:
            logger.info("[%s] if-branch@L2279", "_extract_math_expression")
            return None
        if not re.search(r"\d", expr):
            logger.info("[%s] if-branch@L2281", "_extract_math_expression")
            return None
        if not re.search(r"[\+\-\*/\^%]", expr):
            logger.info("[%s] if-branch@L2283", "_extract_math_expression")
            return None
        return expr.replace("^", "**")

    @_log_call
    def _safe_eval(self, expr: str) -> float:
        node = ast.parse(expr, mode="eval")
        value = self._eval_ast(node.body)
        return float(value)

    @_log_call
    def _eval_ast(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            logger.info("[%s] if-branch@L2295", "_eval_ast")
            if isinstance(node.value, (int, float)):
                logger.info("[%s] if-branch@L2296", "_eval_ast")
                return float(node.value)
            raise ValueError("表达式包含非法常量")

        if isinstance(node, ast.BinOp):
            logger.info("[%s] if-branch@L2300", "_eval_ast")
            op_type = type(node.op)
            if op_type not in self._allowed_ops:
                logger.info("[%s] if-branch@L2302", "_eval_ast")
                raise ValueError("表达式包含不支持的二元运算")
            left = self._eval_ast(node.left)
            right = self._eval_ast(node.right)
            return float(self._allowed_ops[op_type](left, right))

        if isinstance(node, ast.UnaryOp):
            logger.info("[%s] if-branch@L2308", "_eval_ast")
            op_type = type(node.op)
            if op_type not in self._allowed_unary_ops:
                logger.info("[%s] if-branch@L2310", "_eval_ast")
                raise ValueError("表达式包含不支持的一元运算")
            value = self._eval_ast(node.operand)
            return float(self._allowed_unary_ops[op_type](value))

        raise ValueError("表达式语法不受支持")

    _allowed_ops: Dict[Any, Any] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }

    _allowed_unary_ops: Dict[Any, Any] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }


__all__ = ["KimiCalculusAgent"]
