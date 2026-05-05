"""Minimal entry: build dict payload and call agent.solve()."""

from agent import KimiCalculusAgent


def main() -> None:
    agent = KimiCalculusAgent(api_key="sk-45a126c7cbb746e9a1851ae38f5e3240")
    questions =[
    {
        "question_id": "CAL_021",
        "type": "计算题",
        "difficulty": "困难",
        "question": "计算反常积分：∫_{0}^{∞} (ln x) / (x^2 + 1) dx。"
    },
    {
        "question_id": "CAL_022",
        "type": "计算题",
        "difficulty": "困难",
        "question": "计算无穷级数的和：∑_{n=1}^{∞} (-1)^{n+1} / (n^2 * 2^n)。"
    },
    {
        "question_id": "CAL_023",
        "type": "证明题",
        "difficulty": "竞赛",
        "question": "设 f(x) = ∑_{n=0}^{∞} x^n / (n! * (n+1))。求 f(x) 的闭合表达式，并计算 lim_{x→∞} e^{-x} f(x)。"
    },
    {
        "question_id": "CAL_024",
        "type": "证明题",
        "difficulty": "竞赛",
        "question": "计算含参变量积分：I(α) = ∫_{0}^{∞} (e^{-αx} * sin x) / x dx，其中 α ≥ 0。利用对 α 求导的方法求 I(α)，并计算 ∫_{0}^{∞} sin x / x dx 的值。"
    },
    {
        "question_id": "CAL_025",
        "type": "证明题",
        "difficulty": "竞赛",
        "question": "设 f 在 [0,1] 上二阶连续可导，且 f(0)=f(1)=0。证明：∫_{0}^{1} |f''(x)| dx ≥ 4 * max_{x∈[0,1]} |f(x)|。"
    },
    {
        "question_id": "CAL_026",
        "type": "计算题",
        "difficulty": "困难",
        "question": "计算定积分：∫_{0}^{π} x * sin x / (1 + cos^2 x) dx。"
    },
    {
        "question_id": "CAL_027",
        "type": "计算题",
        "difficulty": "困难",
        "question": "计算无穷级数的和：∑_{n=1}^{∞} 1 / (n^2 + n^4)。"
    },
    {
        "question_id": "CAL_028",
        "type": "证明题",
        "difficulty": "竞赛",
        "question": "设 a_n = ∫_{0}^{1} (1 - x^2)^n dx。求 lim_{n→∞} √n * a_n，并判定级数 ∑_{n=1}^{∞} a_n 的敛散性。"
    },
    {
        "question_id": "CAL_029",
        "type": "证明题",
        "difficulty": "竞赛",
        "question": "计算幂级数的和函数：S(x) = ∑_{n=1}^{∞} n^2 * x^n / (n+1)，并求其收敛域。"
    },
    {
        "question_id": "CAL_030",
        "type": "计算题",
        "difficulty": "困难",
        "question": "计算二重积分：∬_{D} e^{-(x^2+y^2)} dxdy，其中 D = {(x,y) | 0 ≤ x ≤ 1, 0 ≤ y ≤ 1}。结果用误差函数erf(x)表示。"
    }
]

    for item in questions:
        result = agent.solve(item)
        print(f"=== {item.get('question_id', '')} ===")
        print(result)


if __name__ == "__main__":
    main()
