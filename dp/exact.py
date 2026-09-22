"""반복 없이 정확한 답을 구하는 두 함수.

:func:`exact_policy_values`
    ``pi``를 고정하면 Bellman 기대 방정식은 연립 일차방정식이 되므로
    ``v_pi = (I - gamma P_pi)^-1 r_pi``를 한 번에 푼다. sweep도 theta도 없다.

:func:`exhaustive_search`
    Bellman 최적 방정식에는 ``max``가 있어 같은 방식의 닫힌 형태가 없다. 대신
    결정적 정책을 전부 나열해 각각을 위 함수로 정확히 평가하고 제일 좋은 것을
    고른다. 4x5 격자에서는 ``4 ** 11 = 4,194,304``개이고 몇 초 걸린다.

    python -m dp.exact              fig02
"""

from __future__ import annotations

import time

import numpy as np

from .gridworld import (
    ACTIONS,
    N_ACTIONS,
    GridWorld,
    greedy_from_q,
    main_grid,
    q_table_from_v,
)


# ----------------------------------------------------------------------
# 정책 하나를 정확히 평가하기
# ----------------------------------------------------------------------
def exact_policy_values(env: GridWorld, pi: np.ndarray, gamma: float) -> np.ndarray:
    """``(I - gamma P_pi) v = r_pi``를 직접 풀어서 얻는 ``v_pi``.

    이 패키지의 어떤 반복 루틴도 쓰지 않으므로 ``test_dp.py``가 반복적 정책
    평가를 대조하는 기준이 된다.
    """
    index = {s: i for i, s in enumerate(env.states)}
    A = np.eye(len(index))
    b = np.zeros(len(index))
    for s, i in index.items():
        if env.is_terminal(s):
            continue  # v = 0이므로 단위행렬 행을 그대로 둔다
        for a in ACTIONS:
            if pi[s, a] == 0:
                continue
            for prob, s_next, reward, _ in env.P[s][a]:
                weight = pi[s, a] * prob
                b[i] += weight * reward
                A[i, index[s_next]] -= gamma * weight
    solution = np.linalg.solve(A, b)
    V = np.zeros(env.n_states)
    for s, i in index.items():
        V[s] = solution[i]
    return V


# ----------------------------------------------------------------------
# 존재하는 모든 정책을 다 해보기
# ----------------------------------------------------------------------
def _linear_system_pieces(env: GridWorld) -> tuple:
    """내부 상태만 대상으로 한 ``(TRANS, REWARD)``. 인덱스는 순서 번호.

    ``TRANS[i, a, j]``는 행동 ``a``로 내부 상태 ``i``에서 ``j``로 갈 확률,
    ``REWARD[i, a]``는 즉시 보상의 기댓값. 종결 상태는 가치가 0이라 연립방정식에
    기여하지 않으므로 둘 다에서 뺀다.
    """
    interior = env.interior_states
    index = {s: i for i, s in enumerate(interior)}
    n = len(interior)
    trans = np.zeros((n, N_ACTIONS, n))
    reward = np.zeros((n, N_ACTIONS))
    for s, i in index.items():
        for a in ACTIONS:
            for prob, s_next, r, _ in env.P[s][a]:
                reward[i, a] += prob * r
                if s_next in index:
                    trans[i, a, index[s_next]] += prob
    return trans, reward


def exhaustive_search(env: GridWorld, gamma: float, batch: int = 20_000,
                      max_policies: int = 20_000_000, verbose: bool = True):
    """결정적 정책을 전부 정확히 평가하고 제일 좋은 것을 고른다.

    ``(V, pi, 정책 수, 초)``를 돌려준다. ``V``와 ``pi``의 모양은 다른 알고리즘이
    돌려주는 것과 같아서 바로 비교할 수 있다.

    점수는 ``sum_s p0(s) v_pi(s)``. ``p0``이 모든 내부 상태에서 양수이고 유한
    MDP에는 모든 상태에서 동시에 최적인 정책이 존재하므로, 이 가중치로 이긴
    정책은 모든 상태에서 최적이다.

    정책을 ``|A|``진수 자릿수로 펼치고 연립방정식을 ``batch``개씩 묶어
    ``np.linalg.solve``에 넘긴다. 4백만 번을 파이썬 루프로 돌리면 몇 시간
    걸린다.
    """
    trans, reward = _linear_system_pieces(env)
    n = trans.shape[0]
    total = N_ACTIONS ** n
    if total > max_policies:
        raise ValueError(
            f"{env.name}: {N_ACTIONS}**{n} = {total:,} policies is past the "
            f"{max_policies:,} limit")

    weights = np.zeros(n)
    positions = {s: i for i, s in enumerate(env.interior_states)}
    for s, w in env.p0.items():
        weights[positions[s]] = w

    identity = np.eye(n)
    place_value = N_ACTIONS ** np.arange(n)
    rows = np.arange(n)

    best_score = -np.inf
    best_actions = None
    started = time.perf_counter()

    for first in range(0, total, batch):
        codes = np.arange(first, min(first + batch, total), dtype=np.int64)
        actions = (codes[:, None] // place_value) % N_ACTIONS   # (B, n)
        A = identity - gamma * trans[rows, actions]             # (B, n, n)
        b = reward[rows, actions]                               # (B, n)
        values = np.linalg.solve(A, b[..., None])[..., 0]       # (B, n)
        scores = values @ weights
        winner = int(scores.argmax())
        if scores[winner] > best_score:
            best_score = float(scores[winner])
            best_actions = actions[winner].copy()
        if verbose and first % (batch * 40) == 0:
            done = min(first + batch, total)
            print(f"  {done:>10,} / {total:,}   best score so far "
                  f"{best_score:9.4f}", flush=True)

    seconds = time.perf_counter() - started

    V = np.zeros(env.n_states)
    pi = np.zeros((env.n_states, N_ACTIONS))
    A = identity - gamma * trans[rows, best_actions]
    values = np.linalg.solve(A, reward[rows, best_actions])
    for s, i in positions.items():
        V[s] = values[i]
        pi[s, best_actions[i]] = 1.0
    return V, pi, total, seconds


def optimal_policy_is_tied_everywhere(env: GridWorld, V: np.ndarray,
                                      gamma: float) -> np.ndarray:
    """최적 ``V``가 주어졌을 때, 동점까지 포함한 전체 최적 정책.

    :func:`exhaustive_search`는 이긴 정책 하나만 돌려주므로, 동점을 도로 채워야
    다른 알고리즘의 결과와 비교할 수 있다.
    """
    return greedy_from_q(env, q_table_from_v(env, V, gamma))


# ----------------------------------------------------------------------
# python -m dp.exact
# ----------------------------------------------------------------------
def main(gamma: float = 0.9, noise: float = 0.0) -> None:
    from . import gridworld as gw
    from . import viz
    from .value_iteration import value_iteration

    viz.begin_demo("2. The exact answer, without iterating")
    env = main_grid(noise=noise)

    print(f"{env.name}, gamma = {gamma}")
    print(f"interior states: {len(env.interior_states)}, actions: {N_ACTIONS}")
    print(f"deterministic policies to try: {N_ACTIONS}**"
          f"{len(env.interior_states)} = {N_ACTIONS ** len(env.interior_states):,}\n")

    V, pi_one, total, seconds = exhaustive_search(env, gamma)
    pi = optimal_policy_is_tied_everywhere(env, V, gamma)

    print(f"\nsearched {total:,} policies in {seconds:.1f}s")
    print(gw.render_values(env, V))
    print()
    print(gw.render_policy(env, pi))
    print(f"\nends at: {gw.describe_outcome(env, pi)}")

    vi = value_iteration(env, gamma=gamma)
    gap = np.abs(vi.V - V).max()
    print(f"\nvalue iteration reaches the same answer in {vi.n_sweeps} sweeps")
    print(f"  largest disagreement in V   {gap:.2e}")
    print(f"  same policy                 {gw.policy_matches(vi.pi, pi)}")
    print(f"\nOne more interior cell multiplies the search by {N_ACTIONS} "
          "and the sweep count by\nalmost nothing.")

    viz.figure_value_policy(
        env, V, pi,
        f"The true optimum, found by trying all {total:,} deterministic policies",
        viz.FIGURES / "fig02_exact_solution.png")
    print(f"\nwrote {viz.FIGURES.name}/fig02_exact_solution.png")


if __name__ == "__main__":
    from . import viz
    main(**vars(viz.demo_args(gamma=0.9, noise=0.0)))
