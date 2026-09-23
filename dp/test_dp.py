"""알고리즘 검증.

    python -m dp.test_dp

각 알고리즘을 *그 알고리즘을 쓰지 않고 구한 답*에 대조한다. 대조 상대는 넷이다.

* 교과서가 인쇄한 값 (아래 ``FIG_4_1``)
* 너비 우선 탐색으로 구한 최단 거리
* 연립방정식 풀이 (``exact.py``)
* 서로 다른 알고리즘끼리의 일치 (V 대 Q, 값 반복 대 정책 반복)
"""

from __future__ import annotations

from collections import deque

import numpy as np

from . import gridworld as gw
from .exact import exact_policy_values, exhaustive_search
from .gridworld import (
    action_values,
    greedy_policy,
    outcome_probabilities,
    policy_matches,
    uniform_random_policy,
)
from .policy_iteration import (expected_policy_evaluation,
                               policy_evaluation, policy_iteration)
from .q_value_iteration import q_policy_evaluation, q_value_iteration
from .q_value_iteration import greedy_policy as q_greedy_policy
from .value_iteration import value_iteration

# 답이 인쇄되어 있는 4x4 판. 실험용 환경이 아니라 이 파일 전용 고정물이다.
# 마주보는 두 모서리가 종결, 매 걸음 -1, 할인 없음.
def book_grid() -> gw.GridWorld:
    return gw.GridWorld(n_rows=4, n_cols=4,
                        terminals={(0, 0): 0.0, (3, 3): 0.0},
                        step_reward=-1.0, noise=0.0, name="4x4 book fixture")


# 그 판에서 등확률 무작위 정책의 v_k를 소수점 한 자리로 인쇄한 값.
# *동기식*(두 배열) 갱신에서 나온 것이므로 아래 테스트들은 in_place=False를 넘긴다.
FIG_4_1 = {
    0: np.zeros((4, 4)),
    1: np.array([[0.0, -1.0, -1.0, -1.0],
                 [-1.0, -1.0, -1.0, -1.0],
                 [-1.0, -1.0, -1.0, -1.0],
                 [-1.0, -1.0, -1.0, 0.0]]),
    2: np.array([[0.0, -1.7, -2.0, -2.0],
                 [-1.7, -2.0, -2.0, -2.0],
                 [-2.0, -2.0, -2.0, -1.7],
                 [-2.0, -2.0, -1.7, 0.0]]),
    3: np.array([[0.0, -2.4, -2.9, -3.0],
                 [-2.4, -2.9, -3.0, -2.9],
                 [-2.9, -3.0, -2.9, -2.4],
                 [-3.0, -2.9, -2.4, 0.0]]),
    10: np.array([[0.0, -6.1, -8.4, -9.0],
                  [-6.1, -7.7, -8.4, -8.4],
                  [-8.4, -8.4, -7.7, -6.1],
                  [-9.0, -8.4, -6.1, 0.0]]),
}

FIG_4_1_CONVERGED = np.array([[0.0, -14.0, -20.0, -22.0],
                              [-14.0, -18.0, -20.0, -20.0],
                              [-20.0, -20.0, -18.0, -14.0],
                              [-22.0, -20.0, -14.0, 0.0]])

# 교차 검증이 돌 (환경, gamma) 짝.
CASES = [
    (gw.main_grid(), 0.9),
    (gw.main_grid(noise=0.2), 0.9),
    (gw.main_grid(noise=0.4, step_reward=-0.2), 0.95),
    (book_grid(), 1.0),
]


def as_grid(env: gw.GridWorld, V: np.ndarray) -> np.ndarray:
    return V.reshape(env.shape)


def shortest_path_values(env: gw.GridWorld) -> np.ndarray:
    """``-1 * (가장 가까운 종결 상태까지의 걸음 수)``. 너비 우선 탐색으로."""
    dist = np.full(env.n_states, np.inf)
    queue = deque()
    for s in env.terminal_states:
        dist[s] = 0.0
        queue.append(s)
    while queue:
        s = queue.popleft()
        rc = env.to_rc(s)
        for a in gw.ACTIONS:
            # 탐색은 거꾸로 돌지만 격자 이동이 대칭이라, 한 칸의 이웃은 곧 그
            # 칸에 한 걸음에 도달할 수 있는 칸들이다.
            s_prev = env.to_s(env._move(rc, a))
            if dist[s_prev] == np.inf:
                dist[s_prev] = dist[s] + 1
                queue.append(s_prev)
    return -dist


# ----------------------------------------------------------------------
# 모델
# ----------------------------------------------------------------------
def test_transition_table_is_a_distribution():
    for env in (gw.main_grid(), gw.main_grid(noise=0.2), book_grid()):
        for s in env.states:
            for a in gw.ACTIONS:
                total = sum(p for p, *_ in env.P[s][a])
                assert abs(total - 1.0) < 1e-12, (env.name, s, a, total)
                for _, s_next, _, _ in env.P[s][a]:
                    assert not env.is_wall(s_next), "transitioned into a wall"


# ----------------------------------------------------------------------
# 4x5 격자
# ----------------------------------------------------------------------
def test_state_numbering():
    """벽이 아닌 15칸에 행 우선으로 s1..s15가 붙고, 보상 칸이 s3/s6/s9/s11."""
    env = gw.main_grid()
    names = env.state_names
    assert len(names) == 15
    assert [names[env.to_s(rc)] for rc in sorted(env.terminals)] == [
        "s3", "s6", "s9", "s11"]


def test_p0_is_uniform_off_the_reward_cells():
    """보상 칸 넷은 종결이라 p0에서 빠지고, 나머지 11칸이 1/11씩."""
    env = gw.main_grid()
    p0 = env.p0
    assert len(p0) == 11
    assert all(np.isclose(w, 1 / 11) for w in p0.values())
    assert not any(env.is_terminal(s) for s in p0)
    names = {env.state_names[s] for s in p0}
    assert names.isdisjoint({"s3", "s6", "s9", "s11"})


def test_discount_flips_the_target_around_the_crossover():
    """s2에서 동전은 1칸, 보석은 9칸이므로 0.1**(1/8)에서 교차한다."""
    env = gw.main_grid()
    s2 = env.to_s((0, 1))
    gem = list(env.terminal_states).index(env.to_s((1, 3)))
    crossover = 0.1 ** (1 / 8)
    for gamma, wants_gem in ((0.1, False), (crossover - 0.05, False),
                             (crossover + 0.05, True), (0.99, True)):
        pi = value_iteration(env, gamma=gamma).pi
        reaches_gem = outcome_probabilities(env, pi)[s2, gem] > 0.5
        assert reaches_gem == wants_gem, (gamma, reaches_gem)


PITS = {(2, 3), (3, 0)}


def pit_probability(env: gw.GridWorld, pi: np.ndarray) -> float:
    """p0으로 평균 낸, 함정에서 끝날 확률."""
    outcome = outcome_probabilities(env, pi)
    terminals = list(env.terminal_states)
    columns = [terminals.index(env.to_s(rc)) for rc in PITS]
    return sum(w * outcome[s, columns].sum() for s, w in env.p0.items())


def test_safety_is_always_available_beside_a_pit():
    """함정에 닿을 수 있는 칸은 정확히 셋이고, 각각 위험 0인 행동을 가진다.

    위험 0인 행동은 벽이나 판 가장자리로 들어가는 쪽이다. 함정에 들어갈 수 없고
    제자리에 머물거나 옆으로 미끄러질 뿐이다.
    """
    for noise in (0.05, 0.1, 0.2, 0.5):
        env = gw.main_grid(noise=noise)
        beside = {env.to_rc(s) for s in env.interior_states
                  for a in gw.ACTIONS
                  if any(env.to_rc(s2) in PITS for _, s2, _, _ in env.P[s][a])}
        assert beside == {(3, 3), (2, 4), (3, 1)}, (noise, sorted(beside))

        for rc in beside:
            s = env.to_s(rc)
            risk_free = [a for a in gw.ACTIONS
                         if not any(env.to_rc(s2) in PITS
                                    for p, s2, _, _ in env.P[s][a] if p > 0)]
            assert risk_free, (noise, rc)


def test_the_agent_gives_up_risk_one_cell_at_a_time():
    """noise를 올리면 s14와 s10이 서로 다른 지점에서 안전한 행동으로 바뀐다.

    s14 = (3,3)은 "right"(함정으로 미끄러질 수 있음)와 "down"(판 가장자리라
    불가능) 사이, s10 = (2,4)는 "up"과 "right" 사이에서 고른다.

        noise 0.05        둘 다 위험             P(함정) ~ 0.04
        noise 0.08-0.10   s14 안전, s10 위험     P(함정) ~ 0.02
        noise >= 0.12     둘 다 안전             P(함정) = 0
    """
    env0 = gw.main_grid()
    s14, s10 = env0.to_s((3, 3)), env0.to_s((2, 4))

    def solve(noise):
        env = gw.main_grid(noise=noise)
        return env, value_iteration(env, gamma=0.9, theta=1e-13).pi

    env, pi = solve(0.05)
    assert pi[s14, gw.RIGHT] > 0 and pi[s10, gw.UP] > 0
    assert 0.02 < pit_probability(env, pi) < 0.1

    for noise in (0.08, 0.10):
        env, pi = solve(noise)
        assert pi[s14, gw.DOWN] > 0, noise      # s14는 이미 물러섰다
        assert pi[s10, gw.UP] > 0, noise        # s10은 아직
        assert 0.0 < pit_probability(env, pi) < 0.05, noise

    for noise in (0.12, 0.15, 0.2, 0.3, 0.5):
        env, pi = solve(noise)
        assert pi[s14, gw.DOWN] > 0, noise
        assert pi[s10, gw.RIGHT] > 0, noise
        assert pit_probability(env, pi) < 1e-9, noise


# ----------------------------------------------------------------------
# 상태가치 알고리즘
# ----------------------------------------------------------------------
def test_policy_evaluation_matches_figure_4_1():
    env = book_grid()
    pi = uniform_random_policy(env)
    result = expected_policy_evaluation(env, pi, gamma=1.0, theta=1e-12,
                                        in_place=False,
                               snapshots_at=(0, 1, 2, 3, 10))
    for k, expected in FIG_4_1.items():
        got = as_grid(env, result.snapshots[k])
        # 책이 소수점 한 자리로 인쇄했으므로 허용 오차를 느슨하게 둔다.
        assert np.allclose(got, expected, atol=0.06), f"k={k}\n{got}\n!=\n{expected}"
    assert np.allclose(as_grid(env, result.V), FIG_4_1_CONVERGED, atol=1e-6)


def test_exhaustive_search_finds_what_value_iteration_finds():
    """전수 탐색이 값 반복과 같은 답을 내는지. 4x5는 무거우므로 2x3 판에서.

    4x5는 정책이 4**11 = 4,194,304개라 테스트로 돌리기엔 느리다. 여기 2x3 판은
    4**4 = 256개이고 확인하는 주장은 같다.
    """
    env = gw.GridWorld(n_rows=2, n_cols=3,
                       terminals={(0, 2): 1.0, (1, 0): -1.0},
                       step_reward=-0.04, noise=0.2, name="2x3 fixture")
    V, _, total, _ = exhaustive_search(env, gamma=0.9, verbose=False)
    vi = value_iteration(env, gamma=0.9, theta=1e-13)
    assert total == 4 ** len(env.interior_states), total
    assert np.allclose(V, vi.V, atol=1e-8), np.abs(V - vi.V).max()


def test_optimal_values_equal_shortest_path():
    env = book_grid()
    expected = shortest_path_values(env)
    for in_place in (True, False):
        vi = value_iteration(env, gamma=1.0, in_place=in_place)
        assert np.allclose(vi.V, expected, atol=1e-9), (in_place, vi.V, expected)


def test_iterative_evaluation_matches_the_exact_linear_solve():
    for env, gamma in CASES:
        for pi in (uniform_random_policy(env),
                   value_iteration(env, gamma=gamma).pi):
            iterative = expected_policy_evaluation(env, pi, gamma, theta=1e-14).V
            exact = exact_policy_values(env, pi, gamma)
            assert np.allclose(iterative, exact, atol=1e-8), (
                env.name, np.abs(iterative - exact).max())


def test_optimal_values_satisfy_the_bellman_optimality_equation():
    """``V*(s) == max_a q(s, a)``. 정의적 성질을 직접 확인한다."""
    for env, gamma in CASES:
        V = value_iteration(env, gamma=gamma, theta=1e-14).V
        residual = max(abs(V[s] - action_values(env, V, s, gamma).max())
                       for s in env.interior_states)
        assert residual < 1e-8, (env.name, residual)


def chosen_actions_are_optimal(env, deterministic_pi, optimal_pi) -> bool:
    """결정론적 정책이 고른 행동이 전부 최적 집합 안에 있는가.

    정책 반복은 칸마다 행동 하나를 돌려주고, 값 반복은 동점을 나눠 담은 분포를
    돌려준다. 동점이 있는 판에서는 두 배열이 글자 그대로 같을 수 없다.
    """
    return all(optimal_pi[s, int(deterministic_pi[s].argmax())] > 0
               for s in env.interior_states)


def test_value_iteration_agrees_with_policy_iteration():
    for env, gamma in CASES:
        vi = value_iteration(env, gamma=gamma)
        pit = policy_iteration(env, gamma=gamma)
        assert np.allclose(vi.V, pit.V, atol=1e-6), env.name
        assert chosen_actions_are_optimal(env, pit.pi, vi.pi), (
            f"{env.name}\n{gw.render_policy(env, vi.pi)}\n"
            f"---\n{gw.render_policy(env, pit.pi)}")


def test_in_place_and_synchronous_reach_the_same_fixed_point():
    env = gw.main_grid(noise=0.2)
    a = value_iteration(env, gamma=0.9, in_place=True)
    b = value_iteration(env, gamma=0.9, in_place=False)
    assert np.allclose(a.V, b.V, atol=1e-6)
    # ...다만 in-place가 거기 도달하는 데 더 많은 sweep을 쓰면 안 된다.
    assert a.n_sweeps <= b.n_sweeps, (a.n_sweeps, b.n_sweeps)


def test_discount_factor_decides_coin_versus_gem():
    """시작 칸 s5의 가치가 어느 종결 상태로 걸어가는지를 못박는다.

    noise = 0에서 동전(+1)은 s5로부터 2칸, 보석(+10)은 8칸이고 step_reward가
    0이므로 두 수익은 ``gamma**1 * 1``과 ``gamma**7 * 10``이다.
    """
    env = gw.main_grid()
    start = env.to_s(env.start)
    assert np.isclose(value_iteration(env, gamma=0.9).V[start], 0.9 ** 7 * 10)
    assert np.isclose(value_iteration(env, gamma=0.5).V[start], 0.5 ** 1 * 1)


def test_terminal_states_keep_zero_value():
    for env, gamma in CASES:
        for result in (value_iteration(env, gamma=gamma),
                       policy_iteration(env, gamma=gamma)):
            for s in env.terminal_states:
                assert result.V[s] == 0.0, (env.name, s, result.V[s])


def test_policy_iteration_terminates_with_ties_everywhere():
    """동점을 균등하게 나누는 처리가 정책 반복의 진동을 막아야 한다.

    4x4 고정물에는 똑같이 최적인 행동이 둘 또는 넷인 상태가 널려 있어서,
    argmax를 임의로 고르면 그 사이를 영원히 오갈 수 있다.
    """
    env = book_grid()
    result = policy_iteration(env, gamma=1.0, max_iterations=50)
    assert result.n_iterations < 50
    # 돌려주는 정책은 결정론적이라 칸마다 행동 하나다.
    for s in env.interior_states:
        assert np.count_nonzero(result.pi[s]) == 1, s
    # (0, 1)에는 최적 행동이 하나뿐이다(left, 종결 상태로).
    assert result.pi[env.to_s((0, 1))].tolist() == [0.0, 0.0, 0.0, 1.0]
    # (1, 1)은 up과 left가 동점이다. 둘 중 하나를 골랐으면 된다.
    optimal = value_iteration(env, gamma=1.0).pi[env.to_s((1, 1))]
    assert np.count_nonzero(optimal) == 2
    assert optimal[int(result.pi[env.to_s((1, 1))].argmax())] > 0


def test_truncated_policy_iteration_saves_sweeps_when_it_works():
    env = gw.main_grid()
    reference = policy_iteration(env, gamma=0.9)
    for cap in (1, 2, 5, 20):
        truncated = policy_iteration(env, gamma=0.9, max_eval_sweeps=cap)
        assert chosen_actions_are_optimal(env, truncated.pi,
                                          value_iteration(env, gamma=0.9).pi), cap
        assert truncated.n_sweeps <= reference.n_sweeps, (cap, truncated.n_sweeps)


def test_truncating_evaluation_too_hard_stops_early():
    """max_eval_sweeps=1이면 V가 v*에서 먼 채로 정책이 굳어 준최적으로 끝난다.

    값 반복은 ``Delta < theta``로 멈추므로 이 실패가 없다.
    """
    env = gw.main_grid(noise=0.2)
    optimal = value_iteration(env, gamma=0.9)

    aggressive = policy_iteration(env, gamma=0.9, max_eval_sweeps=1)
    assert not policy_matches(aggressive.pi, optimal.pi)
    assert np.abs(aggressive.V - optimal.V).max() > 1.0

    patient = policy_iteration(env, gamma=0.9, max_eval_sweeps=20)
    assert policy_matches(patient.pi, optimal.pi)
    assert np.allclose(patient.V, optimal.V, atol=1e-4)


def test_warm_start_does_not_change_the_answer():
    for env, gamma in CASES:
        cold = policy_iteration(env, gamma=gamma, warm_start=False)
        warm = policy_iteration(env, gamma=gamma, warm_start=True)
        assert np.allclose(cold.V, warm.V, atol=1e-6), env.name
        assert policy_matches(cold.pi, warm.pi), env.name


def test_warm_start_can_cost_sweeps_when_the_first_policy_is_terrible():
    """출발 정책이 나쁘면 warm_start가 sweep을 더 쓴다.

    모든 칸에서 왼쪽으로만 가는 정책은 -100 함정으로 걸어 들어가서 값이
    -100 근처까지 내려간다. ``v*``는 +10 언저리라, 그 값에서 이어 시작하면
    0에서 시작하는 것보다 먼 출발점이 된다.
    """
    env = gw.main_grid()
    bad = np.full(env.n_states, gw.LEFT)
    v_bad = policy_evaluation(env, bad, 0.9).V
    v_star = value_iteration(env, gamma=0.9).V
    assert np.abs(v_bad - v_star).max() > np.abs(v_star).max()

    cold = policy_iteration(env, gamma=0.9, warm_start=False, pi0=bad)
    warm = policy_iteration(env, gamma=0.9, warm_start=True, pi0=bad)
    assert warm.n_sweeps > cold.n_sweeps, (warm.n_sweeps, cold.n_sweeps)


# ----------------------------------------------------------------------
# Q-값 반복
# ----------------------------------------------------------------------
def test_q_evaluation_matches_v_evaluation():
    """``sum_a pi(a|s) q_pi(s, a) == v_pi(s)``. 두 표가 일치한다."""
    for env, gamma in CASES:
        for pi in (uniform_random_policy(env),
                   value_iteration(env, gamma=gamma).pi):
            q_result = q_policy_evaluation(env, pi, gamma, theta=1e-13)
            v_result = expected_policy_evaluation(env, pi, gamma, theta=1e-13)
            assert np.allclose(q_result.V, v_result.V, atol=1e-7), (
                env.name, np.abs(q_result.V - v_result.V).max())


def test_q_value_iteration_agrees_with_value_iteration():
    """``max_a q*(s, a) == V*(s)``이고, 탐욕 정책도 일치한다."""
    for env, gamma in CASES:
        q_vi = q_value_iteration(env, gamma=gamma, theta=1e-13)
        v_vi = value_iteration(env, gamma=gamma, theta=1e-13)
        assert np.allclose(q_vi.V, v_vi.V, atol=1e-7), (
            env.name, np.abs(q_vi.V - v_vi.V).max())
        assert policy_matches(q_vi.pi, v_vi.pi), env.name


def test_q_matches_the_one_step_lookahead_from_v():
    """Q-값 반복의 ``q*(s, a)``가 ``V*``로부터 만든 한 걸음 예측과 같다."""
    for env, gamma in CASES:
        q_star = q_value_iteration(env, gamma=gamma, theta=1e-13).Q
        v_star = value_iteration(env, gamma=gamma, theta=1e-13).V
        for s in env.interior_states:
            lookahead = action_values(env, v_star, s, gamma)
            assert np.allclose(q_star[s], lookahead, atol=1e-7), (env.name, s)


def test_q_terminal_rows_stay_zero():
    for env, gamma in CASES:
        for result in (q_value_iteration(env, gamma=gamma),
                       q_policy_evaluation(env, uniform_random_policy(env),
                                               gamma)):
            for s in env.terminal_states:
                assert np.all(result.Q[s] == 0.0), (env.name, s)


def test_q_improvement_needs_no_model():
    """``Q``만 보고 뽑은 정책과, ``V``에서 모델을 훑어 뽑은 정책이 같다."""
    env = gw.main_grid(noise=0.2)
    Q = q_value_iteration(env, gamma=0.9).Q
    from_table = q_greedy_policy(env, Q)
    from_model = greedy_policy(env, value_iteration(env, gamma=0.9).V, 0.9)
    assert policy_matches(from_table, from_model)
    for s in env.interior_states:
        assert np.argmax(from_table[s]) in np.flatnonzero(
            Q[s] >= Q[s].max() - 1e-8)


def test_q_table_is_bigger_and_costs_more_per_sweep():
    """Q 표는 V의 |A|배 크기이고, sweep 횟수는 거의 같다."""
    env = gw.main_grid(noise=0.2)
    v_vi = value_iteration(env, gamma=0.9)
    q_vi = q_value_iteration(env, gamma=0.9)
    assert q_vi.Q.size == v_vi.V.size * len(gw.ACTIONS)
    # 수축률이 같으므로 sweep 횟수는 비슷한 범위에 머문다. 추가 비용은 sweep
    # 하나당 드는 것이지 sweep 횟수에 있지 않다.
    assert abs(q_vi.n_sweeps - v_vi.n_sweeps) <= 3, (q_vi.n_sweeps, v_vi.n_sweeps)


# ----------------------------------------------------------------------
def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {test.__name__}\n      {exc}")
        else:
            print(f"ok    {test.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
