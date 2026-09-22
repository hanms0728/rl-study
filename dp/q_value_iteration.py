"""Q-값 반복. ``V(s)`` 대신 ``Q(s, a)`` 표 위에서 도는 값 반복.

루프도 정지 조건도 ``value_iteration.py``와 같고, ``max``의 위치만 다르다::

    value_iteration.py    V(s)   <- max_a  sum_{s',r} p(s',r|s,a)[r + gamma V(s')]
    q_value_iteration.py  Q(s,a) <-        sum_{s',r} p(s',r|s,a)[r + gamma max_a' Q(s',a')]

그래서 정책을 꺼낼 때 모델이 필요 없다. ``V``는 :func:`gridworld.greedy_policy`
가 모델을 한 번 훑어야 하지만, ``Q``는 ``argmax_a Q(s, a)``로 끝난다. 대신 표가
상태당 ``|A|``배로 커져서 sweep 하나가 그만큼 비싸다.

    python -m dp.q_value_iteration      fig07-09
"""

from __future__ import annotations

import numpy as np

from .gridworld import ACTIONS, N_ACTIONS, DPResult, GridWorld, greedy_from_q


def backup(env: GridWorld, Q: np.ndarray, s: int, a: int, gamma: float,
           pi: np.ndarray | None = None) -> float:
    """``(s, a)`` 항목 하나를 한 번 갱신한 값.

    ``pi``를 생략하면 최적 backup:

    .. math::
        q_*(s, a) = \\sum_{s', r} p(s', r \\mid s, a)
            \\Big[\\, r + \\gamma \\max_{a'} q_*(s', a') \\,\\Big]

    ``pi``를 주면 ``pi``를 평가하는 기대 backup:

    .. math::
        q_\\pi(s, a) = \\sum_{s', r} p(s', r \\mid s, a)
            \\Big[\\, r + \\gamma \\sum_{a'} \\pi(a' \\mid s')\\, q_\\pi(s', a')
            \\,\\Big]
    """
    total = 0.0
    for prob, s_next, reward, _ in env.P[s][a]:
        if pi is None:
            bootstrap = float(Q[s_next].max())
        else:
            bootstrap = float(np.dot(pi[s_next], Q[s_next]))
        total += prob * (reward + gamma * bootstrap)
    return total


def greedy_policy(env: GridWorld, Q: np.ndarray) -> np.ndarray:
    """``Q``에 대한 탐욕 정책. 모델을 보지 않고 argmax만 한다."""
    return greedy_from_q(env, Q)


def state_values(env: GridWorld, Q: np.ndarray,
                 pi: np.ndarray | None = None) -> np.ndarray:
    """``Q``를 ``V``로 접는다. ``max_a Q(s,a)``, ``pi``를 주면 그 기댓값."""
    V = np.zeros(env.n_states)
    for s in env.interior_states:
        V[s] = Q[s].max() if pi is None else float(np.dot(pi[s], Q[s]))
    return V


def q_value_iteration(env: GridWorld, gamma: float, theta: float = 1e-10,
                      in_place: bool = True, max_sweeps: int = 100_000,
                      snapshots_at: tuple = ()) -> DPResult:
    """``q_*``를 구하려고 Bellman 최적 방정식을 고정점까지 반복한다.

    수도코드. ``value_iteration``과 나란히 보도록 줄 번호를 맞췄다::

        Parameter: a small threshold theta > 0
        (P1) Initialise Q(s,a) arbitrarily, except Q(terminal, .) = 0
        (P2) Loop:
        (P3)     Delta <- 0
        (P4)     Loop for each s in S, each a in A:
        (P5)         q <- Q(s,a)
        (P6)         Q(s,a) <- sum_{s',r} p(s',r|s,a)[r + gamma max_a' Q(s',a')]
        (P7)         Delta <- max(Delta, |q - Q(s,a)|)
        (P8) until Delta < theta
        (P9) Output pi(s) = argmax_a Q(s,a)

    다른 줄은 (P4), (P6), (P9)뿐이다. sweep이 ``|S| x |A|``개 항목을 돌고,
    ``max``가 안쪽으로 들어가며, 정책이 표에서 바로 나온다. 종결 행은 갱신하지
    않으므로 ``Q(종결, a) = 0``으로 남는다.
    """
    Q = np.zeros((env.n_states, N_ACTIONS))                      # (P1)
    snapshots = {0: Q.copy()} if 0 in snapshots_at else {}
    deltas = []
    sweep = 0

    while sweep < max_sweeps:                                    # (P2)
        sweep += 1
        delta = 0.0                                              # (P3)
        source = Q if in_place else Q.copy()

        for s in env.interior_states:                            # (P4)
            for a in ACTIONS:
                q_old = Q[s, a]                                  # (P5)
                Q[s, a] = backup(env, source, s, a, gamma)       # (P6)
                delta = max(delta, abs(q_old - Q[s, a]))         # (P7)

        deltas.append(delta)
        if sweep in snapshots_at:
            snapshots[sweep] = Q.copy()
        if delta < theta:                                        # (P8)
            break

    for k in snapshots_at:
        snapshots.setdefault(k, Q.copy())

    pi = greedy_policy(env, Q)                                   # (P9)
    return DPResult(table=Q, V=state_values(env, Q), pi=pi, deltas=deltas,
                    snapshots=snapshots, n_sweeps=sweep)


def q_policy_evaluation(env: GridWorld, pi: np.ndarray, gamma: float,
                        theta: float = 1e-10, in_place: bool = True,
                        max_sweeps: int = 100_000) -> DPResult:
    """고정된 정책에 대한 ``q_pi``. ``test_dp.py``의 교차 검증에만 쓴다.

    ``sum_a pi(a|s) q_pi(s,a)``가 ``v_pi(s)``와 같아야 한다.
    """
    Q = np.zeros((env.n_states, N_ACTIONS))
    deltas = []
    sweep = 0

    while sweep < max_sweeps:
        sweep += 1
        delta = 0.0
        source = Q if in_place else Q.copy()
        for s in env.interior_states:
            for a in ACTIONS:
                q_old = Q[s, a]
                Q[s, a] = backup(env, source, s, a, gamma, pi)
                delta = max(delta, abs(q_old - Q[s, a]))
        deltas.append(delta)
        if delta < theta:
            break

    return DPResult(table=Q, V=state_values(env, Q, pi), pi=pi.copy(),
                    deltas=deltas, n_sweeps=sweep)


# ----------------------------------------------------------------------
# python -m dp.q_value_iteration
# ----------------------------------------------------------------------
def main(gamma: float | None = None, noise: float | None = None) -> None:
    from . import gridworld as gw
    from . import viz
    from .value_iteration import GAMMA_VI, NOISE_VI, value_iteration

    gamma = GAMMA_VI if gamma is None else gamma
    noise = NOISE_VI if noise is None else noise
    viz.begin_demo(f"4. Q-value iteration  "
                   f"(gamma = {gamma:g}, noise = {noise:g})")
    noisy = gw.main_grid(noise=noise)
    v_vi = value_iteration(noisy, gamma=gamma)
    q_vi = q_value_iteration(noisy, gamma=gamma)

    print(f"value iteration:    {v_vi.n_sweeps:>3} sweeps, "
          f"{len(noisy.interior_states)} values")
    print(f"Q-value iteration:  {q_vi.n_sweeps:>3} sweeps, "
          f"{len(noisy.interior_states) * len(gw.ACTIONS)} values "
          f"({len(gw.ACTIONS)} per state)")
    print(f"\n  max_a q*(s,a) == V*(s)   {np.allclose(q_vi.V, v_vi.V)}")
    print(f"  same greedy policy       {gw.policy_matches(q_vi.pi, v_vi.pi)}")

    s12 = noisy.to_s((3, 1))
    print(f"\nq*({noisy.state_names[s12]}, a):")
    for a in gw.ACTIONS:
        mark = "  <- greedy" if q_vi.Q[s12, a] >= q_vi.Q[s12].max() - 1e-9 else ""
        print(f"  {gw.ACTION_NAMES[a]:>5}  {q_vi.Q[s12, a]:8.4f}{mark}")

    print("\nPicking the best action from V needs a one-step lookahead "
          "through the model;\nfrom Q it is argmax_a Q(s,a).")

    viz.figure_v_and_q(
        noisy, v_vi.V, v_vi.pi, q_vi.Q, "The same solution in two tables",
        viz.FIGURES / "fig07_v_and_q.png",
        subtitle=f"gamma = {gamma:g}, noise = {noise:g}.")

    snapshots_at = (1, 2, 3, 5)
    swept = q_value_iteration(noisy, gamma=gamma, in_place=False,
                              snapshots_at=snapshots_at)
    panels = [(f"k = {k}", swept.snapshots[k]) for k in snapshots_at]
    panels.append(("converged", swept.Q))
    viz.figure_q_panels(noisy, panels, "Q-value iteration, sweep by sweep",
                        viz.FIGURES / "fig08_q_sweeps.png", ncols=3)

    # value_iteration.py의 노이즈 실험과 같은 설정을 Q로 다시 그린다.
    viz.banner("4b. The noise experiment in action values")
    q_panels = []
    for noise, caption in ((0.0, "nothing to avoid"),
                           (0.05, "takes the risk"),
                           (0.2, "refuses the risk")):
        e = gw.main_grid(noise=noise)
        q_panels.append((f"noise = {noise:g}\n{caption}",
                         q_value_iteration(e, gamma=gamma).Q))
    viz.figure_q_panels(
        gw.main_grid(), q_panels,
        "The same experiment in action values: the -100 bleeds sideways",
        viz.FIGURES / "fig09_noise_q.png", ncols=3)

    print("With noise = 0 the -100 sits on one wedge per cell; with noise it "
          "spreads to\nthe neighbouring wedges, which can slip into the pit "
          "too.")

    for name in ("fig07_v_and_q", "fig08_q_sweeps", "fig09_noise_q"):
        print(f"wrote {viz.FIGURES.name}/{name}.png")


if __name__ == "__main__":
    from . import viz
    from .value_iteration import GAMMA_VI, NOISE_VI
    main(**vars(viz.demo_args(gamma=GAMMA_VI, noise=NOISE_VI)))
