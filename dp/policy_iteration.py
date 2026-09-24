"""정책 평가와 정책 반복.

:func:`policy_evaluation`
    정책이 주어지면 sweep을 돌려 ``v_pi``를 구한다. 같은 답을 반복 없이 구하는
    것은 :func:`dp.exact.exact_policy_values`.

:func:`policy_iteration`
    평가하고, 그 ``V``에 대해 탐욕적으로 정책을 갈아끼우고, 정책이 더 바뀌지
    않을 때까지 반복한다.

두 함수 모두 정책을 결정적 ``pi(s)``가 아니라 확률 배열로 다루고, 종결 상태는
갱신하지 않는다 (:mod:`dp.gridworld` 참고).

    python -m dp.policy_iteration      fig10-12
"""

from __future__ import annotations

import numpy as np

from .gridworld import (
    ACTIONS,
    N_ACTIONS,
    DPResult,
    GridWorld,
    action_values,
    greedy_policy,
    q_table_from_v,
)


def policy_evaluation(env: GridWorld, pi: np.ndarray, gamma: float,
                      theta: float = 1e-10, in_place: bool = True,
                      max_sweeps: int = 100_000,
                      snapshots_at: tuple = (),
                      V0: np.ndarray | None = None) -> DPResult:
    """고정된 정책 ``pi``의 가치 함수 ``v_pi``를 ``theta``까지 반복해 구한다.

    ``pi``는 칸마다 행동 번호 하나인 ``(n_states,)`` 배열이다. 정책 반복이
    쓰는 정책은 결정론적이므로 수도코드의 ``p(s',r|s,pi(s))``를 그대로 돈다.
    행동 확률로 섞인 정책은 :func:`expected_policy_evaluation`이 맡는다.

    수도코드 (Sutton & Barto 4.1절)::

        Input pi, the policy to be evaluated
        Parameter: a small threshold theta > 0
        Initialise V(s) arbitrarily for all s, except V(terminal) = 0
        Loop:
            Delta <- 0
            Loop for each s in S:
                v <- V(s)
                V(s) <- sum_a pi(a|s) sum_{s',r} p(s',r|s,a)[r + gamma V(s')]
                Delta <- max(Delta, |v - V(s)|)
        until Delta < theta

    Parameters
    ----------
    in_place
        ``True``면 같은 배열을 읽고 쓴다. ``False``면 sweep 전체를 이전 ``V``
        로부터 계산한다(동기식).
    snapshots_at
        ``V``의 사본을 남길 sweep 번호. 0은 초기값.
    V0
        시작 ``V``. :func:`policy_iteration`이 ``warm_start``일 때 이전 정책의
        값을 여기로 넘긴다.
    """
    V = np.zeros(env.n_states) if V0 is None else np.asarray(V0, float).copy()
    V[list(env.terminal_states)] = 0.0
    snapshots = {0: V.copy()} if 0 in snapshots_at else {}
    deltas = []
    sweep = 0

    while sweep < max_sweeps:
        sweep += 1
        delta = 0.0
        # 읽는 쪽은 V_k, 쓰는 쪽은 V. in_place 면 같은 배열이라
        # 이번 sweep에서 갱신된 이웃 값을 바로 읽는다.
        V_k = V if in_place else V.copy()

        for s in env.interior_states:
            v_old = V[s]
            V[s] = sum(p * (r + gamma * V_k[s2])
                       for p, s2, r, _ in env.P[s][pi[s]])
            delta = max(delta, abs(v_old - V[s]))

        deltas.append(delta)
        if sweep in snapshots_at:
            snapshots[sweep] = V.copy()
        if delta < theta:
            break

    # 수렴 이후의 sweep을 요청받았으면 최종 V로 채운다.
    for k in snapshots_at:
        snapshots.setdefault(k, V.copy())

    return DPResult(table=V, V=V, pi=np.asarray(pi).copy(), deltas=deltas,
                    snapshots=snapshots, n_sweeps=sweep)


def expected_policy_evaluation(env: GridWorld, pi: np.ndarray, gamma: float,
                               theta: float = 1e-10, in_place: bool = True,
                               max_sweeps: int = 100_000,
                               snapshots_at: tuple = (),
                               V0: np.ndarray | None = None) -> DPResult:
    """행동 확률로 섞인 정책의 ``v_pi``.

    ``pi``가 ``(n_states, n_actions)`` 확률 배열이라 갱신식이 한 겹 더 있다::

        V(s) <- sum_a pi(a|s) sum_{s',r} p(s',r|s,a)[r + gamma V(s')]

    정책 반복은 이걸 쓰지 않는다. 그쪽 정책은 결정론적이다. 균등 무작위 정책을
    평가해 보일 때와 교과서 그림 4.1을 대조할 때만 쓴다.
    """
    V = np.zeros(env.n_states) if V0 is None else np.asarray(V0, float).copy()
    V[list(env.terminal_states)] = 0.0
    snapshots = {0: V.copy()} if 0 in snapshots_at else {}
    deltas = []
    sweep = 0

    while sweep < max_sweeps:
        sweep += 1
        delta = 0.0
        # 읽는 쪽은 V_k, 쓰는 쪽은 V. in_place 면 같은 배열이라
        # 이번 sweep에서 갱신된 이웃 값을 바로 읽는다.
        V_k = V if in_place else V.copy()

        for s in env.interior_states:
            v_old = V[s]
            V[s] = float(np.dot(pi[s], action_values(env, V_k, s, gamma)))
            delta = max(delta, abs(v_old - V[s]))

        deltas.append(delta)
        if sweep in snapshots_at:
            snapshots[sweep] = V.copy()
        if delta < theta:
            break

    for k in snapshots_at:
        snapshots.setdefault(k, V.copy())

    return DPResult(table=V, V=V, pi=pi.copy(), deltas=deltas,
                    snapshots=snapshots, n_sweeps=sweep)


def policy_iteration(env: GridWorld, gamma: float, theta: float = 1e-10,
                     in_place: bool = True, max_iterations: int = 1_000,
                     max_eval_sweeps: int = 100_000, warm_start: bool = True,
                     pi0: np.ndarray | None = None,
                     tol: float = 1e-8) -> DPResult:
    """평가와 탐욕 개선을 정책이 더 바뀌지 않을 때까지 번갈아 돌린다.

    수도코드 (Sutton & Barto 4.3절)::

        1. Initialisation
           V(s) and pi(s) arbitrary, for all s in S
        2. Policy Evaluation
           (the loop of section 4.1, run to convergence under pi)
        3. Policy Improvement
           policy-stable <- true
           For each s in S:
               old-action <- pi(s)
               pi(s) <- argmax_a sum_{s',r} p(s',r|s,a)[r + gamma V(s')]
               If old-action != pi(s), then policy-stable <- false
           If policy-stable, stop and return V ~ v*, pi ~ pi*;
           else go to 2

    정책은 수도코드대로 결정론적이다. 칸마다 행동 번호 하나를 들고 다니고,
    평가에도 그대로 넘긴다.

    개선 단계에서 ``argmax``를 그냥 쓰면 똑같이 최적인 두 행동 사이를 영원히
    오갈 수 있다. 정지 조건이 "정책이 안 바뀜"이기 때문이다. 그래서 지금 들고
    있는 행동이 여전히 최선이면 그대로 두고, **더 나은 행동이 있을 때만**
    바꾼다. 같은 판정이면서 진동하지 않는다.

    돌려주는 ``pi``도 결정론적이다. 칸마다 행동 하나에만 1이 선다. 값 반복이
    돌려주는 것은 동점을 나눠 담은 분포라, 동점이 있는 판에서는 두 정책이
    글자 그대로 같지는 않다. 대신 정책 반복이 고른 행동이 값 반복이 최적이라
    본 행동들 안에 들어 있으면 된다(``test_dp.py``가 그렇게 대조한다).

    Parameters
    ----------
    max_eval_sweeps
        평가 단계를 이만큼의 sweep에서 끊는다. 1로 두면 갱신식이 값 반복의
        갱신식과 같아진다. 단 정지 조건은 여전히 "정책이 안 바뀜"이라,
        ``V``가 ``v*``에서 먼 채로 멈출 수 있다. 실제로 ``noise=0.2``,
        ``max_eval_sweeps=1``에서 준최적 정책을 돌려준다
        (``test_truncating_evaluation_too_hard_stops_early``).
    warm_start
        각 평가를 0이 아니라 이전 정책의 값에서 시작한다. 이 격자에서는
        ``noise=0.2``면 sweep을 아끼고 ``noise=0``이면 오히려 손해다.
        ``python -m dp.policy_iteration``이 둘 다 찍는다.
    """
    # 칸마다 행동 번호 하나. 시작 정책은 아무거나여도 되므로 전부 같은 쪽.
    actions = np.zeros(env.n_states, dtype=int)
    if pi0 is not None:
        pi0 = np.asarray(pi0)
        actions = pi0.argmax(axis=1) if pi0.ndim == 2 else pi0.astype(int).copy()

    V = np.zeros(env.n_states)
    deltas, eval_sweeps = [], []
    total_sweeps = 0

    for iteration in range(1, max_iterations + 1):
        # --- 2. 평가 -----------------------------------------------
        evaluation = policy_evaluation(env, actions, gamma, theta=theta,
                                       in_place=in_place,
                                       max_sweeps=max_eval_sweeps,
                                       V0=V if warm_start else None)
        V = evaluation.V
        deltas.extend(evaluation.deltas)
        eval_sweeps.append(evaluation.n_sweeps)
        total_sweeps += evaluation.n_sweeps

        # --- 3. 개선 -----------------------------------------------
        # 지금 행동이 여전히 최선이면 그대로 둔다. 동점에서 진동하지 않는다.
        q = q_table_from_v(env, V, gamma)
        new_actions = actions.copy()
        for s in env.interior_states:
            if q[s, actions[s]] < q[s].max() - tol:
                new_actions[s] = int(q[s].argmax())
        policy_stable = np.array_equal(actions, new_actions)
        actions = new_actions
        if policy_stable:
            break

    # 결정론적 정책을 그대로 돌려준다. 칸마다 행동 하나에 1.
    pi = np.zeros((env.n_states, N_ACTIONS))
    for s in env.interior_states:
        pi[s, actions[s]] = 1.0
    return DPResult(table=V, V=V, pi=pi, deltas=deltas,
                    n_sweeps=total_sweeps, n_iterations=iteration,
                    eval_sweeps=eval_sweeps)


# ----------------------------------------------------------------------
# python -m dp.policy_iteration
# ----------------------------------------------------------------------
def _evaluation(viz, gamma, noise, step_reward) -> None:
    """균등 무작위 정책을 평가하면서 sweep별 스냅샷을 찍는다. fig10."""
    from . import gridworld as gw

    viz.banner("5a. Evaluating the uniform random policy")
    env = gw.main_grid(noise=noise, step_reward=step_reward)
    snapshots_at = (0, 1, 2, 3, 10)
    ev = expected_policy_evaluation(env, gw.uniform_random_policy(env), gamma=gamma,
                           theta=1e-12, snapshots_at=snapshots_at)
    print(f"converged in {ev.n_sweeps} sweeps")
    print(gw.render_values(env, ev.V))
    print(f"\nends at: {gw.describe_outcome(env, ev.pi)}")

    panels = [(f"k = {k}", ev.snapshots[k], None) for k in snapshots_at]
    panels.append(("k = inf  (v_pi)", ev.V, gw.greedy_policy(env, ev.V, gamma)))
    viz.figure_panels(
        env, panels,
        path := viz.figure_path("10_policy_evaluation", g=gamma, n=noise,
                                s=step_reward),
        ncols=3, value_fmt="{:.1f}",
        params=viz.params_text(gamma=gamma, noise=noise,
                               step_reward=step_reward))
    viz.wrote(path)


def _cost(viz, gamma, noise, step_reward) -> None:
    """변형별 sweep 수를 재고 수렴 곡선을 그린다. fig11, fig12."""
    from . import gridworld as gw
    from .q_value_iteration import q_value_iteration
    from .value_iteration import value_iteration

    viz.banner("5b. What policy iteration costs")
    noisy = gw.main_grid(noise=noise, step_reward=step_reward)
    theta = 1e-10
    runs = {
        "value iteration (in-place)":
            value_iteration(noisy, gamma=gamma, theta=theta, in_place=True),
        "value iteration (synchronous)":
            value_iteration(noisy, gamma=gamma, theta=theta, in_place=False),
        "modified policy iteration (5 sweeps)":
            policy_iteration(noisy, gamma=gamma, theta=theta,
                             max_eval_sweeps=5),
        "Q-value iteration (in-place)":
            q_value_iteration(noisy, gamma=gamma, theta=theta),
    }
    # 이 둘은 sweep이 너무 길어 곡선으로는 안 그리고 총계만 센다.
    extra = {
        "policy iteration (full evaluation)":
            policy_iteration(noisy, gamma=gamma, theta=theta),
        "policy iteration (from V = 0)":
            policy_iteration(noisy, gamma=gamma, theta=theta,
                             warm_start=False),
    }
    pit = extra["policy iteration (full evaluation)"]
    vi = runs["value iteration (in-place)"]

    print(f"{'algorithm':<38}{'sweeps':>8}{'outer iters':>14}")
    for label, result in {**runs, **extra}.items():
        print(f"{label:<38}{result.n_sweeps:>8}"
              f"{str(result.n_iterations or '-'):>14}")
    print(f"\nsame answer: V {np.allclose(vi.V, pit.V)}, "
          f"pi {gw.policy_matches(vi.pi, pit.pi)}")
    print(f"evaluation sweeps per iteration: {pit.eval_sweeps}")

    viz.figure_panels(
        noisy,
        [(f"policy iteration   {pit.n_sweeps} sweeps", pit.V, pit.pi),
         (f"value iteration   {vi.n_sweeps} sweeps", vi.V, vi.pi)],
        compare_path := viz.figure_path("11_pi_vs_vi", g=gamma, n=noise,
                                        s=step_reward),
        ncols=2, params=viz.params_text(gamma=gamma, noise=noise,
                                        step_reward=step_reward))

    # warm_start의 방향이 격자에 따라 갈리므로 양쪽을 다 측정한다.
    cold = extra["policy iteration (from V = 0)"]
    plain = gw.main_grid(step_reward=step_reward)
    plain_warm = policy_iteration(plain, gamma=gamma, theta=theta)
    plain_cold = policy_iteration(plain, gamma=gamma, theta=theta,
                                  warm_start=False)
    v_rand = expected_policy_evaluation(plain, gw.uniform_random_policy(plain),
                               gamma).V
    v_star = value_iteration(plain, gamma=gamma).V

    print("\nwarm start (each evaluation begins at the previous V):")
    print(f"  noise = {noise:g}:  warm {pit.n_sweeps} vs cold "
          f"{cold.n_sweeps}  -> warm wins by {cold.n_sweeps - pit.n_sweeps}")
    print(f"  noise = 0  :  warm {plain_warm.n_sweeps} vs cold "
          f"{plain_cold.n_sweeps}  -> cold wins by "
          f"{plain_warm.n_sweeps - plain_cold.n_sweeps}")
    print(f"  at noise = 0, v_random bottoms out at {v_rand.min():.1f} and v* "
          f"tops out at {v_star.max():.1f},\n  so the previous V sits "
          f"{np.abs(v_rand - v_star).max():.0f} from v* against "
          f"{np.abs(v_star).max():.0f} for zeros.")

    totals = {label: r.n_sweeps for label, r in {**runs, **extra}.items()}
    viz.figure_convergence(
        {label: runs[label].deltas for label in runs}, totals,
        curve_path := viz.figure_path("12_convergence", g=gamma, n=noise,
                                      s=step_reward),
        subtitle=viz.params_text(gamma=gamma, noise=noise,
                                 step_reward=step_reward, theta=theta),
        theta=theta, xlim=45)
    viz.wrote(compare_path)
    viz.wrote(curve_path)


def main(gamma: float | None = None, noise: float | None = None,
         step_reward: float = 0.0) -> None:
    from . import viz
    from .value_iteration import GAMMA_VI, NOISE_VI

    gamma = GAMMA_VI if gamma is None else gamma
    noise = NOISE_VI if noise is None else noise
    viz.begin_demo("5. Policy iteration")
    _evaluation(viz, gamma, noise, step_reward)
    _cost(viz, gamma, noise, step_reward)


if __name__ == "__main__":
    from . import viz
    from .value_iteration import GAMMA_VI, NOISE_VI
    main(**vars(viz.demo_args(gamma=GAMMA_VI, noise=NOISE_VI,
                              step_reward=0.0)))
