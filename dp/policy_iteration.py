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
    DPResult,
    GridWorld,
    action_values,
    greedy_policy,
    uniform_random_policy,
)


def policy_evaluation(env: GridWorld, pi: np.ndarray, gamma: float,
                      theta: float = 1e-10, in_place: bool = True,
                      max_sweeps: int = 100_000,
                      snapshots_at: tuple = (),
                      V0: np.ndarray | None = None) -> DPResult:
    """고정된 정책 ``pi``의 가치 함수 ``v_pi``를 ``theta``까지 반복해 구한다.

    수도코드 (Sutton & Barto 4.1절)::

        Input pi, the policy to be evaluated
        Parameter: a small threshold theta > 0
        (P1) Initialise V(s) arbitrarily for all s, except V(terminal) = 0
        (P2) Loop:
        (P3)     Delta <- 0
        (P4)     Loop for each s in S:
        (P5)         v <- V(s)
        (P6)         V(s) <- sum_a pi(a|s) sum_{s',r} p(s',r|s,a)[r + gamma V(s')]
        (P7)         Delta <- max(Delta, |v - V(s)|)
        (P8) until Delta < theta

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
    V = np.zeros(env.n_states) if V0 is None else np.asarray(V0, float).copy()  # (P1)
    V[list(env.terminal_states)] = 0.0  # (P1)
    snapshots = {0: V.copy()} if 0 in snapshots_at else {}
    deltas = []
    sweep = 0

    while sweep < max_sweeps:                                    # (P2)
        sweep += 1
        delta = 0.0                                              # (P3)
        source = V if in_place else V.copy()

        for s in env.interior_states:                            # (P4)
            v_old = V[s]                                         # (P5)
            q = action_values(env, source, s, gamma)
            V[s] = float(np.dot(pi[s], q))                       # (P6)
            delta = max(delta, abs(v_old - V[s]))                # (P7)

        deltas.append(delta)
        if sweep in snapshots_at:
            snapshots[sweep] = V.copy()
        if delta < theta:                                        # (P8)
            break

    # 수렴 이후의 sweep을 요청받았으면 최종 V로 채운다.
    for k in snapshots_at:
        snapshots.setdefault(k, V.copy())

    return DPResult(table=V, V=V, pi=pi.copy(), deltas=deltas,
                    snapshots=snapshots, n_sweeps=sweep)


def policy_iteration(env: GridWorld, gamma: float, theta: float = 1e-10,
                     in_place: bool = True, max_iterations: int = 1_000,
                     max_eval_sweeps: int = 100_000, warm_start: bool = True,
                     pi0: np.ndarray | None = None) -> DPResult:
    """평가와 탐욕 개선을 정책이 더 바뀌지 않을 때까지 번갈아 돌린다.

    수도코드 (Sutton & Barto 4.3절)::

        (P1) 1. Initialisation
        (P2)    V(s) and pi(s) arbitrary, for all s in S
        (P3) 2. Policy Evaluation
        (P4)    (the loop of section 4.1, run to convergence under pi)
        (P5) 3. Policy Improvement
        (P6)    policy-stable <- true
        (P7)    For each s in S:
        (P8)        old-action <- pi(s)
        (P9)        pi(s) <- argmax_a sum_{s',r} p(s',r|s,a)[r + gamma V(s')]
        (P10)       If old-action != pi(s), then policy-stable <- false
        (P11)   If policy-stable, stop and return V ~ v*, pi ~ pi*;
                else go to 2

    (P8)-(P10)을 행동 하나가 아니라 행동 분포 전체로 비교한다. 동점을 균등하게
    나누므로 같은 판정이면서, 똑같이 탐욕적인 두 행동 사이를 오가며 끝나지 않는
    일이 생기지 않는다.

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
    pi = uniform_random_policy(env) if pi0 is None else pi0.copy()  # (P1)(P2)
    V = np.zeros(env.n_states)
    deltas, eval_sweeps = [], []
    total_sweeps = 0

    for iteration in range(1, max_iterations + 1):
        # --- 2. 평가 ----------------------------------------------- (P3)(P4)
        evaluation = policy_evaluation(env, pi, gamma, theta=theta,
                                       in_place=in_place,
                                       max_sweeps=max_eval_sweeps,
                                       V0=V if warm_start else None)
        V = evaluation.V
        deltas.extend(evaluation.deltas)
        eval_sweeps.append(evaluation.n_sweeps)
        total_sweeps += evaluation.n_sweeps

        # --- 3. 개선 ----------------------------------------------- (P5)(P6)
        pi_new = greedy_policy(env, V, gamma)                        # (P7)-(P9)
        policy_stable = np.allclose(pi, pi_new)                      # (P10)
        pi = pi_new
        if policy_stable:                                            # (P11)
            break

    return DPResult(table=V, V=V, pi=pi, deltas=deltas,
                    n_sweeps=total_sweeps, n_iterations=iteration,
                    eval_sweeps=eval_sweeps)


# ----------------------------------------------------------------------
# python -m dp.policy_iteration
# ----------------------------------------------------------------------
def _evaluation(viz, gamma, noise) -> None:
    """균등 무작위 정책을 평가하면서 sweep별 스냅샷을 찍는다. fig10."""
    from . import gridworld as gw

    viz.banner("5a. Evaluating the uniform random policy")
    env = gw.main_grid(noise=noise)
    snapshots_at = (0, 1, 2, 3, 10)
    ev = policy_evaluation(env, gw.uniform_random_policy(env), gamma=gamma,
                           theta=1e-12, snapshots_at=snapshots_at)
    print(f"converged in {ev.n_sweeps} sweeps")
    print(gw.render_values(env, ev.V))
    print(f"\nends at: {gw.describe_outcome(env, ev.pi)}")

    panels = [(f"k = {k}", ev.snapshots[k], None) for k in snapshots_at]
    panels.append(("k = inf  (v_pi)", ev.V, gw.greedy_policy(env, ev.V, gamma)))
    viz.figure_panels(
        env, panels,
        "Iterative policy evaluation of the uniform random policy",
        viz.FIGURES / "fig10_policy_evaluation.png", ncols=3,
        value_fmt="{:.1f}")


def _cost(viz, gamma, noise) -> None:
    """변형별 sweep 수를 재고 수렴 곡선을 그린다. fig11, fig12."""
    from . import gridworld as gw
    from .q_value_iteration import q_value_iteration
    from .value_iteration import value_iteration

    viz.banner("5b. What policy iteration costs")
    noisy = gw.main_grid(noise=noise)
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
        [(f"policy iteration\n{pit.n_iterations} iterations / "
          f"{pit.n_sweeps} sweeps", pit.V, pit.pi),
         (f"value iteration\n{vi.n_sweeps} sweeps", vi.V, vi.pi)],
        "Both algorithms reach the same fixed point",
        viz.FIGURES / "fig11_pi_vs_vi.png", ncols=2)

    # warm_start의 방향이 격자에 따라 갈리므로 양쪽을 다 측정한다.
    cold = extra["policy iteration (from V = 0)"]
    plain = gw.main_grid()
    plain_warm = policy_iteration(plain, gamma=gamma, theta=theta)
    plain_cold = policy_iteration(plain, gamma=gamma, theta=theta,
                                  warm_start=False)
    v_rand = policy_evaluation(plain, gw.uniform_random_policy(plain),
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
        "Convergence",
        viz.FIGURES / "fig12_convergence.png",
        subtitle=(f"gamma = {gamma:g}, noise = {noise:g}, theta = {theta:g}. "
                  "Each spike in modified policy iteration is a "
                  "policy-improvement step restarting the error."),
        theta=theta, xlim=45)


def main(gamma: float | None = None, noise: float | None = None) -> None:
    from . import viz
    from .value_iteration import GAMMA_VI, NOISE_VI

    gamma = GAMMA_VI if gamma is None else gamma
    noise = NOISE_VI if noise is None else noise
    viz.begin_demo("5. Policy iteration")
    _evaluation(viz, gamma, noise)
    _cost(viz, gamma, noise)
    for name in ("fig10_policy_evaluation", "fig11_pi_vs_vi",
                 "fig12_convergence"):
        print(f"wrote {viz.FIGURES.name}/{name}.png")


if __name__ == "__main__":
    from . import viz
    from .value_iteration import GAMMA_VI, NOISE_VI
    main(**vars(viz.demo_args(gamma=GAMMA_VI, noise=NOISE_VI)))
