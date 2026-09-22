"""값 반복.

``policy_iteration.policy_evaluation``과 루프가 같고 한 줄만 다르다::

    정책 평가:  V(s) <- sum_a pi(a|s) * q(s, a)
    값 반복:    V(s) <-           max_a q(s, a)

평균 낼 정책이 없으므로 정책은 수렴한 뒤 한 번만 꺼낸다.

    python -m dp.value_iteration    fig03-06
"""

from __future__ import annotations

import numpy as np

from .gridworld import DPResult, GridWorld, action_values, greedy_policy


def value_iteration(env: GridWorld, gamma: float, theta: float = 1e-10,
                    in_place: bool = True, max_sweeps: int = 100_000,
                    snapshots_at: tuple = ()) -> DPResult:
    """Bellman 최적 방정식을 ``theta``까지 반복해 ``v_*``와 탐욕 정책을 얻는다.

    수도코드 (Sutton & Barto 4.4절)::

        Parameter: a small threshold theta > 0
        (P1) Initialise V(s) arbitrarily for all s, except V(terminal) = 0
        (P2) Loop:
        (P3)     Delta <- 0
        (P4)     Loop for each s in S:
        (P5)         v <- V(s)
        (P6)         V(s) <- max_a sum_{s',r} p(s',r|s,a)[r + gamma V(s')]
        (P7)         Delta <- max(Delta, |v - V(s)|)
        (P8) until Delta < theta
        (P9) Output a deterministic policy pi with
                 pi(s) = argmax_a sum_{s',r} p(s',r|s,a)[r + gamma V(s')]

    Parameters
    ----------
    in_place
        ``True``면 같은 배열을 읽고 쓰므로 같은 sweep 안에서 갱신된 값을 쓴다.
        ``False``면 sweep 전체를 이전 ``V``로부터 계산한다(동기식).
    snapshots_at
        ``V``의 사본을 남길 sweep 번호. 0은 초기값.
    """
    V = np.zeros(env.n_states)                                   # (P1)
    snapshots = {0: V.copy()} if 0 in snapshots_at else {}
    deltas = []
    sweep = 0

    while sweep < max_sweeps:                                    # (P2)
        sweep += 1
        delta = 0.0                                              # (P3)
        source = V if in_place else V.copy()

        for s in env.interior_states:                            # (P4)
            v_old = V[s]                                         # (P5)
            V[s] = float(action_values(env, source, s, gamma).max())  # (P6)
            delta = max(delta, abs(v_old - V[s]))                # (P7)

        deltas.append(delta)
        if sweep in snapshots_at:
            snapshots[sweep] = V.copy()
        if delta < theta:                                        # (P8)
            break

    # 수렴 이후의 sweep을 요청받았으면 최종 V로 채운다.
    for k in snapshots_at:
        snapshots.setdefault(k, V.copy())

    pi = greedy_policy(env, V, gamma)                            # (P9)
    return DPResult(table=V, V=V, pi=pi, deltas=deltas,
                    snapshots=snapshots, n_sweeps=sweep)


# ----------------------------------------------------------------------
# python -m dp.value_iteration
# ----------------------------------------------------------------------
# 할인율 실험이 쓰는 두 값. 교차점(0.75)의 양쪽에 하나씩. 이 실험 자체가 gamma를
# 훑는 것이므로 --gamma의 영향을 받지 않는다.
GAMMA_NEAR, GAMMA_FAR = 0.1, 0.99
# 나머지 실험의 기본값. q_value_iteration과 policy_iteration도 가져다 쓰고,
# --gamma / --noise로 덮어쓴다.
GAMMA_VI, NOISE_VI = 0.9, 0.2
# 노이즈 실험이 그림으로 보여주는 세 지점. q_value_iteration도 같은 것을 쓴다.
SHOWN_NOISE = ((0.0, "nothing to avoid"),
               (0.05, "takes the risk"),
               (0.2, "refuses the risk"))


def _discount(env, viz, noise, step_reward) -> None:
    """할인율을 바꾸면 최적 정책이 어느 보상으로 향하는지 본다. fig03, fig04."""
    from . import gridworld as gw

    viz.banner("3a. What the discount factor decides")
    s2 = env.to_s((0, 1))
    near = value_iteration(env, gamma=GAMMA_NEAR)
    far = value_iteration(env, gamma=GAMMA_FAR)

    for gamma, result in ((GAMMA_NEAR, near), (GAMMA_FAR, far)):
        print(f"\n-- gamma = {gamma} --")
        print(gw.render_policy(env, result.pi))
        print(f"  V*(s2) = {result.V[s2]:.4f}")
        print(f"  ends at: {gw.describe_outcome(env, result.pi)}")

    # s2에서 동전(+1)은 1칸, 보석(+10)은 9칸이고 보상은 진입 전이에서 나오므로
    # 두 수익은 gamma**0 * 1과 gamma**8 * 10이다. 같아지는 지점이 교차점.
    if step_reward == 0.0:
        crossover = 0.1 ** (1 / 8)
        print(f"\ns2: coin is 1 step away, gem is 9, so the returns compared "
              f"are 1 and\n10 * gamma**8. They cross at 0.1**(1/8) = "
              f"{crossover:.4f}.\n")
    else:
        print(f"\ns2: coin is 1 step away, gem is 9. With step_reward = "
              f"{step_reward:g} each\nreturn carries a step term too, so the "
              f"closed form above no longer holds.\n")

    viz.figure_panels(
        env,
        [(f"\u03b3 = {GAMMA_NEAR:g}", near.V, near.pi),
         (f"\u03b3 = {GAMMA_FAR:g}", far.V, far.pi)],
        discount_path := viz.figure_path("03_discount", n=noise,
                                         s=step_reward),
        ncols=2, shared_scale=False,
        params=viz.params_text(noise=noise, step_reward=step_reward))

    panels = []
    gem_column = list(env.terminal_states).index(env.to_s((1, 3)))
    gammas = (GAMMA_NEAR, 0.70, 0.75, GAMMA_FAR)
    for gamma in gammas:
        vi = value_iteration(env, gamma=gamma)
        # 가치에 임계값을 두는 대신 흡수 확률로 도착지를 판정한다.
        to_gem = gw.outcome_probabilities(env, vi.pi)[s2, gem_column]
        target = "gem +10" if to_gem > 0.5 else "coin +1"
        print(f"gamma = {gamma:<5} V*(s2) = {vi.V[s2]:8.4f}  -> {target}")
        panels.append((f"\u03b3 = {gamma:g}", vi.V, vi.pi))
    viz.figure_panels(
        env, panels,
        sweep_path := viz.figure_path("04_discount_sweep", n=noise,
                                      s=step_reward),
        ncols=4, value_fmt="{:.2f}",
        params=viz.params_text(noise=noise, step_reward=step_reward))
    viz.wrote(discount_path)
    viz.wrote(sweep_path)


def _sweeps(viz, gamma, noise, step_reward):
    """sweep마다 V의 스냅샷을 찍어 값이 퍼지는 과정을 본다. fig05."""
    from . import gridworld as gw

    viz.banner(f"3b. Sweep by sweep  (gamma = {gamma:g}, noise = {noise:g}, "
               f"step = {step_reward:g})")
    noisy = gw.main_grid(noise=noise, step_reward=step_reward)
    snapshots_at = (0, 1, 2, 3, 5)
    swept = value_iteration(noisy, gamma=gamma, in_place=False,
                            snapshots_at=snapshots_at)
    vi = value_iteration(noisy, gamma=gamma)

    print(f"converged in {vi.n_sweeps} sweeps")
    print(gw.render_values(noisy, vi.V))
    print()
    print(gw.render_policy(noisy, vi.pi))
    print(f"\nends at: {gw.describe_outcome(noisy, vi.pi)}")
    print("sweep 1 reaches only the cells next to a reward; each further "
          "sweep moves the\ninformation one more cell outward.")

    panels = [(f"k = {k}", swept.snapshots[k], None) for k in snapshots_at]
    panels.append(("converged", swept.V, swept.pi))
    viz.figure_panels(
        noisy, panels,
        path := viz.figure_path("05_vi_sweeps", g=gamma, n=noise,
                                s=step_reward), ncols=3,
        params=viz.params_text(gamma=gamma, noise=noise,
                               step_reward=step_reward))
    viz.wrote(path)
    return vi


def _noise(env, viz, gamma, step_reward) -> None:
    """noise를 훑으면서 두 위험한 칸의 행동과 함정 확률을 표로 찍는다. fig06.

    noise를 훑는 것이 이 실험이므로 --noise는 여기에 영향을 주지 않는다.
    """
    from . import gridworld as gw

    viz.banner(f"3c. Sweeping the transition noise  (gamma = {gamma:g}, "
               f"step = {step_reward:g})")
    # 보석으로 가는 길목에서 위험한 행동과 안전한 행동을 함께 가진 두 칸.
    # 안전한 쪽은 벽이나 판 가장자리로 들어가는 행동이라 함정에 닿을 수 없다.
    risky_cells = {(3, 3): "s14", (2, 4): "s10"}
    pits = {(2, 3), (3, 0)}

    def plays(e, pi, rc):
        s = e.to_s(rc)
        allowed = [a for a in gw.ACTIONS if pi[s, a] > 0]
        risky = any(e.to_rc(s2) in pits
                    for a in allowed for p, s2, _, _ in e.P[s][a] if p > 0)
        return "/".join(gw.ACTION_NAMES[a] for a in allowed) + (
            " (risky)" if risky else " (safe)")

    header = "".join(f"{n} plays".rjust(20) for n in risky_cells.values())
    print(f"{'noise':>7}{'V*(s2)':>10}{header}   outcome under p0")
    for noise in (0.0, 0.05, 0.08, 0.1, 0.12, 0.2, 0.5):
        e = gw.main_grid(noise=noise, step_reward=step_reward)
        vi = value_iteration(e, gamma=gamma)
        cells = "".join(plays(e, vi.pi, rc).rjust(20) for rc in risky_cells)
        print(f"{noise:>7.2f}{vi.V[e.to_s((0, 1))]:>10.4f}{cells}   "
              f"{gw.describe_outcome(e, vi.pi)}")

    panels = []
    for noise, caption in SHOWN_NOISE:
        e = gw.main_grid(noise=noise, step_reward=step_reward)
        vi = value_iteration(e, gamma=gamma)
        print(f"\nnoise = {noise:g}   ({caption})")
        print(gw.render_policy(e, vi.pi))
        panels.append((f"noise = {noise:g}", vi.V, vi.pi))

    viz.figure_panels(
        env, panels,
        path := viz.figure_path("06_noise_sweep", g=gamma, s=step_reward),
        ncols=3,
        params=viz.params_text(gamma=gamma, step_reward=step_reward))
    viz.wrote(path)


def main(gamma: float = GAMMA_VI, noise: float = NOISE_VI,
         step_reward: float = 0.0) -> None:
    from . import gridworld as gw
    from . import viz

    viz.begin_demo("3. Value iteration")
    # 할인율 실험은 결정적 격자에서 돈다.
    _discount(gw.main_grid(step_reward=step_reward), viz, noise=0.0,
              step_reward=step_reward)
    _sweeps(viz, gamma, noise, step_reward)
    _noise(gw.main_grid(step_reward=step_reward), viz, gamma, step_reward)


if __name__ == "__main__":
    from . import viz
    main(**vars(viz.demo_args(gamma=GAMMA_VI, noise=NOISE_VI,
                              step_reward=0.0)))
