"""모든 실험을 순서대로 실행한다.

    python -m dp.run
    python -m dp.run --gamma 0.99 --noise 0.1

여기에는 실행 순서 말고는 아무것도 없다. 각 파일이 자기 실험과 자기 그림을
소유하고 있고 따로 실행할 수 있으며, 전부 같은 ``--gamma`` / ``--noise``를
받는다::

    python -m dp.gridworld           MDP 정의와 전이표                 fig01
    python -m dp.exact               전수 탐색으로 구한 진짜 답        fig02
    python -m dp.value_iteration     할인율, sweep, 노이즈             fig03-06
    python -m dp.q_value_iteration   V와 Q, sweep, 노이즈를 Q로        fig07-09
    python -m dp.policy_iteration    정책 평가, 그리고 PI의 비용       fig10-12

정확성 검증은 따로다. ``python -m dp.test_dp``.
"""

from __future__ import annotations

from . import exact, gridworld, policy_iteration, q_value_iteration, viz
from . import value_iteration
from .value_iteration import GAMMA_VI, NOISE_VI


def main(gamma: float = GAMMA_VI, noise: float = NOISE_VI) -> None:
    gridworld.main(noise=noise)
    # exact는 결정적 격자에서 돈다. 노이즈를 보려면 직접 실행한다:
    #     python -m dp.exact --noise 0.2
    exact.main(gamma=gamma)
    value_iteration.main(gamma=gamma, noise=noise)
    q_value_iteration.main(gamma=gamma, noise=noise)
    policy_iteration.main(gamma=gamma, noise=noise)

    viz.banner("figures written")
    for path in sorted(viz.FIGURES.glob("*.png")):
        print(f"  {path.parent.name}/{path.name}")


if __name__ == "__main__":
    main(**vars(viz.demo_args(gamma=GAMMA_VI, noise=NOISE_VI)))
