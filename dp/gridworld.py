"""격자 MDP 정의와, 그 위에서 정책을 다루는 데 필요한 것들.

환경은 ``step()`` 함수가 아니라 전이표로 노출된다::

    P[s][a] -> [(확률, 다음 상태, 보상, 종료 여부), ...]

상태 번호는 행 우선이다(``s = r * n_cols + c``). 벽 칸도 번호는 가지지만
``env.states``에서 빠지고 다음 상태로 나타나지 않는다.

파일 아래쪽에는 알고리즘 파일들이 공통으로 쓰는 것이 모여 있다.
:func:`action_values`, 정책 관련 함수들, :class:`DPResult`,
:func:`outcome_probabilities`.

    python -m dp.gridworld      환경을 출력하고 fig01을 쓴다
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# 시계 방향 순서. 행동 ``a``에 수직인 두 방향이 ``(a - 1) % 4``와
# ``(a + 1) % 4``가 되고, _slip_distribution이 그것을 쓴다.
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTIONS = (UP, RIGHT, DOWN, LEFT)
N_ACTIONS = len(ACTIONS)

ACTION_NAMES = {UP: "up", RIGHT: "right", DOWN: "down", LEFT: "left"}
ACTION_ARROWS = {UP: "↑", RIGHT: "→", DOWN: "↓", LEFT: "←"}
ACTION_DELTAS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}


@dataclass(frozen=True)
class GridWorld:
    """직사각형 격자 위의 유한 MDP.

    Parameters
    ----------
    n_rows, n_cols
        격자 크기.
    walls
        점유할 수 없는 칸. 이 칸으로 이동하면 원래 자리에 머문다.
    terminals
        ``{(행, 열): 도착 보상}``. 이 칸에 들어가면 ``step_reward + 도착 보상``
        을 받고 에피소드가 끝난다. 이후 그 칸은 가치 0으로 흡수된다.
    step_reward
        종결 칸으로 들어가는 것을 포함해 모든 전이에서 지급된다.
    noise
        미끄러질 확률. ``noise = p``면 의도한 방향으로 ``1 - p``, 수직인 두
        방향으로 각각 ``p / 2``. 0이면 결정적.
    start
        그림에 로봇을 그리는 위치. 계산에는 쓰이지 않는다.
    """

    n_rows: int
    n_cols: int
    walls: frozenset = frozenset()
    terminals: dict = field(default_factory=dict)
    step_reward: float = 0.0
    noise: float = 0.0
    start: tuple | None = None
    name: str = "gridworld"

    # __post_init__에서 채우는 파생 필드.
    P: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "walls", frozenset(self.walls))
        object.__setattr__(self, "terminals", dict(self.terminals))
        if not 0.0 <= self.noise <= 1.0:
            raise ValueError(f"noise must be in [0, 1], got {self.noise}")
        overlap = self.walls & set(self.terminals)
        if overlap:
            raise ValueError(f"cells are both wall and terminal: {sorted(overlap)}")
        object.__setattr__(self, "P", self._build_transition_table())

    # ------------------------------------------------------------------
    # 인덱스 도우미
    # ------------------------------------------------------------------
    @property
    def n_states(self) -> int:
        return self.n_rows * self.n_cols

    @property
    def shape(self) -> tuple:
        return (self.n_rows, self.n_cols)

    def to_s(self, rc: tuple) -> int:
        r, c = rc
        return r * self.n_cols + c

    def to_rc(self, s: int) -> tuple:
        return divmod(s, self.n_cols)

    def is_wall(self, s: int) -> bool:
        return self.to_rc(s) in self.walls

    def is_terminal(self, s: int) -> bool:
        return self.to_rc(s) in self.terminals

    @property
    def states(self) -> tuple:
        """벽을 제외한 모든 상태."""
        return tuple(s for s in range(self.n_states) if not self.is_wall(s))

    @property
    def interior_states(self) -> tuple:
        """알고리즘이 갱신하는 상태. 벽도 종결도 아닌 칸.

        종결 상태를 빼는 이유는 그 가치가 0으로 고정이라 갱신할 것이 없기
        때문이다. sweep 횟수도 그만큼 정직해진다.
        """
        return tuple(s for s in self.states if not self.is_terminal(s))

    @property
    def terminal_states(self) -> tuple:
        return tuple(s for s in self.states if self.is_terminal(s))

    @property
    def state_names(self) -> dict:
        """``{상태: "s1"}``. 벽이 아닌 칸을 행 우선으로 1부터 번호 매긴다."""
        return {s: f"s{i + 1}" for i, s in enumerate(self.states)}

    @property
    def p0(self) -> dict:
        interior = self.interior_states
        return {s: 1.0 / len(interior) for s in interior}

    # ------------------------------------------------------------------
    # 전이표 구성
    # ------------------------------------------------------------------
    def _slip_distribution(self, a: int) -> tuple:
        """의도한 행동에 대한 ``[(실제 행동, 확률), ...]``."""
        if self.noise == 0.0:
            return ((a, 1.0),)
        half = self.noise / 2.0
        return (
            (a, 1.0 - self.noise),
            ((a - 1) % N_ACTIONS, half),
            ((a + 1) % N_ACTIONS, half),
        )

    def _move(self, rc: tuple, a: int) -> tuple:
        """``rc``에서 행동 ``a``가 도착하는 칸. 막히면 제자리."""
        dr, dc = ACTION_DELTAS[a]
        r, c = rc[0] + dr, rc[1] + dc
        if not (0 <= r < self.n_rows and 0 <= c < self.n_cols):
            return rc  # 판 밖
        if (r, c) in self.walls:
            return rc  # 벽
        return (r, c)

    def _build_transition_table(self) -> dict:
        P = {}
        for s in range(self.n_states):
            if self.is_wall(s):
                continue
            P[s] = {}
            if self.is_terminal(s):
                # 보상 0으로 자기 자신에 흡수. V(종결) = 0이 여기서 나온다.
                for a in ACTIONS:
                    P[s][a] = [(1.0, s, 0.0, True)]
                continue

            rc = self.to_rc(s)
            for a in ACTIONS:
                merged = {}
                for a_eff, prob in self._slip_distribution(a):
                    if prob == 0.0:
                        continue
                    rc2 = self._move(rc, a_eff)
                    # 보상이 도착 칸에만 의존하므로 같은 칸에 떨어지는 항은
                    # 하나로 합친다.
                    merged[rc2] = merged.get(rc2, 0.0) + prob
                P[s][a] = [
                    (
                        prob,
                        self.to_s(rc2),
                        self.step_reward + self.terminals.get(rc2, 0.0),
                        rc2 in self.terminals,
                    )
                    for rc2, prob in merged.items()
                ]
        return P


def main_grid(noise: float = 0.0, step_reward: float = 0.0, **kwargs) -> GridWorld:

    return GridWorld(
        n_rows=4,
        n_cols=5,
        walls={(0, 3), (0, 4), (1, 2), (2, 0), (2, 2)},
        terminals={(0, 2): 1.0, (1, 3): 10.0, (2, 3): -100.0, (3, 0): -100.0},
        step_reward=step_reward,
        noise=noise,
        start=(1, 1),
        name=f"4x5 (noise={noise:g}, step={step_reward:g})",
        **kwargs,
    )


# ----------------------------------------------------------------------
# 텍스트 출력
# ----------------------------------------------------------------------
def render_values(env: GridWorld, V: np.ndarray, width: int = 8, prec: int = 2) -> str:
    """가치 함수를 숫자 격자 문자열로."""
    lines = []
    for r in range(env.n_rows):
        cells = []
        for c in range(env.n_cols):
            if (r, c) in env.walls:
                cells.append("#".rjust(width))
            else:
                cells.append(f"{V[env.to_s((r, c))]:{width}.{prec}f}")
        lines.append(" ".join(cells))
    return "\n".join(lines)


def render_policy(env: GridWorld, pi: np.ndarray, width: int = 8) -> str:
    """정책을 화살표 격자 문자열로. 동점인 행동은 전부 표시한다."""
    lines = []
    for r in range(env.n_rows):
        cells = []
        for c in range(env.n_cols):
            rc = (r, c)
            if rc in env.walls:
                cells.append("#".center(width))
            elif rc in env.terminals:
                cells.append(f"{env.terminals[rc]:+g}".center(width))
            else:
                s = env.to_s(rc)
                arrows = "".join(ACTION_ARROWS[a] for a in ACTIONS if pi[s, a] > 0)
                cells.append(arrows.center(width))
        lines.append(" ".join(cells))
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 알고리즘의 반환값
# ----------------------------------------------------------------------
@dataclass
class DPResult:
    """알고리즘이 돌려주는 것.

    ``table``은 그 알고리즘이 반복한 표다. ``(n_states,)``인 ``V``이거나
    ``(n_states, n_actions)``인 ``Q``. ``V``는 어느 쪽이든 채워지므로 결과를
    서로 비교할 때 표의 종류를 신경 쓰지 않아도 된다.
    """

    table: np.ndarray
    V: np.ndarray
    pi: np.ndarray
    deltas: list = field(default_factory=list)
    """각 sweep 후의 ``max |표의 변화량|``."""
    snapshots: dict = field(default_factory=dict)
    """``{sweep 번호: table.copy()}``. ``snapshots_at``으로 요청한 것만."""
    n_sweeps: int = 0
    """수행한 sweep의 총 횟수."""
    n_iterations: int = 0
    """바깥쪽 반복 횟수. 정책 반복만 채우고 나머지는 0."""
    eval_sweeps: list = field(default_factory=list)
    """평가 단계마다 쓴 sweep 수. 정책 반복 전용."""

    @property
    def Q(self) -> np.ndarray:
        """행동가치 표. ``V``를 반복한 결과면 예외를 던진다."""
        if self.table.ndim != 2:
            raise AttributeError("this result iterated on V, not Q")
        return self.table


# ----------------------------------------------------------------------
# 모델 조회
# ----------------------------------------------------------------------
def action_values(env: GridWorld, V: np.ndarray, s: int, gamma: float) -> np.ndarray:
    """``V``가 주어졌을 때 모든 행동에 대한 ``q(s, a)``.

    .. math::
        q(s, a) = \\sum_{s', r} p(s', r \\mid s, a)\\,[\\,r + \\gamma V(s')\\,]

    상태가치 기반 알고리즘이 ``env.P``를 건드리는 유일한 지점이다. 정책 평가는
    이 값을 ``pi``로 평균 내고, 값 반복은 최댓값을, 정책 개선은 argmax를 취한다.
    """
    q = np.zeros(N_ACTIONS)
    for a in ACTIONS:
        q[a] = sum(prob * (reward + gamma * V[s_next])
                   for prob, s_next, reward, _ in env.P[s][a])
    return q


def q_table_from_v(env: GridWorld, V: np.ndarray, gamma: float) -> np.ndarray:
    """모든 상태에 대해 :func:`action_values`를 한 번에."""
    Q = np.zeros((env.n_states, N_ACTIONS))
    for s in env.interior_states:
        Q[s] = action_values(env, V, s, gamma)
    return Q


# ----------------------------------------------------------------------
# 정책
# ----------------------------------------------------------------------
def uniform_random_policy(env: GridWorld) -> np.ndarray:
    """모든 상태에서 모든 행동이 같은 확률인 정책."""
    return np.full((env.n_states, N_ACTIONS), 1.0 / N_ACTIONS)


def greedy_from_q(env: GridWorld, Q: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """``Q``에 대한 탐욕 정책. 동점은 균등하게 나눈다.

    정책은 결정적 ``pi(s)``가 아니라 ``(n_states, n_actions)`` 확률 배열로
    둔다. 균등 무작위 정책을 그대로 표현할 수 있고, 동점을 임의로 깨지 않아도
    된다. 동점을 임의로 깨면 똑같이 최적인 두 정책 사이에서 정책 반복이 멈추지
    않을 수 있다.
    """
    pi = np.zeros((env.n_states, N_ACTIONS))
    for s in env.states:
        if env.is_terminal(s):
            continue  # 종결 상태에서는 행동하지 않는다
        best = Q[s] >= Q[s].max() - tol
        pi[s] = best / best.sum()
    return pi


def greedy_policy(env: GridWorld, V: np.ndarray, gamma: float) -> np.ndarray:
    """``V``에 대한 탐욕 정책. 모델을 한 번 훑어 ``Q``를 만든 뒤 argmax."""
    return greedy_from_q(env, q_table_from_v(env, V, gamma))


def policy_matches(pi_a: np.ndarray, pi_b: np.ndarray) -> bool:
    """두 정책이 모든 상태에서 같은 행동 집합을 허용하는가."""
    return np.array_equal(pi_a > 0, pi_b > 0)


# ----------------------------------------------------------------------
# 정책의 도착지
# ----------------------------------------------------------------------
def outcome_probabilities(env: GridWorld, pi: np.ndarray) -> np.ndarray:
    """``P[s, t]``: ``s``에서 출발해 종결 상태 ``t``에서 끝날 확률.

    정책이 만드는 마르코프 연쇄의 흡수 확률을 선형 풀이로 구한다. 시뮬레이션은
    하지 않는다. 열 순서는 ``env.terminal_states``를 따른다.
    """
    index = {s: i for i, s in enumerate(env.states)}
    terminals = list(env.terminal_states)
    A = np.eye(len(index))
    B = np.zeros((len(index), len(terminals)))
    for s, i in index.items():
        if env.is_terminal(s):
            B[i, terminals.index(s)] = 1.0
            continue
        for a in ACTIONS:
            if pi[s, a] == 0:
                continue
            for prob, s_next, _, _ in env.P[s][a]:
                A[i, index[s_next]] -= pi[s, a] * prob
    solution = np.linalg.solve(A, B)
    out = np.zeros((env.n_states, len(terminals)))
    for s, i in index.items():
        out[s] = solution[i]
    return out


def outcome_under_p0(env: GridWorld, pi: np.ndarray) -> dict:
    """``env.p0``으로 평균 낸 ``{종결 상태: 확률}``."""
    probabilities = outcome_probabilities(env, pi)
    return {t: sum(w * probabilities[s, j] for s, w in env.p0.items())
            for j, t in enumerate(env.terminal_states)}


def describe_outcome(env: GridWorld, pi: np.ndarray) -> str:
    """:func:`outcome_under_p0`를 한 줄 문자열로. 확률 0.005 이하는 생략."""
    names = env.state_names
    return "  ".join(
        f"{names[t]}({env.terminals[env.to_rc(t)]:+g}) {p:.2f}"
        for t, p in outcome_under_p0(env, pi).items() if p > 5e-3)


# ----------------------------------------------------------------------
# python -m dp.gridworld
# ----------------------------------------------------------------------
def main(noise: float = 0.2, step_reward: float = 0.0) -> None:
    from . import viz  # viz가 이 모듈을 import 하므로 여기서 불러온다

    viz.begin_demo("1. The MDP")
    env = main_grid(step_reward=step_reward)
    names = env.state_names

    print(f"S: {len(env.states)} states, " + ", ".join(
        names[s] for s in env.states))
    print("A: " + ", ".join(ACTION_NAMES[a] for a in ACTIONS))
    print("\nlayout (### = wall):")
    for r in range(env.n_rows):
        print("   " + " ".join(
            ("###" if (r, c) in env.walls else names[env.to_s((r, c))]).rjust(5)
            for c in range(env.n_cols)))

    # 종결 칸에 들어가면 step_reward + 도착 보상을 받는다. 전이표와 같은 수를
    # 찍어야 하므로 도착 보상만 따로 보여주지 않는다.
    arrivals = ", ".join(f"{names[env.to_s(rc)]} = {step_reward + v:+g}"
                         for rc, v in sorted(env.terminals.items()))
    print(f"\nR(s): {arrivals}, and {step_reward:g} elsewhere")
    print(f"p0: uniform 1/{len(env.p0)} over the non-terminal states")

    s5, s7 = env.to_s((1, 1)), env.to_s((1, 4))
    print("\nP[s][a], deterministic:")
    for label, s, a in (("s5", s5, RIGHT), ("s7", s7, LEFT)):
        for prob, s2, reward, done in env.P[s][a]:
            note = ("   <- blocked, stays put" if s2 == s
                    else "   <- enters a reward cell, episode ends" if done else "")
            print(f"  P[{label}][{ACTION_NAMES[a]}] = ({prob:g}, {names[s2]}, "
                  f"{reward:+g}, {done}){note}")

    noisy = main_grid(noise=noise, step_reward=step_reward)
    print(f"\nThe same two with noise = {noise:g}:")
    for label, s, a in (("s5", s5, RIGHT), ("s7", s7, LEFT)):
        outcomes = ", ".join(f"({p:g}, {names[s2]}, {r:+g}, {d})"
                             for p, s2, r, d in noisy.P[s][a])
        print(f"  P[{label}][{ACTION_NAMES[a]}] = [{outcomes}]")

    # 이 그림은 값도 정책도 안 그리므로 gamma/noise와 무관하다.
    path = viz.figure_path("01_environment")
    viz.figure_environment(env, path)
    print()
    viz.wrote(path)


if __name__ == "__main__":
    from . import viz
    main(**vars(viz.demo_args(noise=0.2, step_reward=0.0)))
