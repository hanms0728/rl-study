"""격자 DP 실험을 위한 그림.

그림마다 다시 따지지 않도록 여기에 설계 규칙을 적어둔다.

* **가치는 하나의 규칙으로 색에 싣는다.** 값이 0을 가로지르면 *발산형*
  스케일을 쓴다. 중간점을 0에 고정한 회색으로 두고 양쪽에 색상 둘을 두어
  "좋음"과 "나쁨"이 반대로 읽히게 한다. 값의 부호가 한쪽뿐이면 *순차형*
  이다. 단일 색상을 밝은 쪽에서 어두운 쪽으로. 보여줄 극성이 없고 크기만
  있기 때문이다. 무지개색은 절대 쓰지 않는다.
* **발산형 스케일의 양팔은 따로 스케일한다** (``TwoSlopeNorm``). 이 격자에는
  +10 보석 옆에 -100 함정이 있어서, 대칭 스케일을 쓰면 나머지 칸이 전부
  회색으로 뭉개진다.
* **칸은 테두리가 아니라 간격으로 나눈다.** 모든 칸에 테두리를 두르면 격자가
  스프레드시트처럼 읽힌다. 둥근 타일 사이로 바탕이 비치게 두면 정보를 나르는
  것이 색 하나로 남는다.
* **벽은 데이터가 아니므로** 가치 색상표에서 색을 가져오지 않는다. 종결 상태도
  데이터가 아니다. ``V(종결)``은 구성상 0이라, +10 보석을 가치 스케일에 올리면
  판에서 가장 비어 있는 칸으로 칠해진다. 둘 다 무채색으로 처리한다.
* **글자는 잉크 색을 입고 계열 색을 입지 않는다.** 칸의 숫자가 흰색으로
  뒤집히는 경우는 바탕이 너무 어두워 읽히지 않을 때뿐이며, 이는 가독성 보정이지
  부호화가 아니다.
* **비교할 패널들은 하나의 색 스케일을 공유한다.**

그림 안의 글자를 영어로 두는 것은 의도적이다. Windows의 기본 matplotlib 폰트에
한글 글리프가 없어서 한글 라벨이 네모로 깨지고, 대체 폰트도 연구실 PC와 맥북이
서로 다르다. 한국어 설명은 이렇게 주석으로 둔다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

from .gridworld import ACTION_DELTAS, ACTIONS, N_ACTIONS, GridWorld

# --- 팔레트 -------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#eceae3"
BASELINE = "#dedcd4"
WALL_FILL = "#d8d5cb"
NEUTRAL = "#f0efec"
ON_DARK = "#ffffff"

# 범주형 색 슬롯. 고정된 순서로 쓰고 절대 순환시키지 않는다.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

# 크기를 나타내는 단일 색상 램프. 밝은 쪽에서 어두운 쪽으로.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#f4f8fe", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"],
)

# 극성을 나타내는 두 극과 중립 회색 중간점.
DIVERGING = LinearSegmentedColormap.from_list(
    "div_red_blue",
    ["#7d1c1c", "#b03232", "#e34948", "#f09b9a", "#f8d4d3", NEUTRAL,
     "#d6e6fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"],
)

# --- 타일 기하 ----------------------------------------------------------
GAP = 0.045        # 이웃한 타일 사이로 비치는 바탕. 칸 단위.
CORNER = 0.09      # 타일 모서리 반지름
WEDGE_GAP = 0.018  # Q 칸의 네 쐐기 사이로 비치는 실선 굵기의 바탕


def apply_style() -> None:
    """눈에 띄지 않는 장식, 시스템 산세리프, 밝은 바탕, 여백 최소."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.03,
        "figure.constrained_layout.wspace": 0.03,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.labelsize": 9,
        "axes.edgecolor": BASELINE,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.9,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "axes.titlecolor": INK_SECONDARY,
        "figure.dpi": 130,
        "savefig.dpi": 200,
    })


# --- 색 스케일 ----------------------------------------------------------
def value_scale(values: np.ndarray):
    """값 집합에 맞는 ``(cmap, norm)``을 고른다.

    0을 가로지르면 발산형, 아니면 순차형.
    """
    finite = np.asarray(values, dtype=float)
    lo, hi = float(finite.min()), float(finite.max())
    if lo < -1e-9 and hi > 1e-9:
        return DIVERGING, TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
    if hi <= 1e-9:
        # 전부 0 이하. 크기가 아래로 자라므로 vmin에서 가장 어둡게.
        return SEQUENTIAL.reversed(), Normalize(vmin=min(lo, -1e-9), vmax=0.0)
    return SEQUENTIAL, Normalize(vmin=0.0, vmax=max(hi, 1e-9))


def _readable_ink(rgba) -> str:
    """``rgba`` 위에서 읽히는 잉크 색."""
    r, g, b = rgba[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return INK if luminance > 0.55 else ON_DARK


# --- 타일 ---------------------------------------------------------------
def _tile(ax, x, y, facecolor, *, edgecolor="none", lw=0.0, zorder=2):
    """둥근 칸 타일. 이웃 사이로 바탕이 비치도록 안쪽으로 들여 그린다."""
    patch = FancyBboxPatch(
        (x + GAP, y + GAP), 1 - 2 * GAP, 1 - 2 * GAP,
        boxstyle=f"round,pad=0,rounding_size={CORNER}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
        mutation_aspect=1, zorder=zorder)
    ax.add_patch(patch)
    return patch


def _shrink(points, amount):
    """다각형의 꼭짓점을 무게중심 쪽으로 당겨 간격을 낸다."""
    pts = np.asarray(points, dtype=float)
    centre = pts.mean(axis=0)
    span = np.abs(pts - centre).max()
    return centre + (pts - centre) * (1.0 - amount / max(span, 1e-9))


def _draw_wall(ax, x, y):
    """벽은 둥글지 않고 각지게 그린다. 데이터가 아니라 구조이기 때문이다.

    둥근 타일은 값을 나른다. 각진 블록은 판 위에서 에이전트가 절대 점유할 수
    없는 유일한 것이므로, 애초에 칸처럼 보이면 안 된다.
    """
    ax.add_patch(Rectangle((x + GAP, y + GAP), 1 - 2 * GAP, 1 - 2 * GAP,
                           facecolor=WALL_FILL, edgecolor="none", zorder=1))


def _draw_terminal(ax, env, rc, x, y):
    """종결 상태는 상태가치가 아니라 보상을 나른다. 무채색 채움에 굵은 라벨."""
    _tile(ax, x, y, "#fbfaf7", edgecolor=INK, lw=1.7, zorder=3)
    reward = env.terminals[rc]
    ax.text(x + 0.5, y + 0.5, f"{reward:+g}" if reward else "0",
            ha="center", va="center", color=INK, fontsize=12,
            fontweight="semibold", zorder=4)


def _draw_start(ax, rc):
    """시작 칸 모서리에 작은 고리 하나."""
    r, c = rc
    ax.add_patch(Circle((c + 0.165, r + 0.835), 0.058, facecolor=SURFACE,
                        edgecolor=INK_SECONDARY, linewidth=1.5, zorder=6))


def _draw_arrows(ax, pi, s, x, y, colour):
    """정책이 허용하는 모든 행동을, 칸 중심에서 뻗는 화살표로."""
    for a in ACTIONS:
        if pi[s, a] <= 0:
            continue
        dr, dc = ACTION_DELTAS[a]
        ax.annotate(
            "", xytext=(x + 0.5 + 0.09 * dc, y + 0.60 + 0.09 * dr),
            xy=(x + 0.5 + 0.31 * dc, y + 0.60 + 0.31 * dr),
            arrowprops=dict(arrowstyle="-|>,head_width=0.19,head_length=0.34",
                            color=colour, linewidth=1.5,
                            shrinkA=0, shrinkB=0,
                            joinstyle="miter", capstyle="butt"),
            zorder=5)


def _finish_axes(ax, env, title=None):
    ax.set_xlim(-0.02, env.n_cols + 0.02)
    ax.set_ylim(env.n_rows + 0.02, -0.02)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, pad=6, loc="center")


# --- 격자 패널 ----------------------------------------------------------
def draw_grid(ax, env: GridWorld, V=None, pi=None, *, title=None, cmap=None,
              norm=None, value_fmt="{:.2f}", show_values=True,
              show_start=True) -> None:
    """상태가치 격자 하나를 그린다. 색 = V(s), 화살표 = 정책."""
    if V is not None and cmap is None:
        cmap, norm = value_scale(V[list(env.interior_states)])

    for r in range(env.n_rows):
        for c in range(env.n_cols):
            rc, x, y = (r, c), c, r

            if rc in env.walls:
                _draw_wall(ax, x, y)
                continue
            if rc in env.terminals:
                _draw_terminal(ax, env, rc, x, y)
                continue

            s = env.to_s(rc)
            face = cmap(norm(V[s])) if V is not None else SURFACE
            _tile(ax, x, y, face, edgecolor=BASELINE if V is None else "none",
                  lw=1.0 if V is None else 0.0)
            ink = _readable_ink(mpl.colors.to_rgba(face))

            # 화살표가 있으면 숫자를 타일 위쪽에 올려 겹치지 않게 하고,
            # 없으면 가운데를 차지한다.
            arrows = pi is not None
            if show_values and V is not None:
                ax.text(x + 0.5, y + (0.17 if arrows else 0.5),
                        value_fmt.format(V[s]), ha="center", va="center",
                        color=ink, fontsize=8.5 if arrows else 10,
                        zorder=4)
            if arrows:
                _draw_arrows(ax, pi, s, x, y, ink)

    if show_start and env.start is not None and env.start not in env.terminals:
        _draw_start(ax, env.start)
    _finish_axes(ax, env, title)


# 한 칸의 네 행동을 중심에서 만나는 네 개의 쐐기로 그린다. 각 쐐기는 그
# 행동이 향하는 쪽을 가리킨다. Q 표는 V가 하나를 담는 자리에 상태당 |A|개를
# 담으므로, 달리 넣을 곳이 없다.
Q_WEDGES = {
    0: ((0.0, 0.0), (1.0, 0.0), (0.5, 0.5)),   # up    -- 위쪽 쐐기
    1: ((1.0, 0.0), (1.0, 1.0), (0.5, 0.5)),   # right -- 오른쪽 쐐기
    2: ((0.0, 1.0), (1.0, 1.0), (0.5, 0.5)),   # down  -- 아래쪽 쐐기
    3: ((0.0, 0.0), (0.0, 1.0), (0.5, 0.5)),   # left  -- 왼쪽 쐐기
}
Q_LABEL_POS = {0: (0.50, 0.20), 1: (0.78, 0.50), 2: (0.50, 0.80), 3: (0.22, 0.50)}


def draw_q_grid(ax, env: GridWorld, Q, *, title=None, cmap=None, norm=None,
                value_fmt="{:.2f}", highlight_greedy=True, show_values=True,
                show_start=True) -> None:
    """행동가치 격자 하나를 그린다. 행동마다 쐐기 하나, 색은 q(s, a)."""
    if cmap is None:
        cmap, norm = value_scale(Q[list(env.interior_states)].ravel())

    for r in range(env.n_rows):
        for c in range(env.n_cols):
            rc, x, y = (r, c), c, r

            if rc in env.walls:
                _draw_wall(ax, x, y)
                continue
            if rc in env.terminals:
                _draw_terminal(ax, env, rc, x, y)
                continue

            s = env.to_s(rc)
            best = Q[s].max()
            # 탐욕 행동 표시는 승자를 하나 골라낼 때만 의미가 있다. 값
            # 반복 초반에는 대부분의 항목이 아직 0으로 동점이라, 네 쐐기를 전부
            # 테두리 치면 칸마다 X를 그리는 꼴이 된다. 그래서 대부분의 행동이
            # 동점이면 강조를 끈다.
            discriminates = int((Q[s] >= best - 1e-8).sum()) <= N_ACTIONS // 2
            # 쐐기를 칸 전체에서 잘라낸 뒤 둥근 타일로 클리핑한다. 그래야 Q
            # 칸이 V 칸과 정확히 같은 실루엣을 갖고 두 패널이 나란히 맞는다.
            tile = _tile(ax, x, y, SURFACE, zorder=2)
            for a in ACTIONS:
                face = cmap(norm(Q[s, a]))
                corners = _shrink([(x + dx, y + dy) for dx, dy in Q_WEDGES[a]],
                                  WEDGE_GAP)
                greedy = (highlight_greedy and discriminates
                          and Q[s, a] >= best - 1e-8)
                wedge = Polygon(corners, closed=True, facecolor=face,
                                edgecolor=INK if greedy else "none",
                                linewidth=1.3 if greedy else 0.0,
                                joinstyle="round", zorder=4 if greedy else 3)
                ax.add_patch(wedge)
                wedge.set_clip_path(tile)
                if show_values:
                    dx, dy = Q_LABEL_POS[a]
                    ax.text(x + dx, y + dy, value_fmt.format(Q[s, a]),
                            ha="center", va="center", fontsize=6.8,
                            fontweight="semibold" if greedy else "normal",
                            color=_readable_ink(mpl.colors.to_rgba(face)),
                            zorder=5)

    if show_start and env.start is not None and env.start not in env.terminals:
        _draw_start(ax, env.start)
    _finish_axes(ax, env, title)


# --- 그림 장식 ----------------------------------------------------------
TERMINAL_NOTE = ("Terminal cells show their reward and stay off the colour "
                 "scale — V = 0 there by definition.   Ring marks the start.")


def _titles(fig, title, subtitle=None):
    fig.suptitle(title, fontsize=12.5, fontweight="semibold", color=INK)
    if subtitle:
        fig.supxlabel(subtitle, fontsize=8.8, color=INK_SECONDARY, x=0.5)


def _footnote(fig, text=TERMINAL_NOTE):
    fig.supxlabel(text, fontsize=7.6, color=INK_MUTED, x=0.5)


def _colorbar(fig, axes, cmap, norm, label="state value  V(s)"):
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    bar = fig.colorbar(mappable, ax=axes, shrink=0.62, aspect=22, pad=0.015)
    bar.set_label(label, color=INK_SECONDARY, fontsize=8.5, labelpad=6)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=2, width=0.8)
    return bar


def _panel_figsize(env, ncols, nrows, cell, title_pad=0.42):
    return (ncols * env.n_cols * cell + 1.5,
            nrows * (env.n_rows * cell + title_pad) + 0.85)


# --- 그림들 -------------------------------------------------------------
def figure_environment(env: GridWorld, title: str, path):
    """판 자체만. 보상, 벽, 시작 위치. 값도 정책도 없다."""
    cell = 0.72
    fig, ax = plt.subplots(figsize=(env.n_cols * cell + 0.5,
                                    env.n_rows * cell + 1.0))
    draw_grid(ax, env, V=None, pi=None)
    _titles(fig, title)
    _footnote(fig)
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_value_policy(env: GridWorld, V, pi, title: str, path,
                        subtitle: str | None = None, value_fmt="{:.2f}"):
    """격자 하나. 최적 가치는 색과 숫자로, 최적 정책은 화살표로."""
    cmap, norm = value_scale(V[list(env.interior_states)])
    cell = 0.86
    fig, ax = plt.subplots(figsize=(env.n_cols * cell + 1.6,
                                    env.n_rows * cell + 1.25))
    draw_grid(ax, env, V, pi, cmap=cmap, norm=norm, value_fmt=value_fmt,
              title=subtitle)
    _colorbar(fig, ax, cmap, norm)
    _titles(fig, title)
    _footnote(fig)
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_panels(env: GridWorld, panels, title: str, path, *,
                  ncols=None, shared_scale=True, value_fmt="{:.2f}",
                  show_values=True):
    """상태가치 패널을 줄지어 놓는다. ``panels``은 ``[(소제목, V, pi)]``.

    ``shared_scale``이면 모든 패널이 색 스케일 하나를 쓴다. 연속된 sweep들을
    비교 가능하게 만드는 것이 바로 이것이다.
    """
    if ncols is None:
        ncols = min(len(panels), 3 if env.n_cols > 6 else 6)
    nrows = int(np.ceil(len(panels) / ncols))
    interior = list(env.interior_states)

    if shared_scale:
        stacked = np.concatenate([V[interior] for _, V, _ in panels])
        cmap, norm = value_scale(stacked)
    else:
        cmap = norm = None

    cell = 0.62
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=_panel_figsize(env, ncols, nrows, cell))
    axes = np.atleast_1d(axes).ravel()

    for ax, (subtitle, V, pi) in zip(axes, panels):
        pc, pn = (cmap, norm) if shared_scale else value_scale(V[interior])
        draw_grid(ax, env, V, pi, title=subtitle, cmap=pc, norm=pn,
                  value_fmt=value_fmt, show_values=show_values)
    for ax in axes[len(panels):]:
        ax.set_visible(False)

    if shared_scale:
        _colorbar(fig, list(axes[:len(panels)]), cmap, norm)
    _titles(fig, title)
    _footnote(fig)
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_q_values(env: GridWorld, Q, title: str, path,
                    subtitle: str | None = None, value_fmt="{:.2f}"):
    """행동가치 격자 하나만."""
    cmap, norm = value_scale(Q[list(env.interior_states)].ravel())
    cell = 1.0
    fig, ax = plt.subplots(figsize=(env.n_cols * cell + 1.7,
                                    env.n_rows * cell + 1.25))
    draw_q_grid(ax, env, Q, cmap=cmap, norm=norm, value_fmt=value_fmt,
                title=subtitle)
    _colorbar(fig, ax, cmap, norm, label="action value  q(s, a)")
    _titles(fig, title)
    _footnote(fig, "Outlined wedge is the greedy action, shown only where it "
              "picks a winner.   " + TERMINAL_NOTE)
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_q_panels(env: GridWorld, panels, title: str, path, *, ncols=3,
                    value_fmt="{:.1f}", shared_scale=True,
                    highlight_greedy=True):
    """행동가치 패널을 줄지어 놓는다. ``panels``은 ``[(소제목, Q), ...]``."""
    interior = list(env.interior_states)
    nrows = int(np.ceil(len(panels) / ncols))
    if shared_scale:
        stacked = np.concatenate([Q[interior].ravel() for _, Q in panels])
        cmap, norm = value_scale(stacked)
    else:
        cmap = norm = None

    cell = 0.80
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=_panel_figsize(env, ncols, nrows, cell))
    axes = np.atleast_1d(axes).ravel()

    for ax, (subtitle, Q) in zip(axes, panels):
        pc, pn = ((cmap, norm) if shared_scale
                  else value_scale(Q[interior].ravel()))
        draw_q_grid(ax, env, Q, title=subtitle, cmap=pc, norm=pn,
                    value_fmt=value_fmt,
                    highlight_greedy=highlight_greedy)
    for ax in axes[len(panels):]:
        ax.set_visible(False)

    if shared_scale:
        _colorbar(fig, list(axes[:len(panels)]), cmap, norm,
                  label="action value  q(s, a)")
    _titles(fig, title)
    _footnote(fig, "Outlined wedge is the greedy action, shown only where it "
              "picks a winner.   " + TERMINAL_NOTE)
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_v_and_q(env: GridWorld, V, pi, Q, title: str, path,
                   subtitle: str | None = None):
    """두 표를 나란히, 공유된 색 스케일 하나 위에.

    같은 해를 두 가지로 쓴 것이다. 오른쪽의 모든 쐐기가 ``q(s, a)``이고,
    왼쪽의 상태가치는 그 칸에서 가장 큰 쐐기다. 그것을 한눈에 읽히게 만드는
    것이 공유 스케일이다.
    """
    interior = list(env.interior_states)
    cmap, norm = value_scale(np.concatenate([V[interior], Q[interior].ravel()]))

    cell = 0.92
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(2 * env.n_cols * cell + 1.8, env.n_rows * cell + 1.6))
    draw_grid(ax, env, V, pi, title="state values  V(s)  +  greedy policy",
              cmap=cmap, norm=norm)
    draw_q_grid(bx, env, Q, title="action values  q(s, a)", cmap=cmap,
                norm=norm)
    _colorbar(fig, [ax, bx], cmap, norm, label="value")
    _titles(fig, title)
    _footnote(fig, (subtitle + "   " if subtitle else "")
              + "V(s) is the largest wedge of the same cell.")
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_convergence(curves: dict, totals: dict, title: str, path, *,
                       subtitle: str | None = None, theta: float | None = None,
                       xlim: float | None = None):
    """패널 둘. 여기에 서로 다른 질문이 두 개 있기 때문이다.

    *sweep마다 오차가 얼마나 빨리 떨어지는가?* 는 시간에 따른 변화이므로 선.
    *각 변형이 총 얼마나 일했는가?* 는 범주당 숫자 하나이므로 막대. 둘을 한
    선 그래프에 억지로 넣은 것이 이 그림의 첫 버전을 읽을 수 없게 만든
    원인이었다. 정책 반복이 값 반복보다 마흔 배 길게 도는 바람에 정작 궁금한
    곡선들이 왼쪽 가장자리에 붙어 뭉갰다.

    ``curves``는 sweep별로 그릴 만한 계열을 담고, ``totals``는 그리기엔 너무
    긴 것까지 포함한 모든 변형의 총 sweep 수를 담는다.
    """
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(11.4, 4.1), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # --- 왼쪽: sweep당 오차 --------------------------------------------
    ax.set_yscale("log")
    seen = []
    for i, (label, deltas) in enumerate(curves.items()):
        deltas = np.asarray(deltas, dtype=float)
        # 이 격자들에서 값 반복은 *정확히* 수렴한다. 마지막 sweep이 아무것도
        # 바꾸지 않아 delta가 0이 되는데, 로그 축은 그것을 그릴 수 없다. 양수인
        # delta만 그리고 0에 닿은 지점을 따로 표시한다.
        x = np.arange(1, len(deltas) + 1)
        positive = deltas > 0
        colour = SERIES[i % len(SERIES)]
        ax.plot(x[positive], deltas[positive], color=colour, linewidth=2.0,
                label=label, solid_capstyle="round",
                marker="o" if positive.sum() <= 34 else None, markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=0.9,
                zorder=3 - 0.1 * i)
        seen.append(deltas[positive])
        if not deltas[-1] > 0:
            # 여러 실행이 같은 sweep에서 끝나므로 라벨을 펼쳐 놓는다.
            dy = (-13, 10, -25, 22)[i % 4]
            ax.annotate("Δ = 0", (x[positive][-1], deltas[positive][-1]),
                        xytext=(6, dy), textcoords="offset points",
                        fontsize=8, color=colour, fontweight="semibold")

    flat = np.concatenate(seen)
    ax.set_ylim(flat.min() * 0.35, flat.max() * 3.2)
    if theta is not None and theta >= flat.min() * 0.35:
        ax.axhline(theta, color=INK_MUTED, linewidth=1.0,
                   linestyle=(0, (4, 3)), zorder=1)

    ax.set_xlabel("Bellman sweep")
    ax.set_ylabel("max |V(s) change| over the sweep")
    ax.set_title("Error per sweep", pad=6, loc="left")
    if xlim:
        ax.set_xlim(0, xlim)
    ax.grid(True, axis="y", alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(which="both", length=0)
    legend = ax.legend(frameon=False, fontsize=8.8, loc="lower left",
                       handlelength=1.6, borderpad=0.2, labelspacing=0.35)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    # --- 오른쪽: 총 작업량 ----------------------------------------------
    labels = sorted(totals, key=totals.get)
    values = [totals[k] for k in labels]
    ypos = np.arange(len(labels))

    bx.barh(ypos, values, height=0.58, color=SERIES[0], zorder=3)
    for y, value in zip(ypos, values):
        bx.annotate(f"{value}", (value, y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.8,
                    fontweight="semibold", color=INK_SECONDARY)
    bx.set_yticks(ypos, labels, fontsize=8.8)
    bx.tick_params(axis="y", which="both", length=0,
                   labelcolor=INK_SECONDARY)
    bx.tick_params(axis="x", which="both", length=0)
    bx.set_xlabel("total Bellman sweeps to convergence")
    bx.set_title("Total work", pad=6, loc="left")
    bx.set_xlim(0, max(values) * 1.16)
    bx.grid(True, axis="x", alpha=1.0)
    bx.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        bx.spines[side].set_visible(False)

    _titles(fig, title, subtitle)
    fig.savefig(path)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# 실행 가능한 데모들이 공유하는 작은 도우미
# ----------------------------------------------------------------------
# 모든 알고리즘 파일은 따로 실행할 수 있고(``python -m dp.value_iteration``)
# 그림을 여기에 쓴다. 이 둘이 이 모듈에 있는 이유는 계산이 아니라 결과를
# 보여주는 일에 속하기 때문이다.
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def begin_demo(text: str) -> None:
    """실행 데모 시작. 콘솔 인코딩, 그림 스타일, 출력 폴더를 준비한다."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔에서 화살표 출력
    FIGURES.mkdir(exist_ok=True)
    apply_style()
    banner(text)


def banner(text: str) -> None:
    rule = "=" * 72
    print(f"\n{rule}\n{text}\n{rule}")


def demo_args(argv=None, **defaults):
    """``defaults``에 준 이름을 그대로 ``--이름`` 옵션으로 만든다.

    ``viz.demo_args(gamma=0.9, noise=0.2)`` -> ``--gamma`` 와 ``--noise``를 받는
    네임스페이스. ``vars()``로 풀어 ``main()``에 그대로 넘긴다.
    """
    parser = argparse.ArgumentParser()
    for name, value in defaults.items():
        parser.add_argument(f"--{name}", type=float, default=value)
    return parser.parse_args(argv)
