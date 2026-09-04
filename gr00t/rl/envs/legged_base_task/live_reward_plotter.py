# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# RewardPlotter v3: color‑coded (greens for positives, reds for negatives) + labeled hover
from collections import deque
from itertools import cycle

import plotly.express as px
import plotly.graph_objects as go

NEG_KEYWORDS = ("limit", "penalty", "termination")


class RewardPlotter:
    """Live rolling stacked-area chart with separate pos/neg stacks and color families."""

    def __init__(self, window=200):
        self.window = window
        self.step_idx = 0
        self.time = deque(maxlen=window)
        self.buffers = {}  # key → deque

        # color cycles
        self.pos_colors = cycle(px.colors.sequential.Greens[::-1])  # darker→lighter
        self.neg_colors = cycle(px.colors.sequential.Reds)  # lighter→darker

        self.fig = go.Figure(
            layout=dict(
                title="Reward components (positive vs negative stacks)",
                xaxis_title="Environment step",
                yaxis_title="Reward value",
                hovermode="x unified",
            )
        )

    # ---------------------------------------------------------------- utilities
    def _is_negative(self, name: str) -> bool:
        return any(kw in name.lower() for kw in NEG_KEYWORDS)

    def _ensure_trace(self, key: str):
        if key in self.buffers:
            return
        self.buffers[key] = deque(maxlen=self.window)

        is_neg = self._is_negative(key)
        color = next(self.neg_colors) if is_neg else next(self.pos_colors)
        group = "negative" if is_neg else "positive"

        self.fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name=key,
                mode="lines",
                stackgroup=group,
                line=dict(width=0.5, color=color),
                fillcolor=color,
                hovertemplate=f"%{{y:.4f}} - <b>{key}</b><extra></extra>",
            )
        )

    # ---------------------------------------------------------------- public API
    def add_step(self, reward_dict: dict[str, float]):
        self.time.append(self.step_idx)
        self.step_idx += 1

        for k, v in reward_dict.items():
            self._ensure_trace(k)
            self.buffers[k].append(v)

        # update existing traces
        for i, k in enumerate(self.buffers):
            self.fig.data[i].x = list(self.time)
            self.fig.data[i].y = list(self.buffers[k])

    def show(self):
        """Call once inside a notebook to display the live widget."""
        return self.fig
