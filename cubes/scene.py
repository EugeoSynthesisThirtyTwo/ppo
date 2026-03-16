import numpy as np
from manim import *
import torch

import algo

# Define cube vertices
vertices = np.array([
    [1, 1, 0],
    [1, 1, 1],
    [0, 1, 1],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 1],
    [0, 0, 1],
    [0, 0, 0],
])

edges_idx = [
    (3, 0), (0, 1), (1, 2), (2, 3),  # top
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    (7, 4), (4, 5), (5, 6), (6, 7),  # bottom
]

class CubeEdges(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=45 * DEGREES)

        # Load all cube configs
        edges = algo.build_every_cube()
        edges = algo.filtrer_2d(edges)
        edges = algo.filter_non_connexe(edges)
        edges = algo.filter_rotations(edges) # edges is a tensor of shape [n, 12]

        for edge in edges:
            lines = VGroup()

            for active, (i, j) in zip(edge, edges_idx):
                if active:
                    line = Line3D(
                        start=vertices[i],
                        end=vertices[j],
                        color=BLUE,
                        stroke_width=6,
                    )
                    lines.add(line)

            self.play(Create(lines, lag_ratio=0.5))
            self.wait(0.3)
            self.play(FadeOut(lines))

        self.wait()
