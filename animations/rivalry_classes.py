from __future__ import annotations

import numpy as np
from manim import *

"""
Classification classes comparison diagram
Run: manim -qh -s animations/rivalry_classes.py BinocularRivalryClasses
"""


class BinocularRivalryClasses(Scene):
    RED_STIM = "#d7191c"
    GREEN_STIM = "#1aaf54"
    DARK = "#151515"
    GABOR_OPACITY = 0.34

    def construct(self):
        self.camera.background_color = WHITE
        
        # 1. Vertical Divider
        divider = Line(UP * 3.3, DOWN * 3.3, color=GRAY_B, stroke_width=2)
        self.add(divider)

        # Left Column: C vs IP
        left_center = -3.55
        left_col_title = Text("C vs IP", font_size=26, color=self.DARK, weight=BOLD)
        left_col_title.move_to([left_center, 2.8, 0])
        self.add(left_col_title)

        # Left Card 1: Congruent (C)
        card_c = self.create_card("Congruent (C)", left_center, 1.15)
        c_elements = self.draw_congruent_stimuli(left_center, 1.15)
        self.add(card_c, c_elements)

        # VS label left
        vs_left = Text("vs", font_size=20, color=GRAY_C, weight=BOLD).move_to([left_center, -0.1, 0])
        self.add(vs_left)

        # Left Card 2: Incongruent Pure (IP)
        card_ip_left = self.create_card("Incongruent Pur (IP)", left_center, -1.35)
        ip_left_elements = self.draw_pure_interpretation(left_center, -1.35)
        self.add(card_ip_left, ip_left_elements)

        # Right Column: IM vs IP
        right_center = 3.55
        right_col_title = Text("IM vs IP", font_size=26, color=self.DARK, weight=BOLD)
        right_col_title.move_to([right_center, 2.8, 0])
        self.add(right_col_title)

        # Right Card 1: Incongruent Mixed (IM)
        card_im = self.create_card("Incongruent Mixt (IM)", right_center, 1.15)
        im_elements = self.draw_mixed_interpretation(right_center, 1.15)
        self.add(card_im, im_elements)

        # VS label right
        vs_right = Text("vs", font_size=20, color=GRAY_C, weight=BOLD).move_to([right_center, -0.1, 0])
        self.add(vs_right)

        # Right Card 2: Incongruent Pure (IP)
        card_ip_right = self.create_card("Incongruent Pur (IP)", right_center, -1.35)
        ip_right_elements = self.draw_pure_interpretation(right_center, -1.35)
        self.add(card_ip_right, ip_right_elements)

    def create_card(self, title_text, x_center, y_center):
        box = RoundedRectangle(width=4.8, height=1.75, corner_radius=0.08, color=GRAY_B, stroke_width=2.5)
        box.set_fill(WHITE, opacity=1).move_to([x_center, y_center, 0])
        title = Text(title_text, font_size=18, color=self.DARK, weight=BOLD)
        title.move_to(box.get_top() + DOWN * 0.30)
        return VGroup(box, title)

    def draw_congruent_stimuli(self, x_center, y_center):
        # Congruent Stimulus (Green-Green)
        stim = self.stimulus_pair(self.GREEN_STIM, self.GREEN_STIM, radius=0.28, gap=0.65)
        stim.move_to([x_center - 1.275, y_center - 0.15, 0])
        
        # Arrow: starts at x_center - 0.52 (gap of 0.15 from right edge of stim at x_center - 0.67),
        # ends at x_center + 0.67 (gap of 0.15 from left edge of pure_green at x_center + 0.82).
        arrow = Arrow([x_center - 0.52, y_center - 0.15, 0], [x_center + 0.67, y_center - 0.15, 0], color=GRAY_D, buff=0.0, stroke_width=2.5, tip_length=0.15)
        
        # Outcome: Green octagon
        pure_green = self.response_disc(self.GREEN_STIM, radius=0.28).move_to([x_center + 1.10, y_center - 0.15, 0])
        
        # Labels: L and R under corresponding Gabors
        lbl_l = Text("L", font_size=14, color=GRAY_D).move_to([x_center - 1.60, y_center - 0.58, 0])
        lbl_r = Text("R", font_size=14, color=GRAY_D).move_to([x_center - 0.95, y_center - 0.58, 0])

        return VGroup(stim, arrow, pure_green, lbl_l, lbl_r)

    def draw_pure_interpretation(self, x_center, y_center):
        # Incongruent Stimulus (Red-Green)
        stim = self.stimulus_pair(self.RED_STIM, self.GREEN_STIM, radius=0.28, gap=0.65)
        stim.move_to([x_center - 1.275, y_center - 0.15, 0])
        
        # Arrow: starts at x_center - 0.52 (gap of 0.15 from right edge of stim at x_center - 0.67),
        # ends at x_center + 0.17 (gap of 0.15 from left edge of pure_red at x_center + 0.32).
        arrow = Arrow([x_center - 0.52, y_center - 0.15, 0], [x_center + 0.17, y_center - 0.15, 0], color=GRAY_D, buff=0.0, stroke_width=2.5, tip_length=0.15)
        
        # Outcomes: Red octagon OR Green octagon
        pure_red = self.response_disc(self.RED_STIM, radius=0.28).move_to([x_center + 0.60, y_center - 0.15, 0])
        or_text = Text("o", font_size=16, color=self.DARK).move_to([x_center + 1.10, y_center - 0.15, 0])
        pure_green = self.response_disc(self.GREEN_STIM, radius=0.28).move_to([x_center + 1.60, y_center - 0.15, 0])
        
        # Labels
        lbl_l = Text("L", font_size=14, color=GRAY_D).move_to([x_center - 1.60, y_center - 0.58, 0])
        lbl_r = Text("R", font_size=14, color=GRAY_D).move_to([x_center - 0.95, y_center - 0.58, 0])
        lbl_out = Text("Percepció pura", font_size=16, color=GRAY_D).move_to([x_center + 1.10, y_center - 0.58, 0])

        return VGroup(stim, arrow, pure_red, or_text, pure_green, lbl_l, lbl_r, lbl_out)

    def draw_mixed_interpretation(self, x_center, y_center):
        # Incongruent Stimulus (Red-Green)
        stim = self.stimulus_pair(self.RED_STIM, self.GREEN_STIM, radius=0.28, gap=0.65)
        stim.move_to([x_center - 1.275, y_center - 0.15, 0])
        
        # Arrow: starts at x_center - 0.52 (gap of 0.15 from right edge of stim at x_center - 0.67),
        # ends at x_center + 0.67 (gap of 0.15 from left edge of mixed_disc at x_center + 0.82).
        arrow = Arrow([x_center - 0.52, y_center - 0.15, 0], [x_center + 0.67, y_center - 0.15, 0], color=GRAY_D, buff=0.0, stroke_width=2.5, tip_length=0.15)
        
        # Outcome: Mixed split disc
        mixed_disc = self.split_response_disc(self.RED_STIM, self.GREEN_STIM, radius=0.28).move_to([x_center + 1.10, y_center - 0.15, 0])
        
        # Labels
        lbl_l = Text("L", font_size=14, color=GRAY_D).move_to([x_center - 1.60, y_center - 0.58, 0])
        lbl_r = Text("R", font_size=14, color=GRAY_D).move_to([x_center - 0.95, y_center - 0.58, 0])
        lbl_out = Text("Percepció mixta", font_size=16, color=GRAY_D).move_to([x_center + 1.10, y_center - 0.58, 0])

        return VGroup(stim, arrow, mixed_disc, lbl_l, lbl_r, lbl_out)

    def rivalry_disc(self, color, radius=0.55):
        base_color = ManimColor(color).interpolate(WHITE, 0.18)
        base = Circle(radius=radius, color=BLACK, stroke_width=1.5).set_fill(base_color, opacity=1)
        stripes = VGroup()
        spacing = radius * 0.12
        if color == self.RED_STIM:
            angle = -25 * DEGREES
        else:
            angle = 25 * DEGREES
        direction = np.array([np.cos(angle), np.sin(angle), 0])
        normal = np.array([-np.sin(angle), np.cos(angle), 0])
        for i in range(-7, 8):
            offset = i * spacing
            if abs(offset) >= radius:
                continue
            half_width = np.sqrt(radius**2 - offset**2)
            start = offset * normal - half_width * direction
            end = offset * normal + half_width * direction
            stripe = Line(start, end, color=BLACK, stroke_width=radius * 2.2)
            stripe.set_opacity(self.GABOR_OPACITY)
            stripes.add(stripe)
        center = Circle(radius=radius * 0.18, color=BLACK, stroke_width=1.0).set_fill(BLACK, opacity=1)
        dot = Dot(radius=radius * 0.035, color=GRAY_C)
        return VGroup(base, stripes, center, dot)

    def stimulus_pair(self, left_color, right_color, radius=0.55, gap=1.6):
        left = self.rivalry_disc(left_color, radius=radius)
        right = self.rivalry_disc(right_color, radius=radius)
        left.move_to(LEFT * gap / 2)
        right.move_to(RIGHT * gap / 2)
        return VGroup(left, right)

    def response_disc(self, color, radius=0.26):
        return RegularPolygon(n=8, radius=radius, color=color, stroke_width=0, start_angle=np.pi/8).set_fill(color, opacity=1)

    def split_response_disc(self, left_color, right_color, radius=0.26):
        r = radius
        cos_val = np.cos(np.pi / 8)
        sin_val = np.sin(np.pi / 8)
        
        # Right half vertices
        right_verts = [
            [0, r * cos_val, 0],
            [r * sin_val, r * cos_val, 0],
            [r * cos_val, r * sin_val, 0],
            [r * cos_val, -r * sin_val, 0],
            [r * sin_val, -r * cos_val, 0],
            [0, -r * cos_val, 0]
        ]
        # Left half vertices
        left_verts = [
            [0, r * cos_val, 0],
            [-r * sin_val, r * cos_val, 0],
            [-r * cos_val, r * sin_val, 0],
            [-r * cos_val, -r * sin_val, 0],
            [-r * sin_val, -r * cos_val, 0],
            [0, -r * cos_val, 0]
        ]
        
        left_poly = Polygon(*left_verts, color=left_color, stroke_width=0).set_fill(left_color, opacity=1)
        right_poly = Polygon(*right_verts, color=right_color, stroke_width=0).set_fill(right_color, opacity=1)
        divider = Line([0, r * cos_val, 0], [0, -r * cos_val, 0], color=WHITE, stroke_width=1.5)
        return VGroup(left_poly, right_poly, divider)
