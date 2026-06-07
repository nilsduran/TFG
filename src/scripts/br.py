from __future__ import annotations

from manim import *

"""
Binocular rivalry: typical trial and interpretations
Run: manim -pql src\\scripts\\br.py BinocularRivalry
"""


class BinocularRivalry(MovingCameraScene):
    """Schematic visualization of a binocular rivalry trial."""

    RED_STIM = "#d7191c"
    GREEN_STIM = "#1aaf54"
    SCREEN_BLUE = "#dcecf7"
    DARK = "#151515"
    GABOR_OPACITY = 0.34
    TEXTS = {
        "main_title": "Binocular rivalry",
        "example_title": "Incongruent Example",
        "blank": "Blank screen",
        "ramp": "Ramp up",
        "stim_on": "Stimulus onset",
        "stim_off": "Stimulus offset",
        "response_choice": "Perceptual response",
        "variations_title": "Four trial variations",
        "congruent": "Congruent",
        "incongruent": "Incongruent",
        "interpretation_title": "Interpretation of incongruent trials",
        "incongruent_stimulus": "Incongruent stimulus",
        "or": "or",
        "pure": "Pure incongruent",
        "mixed": "Mixed incongruent",
        "response": "Response",
    }

    def text(self, key):
        return self.TEXTS[key]

    def construct(self):
        self.camera.background_color = WHITE

        screen = self.intro_observer_to_screen()
        self.full_trial_example(screen)
        self.show_variations()
        self.show_interpretation()

    def intro_observer_to_screen(self):
        title = Text(self.text("main_title"), font_size=42, color=self.DARK)
        title.to_edge(UP, buff=0.45)

        head = Circle(radius=0.55, color=self.DARK, stroke_width=4)
        head.set_fill("#f0c7a8", opacity=1)
        head.move_to(LEFT * 3.35 + UP * 0.15)
        neck = Line(head.get_bottom(), head.get_bottom() + DOWN * 0.6, color=self.DARK, stroke_width=6)
        torso = Polygon(
            [-4.05, -2.2, 0],
            [-2.85, -2.2, 0],
            [-3.1, -0.35, 0],
            [-3.8, -0.35, 0],
            color=self.DARK,
            stroke_width=4,
        ).set_fill("#f6b2a6", opacity=1)
        left_eye = Dot(head.get_center() + RIGHT * 0.2 + UP * 0.08, radius=0.04, color=BLACK)
        right_eye = Dot(head.get_center() + RIGHT * 0.42 + UP * 0.08, radius=0.04, color=BLACK)

        desk = Line(LEFT * 5.7 + DOWN * 2.2, RIGHT * 5.4 + DOWN * 2.2, color=GRAY_D, stroke_width=6)

        screen = RoundedRectangle(width=3.25, height=2.05, corner_radius=0.06, color=self.DARK, stroke_width=5)
        screen.set_fill(self.SCREEN_BLUE, opacity=1).move_to(RIGHT * 1.55 + UP * 0.05)
        stand = VGroup(
            Line(screen.get_bottom(), screen.get_bottom() + DOWN * 0.55, color=self.DARK, stroke_width=5),
            Line(
                screen.get_bottom() + DOWN * 0.55 + LEFT * 0.55,
                screen.get_bottom() + DOWN * 0.55 + RIGHT * 0.55,
                color=self.DARK,
                stroke_width=5,
            ),
        )
        screen_stim = self.stimulus_pair(self.RED_STIM, self.GREEN_STIM, radius=0.25, gap=0.85)
        screen_stim.move_to(screen.get_center())
        screen_stim.set_opacity(0)
        red_target = screen_stim[0].get_center()
        green_target = screen_stim[1].get_center()
        green_path = CubicBezier(
            left_eye.get_center(),
            left_eye.get_center() + RIGHT * 1.1 + UP * 0.64,
            green_target + LEFT * 1.05 + UP * 0.52,
            green_target,
        )
        eye_paths = VGroup(
            DashedLine(right_eye.get_center(), red_target, color=self.RED_STIM, dash_length=0.1).set_opacity(0.6),
            DashedVMobject(green_path, num_dashes=18).set_color(self.GREEN_STIM).set_opacity(0.6),
        )
        eye_paths.set_opacity(0)

        scene = VGroup(head, neck, torso, left_eye, right_eye, desk, screen, stand, screen_stim, eye_paths)

        self.play(FadeIn(title), FadeIn(scene), run_time=1.2)
        self.wait(0.5)
        self.external_trial_preview(screen_stim, eye_paths)
        self.play(
            self.camera.frame.animate.scale(0.38).move_to(screen.get_center()),
            screen.animate.set_fill(WHITE, opacity=1),
            FadeOut(title),
            run_time=1.5,
        )
        self.play(
            FadeOut(VGroup(head, neck, torso, left_eye, right_eye, desk, stand, eye_paths, screen_stim)), run_time=0.6
        )
        return screen

    def external_trial_preview(self, screen_stim, eye_paths):
        self.wait(1.0)
        self.play(screen_stim.animate.set_opacity(1), run_time=0.24)
        self.play(eye_paths.animate.set_opacity(1), run_time=0.2)
        self.wait(1.25)
        self.play(screen_stim.animate.set_opacity(0), eye_paths.animate.set_opacity(0), run_time=0.25)
        self.wait(0.6)

    def full_trial_example(self, screen):
        full_screen = RoundedRectangle(width=8.6, height=4.45, corner_radius=0.04, color=BLACK, stroke_width=3)
        full_screen.set_fill(WHITE, opacity=1).move_to(DOWN * 0.25)
        self.play(
            Transform(screen, full_screen),
            self.camera.frame.animate.scale(1 / 0.38).move_to(ORIGIN),
            run_time=1.0,
        )

        header = Text(self.text("example_title"), font_size=34, color=self.DARK)
        header.to_edge(UP, buff=0.22)

        stage = self.stage_label(self.text("blank"), "1250 +/- 250 ms", header=header)
        self.play(FadeIn(header), FadeIn(stage), run_time=0.5)
        self.wait(1.25)

        pair = self.stimulus_pair(self.RED_STIM, self.GREEN_STIM, radius=0.78, gap=2.6).move_to(
            screen.get_center() + UP * 0.15
        )
        pair.set_opacity(0)
        self.play(
            Transform(stage, self.stage_label(self.text("ramp"), "240 ms", header=header)), FadeIn(pair), run_time=0.5
        )
        self.play(pair.animate.set_opacity(1), run_time=1.0)

        self.play(Transform(stage, self.stage_label(self.text("stim_on"), "1500 ms", header=header)), run_time=0.5)
        self.wait(1.5)

        self.play(
            Transform(stage, self.stage_label(self.text("stim_off"), "600 +/- 100 ms", header=header)),
            FadeOut(pair),
            run_time=0.5,
        )
        self.wait(1.0)

        response = self.response_choices(screen)
        self.play(
            Transform(stage, self.stage_label(self.text("response_choice"), "", header=header)),
            FadeIn(response),
            run_time=1.2,
        )
        self.wait(1.5)
        self.play(FadeOut(VGroup(header, screen, stage, response)), run_time=0.7)

    def show_variations(self):
        title = Text(self.text("variations_title"), font_size=36, color=self.DARK).to_edge(UP, buff=0.4)
        self.play(FadeIn(title), run_time=0.5)

        congruent_label = Text(self.text("congruent"), font_size=26, color=self.DARK)
        incongruent_label = Text(self.text("incongruent"), font_size=26, color=self.DARK)
        congruent_examples = VGroup(
            self.stimulus_pair(self.GREEN_STIM, self.GREEN_STIM, radius=0.5, gap=1.45),
            self.stimulus_pair(self.RED_STIM, self.RED_STIM, radius=0.5, gap=1.45),
        ).arrange(DOWN, buff=0.45)
        incongruent_examples = VGroup(
            self.stimulus_pair(self.GREEN_STIM, self.RED_STIM, radius=0.5, gap=1.45),
            self.stimulus_pair(self.RED_STIM, self.GREEN_STIM, radius=0.5, gap=1.45),
        ).arrange(DOWN, buff=0.45)

        congruent_group = (
            VGroup(congruent_label, congruent_examples).arrange(DOWN, buff=0.38).move_to(LEFT * 2.8 + DOWN * 0.05)
        )
        incongruent_group = (
            VGroup(incongruent_label, incongruent_examples).arrange(DOWN, buff=0.38).move_to(RIGHT * 2.8 + DOWN * 0.05)
        )
        congruent_split = Line(LEFT * 1.35, RIGHT * 1.35, color=GRAY_C)
        congruent_split.move_to((congruent_examples[0].get_bottom() + congruent_examples[1].get_top()) / 2)
        incongruent_split = Line(LEFT * 1.35, RIGHT * 1.35, color=GRAY_C)
        incongruent_split.move_to((incongruent_examples[0].get_bottom() + incongruent_examples[1].get_top()) / 2)
        divider = Line(UP * 1.4, DOWN * 1.7, color=GRAY_C)
        cards = VGroup(congruent_group, congruent_split, divider, incongruent_group, incongruent_split)

        self.play(FadeIn(cards, shift=UP * 0.2), run_time=1.0)
        self.wait(2.5)
        self.play(FadeOut(VGroup(title, cards)), run_time=0.7)

    def show_interpretation(self):
        title = Text(self.text("interpretation_title"), font_size=34, color=self.DARK)
        title.to_edge(UP, buff=0.4)

        incongruent = self.stimulus_pair(self.RED_STIM, self.GREEN_STIM, radius=0.48, gap=1.35).move_to(UP * 0.88)
        incongruent_label = Text(self.text("incongruent_stimulus"), font_size=24, color=GRAY_D).next_to(
            incongruent, UP, buff=0.25
        )

        pure = self.response_disc(self.RED_STIM, radius=0.42)
        or_text = Text(self.text("or"), font_size=30, color=self.DARK)
        pure_2 = self.response_disc(self.GREEN_STIM, radius=0.42)
        pure_group = VGroup(pure, or_text, pure_2).arrange(RIGHT, buff=0.28)
        pure_box = self.interpretation_box(self.text("pure"), pure_group, LEFT * 2.5 + DOWN * 1.8)

        mixed_disc = self.split_response_disc(self.RED_STIM, self.GREEN_STIM, radius=0.42)
        mixed_group = VGroup(mixed_disc)
        mixed_box = self.interpretation_box(self.text("mixed"), mixed_group, RIGHT * 2.5 + DOWN * 1.8)

        arrows = VGroup(
            Arrow(incongruent.get_bottom(), pure_box.get_top(), color=GRAY_D, buff=0.15),
            Arrow(incongruent.get_bottom(), mixed_box.get_top(), color=GRAY_D, buff=0.15),
        )

        self.play(FadeIn(title), FadeIn(incongruent), FadeIn(incongruent_label), run_time=0.8)
        self.play(Create(arrows), FadeIn(pure_box), FadeIn(mixed_box), run_time=1.2)
        self.wait(3)

    def rivalry_disc(self, color, radius=0.55):
        base_color = ManimColor(color).interpolate(WHITE, 0.18)
        base = Circle(radius=radius, color=BLACK, stroke_width=2).set_fill(base_color, opacity=1)
        stripes = VGroup()
        spacing = radius * 0.12
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
            stripe = Line(start, end, color=BLACK, stroke_width=radius * 2.6)
            stripe.set_opacity(self.GABOR_OPACITY)
            stripes.add(stripe)
        center = Circle(radius=radius * 0.18, color=BLACK, stroke_width=1.5).set_fill(BLACK, opacity=1)
        dot = Dot(radius=radius * 0.035, color=GRAY_C)
        return VGroup(base, stripes, center, dot)

    def stimulus_pair(self, left_color, right_color, radius=0.55, gap=1.6):
        left = self.rivalry_disc(left_color, radius=radius)
        right = self.rivalry_disc(right_color, radius=radius)
        left.move_to(LEFT * gap / 2)
        right.move_to(RIGHT * gap / 2)
        return VGroup(left, right)

    def response_disc(self, color, radius=0.26):
        return Circle(radius=radius, color=color, stroke_width=0).set_fill(color, opacity=1)

    def split_response_disc(self, left_color, right_color, radius=0.26):
        left = AnnularSector(inner_radius=0, outer_radius=radius, angle=PI, start_angle=PI / 2, color=left_color)
        right = AnnularSector(inner_radius=0, outer_radius=radius, angle=PI, start_angle=-PI / 2, color=right_color)
        divider = Line(UP * radius, DOWN * radius, color=WHITE, stroke_width=2)
        return VGroup(left, right, divider)

    def response_choices(self, screen):
        dots = VGroup()
        nums = VGroup()
        dot_radius = 0.5
        y = screen.get_center()[1]
        xs = [-1.42, 0, 1.42]
        response_items = [
            self.response_disc(self.RED_STIM, radius=dot_radius),
            self.split_response_disc(self.RED_STIM, self.GREEN_STIM, radius=dot_radius),
            self.response_disc(self.GREEN_STIM, radius=dot_radius),
        ]
        for i, (x, dot) in enumerate(zip(xs, response_items), start=1):
            dot.move_to([screen.get_center()[0] + x, y + 0.22, 0])
            num = Text(str(i), font_size=39, color=BLACK).move_to([screen.get_center()[0] + x, y - 0.64, 0])
            dots.add(dot)
            nums.add(num)
        return VGroup(dots, nums)

    def response_panel(self, anchor=None):
        panel = RoundedRectangle(width=1.78, height=0.48, corner_radius=0.025, color=BLACK, stroke_width=1.2)
        if anchor is None:
            panel.move_to(DOWN * 2.0)
            font_size = 22
            label_size = 24
            dot_radius = 0.26
            xs = [-0.78, 0, 0.78]
        else:
            panel.next_to(anchor, DOWN, buff=0.12)
            font_size = 9
            label_size = 10
            dot_radius = 0.105
            xs = [-0.32, 0, 0.32]
        panel.set_fill(WHITE, opacity=1)
        dots = VGroup()
        nums = VGroup()
        response_items = [
            self.response_disc(self.RED_STIM, radius=dot_radius),
            self.split_response_disc(self.RED_STIM, self.GREEN_STIM, radius=dot_radius),
            self.response_disc(self.GREEN_STIM, radius=dot_radius),
        ]
        for i, (x, dot) in enumerate(zip(xs, response_items), start=1):
            dot.move_to(panel.get_center() + RIGHT * x + UP * (dot_radius * 0.68))
            num = Text(str(i), font_size=font_size, color=BLACK).move_to(
                panel.get_center() + RIGHT * x + DOWN * (dot_radius * 1.25)
            )
            dots.add(dot)
            nums.add(num)
        label = Text(self.text("response"), font_size=label_size, color=BLACK).next_to(
            panel, RIGHT, buff=0.08 if anchor is not None else 0.25
        )
        return VGroup(panel, dots, nums, label)

    def stage_label(self, name, timing, anchor=None, header=None):
        font_size = 11 if anchor is not None else 27
        if timing:
            group = Text(f"{name}   {timing}", font_size=font_size, color=self.DARK)
        else:
            group = Text(name, font_size=font_size, color=self.DARK)
        if anchor is None:
            if header is None:
                group.to_edge(RIGHT, buff=0.8).shift(UP * 0.95)
            else:
                group.next_to(header, DOWN, buff=0.24)
        else:
            group.next_to(anchor, RIGHT, buff=0.16).shift(UP * 0.35)
        return group

    def variation_card(self, name, left_color, right_color):
        title = Text(name, font_size=20, color=self.DARK)
        pair = self.stimulus_pair(left_color, right_color, radius=0.38, gap=1.15)
        group = VGroup(title, pair).arrange(DOWN, buff=0.28)
        return group

    def interpretation_box(self, label, content, position):
        title = Text(label, font_size=26, color=self.DARK)
        content.next_to(title, DOWN, buff=0.35)
        box = RoundedRectangle(width=3.7, height=1.85, corner_radius=0.05, color=GRAY_B, stroke_width=2)
        group = VGroup(box, title, content)
        group.move_to(position)
        title.move_to(box.get_top() + DOWN * 0.35)
        content.move_to(box.get_center() + DOWN * 0.3)
        return group
