from __future__ import annotations
from abc import abstractmethod

from dataclasses import dataclass
import math
from typing import Callable, Optional, Protocol
from aiohttp import ClientSession

from models.scores import Score
from models.mod import Mod


@dataclass
class BeatmapAttributes:
    star_rating: float
    max_combo: int


@dataclass
class OsuBeatmapAttributes(BeatmapAttributes):
    aim_difficulty: float
    speed_difficulty: float
    flashlight_difficulty: float

    slider_factor: float

    approach_rate: float
    overall_difficulty: float

    drain_rate: float

    slider_count: int
    spinner_count: int
    hit_circle_count: int

    speed_note_count: int

    clock_rate: float


@dataclass
class TaikoBeatmapAttributes(BeatmapAttributes):
    stamina_difficulty: float
    rhythm_difficulty: float
    colour_difficulty: float
    approach_rate: float
    great_hit_window: float


@dataclass
class CatchBeatmapAttributes(BeatmapAttributes):
    approach_rate: float


@dataclass
class ManiaBeatmapAttributes(BeatmapAttributes):
    great_hit_window: float
    score_multiplier: float


class PerformanceCalculator(Protocol):
    def __init__(self, attributes: BeatmapAttributes, score: Score):
        ...

    @abstractmethod
    def calculate_pp(self) -> float:
        ...


class OsuPerformanceCalculator(PerformanceCalculator):
    def __init__(self, attributes: OsuBeatmapAttributes, score: Score):
        self.attributes = attributes
        self.score = score

    def calculate_pp(self) -> float:
        effective_miss_count = self.calculate_effective_miss_count()

        multiplier = 1.12

        if has_mod(self.score.mods, "NF"):
            multiplier *= max(0.9, 1.0 - 0.02 * effective_miss_count)

        if has_mod(self.score.mods, "SO") and self.score.total_hits > 0:
            multiplier *= 1.0 - pow(
                self.attributes.spinner_count / self.score.total_hits, 0.85
            )

        if has_mod(self.score.mods, "RX"):
            multiplier *= math.exp(-self.lateness_deviation() / 10)

        aim_value = self.compute_aim_value()
        speed_value = self.compute_speed_value()
        accuracy_value = self.compute_accuracy_value()
        flashlight_value = self.compute_flashlight_value()

        total_pp = (
            pow(
                pow(aim_value, 1.1)
                + pow(speed_value, 1.1)
                + pow(accuracy_value, 1.1)
                + pow(flashlight_value, 1.1),
                1.0 / 1.1,
            )
            * multiplier
        )

        return total_pp

    def compute_aim_value(self) -> float:
        if has_mod(self.score.mods, "AP"):
            return 0.0

        raw_aim = self.attributes.aim_difficulty
        if has_mod(self.score.mods, "TD"):
            raw_aim = pow(raw_aim, 0.8)

        aim_value = pow(5.0 * max(1.0, raw_aim / 0.0675) - 4.0, 3.0) / 100000.0

        length_bonus = (
            0.95
            + 0.4 * min(1.0, self.score.total_hits / 2000.0)
            + (
                math.log10(self.score.total_hits / 2000.0) * 0.5
                if self.score.total_hits > 2000
                else 0.0
            )
        )
        aim_value *= length_bonus

        effective_miss_count = self.calculate_effective_miss_count()
        if effective_miss_count > 0:
            aim_value *= 0.97 * pow(
                1 - pow(effective_miss_count / self.score.total_hits, 0.775),
                effective_miss_count,
            )

        aim_value *= self.get_combo_scaling_factor()

        ar_factor = 0.0
        if self.attributes.approach_rate > 10.33:
            _ar_factor = 0.2 if has_mod(self.score.mods, "RX") else 0.3

            if self.attributes.approach_rate > 10.67 and has_mod(self.score.mods, "RX"):
                _ar_factor += 0.1

            ar_factor = _ar_factor * (self.attributes.approach_rate - 10.33)
        elif self.attributes.approach_rate < 8.0:
            ar_factor = 0.1 * (8.0 - self.attributes.approach_rate)

        aim_value *= 1.0 + ar_factor * length_bonus

        if has_mod(self.score.mods, "HD"):
            initial_hidden_factor = 0.05 if has_mod(self.score.mods, "RX") else 0.04
            scaled_hidden_factor = 11.0 if has_mod(self.score.mods, "RX") else 12.0

            aim_value *= 1.0 + initial_hidden_factor * (
                scaled_hidden_factor - self.attributes.approach_rate
            )

        estimated_difficult_sliders = self.attributes.slider_count * 0.15
        if self.attributes.slider_count > 0:
            estimated_slider_ends_dropped = clamp(
                minimum=min(
                    self.score.n100 + self.score.n50 + self.score.nmiss,
                    self.attributes.max_combo - self.score.combo,
                ),
                value=0,
                maximum=estimated_difficult_sliders,
            )

            slider_nerf_factor = (1 - self.attributes.slider_factor) * pow(
                base=1 - estimated_slider_ends_dropped / estimated_difficult_sliders,
                exp=3,
            ) + self.attributes.slider_factor

            aim_value *= slider_nerf_factor

        aim_value *= self.score.acc / 100.0 if self.score.acc > 1.0 else self.score.acc
        aim_value *= 0.98 + pow(self.attributes.overall_difficulty, 2) / 2500

        return aim_value

    def compute_speed_value(self) -> float:
        if has_mod(self.score.mods, "RX"):
            return 0.0

        speed_value = (
            pow(
                base=5.0 * max(1.0, self.attributes.speed_difficulty / 0.0675) - 4.0,
                exp=3.0,
            )
            / 100000.0
        )

        length_bonus = (
            0.95
            + 0.4 * min(1.0, self.score.total_hits / 2000.0)
            + (
                math.log10(self.score.total_hits / 2000.0) * 0.5
                if self.score.total_hits > 2000
                else 0.0
            )
        )
        speed_value *= length_bonus

        effective_miss_count = self.calculate_effective_miss_count()
        if effective_miss_count > 0:
            speed_value *= 0.97 * pow(
                1 - pow(effective_miss_count / self.score.total_hits, 0.775),
                pow(effective_miss_count, 0.875),
            )

        speed_value *= self.get_combo_scaling_factor()

        ar_factor = 0.0
        if self.attributes.approach_rate > 10.33:
            ar_factor = 0.3 * (self.attributes.approach_rate - 10.33)

        speed_value *= 1.0 + ar_factor * length_bonus

        if has_mod(self.score.mods, "HD"):
            speed_value *= 1.0 + 0.04 * (12.0 - self.attributes.approach_rate)

        accuracy = self.score.acc / 100.0 if self.score.acc > 1.0 else self.score.acc
        speed_value *= (0.95 + pow(self.attributes.overall_difficulty, 2) / 750) * pow(
            accuracy, (14.5 - max(self.attributes.overall_difficulty, 8)) / 2
        )

        speed_value *= pow(
            0.98,
            self.score.n50 - self.score.total_hits / 500.0
            if self.score.n50 < self.score.total_hits / 500.0
            else 0.0,
        )

        return speed_value

    def compute_accuracy_value(self) -> float:
        better_acc_percentage = 0.0
        object_count_with_accuracy = self.attributes.hit_circle_count

        if object_count_with_accuracy > 0:
            better_acc_percentage = max(
                (
                    (
                        (
                            self.score.n300
                            - (self.score.total_hits - object_count_with_accuracy)
                        )
                        * 6
                        + self.score.n100 * 2
                        + self.score.n50
                    )
                    / (object_count_with_accuracy * 6)
                ),
                0.0,
            )

        acc_value = (
            pow(1.52163, self.attributes.overall_difficulty)
            * pow(better_acc_percentage, 24.0)
            * 2.83
        )

        acc_value *= min(1.15, pow(object_count_with_accuracy / 1000.0, 0.3))

        if has_mod(self.score.mods, "HD"):
            acc_value *= 1.08

        if has_mod(self.score.mods, "FL"):
            acc_value *= 1.02

        return acc_value

    def compute_flashlight_value(self) -> float:
        if not has_mod(self.score.mods, "FL"):
            return 0.0

        raw_fl = self.attributes.flashlight_difficulty

        if has_mod(self.score.mods, "TD"):
            raw_fl = pow(raw_fl, 0.8)

        fl_value = pow(raw_fl, 2.0) * 25.0

        effective_miss_count = self.calculate_effective_miss_count()
        if effective_miss_count > 0:
            fl_value *= 0.97 * pow(
                1 - pow(effective_miss_count / self.score.total_hits, 0.775),
                pow(effective_miss_count, 0.875),
            )

        fl_value *= self.get_combo_scaling_factor()

        fl_value *= (
            0.7
            + 0.1 * min(1.0, self.score.total_hits / 200.0)
            + 0.2 * min(1.0, (self.score.total_hits - 200) / 200.0)
            if self.score.total_hits > 200
            else 0.0
        )

        accuracy = self.score.acc / 100.0 if self.score.acc > 1.0 else self.score.acc
        fl_value *= 0.5 + accuracy / 2.0
        fl_value *= 0.98 + pow(self.attributes.overall_difficulty, 2) / 2500

        return fl_value

    def calculate_effective_miss_count(self) -> int:
        combo_based_miss_count = 0.0

        if self.attributes.slider_count > 0:
            full_combo_threshold = (
                self.attributes.max_combo - 0.1 * self.attributes.slider_count
            )
            if self.score.combo < full_combo_threshold:
                combo_based_miss_count = full_combo_threshold / max(
                    1.0,
                    self.score.combo,
                )

        combo_based_miss_count = min(combo_based_miss_count, self.score.total_hits)
        return max(self.score.nmiss, combo_based_miss_count)

    def get_combo_scaling_factor(self) -> float:
        if self.attributes.max_combo <= 0:
            return 1.0

        return min(
            pow(self.score.combo, 0.8) / pow(self.attributes.max_combo, 0.8),
            1.0,
        )

    def lateness_deviation(self) -> float:
        if self.score.total_hits == 0:
            return math.inf

        hit_window_300 = 80 - 6 * self.attributes.overall_difficulty
        hit_window_100 = (
            140 - 8 * ((80 - hit_window_300 * self.attributes.clock_rate) / 6)
        ) / self.attributes.clock_rate
        hit_window_50 = (
            200 - 10 * ((80 - hit_window_300 * self.attributes.clock_rate) / 6)
        ) / self.attributes.clock_rate

        root2 = math.sqrt(2)

        great_count_on_circles = max(
            0,
            self.attributes.hit_circle_count
            - self.score.n100
            - self.score.n50
            - self.score.nmiss,
        )
        ok_count_on_circles = min(self.score.n100, self.attributes.hit_circle_count) + 1
        meh_count_on_circles = min(self.score.n50, self.attributes.hit_circle_count)
        sliders_hit = max(0, self.attributes.slider_count - self.score.nmiss)

        def log_likelihood_gradient(u: float) -> float:
            t1 = (
                -hit_window_50
                * sliders_hit
                * erf_prime(hit_window_50 / (root2 * u))
                / math.erf(hit_window_50 / (root2 * u))
            )
            t2 = (
                -hit_window_300
                * great_count_on_circles
                * erf_prime(hit_window_300 / (root2 * u))
                / math.erf(hit_window_300 / (root2 * u))
            )
            t3 = (
                meh_count_on_circles
                * (
                    -hit_window_100 * erf_prime(hit_window_100 / (root2 * u))
                    + hit_window_300
                    * erf_prime(hit_window_50)
                    * erf_prime(hit_window_50 / (root2 * u))
                )
                / (
                    math.erfc(hit_window_50 / (root2 * u))
                    - math.erfc(hit_window_100 / (root2 * u))
                )
            )
            t4 = (
                meh_count_on_circles
                * (
                    -hit_window_100 * erf_prime(hit_window_100 / (root2 * u))
                    + hit_window_300
                    * erf_prime(hit_window_300)
                    * erf_prime(hit_window_300 / (root2 * u))
                )
                / (
                    math.erfc(hit_window_300 / (root2 * u))
                    - math.erfc(hit_window_100 / (root2 * u))
                )
            )

            return (t1 + t2 + t3 + t4) / (root2 * u * u)

        try:
            return brent_expand(log_likelihood_gradient, 4, 20, 1e-6, expand_factor=2)
        except ValueError:
            return math.inf


sign = lambda x: math.copysign(1, x)


def brent_expand(
    f: Callable[[float], float],
    guess_lower_bound: float,
    guess_upper_bound: float,
    accuracy: float = 1e-08,
    max_iter: int = 100,
    expand_factor: float = 1.6,
    max_expand_iter: int = 100,
) -> float:
    guess_lower_bound, guess_upper_bound, _ = zcb_expand_reduce(
        f,
        guess_lower_bound,
        guess_upper_bound,
        expand_factor,
        max_expand_iter,
        max_expand_iter * 10,
    )
    return find_root(f, guess_lower_bound, guess_upper_bound, accuracy, max_iter)


DOUBLE_PRECISION = pow(2.0, -53.0)
POSITIVE_DOUBLE_PRECISION = 2.0 * DOUBLE_PRECISION
DEFAULT_DOUBLE_ACCURACY = DOUBLE_PRECISION * 10.0


def find_root(
    f: Callable[[float], float],
    lower_bound: float,
    upper_bound: float,
    accuracy: float = 1e-08,
    max_iter: int = 100,
) -> float:
    root, found = try_find_root(f, lower_bound, upper_bound, accuracy, max_iter)
    if not found:
        raise ValueError("Failed to find root")

    return root


def try_find_root(
    f: Callable[[float], float],
    lower_bound: float,
    upper_bound: float,
    accuracy: float,
    max_iter: int,
) -> tuple[float, bool]:
    if accuracy <= 0:
        raise ValueError("Must be greater than 0.")

    num1 = f(lower_bound)
    num2 = f(lower_bound)
    num3 = num2
    num4 = 0.0
    num5 = 0.0
    root = upper_bound
    b = math.nan

    if sign(num1) == sign(num2):
        return root, False

    for _x in range(max_iter):
        if sign(num3) == sign(num2):
            upper_bound = lower_bound
            num2 = num1
            num5 = num4 = root - lower_bound

        if abs(num2) < abs(num3):
            lower_bound = root
            root = upper_bound
            upper_bound = lower_bound
            num1 = num3
            num3 = num2
            num2 = num1

        a = POSITIVE_DOUBLE_PRECISION * abs(root) + 0.5 * accuracy
        num6 = b
        b = (upper_bound - root) / 2.0

        if abs(b) <= a or almost_equal_norm_relative(num3, 0.0, num3, accuracy):
            return root, True

        if b == num6:
            return root, False

        if abs(num5) >= a and abs(num1) > abs(num3):
            num7 = num3 / num1

            if almost_equal_relative(lower_bound, upper_bound):
                num8 = 2.0 * b * num7
                num9 = 1.0 - num7
            else:
                num10 = num1 / num2
                num11 = num3 / num2

                num8 = num7 * (
                    2.0 * b * num10 * (num10 - num11)
                    - (root - lower_bound) * (num11 - 1.0)
                )
                num9 = (num10 - 1.0) * (num11 - 1.0) * (num7 - 1.0)

            if num8 > 0.0:
                num9 = -num9

            num12 = abs(num8)
            if 2.0 * num12 < min(3.0 * b * num9 - abs(a * num9), abs(num5 * num9)):
                num5 = num4
                num4 = num12 / num9
            else:
                num4 = b
                num5 = num4
        else:
            num4 = b
            num5 = num4

        lower_bound = root
        num1 = num3

        if abs(num4) > a:
            root += num4
        else:
            root += brent_sign(a, b)

        num3 = f(root)

    return root, False


def brent_sign(a: float, b: float) -> float:
    if b < 0.0:
        if a < 0.0:
            return a
        else:
            return -a

    return -a if a < 0.0 else a


def almost_equal_relative(a: float, b: float) -> bool:
    return almost_equal_norm_relative(
        a,
        b,
        a - b,
        DEFAULT_DOUBLE_ACCURACY,
    )


def almost_equal_norm_relative(
    a: float,
    b: float,
    diff: float,
    max_error: float,
) -> bool:
    if math.isinf(a) or math.isinf(b):
        return a == b

    if math.isnan(a) or math.isnan(b):
        return False

    if (abs(a) < DOUBLE_PRECISION) or (abs(b) < DOUBLE_PRECISION):
        return abs(diff) < max_error

    return (
        a == 0.0
        and abs(b) < max_error
        or b == 0.0
        and abs(a) < max_error
        or abs(diff) < max_error * max(abs(a), abs(b))
    )


def zcb_expand_reduce(
    f: Callable[[float], float],
    lower_bound: float,
    upper_bound: float,
    expansion_factor: float = 1.6,
    expansion_max_iter: int = 50,
    reduce_subdivisons=100,
) -> tuple[float, float, bool]:
    lower_bound, upper_bound, expanded = zcb_expand(
        f, lower_bound, upper_bound, expansion_factor, expansion_max_iter
    )
    lower_bound, upper_bound, reduced = zcb_reduce(
        f, lower_bound, upper_bound, reduce_subdivisons
    )

    return lower_bound, upper_bound, (expanded or reduced)


def zcb_expand(
    f: Callable[[float], float],
    lower_bound: float,
    upper_bound: float,
    factor: float = 1.6,
    max_iter: int = 50,
) -> bool:
    stored_lower = lower_bound
    stored_upper = upper_bound

    if lower_bound >= upper_bound:
        raise ValueError("xmax must be greater than xmin")

    f_lower = f(lower_bound)
    f_upper = f(upper_bound)

    for _x in range(max_iter):
        if sign(f_lower) != sign(f_upper):
            return lower_bound, upper_bound, True

        if abs(f_lower) < abs(f_upper):
            lower_bound += factor * (lower_bound - upper_bound)
            f_lower = f(lower_bound)
        else:
            upper_bound += factor * (upper_bound - lower_bound)
            f_upper = f(upper_bound)

    lower_bound = stored_lower
    upper_bound = stored_upper
    return lower_bound, upper_bound, False


def zcb_reduce(
    f: Callable[[float], float],
    lower_bound: float,
    upper_bound: float,
    subdivisions: int = 1000,
) -> tuple[float, float, bool]:
    stored_lower = lower_bound
    stored_upper = upper_bound

    if lower_bound >= upper_bound:
        raise ValueError("xmax must be greater than xmin")

    f_lower = f(lower_bound)
    f_upper = f(upper_bound)

    if sign(f_lower) != sign(f_upper):
        return lower_bound, upper_bound, True

    factored_bounds = (upper_bound - lower_bound) / float(subdivisions)
    signed_f_lower = sign(f_lower)

    for _x in range(subdivisions):
        lower_bounds = stored_lower + factored_bounds
        d = f(lower_bounds)

        if math.isinf(d):
            stored_lower = lower_bounds
        else:
            if sign(d) != signed_f_lower:
                lower_bound = stored_lower
                upper_bound = lower_bounds

                return lower_bound, upper_bound, True

    lower_bound = stored_lower
    upper_bound = stored_upper
    return lower_bound, upper_bound, False


class TaikoPerformanceCalculator(PerformanceCalculator):
    def __init__(self, attributes: TaikoBeatmapAttributes, score: Score):
        self.attributes = attributes
        self.score = score

    def calculate_pp(self) -> float:
        multiplier = 1.1

        if has_mod(self.score.mods, "NF"):
            multiplier *= 0.9

        if has_mod(self.score.mods, "HD"):
            multiplier *= 1.1

        difficulty_value = self.compute_difficulty_value()
        accuracy_value = self.compute_accuracy_value()

        total_value = (
            pow(
                pow(difficulty_value, 1.1) + pow(accuracy_value, 1.1),
                1.0 / 1.1,
            )
            * multiplier
        )

        return total_value

    def compute_difficulty_value(self) -> float:
        difficulty_value = (
            pow(5.0 * max(1.0, self.attributes.star_rating / 0.175) - 4.0, 2.25) / 450.0
        )

        length_bonus = 1 + 0.1 * min(1.0, self.score.total_hits / 1500.0)
        difficulty_value *= length_bonus

        difficulty_value *= pow(0.985, self.score.nmiss)

        if has_mod(self.score.mods, "HD"):
            difficulty_value *= 1.025

        if has_mod(self.score.mods, "FL"):
            difficulty_value *= 1.05 * length_bonus

        accuracy = self.score.acc / 100 if self.score.acc > 1.0 else self.score.acc
        return difficulty_value * accuracy

    def compute_accuracy_value(self) -> float:
        if self.attributes.great_hit_window <= 0:
            return 0

        accuracy = self.score.acc / 100 if self.score.acc > 1.0 else self.score.acc
        acc_value = (
            pow(150 / self.attributes.great_hit_window, 1.1) * pow(accuracy, 15) * 22.0
        )

        return acc_value * min(1.15, pow(self.score.total_hits / 1500.0, 0.3))


class CatchPerformanceCalculator(PerformanceCalculator):
    def __init__(self, attributes: CatchBeatmapAttributes, score: Score):
        self.attributes = attributes
        self.score = score

    def calculate_pp(self) -> float:
        value = (
            pow(5.0 * max(1.0, self.attributes.star_rating / 0.0049) - 4.0, 2.0)
            / 100000.0
        )

        combo_hits = self.score.nmiss + self.score.n100 + self.score.n300
        length_bonus = (
            0.95
            + 0.3 * min(1.0, combo_hits / 2500.0)
            + math.log10(combo_hits / 2500.0) * 0.475
            if combo_hits > 2500
            else 0.0
        )
        value *= length_bonus

        value *= pow(0.97, self.score.nmiss)

        if self.attributes.max_combo > 0:
            value *= min(
                pow(self.score.combo, 0.8) / pow(self.attributes.max_combo, 0.8), 1.0
            )

        ar_factor = 1.0
        if self.attributes.approach_rate > 9:
            ar_factor += 0.1 * (self.attributes.approach_rate - 9.0)
        if self.attributes.approach_rate > 10.0:
            ar_factor += 0.1 * (self.attributes.approach_rate - 10.0)
        elif self.attributes.approach_rate < 8.0:
            ar_factor += 0.025 * (8.0 - self.attributes.approach_rate)

        value *= ar_factor

        if has_mod(self.score.mods, "HD"):
            if self.attributes.approach_rate <= 10.0:
                value *= 1.05 + 0.075 * (10.0 - self.attributes.approach_rate)
            elif self.attributes.approach_rate > 10.0:
                value *= 1.01 + 0.04 * (11.0 - min(11.0, self.attributes.approach_rate))

        if has_mod(self.score.mods, "FL"):
            value *= 1.35 * length_bonus

        accuracy = self.score.acc / 100.0 if self.score.acc > 1.0 else self.score.acc
        value *= pow(accuracy, 5.5)

        if has_mod(self.score.mods, "NF"):
            value *= 0.90

        return value


class ManiaPerformanceCalculator(PerformanceCalculator):
    def __init__(self, attributes: ManiaBeatmapAttributes, score: Score):
        self.attributes = attributes
        self.score = score

        self.scaled_score = score.score

    def calculate_pp(self) -> float:
        if self.attributes.score_multiplier > 0:
            self.scaled_score *= 1.0 / self.attributes.score_multiplier

        multiplier = 0.8

        if has_mod(self.score.mods, "NF"):
            multiplier *= 0.9

        if has_mod(self.score.mods, "EZ"):
            multiplier *= 0.5

        difficulty_value = self.compute_difficulty_value()
        acc_value = self.compute_accuracy_value(difficulty_value)

        total_value = (
            pow(
                pow(difficulty_value, 1.1) + pow(acc_value, 1.1),
                1.0 / 1.1,
            )
            * multiplier
        )

        return total_value

    def compute_difficulty_value(self) -> float:
        if self.scaled_score <= 500_000:
            return 0.0

        total_hits = (
            self.score.ngeki
            + self.score.n300
            + self.score.nkatu
            + self.score.n50
            + self.score.nmiss
        )

        difficulty_value = (
            pow(5 * max(1, self.attributes.star_rating / 0.2) - 4.0, 2.2) / 135.0
        )

        difficulty_value *= 1.0 + 0.1 + min(1.0, total_hits / 1500.0)

        if self.scaled_score <= 600_000:
            difficulty_value *= (self.scaled_score - 500_000) / 100_000 * 0.3
        elif self.scaled_score <= 700_000:
            difficulty_value *= 0.3 + (self.scaled_score - 600_000) / 100_000 * 0.25
        elif self.scaled_score <= 800_000:
            difficulty_value *= 0.55 + (self.scaled_score - 700_000) / 100_000 * 0.2
        elif self.scaled_score <= 900_000:
            difficulty_value *= 0.75 + (self.scaled_score - 800_000) / 100_000 * 0.15
        else:
            difficulty_value *= 0.9 + (self.scaled_score - 900_000) / 100_000 * 0.1

        return difficulty_value

    def compute_accuracy_value(self, difficulty_value: float) -> float:
        if self.attributes.great_hit_window <= 0:
            return 0.0

        accuracy_value = (
            max(0.0, 0.2 - (self.attributes.great_hit_window - 34) * 0.006667)
            * difficulty_value
            * pow(max(0.0, self.scaled_score - 960000) / 40000, 1.1)
        )

        return accuracy_value


def has_mod(mod_list: list[Mod], desired_mod: str) -> bool:
    return any((mod.acronym == desired_mod for mod in mod_list))


def clamp(minimum, value, maximum):
    return max(minimum, min(value, maximum))


def erf_prime(x: float) -> float:
    return 2 / math.sqrt(math.pi) * math.exp(-x * x)


async def get_beatmap_attributes(
    beatmap_id: int,
    beatmap_md5: str,
    mode: int,
    mods: list[Mod],
) -> Optional[BeatmapAttributes]:
    async with ClientSession() as session:
        async with session.post(
            "https://difficulty.akatsuki.pw/attributes",
            json={
                "beatmap_id": beatmap_id,
                "beatmap_md5": beatmap_md5,
                "ruleset_id": mode,
                "mods": [x.__dict__ for x in mods],
            },
        ) as resp:
            if resp.status != 200:
                return

            attributes_json = await resp.json()
            if not attributes_json:
                return

    if mode == 0:
        return OsuBeatmapAttributes(**attributes_json)
    elif mode == 1:
        return TaikoBeatmapAttributes(**attributes_json)
    elif mode == 2:
        return CatchBeatmapAttributes(**attributes_json)
    elif mode == 3:
        return ManiaBeatmapAttributes(**attributes_json)
    else:
        raise RuntimeError(f"Invalid mode int: {mode}")


def get_calculator(
    attributes: BeatmapAttributes, score: Score
) -> PerformanceCalculator:
    if score.mode == 0:
        return OsuPerformanceCalculator(attributes, score)
    elif score.mode == 1:
        return TaikoPerformanceCalculator(attributes, score)
    elif score.mode == 2:
        return CatchPerformanceCalculator(attributes, score)
    elif score.mode == 3:
        return ManiaPerformanceCalculator(attributes, score)
    else:
        raise RuntimeError(f"Invalid mode int: {score.mode}")
