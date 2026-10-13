# SPDX-License-Identifier: Apache-2.0
"""The Fastest-10 frames in openftgs.data (D-69; Q-59).

Expected values come from sequences built so that the window of largest motion is known by
construction: frames that stay constant except at chosen transitions, whose mean absolute
differences are worked by hand.
"""

from collections.abc import Iterator

import pytest
import torch

from openftgs.data import fastest_frames

F32 = torch.float32


def _frames(levels: list[float], *, shape: tuple[int, int] = (4, 6)) -> list[torch.Tensor]:
    """Constant frames: frame i is `levels[i]` everywhere, so consecutive frames differ by
    |levels[i + 1] − levels[i]| on average."""
    return [torch.full((*shape, 3), level, dtype=F32) for level in levels]


def test_fastest_frames_finds_the_window_of_largest_motion() -> None:
    """D-69: the 10 consecutive frames whose consecutive frames differ most. Twenty frames,
    constant except for jumps between frames 7 and 8, 9 and 10 and 12 and 13: the windows
    containing all three jumps start at 4 … 7, so the lowest, 4, wins (Q-59)."""
    levels = [0.0] * 8 + [0.5] * 2 + [0.1] * 3 + [0.9] * 7
    assert fastest_frames(_frames(levels)) == 4


def test_fastest_frames_counts_only_differences_inside_the_window() -> None:
    """Q-59: a window scores its count − 1 internal differences. With count = 3, the jump
    from frame 5 to 6 lies inside the windows starting at 4 and 5, not at 6, where frame 6 is
    the window's first frame; the larger jump from 9 to 10 is caught by windows 8 and 9."""
    levels = [0.0] * 6 + [0.3] * 4 + [1.0] * 4
    assert fastest_frames(_frames(levels), count=3) == 8


def test_fastest_frames_breaks_ties_to_the_lowest_start() -> None:
    """Q-59: equal scores go to the lowest start (as D-44 breaks ties). Two equal jumps, at
    2→3 and 10→11, each alone in their windows of 4 frames."""
    levels = [0.0] * 3 + [0.5] * 8 + [1.0] * 4
    assert fastest_frames(_frames(levels), count=4) == 0


def test_fastest_frames_streams_frames_and_takes_uint8() -> None:
    """Q-59: frames may come from a generator, and uint8 frames give the start their floating
    counterparts give."""
    levels = [0.0] * 8 + [0.5] * 2 + [0.1] * 3 + [0.9] * 7

    def stream() -> Iterator[torch.Tensor]:
        for frame in _frames(levels):
            yield (frame * 255).round().to(torch.uint8)

    assert fastest_frames(stream()) == fastest_frames(_frames(levels)) == 4


def test_fastest_frames_of_exactly_count_frames_is_zero() -> None:
    assert fastest_frames(_frames([0.0, 1.0] * 5)) == 0


@pytest.mark.parametrize("count", [1, 0, -3, True, 2.0, "10"])
def test_fastest_frames_needs_a_count_of_at_least_two(count: object) -> None:
    with pytest.raises(ValueError, match="count"):
        fastest_frames(_frames([0.0] * 12), count=count)  # type: ignore[arg-type]


def test_fastest_frames_needs_count_frames() -> None:
    with pytest.raises(ValueError, match="frames"):
        fastest_frames(_frames([0.0] * 9))


@pytest.mark.parametrize(
    "bad", [torch.zeros(4, 7, 3), torch.zeros(4, 6, 4), torch.zeros(4, 6), torch.zeros(1, 4, 6, 3)]
)
def test_fastest_frames_needs_frames_of_one_hw3_shape(bad: torch.Tensor) -> None:
    frames = _frames([0.0] * 12)
    frames[5] = bad
    with pytest.raises(ValueError, match="shape"):
        fastest_frames(frames)
