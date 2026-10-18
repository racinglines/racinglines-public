"""
L1 input frames (docs/frames.md, docs/engine-roadmap.md): each sport's data in the engine's one shape.

    frames_for("f1", meas)          {name: DataFrame} for exactly the frames sports/f1.toml [data] frames lists,
                                    each checked against its declaration in racinglines.frames.schema
    validate("laps", df)            check one frame (re-exported from schema)

A sport's adapter is the module racinglines/frames/<sport code>.py: a `BUILDERS = {frame name: function(data)}` dict,
where `data` is what that sport's model `load()` returns (`Measurements` for F1, the tidy runs frame for downhill).
So a new sport needs its adapter and its `[data] frames` line, and nothing here.
"""

from importlib import import_module

from racinglines import sports
from racinglines.frames.schema import (FRAMES, SCHEMAS, Column, FrameError, FrameSchema, check_release,  # noqa: F401
                                      schema, validate)


def adapter(sport):
    """The sport's adapter module (racinglines/frames/<code>.py)."""
    name = f"{__name__}.{sport}"
    try:
        return import_module(name)
    except ModuleNotFoundError as ex:
        if ex.name != name:
            raise
        raise FrameError(f"sports/{sport}.toml declares frames, but there is no adapter racinglines/frames/{sport}.py") from None


def frames_for(sport, data, check=True):
    """The frames a sport declares ([data] frames in its schema), built from `data` by its adapter, in the declared
    order. A sport with no `[data]` block gets {} and its adapter isn't loaded. Each frame is validated, and
    session-keyed frames must be released a whole session at a time, after it ends, unless `[data] live` lists them
    (schema.check_release). `check=False` skips both."""
    try:
        names = sports.frames(sport)
    except FileNotFoundError:
        raise FrameError(f"unknown sport {sport!r}: there is no sports/{sport}.toml") from None
    unknown = [n for n in names if n not in SCHEMAS]
    if unknown:
        raise FrameError(f"sports/{sport}.toml declares unknown frames {unknown}; the declared frames are {list(FRAMES)}")
    if not names:
        return {}
    builders = adapter(sport).BUILDERS
    cannot = [n for n in names if n not in builders]
    if cannot:
        raise FrameError(f"sports/{sport}.toml declares {cannot}, which racinglines/frames/{sport}.py can't build "
                         f"(it builds {sorted(builders)})")
    out = {n: builders[n](data) for n in names}
    if check:
        for n, df in out.items():
            validate(n, df)
        check_release(out, live=sports.live_frames(sport))
    return out
