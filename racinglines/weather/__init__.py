"""
Weather forecasts the models can read (docs/weather.md): one provider-agnostic frame, saved per event and per issue.

    schema.py        the forecast frame: COLUMNS, validate, summary, save / load
    open_meteo.py    the default provider: Open-Meteo (no key; free for non-commercial use), verified by the probe
                     of 2026-10-06.
    wunderground.py  optional: Weather Underground (The Weather Company's v3 API, a PWS owner's key). Its endpoints
                     and field names are unverified: `racinglines weather probe --provider wunderground` confirms them.

Nothing here runs by default, writes to the database, or touches the VM. The consumers (props.py rain and red-flag
rates, position_sim's disruption mixture) are wired separately.
"""

from racinglines.weather.schema import (  # noqa: F401
    COLUMNS,
    SESSIONS,
    load,
    save,
    summary,
    validate,
)
