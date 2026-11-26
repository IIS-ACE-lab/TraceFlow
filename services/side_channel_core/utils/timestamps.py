# services/side_channel_core/utils/timestamps.py
"""
System-wide timestamps in the configured local timezone.

The zone is derived from services/side_channel_core/config/master.yaml:

    timezone:
      current_location: "Taipei"     # a city
      current_country: "Taiwan"      # a country name or ISO code

and resolved to an IANA zone (Taipei, Taiwan -> Asia/Taipei):

1. the location is itself an IANA name ("Asia/Taipei")        -> use it;
2. the location matches the city part of a zone of that country -> that zone;
3. the country has one zone, or all its zones keep the same clock -> its
   principal zone (Bochum, Germany -> Europe/Berlin);
4. otherwise the zones of the country are listed in the error.

Every timestamp TraceFlow writes (registries, reports, metric histories,
logs) comes from ``now()`` / ``now_iso()`` here, so all carry the same
offset, e.g. 2026-09-21T19:01:57+08:00. ``apply_process_timezone()`` also
sets ``TZ`` so logging, ``time.strftime`` and shell ``date`` agree.

Author: Shraddha Mishra
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo, available_timezones

import yaml

_MASTER_YAML = Path(__file__).resolve().parents[1] / "config" / "master.yaml"


class TimezoneConfigError(ValueError):
    """master.yaml timezone cannot be resolved to one IANA zone."""


def _country_code(country: str) -> Optional[str]:
    try:
        import pytz
    except ImportError:  # city matching still works without pytz
        return None
    wanted = country.strip().lower()
    for code, name in pytz.country_names.items():
        if wanted in (code.lower(), name.lower()):
            return code
    raise TimezoneConfigError(f"Unknown country in master.yaml timezone.current_country: {country!r}")


def _country_zones(code: Optional[str]) -> list[str]:
    if code is None:
        return []
    import pytz

    return list(pytz.country_timezones.get(code, []))


def _same_clock(zones: list[str]) -> bool:
    """True when all zones keep the same UTC offset in January and July."""
    year = datetime.now().year
    probes = [datetime(year, 1, 15, 12), datetime(year, 7, 15, 12)]
    offsets = {
        tuple(probe.replace(tzinfo=ZoneInfo(zone)).utcoffset() for probe in probes)
        for zone in zones
    }
    return len(offsets) == 1


def resolve_timezone(location: Optional[str], country: Optional[str]) -> str:
    """IANA zone name for a city and country (see module docstring)."""
    zones = available_timezones()
    location = (location or "").strip()
    country = (country or "").strip()
    if location in zones:
        return location

    code = _country_code(country) if country else None
    country_zones = _country_zones(code)
    city = location.replace(" ", "_").lower()
    if city:
        pool = country_zones or sorted(zones)
        matches = [z for z in pool if z.rsplit("/", 1)[-1].lower() == city]
        if len(matches) == 1:
            return matches[0]
    if len(country_zones) == 1 or (country_zones and _same_clock(country_zones)):
        return country_zones[0]  # pytz lists the principal zone first
    if country_zones:
        raise TimezoneConfigError(
            f"Cannot tell which zone of {country} contains {location!r}. Set "
            f"timezone.current_location to the nearest of: {', '.join(country_zones)}"
        )
    raise TimezoneConfigError(
        f"Cannot resolve a timezone from location={location!r} country={country!r}; "
        "use a city name such as 'Taipei' or an IANA name such as 'Asia/Taipei'"
    )


@lru_cache(maxsize=1)
def timezone_name() -> str:
    """The configured IANA zone; the system zone when master.yaml has none."""
    path = Path(os.environ.get("SCC_MASTER_YAML") or _MASTER_YAML)
    block = {}
    if path.is_file():
        with open(path, "r", encoding="utf-8") as handle:
            block = (yaml.safe_load(handle) or {}).get("timezone") or {}
    if not block:
        return str(datetime.now().astimezone().tzinfo)
    return resolve_timezone(block.get("current_location"), block.get("current_country"))


@lru_cache(maxsize=1)
def tzinfo() -> ZoneInfo:
    return ZoneInfo(timezone_name())


def now() -> datetime:
    """Current time, timezone-aware, in the configured zone."""
    return datetime.now(tzinfo())


def now_iso(timespec: str = "seconds") -> str:
    """ISO 8601 with offset, e.g. 2026-09-21T19:01:57+08:00."""
    return now().isoformat(timespec=timespec)


def now_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Formatted current time in the configured zone."""
    return now().strftime(fmt)


def apply_process_timezone() -> str:
    """Set TZ for this process so logging and time.strftime use the zone."""
    name = timezone_name()
    if "/" in name and os.environ.get("TZ") != name:
        os.environ["TZ"] = name
        if hasattr(time, "tzset"):
            time.tzset()
    return name


__all__ = [
    "TimezoneConfigError",
    "apply_process_timezone",
    "now",
    "now_iso",
    "now_str",
    "resolve_timezone",
    "timezone_name",
    "tzinfo",
]
