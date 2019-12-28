"""
Helpers to deal with timestamps
"""
# from dateutil.parser import parse as dateutil_parse
import re
from datetime import datetime, timedelta

TIMEDELTA_PATTERN = r"(?=\d+[dhms])(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?"

def convert_datetime(timestamp):
    """
    Convert object to datetime, if needed.

    :param timestamp: A timestamp
    :type timestamp: `datetime` or `str`
    :return: A `datetime` object
    :rtype: `datetime.datetime`
    """
    if isinstance(timestamp, datetime):
        return timestamp

    return datetime.fromisoformat(timestamp)

def parse_timedelta(delta):
    """
    Parse a time delta. It could be in the standard `datetime` format, or in the
    simpler notation `1d2h3m4s`. Valid deltas include:

    ```
    "42s"      # 42 seconds
    "1h2m30s"  # 1 hour, 2 minutes, 30 seconds
    "1d2s"     # 1 day, 2 seconds
    42         # 42 seconds
    "24"       # 24 seconds
    ```

    :param delta: a delta expressed in digits, or in the supported format
    :type delta: `str` or `int`
    :return: a valid `timedelta` object
    :rtype: `datetime.timedelta`
    """
    assert isinstance(delta, int) or isinstance(delta, str)

    if isinstance(delta, int) or delta.isdigit():
        return timedelta(seconds=int(delta))

    match = re.match(TIMEDELTA_PATTERN, delta)
    if not match:
        raise ValueError("Timestamp must be a valid ISO string, or a delta in the format 1d2h3m4s")
    groups = match.groups(default=0)
    return timedelta(
        days=int(groups[0]),
        hours=int(groups[1]),
        minutes=int(groups[2]),
        seconds=int(groups[3])
    )
