from datetime import datetime, timedelta

from due.util.time import convert_datetime, parse_timedelta

class TestConvertDatetime(object):
    def test_datetime(self):
        dt = datetime.now()
        assert convert_datetime(dt) is dt

    def test_datetime_str(self):
        dt = "2019-12-28T22:18:41.817246"
        assert convert_datetime(dt) == datetime(2019, 12, 28, 22, 18, 41, 817246)

class TestParseTimedelta(object):
    def test_seconds_int(self):
        delta = 124
        assert parse_timedelta(delta) == timedelta(minutes=2, seconds=4)

    def test_seconds_str(self):
        delta = "62"
        assert parse_timedelta(delta) == timedelta(minutes=1, seconds=2)

    def test_formatted(self):
        delta = "32s"
        assert parse_timedelta(delta) == timedelta(seconds=32)

        delta = "1m32s"
        assert parse_timedelta(delta) == timedelta(minutes=1, seconds=32)

        delta = "1h2m50s"
        assert parse_timedelta(delta) == timedelta(hours=1, minutes=2, seconds=50)

        delta = "1d22h3m44s"
        assert parse_timedelta(delta) == timedelta(days=1, hours=22, minutes=3, seconds=44)

        delta = "1d42s"
        assert parse_timedelta(delta) == timedelta(days=1, seconds=42)
