import pandas as pd
import pytest

from salmon_price_estimator.data.feed_cost_fishmeal import build_page_url, parse_price_table

GVPRICES_HTML_FIXTURE = """
<html><body>
<div>Some unrelated table<table><tr><td>not this one</td></tr></table></div>
<table class="tblData" id="gvPrices">
<tr style="background-color:#E5ECF9;">
<th scope="col">Month</th><th scope="col">Price</th><th scope="col">Change</th>
</tr><tr align="right">
<td>Jul 1996</td><td>547.00</td><td>-</td>
</tr><tr align="right" style="background-color:#EFEFEF;">
<td>Aug 1996</td><td>539.00</td><td>-1.46%</td>
</tr><tr align="right">
<td>Jan 2026</td><td>1,820.21</td><td>0.50%</td>
</tr>
</table>
</body></html>
"""


def test_build_page_url():
    url = build_page_url("https://www.indexmundi.com/commodities/", "fish-meal", 360)
    assert url == "https://www.indexmundi.com/commodities/?commodity=fish-meal&months=360"


def test_parse_price_table_extracts_rows_and_strips_commas():
    df = parse_price_table(GVPRICES_HTML_FIXTURE)

    assert list(df.columns) == ["month_date", "fishmeal_price_usd_per_tonne"]
    assert len(df) == 3
    assert df["month_date"].tolist() == [
        pd.Timestamp("1996-07-01"),
        pd.Timestamp("1996-08-01"),
        pd.Timestamp("2026-01-01"),
    ]
    assert df["fishmeal_price_usd_per_tonne"].tolist() == [547.00, 539.00, 1820.21]


def test_parse_price_table_ignores_unrelated_tables():
    df = parse_price_table(GVPRICES_HTML_FIXTURE)
    assert "not this one" not in df.to_string()


def test_parse_price_table_missing_table_raises():
    with pytest.raises(ValueError, match="gvPrices"):
        parse_price_table("<html><body>no table here</body></html>")
