FROM freqtradeorg/freqtrade:stable

USER root
RUN pip install --no-cache-dir "aiohttp>=3.10,<4"

COPY strategies /opt/rmv5/strategies
COPY tools /opt/rmv5/tools
COPY config-v5.base.json /opt/rmv5/config-v5.base.json
COPY entrypoint-v5.sh /opt/rmv5/entrypoint-v5.sh

# Collector stores derivatives data using the canonical SQLite column names
# bucket_ms/funding_rate/*_liq_usdt. Keep the strategy query compatible.
RUN sed -i \
    -e 's/SELECT ts, oi, funding, long_liq, short_liq, taker_ratio, top_ls_ratio/SELECT bucket_ms AS ts, oi, funding_rate AS funding, long_liq_usdt AS long_liq, short_liq_usdt AS short_liq, taker_ratio, top_ls_ratio/' \
    -e 's/WHERE symbol = ? AND ts BETWEEN ? AND ?/WHERE symbol = ? AND bucket_ms BETWEEN ? AND ?/' \
    -e 's/ORDER BY ts/ORDER BY bucket_ms/' \
    /opt/rmv5/strategies/RegimeMomentumV5.py \
 && chown -R ftuser:ftuser /opt/rmv5

USER ftuser
ENTRYPOINT ["/bin/bash", "/opt/rmv5/entrypoint-v5.sh"]
