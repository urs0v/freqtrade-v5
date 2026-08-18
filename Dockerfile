FROM freqtradeorg/freqtrade:stable
USER root
RUN pip install --no-cache-dir "aiohttp>=3.10,<4"
USER ftuser
COPY strategies /opt/rmv5/strategies
COPY tools /opt/rmv5/tools
COPY config-v5.base.json /opt/rmv5/config-v5.base.json
COPY entrypoint-v5.sh /opt/rmv5/entrypoint-v5.sh
ENTRYPOINT ["/bin/bash", "/opt/rmv5/entrypoint-v5.sh"]
