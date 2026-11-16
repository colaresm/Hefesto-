ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

ARG EXTRA_PACKAGES=""
RUN if [ -n "$EXTRA_PACKAGES" ]; then pip install $EXTRA_PACKAGES; fi

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps .

FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="Hefesto" \
      org.opencontainers.image.description="Surrogate regression models for ANSYS semiconductor packaging simulations" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="MIT"

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ARG UID=1000
ARG GID=1000
RUN groupadd --gid "$GID" hefesto && useradd --uid "$UID" --gid "$GID" --create-home hefesto

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    HEFESTO_OUTPUT_DIR=/app/outputs \
    HEFESTO_EXAMPLES_DIR=/app/examples \
    HEFESTO_LOG_LEVEL=INFO

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=hefesto:hefesto app ./app
COPY --chown=hefesto:hefesto .streamlit ./.streamlit
COPY --chown=hefesto:hefesto examples ./examples
RUN mkdir -p /app/outputs /app/data && chown -R hefesto:hefesto /app/outputs /app/data

USER hefesto
EXPOSE 8501
VOLUME ["/app/outputs", "/app/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
