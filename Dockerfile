# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM rust:bookworm AS builder

RUN apt-get update && apt-get install -y protobuf-compiler && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy manifests first for layer caching
COPY Cargo.toml Cargo.lock rust-toolchain.toml deny.toml ./
COPY crates crates

# Build the hirnd binary in release mode
RUN cargo build --release --bin hirnd

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 1000 hirn && \
    useradd --uid 1000 --gid hirn --create-home hirn

COPY --from=builder /build/target/release/hirnd /usr/local/bin/hirnd

# Default data directory
RUN mkdir -p /data && chown hirn:hirn /data

USER hirn

EXPOSE 3000

ENV HIRND_DATA_DIR=/data

ENTRYPOINT ["hirnd"]
CMD ["--data", "/data"]
