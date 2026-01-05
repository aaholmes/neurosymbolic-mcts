FROM rust:1.75-bookworm

# Install Python for training
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy Rust project
WORKDIR /app
COPY . .

# Build Rust project
RUN cargo build --release

# Default command: run tournament
CMD ["./target/release/elo_tournament", "--games", "100"]
