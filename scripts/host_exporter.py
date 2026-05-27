#!/usr/bin/env python3
"""Minimal Prometheus exporter for macOS host disk usage.

Runs alongside the Docker stack to expose host disk metrics
that node-exporter cannot see inside Docker Desktop.

Usage:
    python scripts/host_exporter.py  [--port 9101]
"""
import argparse
import time
import psutil
from prometheus_client import start_http_server, Gauge, REGISTRY


disk_total = Gauge("host_disk_total_bytes", "Total disk space", ["mountpoint", "fstype", "device"])
disk_used = Gauge("host_disk_used_bytes", "Used disk space", ["mountpoint", "fstype", "device"])
disk_free = Gauge("host_disk_free_bytes", "Free disk space", ["mountpoint", "fstype", "device"])
disk_percent = Gauge("host_disk_usage_percent", "Disk usage percentage", ["mountpoint", "fstype", "device"])


def collect():
    for part in psutil.disk_partitions():
        # Skip irrelevant filesystems
        if part.fstype in ("tmpfs", "devtmpfs", "squashfs", "autofs", "proc", "sysfs", "cgroup2"):
            continue
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except PermissionError:
            continue
        labels = [part.mountpoint, part.fstype, part.device]
        disk_total.labels(*labels).set(usage.total)
        disk_used.labels(*labels).set(usage.used)
        disk_free.labels(*labels).set(usage.free)
        disk_percent.labels(*labels).set(usage.percent)


def main():
    parser = argparse.ArgumentParser(description="Host Prometheus exporter for macOS")
    parser.add_argument("--port", type=int, default=9101, help="HTTP server port")
    args = parser.parse_args()

    start_http_server(args.port)
    print(f"[host_exporter] Serving on http://localhost:{args.port}/metrics")

    while True:
        collect()
        time.sleep(15)


if __name__ == "__main__":
    main()
