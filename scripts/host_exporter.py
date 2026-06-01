#!/usr/bin/env python3
"""Prometheus exporter for macOS host metrics.

Exports CPU, memory, network, and disk I/O metrics from the macOS host
so Grafana can show real system data rather than Docker Desktop VM stats.

Usage:
    python scripts/host_exporter.py  [--port 9101]
"""
import argparse
import re
import subprocess
import time
import psutil
from prometheus_client import start_http_server, Gauge

# ── Disk ─────────────────────────────────────────────────────────────────────
disk_total = Gauge("host_disk_total_bytes", "Total disk space", ["mountpoint", "fstype", "device"])
disk_used = Gauge("host_disk_used_bytes", "Used disk space", ["mountpoint", "fstype", "device"])
disk_free = Gauge("host_disk_free_bytes", "Free disk space", ["mountpoint", "fstype", "device"])
disk_percent = Gauge("host_disk_usage_percent", "Disk usage %", ["mountpoint", "fstype", "device"])

# ── CPU ───────────────────────────────────────────────────────────────────────
cpu_percent = Gauge("host_cpu_percent", "Overall CPU usage %")
cpu_core_percent = Gauge("host_cpu_core_percent", "Per-core CPU usage %", ["core"])

# ── Memory ────────────────────────────────────────────────────────────────────
mem_total = Gauge("host_memory_total_bytes", "Total physical RAM")
mem_used = Gauge("host_memory_used_bytes", "Used physical RAM")
mem_available = Gauge("host_memory_available_bytes", "Available RAM")
mem_percent = Gauge("host_memory_percent", "RAM usage %")
mem_wired = Gauge("host_memory_wired_bytes", "Wired (kernel-locked) RAM")
swap_total = Gauge("host_swap_total_bytes", "Total swap space")
swap_used = Gauge("host_swap_used_bytes", "Used swap space")
swap_percent = Gauge("host_swap_percent", "Swap usage %")

# ── Network ───────────────────────────────────────────────────────────────────
net_bytes_recv = Gauge("host_network_bytes_recv_total", "Network bytes received (cumulative)", ["interface"])
net_bytes_sent = Gauge("host_network_bytes_sent_total", "Network bytes sent (cumulative)", ["interface"])
net_packets_recv = Gauge("host_network_packets_recv_total", "Network packets received", ["interface"])
net_packets_sent = Gauge("host_network_packets_sent_total", "Network packets sent", ["interface"])
net_errin = Gauge("host_network_errin_total", "Network inbound errors", ["interface"])
net_errout = Gauge("host_network_errout_total", "Network outbound errors", ["interface"])

# ── Disk I/O ──────────────────────────────────────────────────────────────────
diskio_read_bytes = Gauge("host_diskio_read_bytes_total", "Disk bytes read (cumulative)", ["device"])
diskio_write_bytes = Gauge("host_diskio_write_bytes_total", "Disk bytes written (cumulative)", ["device"])
diskio_read_count = Gauge("host_diskio_read_ops_total", "Disk read operations (cumulative)", ["device"])
diskio_write_count = Gauge("host_diskio_write_ops_total", "Disk write operations (cumulative)", ["device"])

# Exclude virtual/container interfaces
_IFACE_EXCLUDE = {"lo0", "gif0", "stf0", "XHC0", "XHC1", "XHC20"}
_IFACE_SKIP_PREFIX = ("utun", "llw", "awdl", "bridge", "p2p", "veth", "docker", "vmnet")

# Physical disk prefix on macOS
_DISK_PREFIX = "disk"

# ── Apple Silicon GPU (via ioreg — no sudo required) ──────────────────────────
gpu_device_percent   = Gauge("host_gpu_device_percent",   "Apple Silicon GPU overall utilisation %")
gpu_renderer_percent = Gauge("host_gpu_renderer_percent", "Apple Silicon GPU renderer pipeline utilisation %")
gpu_tiler_percent    = Gauge("host_gpu_tiler_percent",    "Apple Silicon GPU tiler pipeline utilisation %")
gpu_mem_used_bytes   = Gauge("host_gpu_mem_used_bytes",   "Apple Silicon GPU shared memory in use (bytes)")
gpu_mem_alloc_bytes  = Gauge("host_gpu_mem_alloc_bytes",  "Apple Silicon GPU shared memory allocated (bytes)")


def collect_gpu():
    """Read Apple Silicon GPU stats via ioreg (no sudo required)."""
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "AGXAccelerator"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0:
            return
        match = re.search(r'"PerformanceStatistics"\s*=\s*\{([^}]+)\}', result.stdout)
        if not match:
            return
        stats = match.group(1)

        def val(key):
            m = re.search(rf'"{re.escape(key)}"\s*=\s*(\d+)', stats)
            return int(m.group(1)) if m else 0

        gpu_device_percent.set(val("Device Utilization %"))
        gpu_renderer_percent.set(val("Renderer Utilization %"))
        gpu_tiler_percent.set(val("Tiler Utilization %"))
        gpu_mem_used_bytes.set(val("In use system memory"))
        gpu_mem_alloc_bytes.set(val("Alloc system memory"))
    except Exception:
        pass


def collect():
    # ── Disk usage ───────────────────────────────────────────────────────────
    for part in psutil.disk_partitions():
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

    # ── CPU ──────────────────────────────────────────────────────────────────
    cpu_percent.set(psutil.cpu_percent(interval=None))
    for i, pct in enumerate(psutil.cpu_percent(interval=None, percpu=True)):
        cpu_core_percent.labels(core=str(i)).set(pct)

    # ── Memory ───────────────────────────────────────────────────────────────
    mem = psutil.virtual_memory()
    mem_total.set(mem.total)
    mem_used.set(mem.used)
    mem_available.set(mem.available)
    mem_percent.set(mem.percent)
    # wired is macOS-specific (kernel memory that can't be paged out)
    mem_wired.set(getattr(mem, "wired", 0))

    swap = psutil.swap_memory()
    swap_total.set(swap.total)
    swap_used.set(swap.used)
    swap_percent.set(swap.percent)

    # ── Network ──────────────────────────────────────────────────────────────
    for iface, stats in psutil.net_io_counters(pernic=True).items():
        if iface in _IFACE_EXCLUDE:
            continue
        if iface.startswith(_IFACE_SKIP_PREFIX):
            continue
        net_bytes_recv.labels(interface=iface).set(stats.bytes_recv)
        net_bytes_sent.labels(interface=iface).set(stats.bytes_sent)
        net_packets_recv.labels(interface=iface).set(stats.packets_recv)
        net_packets_sent.labels(interface=iface).set(stats.packets_sent)
        net_errin.labels(interface=iface).set(stats.errin)
        net_errout.labels(interface=iface).set(stats.errout)

    # ── GPU ──────────────────────────────────────────────────────────────────
    collect_gpu()

    # ── Disk I/O ─────────────────────────────────────────────────────────────
    try:
        for device, stats in psutil.disk_io_counters(perdisk=True).items():
            if not device.startswith(_DISK_PREFIX):
                continue
            diskio_read_bytes.labels(device=device).set(stats.read_bytes)
            diskio_write_bytes.labels(device=device).set(stats.write_bytes)
            diskio_read_count.labels(device=device).set(stats.read_count)
            diskio_write_count.labels(device=device).set(stats.write_count)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Host Prometheus exporter for macOS")
    parser.add_argument("--port", type=int, default=9101, help="HTTP server port")
    parser.add_argument("--interval", type=int, default=5, help="Collect interval in seconds")
    args = parser.parse_args()

    # Warm up cpu_percent (first call always returns 0 in non-blocking mode)
    psutil.cpu_percent(interval=0.1)
    psutil.cpu_percent(interval=0.1, percpu=True)

    start_http_server(args.port)
    print(f"[host_exporter] Serving on http://localhost:{args.port}/metrics (interval={args.interval}s)")

    while True:
        collect()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
