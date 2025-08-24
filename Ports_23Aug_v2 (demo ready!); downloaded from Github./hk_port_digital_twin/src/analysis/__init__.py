# Analysis and Performance Monitoring Module
# This module provides performance analysis and benchmarking capabilities

from .performance_benchmarking import (
    PerformanceBenchmarking,
    BenchmarkReport,
    BenchmarkMetric,
    BenchmarkCategory,
    PerformanceLevel
)

# Create aliases for compatibility with main module imports
PerformanceAnalyzer = PerformanceBenchmarking  # Main performance analyzer
BenchmarkReporter = BenchmarkReport  # Benchmark reporting
RealTimeMonitor = None  # To be implemented
MetricsCollector = None  # To be implemented

__all__ = [
    'PerformanceBenchmarking',
    'BenchmarkReport',
    'BenchmarkMetric',
    'BenchmarkCategory',
    'PerformanceLevel',
    'PerformanceAnalyzer',
    'BenchmarkReporter',
    'RealTimeMonitor',
    'MetricsCollector'
]