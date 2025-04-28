"""
Metrics collection system for operational monitoring.

This module provides a metrics collection system that can be integrated
with various monitoring backends (Prometheus, StatsD, CloudWatch, etc.)
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, List, Callable, Set, Union
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..utils.logging import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """Value of a metric with metadata."""
    name: str
    value: Union[int, float]
    type: MetricType
    tags: Dict[str, str]
    timestamp: float


class MetricsBackend(ABC):
    """Abstract base class for metrics backends."""
    
    @abstractmethod
    def record_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric."""
        pass
    
    @abstractmethod
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric."""
        pass
    
    @abstractmethod
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        pass
    
    @abstractmethod
    def record_summary(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a summary metric."""
        pass
    
    @abstractmethod
    def flush(self) -> None:
        """Flush metrics to the backend."""
        pass


class LoggingMetricsBackend(MetricsBackend):
    """Metrics backend that logs metrics to a logger."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize a logging metrics backend.
        
        Args:
            logger: Optional logger instance (will use default if not provided)
        """
        self.logger = logger or get_logger("metrics")
    
    def record_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric."""
        self.logger.info(
            f"METRIC counter:{name} value:{value} tags:{tags or {}} type:counter"
        )
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric."""
        self.logger.info(
            f"METRIC gauge:{name} value:{value} tags:{tags or {}} type:gauge"
        )
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        self.logger.info(
            f"METRIC histogram:{name} value:{value} tags:{tags or {}} type:histogram"
        )
    
    def record_summary(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a summary metric."""
        self.logger.info(
            f"METRIC summary:{name} value:{value} tags:{tags or {}} type:summary"
        )
    
    def flush(self) -> None:
        """Flush metrics to the backend."""
        pass  # No need to flush for logging backend


class InMemoryMetricsBackend(MetricsBackend):
    """Metrics backend that stores metrics in memory."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize an in-memory metrics backend.
        
        Args:
            max_size: Maximum number of metrics to store
        """
        self.metrics: List[MetricValue] = []
        self.max_size = max_size
        self._lock = threading.Lock()
    
    def record_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric."""
        self._record_metric(name, value, MetricType.COUNTER, tags)
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric."""
        self._record_metric(name, value, MetricType.GAUGE, tags)
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        self._record_metric(name, value, MetricType.HISTOGRAM, tags)
    
    def record_summary(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a summary metric."""
        self._record_metric(name, value, MetricType.SUMMARY, tags)
    
    def _record_metric(self, name: str, value: Union[int, float], type: MetricType, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric."""
        with self._lock:
            metric = MetricValue(
                name=name,
                value=value,
                type=type,
                tags=tags or {},
                timestamp=time.time()
            )
            self.metrics.append(metric)
            
            # Truncate if we exceed max size
            if len(self.metrics) > self.max_size:
                self.metrics = self.metrics[-self.max_size:]
    
    def get_metrics(self) -> List[MetricValue]:
        """Get all recorded metrics."""
        with self._lock:
            return list(self.metrics)
    
    def clear(self) -> None:
        """Clear all recorded metrics."""
        with self._lock:
            self.metrics.clear()
    
    def flush(self) -> None:
        """Flush metrics to the backend."""
        pass  # No need to flush for in-memory backend


class CompositeMetricsBackend(MetricsBackend):
    """Metrics backend that forwards to multiple backends."""
    
    def __init__(self, backends: List[MetricsBackend]):
        """
        Initialize a composite metrics backend.
        
        Args:
            backends: List of backends to forward metrics to
        """
        self.backends = backends
    
    def record_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric."""
        for backend in self.backends:
            backend.record_counter(name, value, tags)
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric."""
        for backend in self.backends:
            backend.record_gauge(name, value, tags)
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        for backend in self.backends:
            backend.record_histogram(name, value, tags)
    
    def record_summary(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a summary metric."""
        for backend in self.backends:
            backend.record_summary(name, value, tags)
    
    def flush(self) -> None:
        """Flush metrics to all backends."""
        for backend in self.backends:
            backend.flush()


class MetricsCollector:
    """
    Metrics collector for recording application metrics.
    
    This class provides methods for recording various types of metrics,
    and forwarding them to one or more metrics backends.
    """
    
    def __init__(self, backend: Optional[MetricsBackend] = None):
        """
        Initialize a metrics collector.
        
        Args:
            backend: Optional metrics backend (will use logging backend if not provided)
        """
        self.backend = backend or LoggingMetricsBackend()
        self._common_tags: Dict[str, str] = {}
    
    def add_common_tag(self, key: str, value: str) -> None:
        """
        Add a common tag to all metrics.
        
        Args:
            key: Tag key
            value: Tag value
        """
        self._common_tags[key] = value
    
    def remove_common_tag(self, key: str) -> None:
        """
        Remove a common tag.
        
        Args:
            key: Tag key
        """
        if key in self._common_tags:
            del self._common_tags[key]
    
    def clear_common_tags(self) -> None:
        """Clear all common tags."""
        self._common_tags.clear()
    
    def _merge_tags(self, tags: Optional[Dict[str, str]]) -> Dict[str, str]:
        """
        Merge common tags with provided tags.
        
        Args:
            tags: Tags to merge with common tags
            
        Returns:
            Merged tags
        """
        result = dict(self._common_tags)
        if tags:
            result.update(tags)
        return result
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Metric name
            value: Increment value
            tags: Metric tags
        """
        self.backend.record_counter(name, value, self._merge_tags(tags))
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Set a gauge metric.
        
        Args:
            name: Metric name
            value: Gauge value
            tags: Metric tags
        """
        self.backend.record_gauge(name, value, self._merge_tags(tags))
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a histogram metric.
        
        Args:
            name: Metric name
            value: Histogram value
            tags: Metric tags
        """
        self.backend.record_histogram(name, value, self._merge_tags(tags))
    
    def record_summary(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a summary metric.
        
        Args:
            name: Metric name
            value: Summary value
            tags: Metric tags
        """
        self.backend.record_summary(name, value, self._merge_tags(tags))
    
    def record_latency(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a latency metric as a histogram.
        
        Args:
            name: Metric name
            value: Latency value in seconds
            tags: Metric tags
        """
        # Record in milliseconds for better granularity
        self.record_histogram(f"{name}_ms", value * 1000, tags)
    
    def measure_latency(self, name: str, tags: Optional[Dict[str, str]] = None) -> 'LatencyMeasurement':
        """
        Create a context manager for measuring latency.
        
        Args:
            name: Metric name
            tags: Metric tags
            
        Returns:
            Context manager for measuring latency
        """
        return LatencyMeasurement(self, name, tags)
    
    def count_calls(self, name: str, tags: Optional[Dict[str, str]] = None) -> Callable:
        """
        Decorator for counting function calls.
        
        Args:
            name: Metric name
            tags: Metric tags
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                self.increment_counter(name, tags=self._merge_tags(tags))
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def measure_call_latency(self, name: str, tags: Optional[Dict[str, str]] = None) -> Callable:
        """
        Decorator for measuring function call latency.
        
        Args:
            name: Metric name
            tags: Metric tags
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    self.record_latency(name, time.time() - start_time, self._merge_tags(tags))
                    return result
                except Exception as e:
                    # Record error and re-raise
                    error_tags = dict(self._merge_tags(tags))
                    error_tags['error_type'] = type(e).__name__
                    self.increment_counter(f"{name}_error", tags=error_tags)
                    self.record_latency(name, time.time() - start_time, error_tags)
                    raise
            return wrapper
        return decorator
    
    def flush(self) -> None:
        """Flush metrics to the backend."""
        self.backend.flush()


class LatencyMeasurement:
    """Context manager for measuring latency."""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Optional[Dict[str, str]] = None):
        """
        Initialize a latency measurement.
        
        Args:
            collector: Metrics collector
            name: Metric name
            tags: Metric tags
        """
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = 0.0
        self.error: Optional[Exception] = None
    
    def __enter__(self) -> 'LatencyMeasurement':
        """Enter the context manager."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager."""
        latency = time.time() - self.start_time
        if exc_val:
            # Record error and latency with error tag
            error_tags = dict(self.collector._merge_tags(self.tags) or {})
            error_tags['error_type'] = exc_type.__name__ if exc_type else 'unknown'
            self.collector.increment_counter(f"{self.name}_error", tags=error_tags)
            self.collector.record_latency(self.name, latency, error_tags)
        else:
            # Record successful latency
            self.collector.record_latency(self.name, latency, self.tags)
    
    def set_tag(self, key: str, value: str) -> None:
        """
        Set a tag for this measurement.
        
        Args:
            key: Tag key
            value: Tag value
        """
        if self.tags is None:
            self.tags = {}
        self.tags[key] = value


# Create default metrics collector
metrics_collector = MetricsCollector() 