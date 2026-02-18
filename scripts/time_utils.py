#!/usr/bin/env python3
"""
Utility functions for time formatting.
"""


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.

    Parameters
    ----------
    seconds : float
        Time duration in seconds.

    Returns
    -------
    str
        Formatted time string in one of the following formats:
        - "X.XXs" for durations less than 60 seconds
        - "Xm X.XXs" for durations less than 1 hour
        - "Xh Xm X.XXs" for durations 1 hour or more

    Examples
    --------
    >>> format_time(45.5)
    '45.50s'
    >>> format_time(125.3)
    '2m 5.30s'
    >>> format_time(3725.8)
    '1h 2m 5.80s'
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"

