"""Run-handler registry shared by feature packages."""

from .runner import dispatch_run, register_run_handler

__all__ = ["dispatch_run", "register_run_handler"]
