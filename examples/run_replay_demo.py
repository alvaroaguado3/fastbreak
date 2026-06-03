"""Minimal end-to-end example. Run: python examples/run_replay_demo.py

Uses only the stdlib core (no torch / no network). Swap ReplayFeed for
SportradarPushFeed and ConsolePublisher for XPublisher to go live.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastbreak.cli import demo

if __name__ == "__main__":
    demo()
