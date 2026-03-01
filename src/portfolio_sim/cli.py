"""CLI dispatcher for portfolio_sim."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="portfolio",
        description="Hybrid R\u00b2 Momentum + Vol-Targeting Strategy toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    from src.portfolio_sim import command as walk_forward

    commands: dict[str, object] = {}
    walk_forward.register(subparsers)
    commands[walk_forward.COMMAND_NAME] = walk_forward.run

    args = parser.parse_args(argv)
    commands[args.command](args)
