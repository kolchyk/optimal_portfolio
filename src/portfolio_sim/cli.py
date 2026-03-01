"""CLI dispatcher with subcommands for portfolio_sim."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="portfolio",
        description="R\u00b2 Momentum Strategy toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    from src.portfolio_sim.commands import walk_forward
    from src.portfolio_sim.strategy_v2 import command as walk_forward_v2

    commands: dict[str, object] = {}
    for mod in [walk_forward, walk_forward_v2]:
        mod.register(subparsers)
        commands[mod.COMMAND_NAME] = mod.run

    args = parser.parse_args(argv)
    commands[args.command](args)
