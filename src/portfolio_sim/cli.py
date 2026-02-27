"""CLI dispatcher with subcommands for portfolio_sim."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="portfolio",
        description="KAMA Momentum Strategy toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    from src.portfolio_sim.commands import walk_forward

    commands: dict[str, object] = {}
    for mod in [walk_forward]:
        mod.register(subparsers)
        commands[mod.COMMAND_NAME] = mod.run

    args = parser.parse_args(argv)
    commands[args.command](args)
