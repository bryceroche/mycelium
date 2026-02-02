"""CLI entry point for Mycelium."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="mycelium",
        description="Mycelium attention-based math decomposer",
    )
    args = parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()
