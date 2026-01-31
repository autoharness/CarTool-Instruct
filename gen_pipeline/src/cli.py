import argparse
import logging
from pathlib import Path
from generation_pipeline import GenerationPipeline
from refine import PostProcess


def main():
    parser = argparse.ArgumentParser(description="Car tool datasets utility.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable verbose logging"
    )
    # Subcommand: `generate`
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="available sub-commands"
    )
    parser_gen = subparsers.add_parser("generate", help="Generate dataset samples.")
    parser_gen.add_argument(
        "--num-samples",
        required=True,
        type=int,
        help="the number of data samples to generate",
    )
    parser_gen.add_argument(
        "--output", required=True, type=Path, help="the output file path"
    )
    parser_gen.add_argument(
        "--save-interval",
        type=int,
        default=5,
        help="save intermediate results every N loops (0 to disable intermediate saving)",
    )
    parser_gen.set_defaults(func=handle_generate)
    # Subcommand: `refine`
    parser_refine = subparsers.add_parser(
        "refine", help="Split and transform generated data into target structure."
    )
    parser_refine.add_argument(
        "--data-file",
        required=True,
        type=Path,
        help="path to the generated data JSON file to process",
    )
    parser_refine.add_argument(
        "--num-test",
        required=True,
        type=int,
        help="the number of samples in test set",
    )
    parser_refine.add_argument(
        "--output", required=True, type=Path, help="the output file path"
    )
    parser_refine.set_defaults(func=handle_refine)

    args = parser.parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format="[%(levelname)s] [%(name)s] %(message)s", level=log_level
    )

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


def handle_generate(args):
    pipeline = GenerationPipeline(args.num_samples, args.output, args.save_interval)
    pipeline.run()


def handle_refine(args):
    p = PostProcess(args.data_file, args.num_test, args.output)
    p.refine()


if __name__ == "__main__":
    main()
