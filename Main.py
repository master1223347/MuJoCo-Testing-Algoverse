#!/usr/bin/env python3
"""
Experiment Runner Script.

Description:
    Iterates through a list of scene IDs, runs experiments using the Experimental class,
    aggregates results, and optionally saves them to a JSON file.

Usage:
    python3 experiment_runner.py [--scenes SCENES] [--output OUTPUT] [--verbose]

Arguments:
    --scenes    Comma-separated list of scene IDs (default: "Scene_1").
    --output    Path to save the aggregated results JSON file (default: aggregated_results.json).
    --verbose   Enable verbose logging.

Dependencies:
    openai_agent.py
    scene.py
    simulator.py
    experimental.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Assuming these modules are implemented in the same directory
from OpenAIAgent import OpenAIAgent
from Scene import Scene
from Simulator import Simulator
from Experimental import Experimental

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run experiments for specified scenes."
    )
    parser.add_argument(
        "--scenes", "-s",
        type=str,
        default="Scene_1",
        help="Comma-separated scene IDs (e.g., Scene_1,Scene_2)."
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("aggregated_results.json"),
        help="Output JSON file path."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging."
    )
    return parser.parse_args()

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def run_experiments(scene_ids: list[str]) -> dict[str, dict]:
    aggregated_results: dict[str, dict] = {}
    for scene_id in scene_ids:
        logging.info(f"Starting experiment for scene: {scene_id}")
        try:
            simulator = Simulator(scene_id)
            scene = Scene(scene_id, simulator)
            prompt = scene.generate_prompt()
            logging.debug(f"Generated prompt for {scene_id}:\n{prompt}")

            experiment = Experimental(scene_id)
            results = experiment.run_experiment()

            if results.get("answer_found"):
                logging.info("Answer Summary")
                logging.info(f"LLM's Answer: {results.get('llm_answer')}")
                logging.info(f"Correct Answer: {results.get('correct_answer')}")
                logging.info(f"Answer Correct: {results.get('correct')}")
            else:
                logging.warning("No answer was provided by the LLM.")

            aggregated_results[scene_id] = results
        except Exception as e:
            logging.error(f"Experiment for scene {scene_id} failed: {e}", exc_info=True)
            aggregated_results[scene_id] = {"error": str(e)}
    return aggregated_results

def save_results(results: dict, output_path: Path) -> None:
    try:
        with output_path.open("w") as f:
            json.dump(results, f, indent=4)
        logging.info(f"Aggregated results saved to {output_path}")
    except IOError as e:
        logging.error(f"Failed to save results to {output_path}: {e}")

def main():
    args = parse_args()
    setup_logging(args.verbose)
    scene_ids = [s.strip() for s in args.scenes.split(",") if s.strip()]
    logging.info(f"Scene IDs to process: {scene_ids}")

    results = run_experiments(scene_ids)
    save_results(results, args.output)

if __name__ == "__main__":
    main()
