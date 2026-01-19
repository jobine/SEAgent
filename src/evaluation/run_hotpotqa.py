'''
HotpotQA Benchmark Runner

Example usage:
    python -m run_hotpotqa --model gpt-4o-mini --num-samples 10 --verbose
'''

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from src.benchmarks.hotpotqa import HotpotQA
from src.benchmarks.benchmark import DatasetType
from src.agents.hotpotqa_agent import HotpotQAAgent
from src.agents.agent import AgentConfig
from src.utils import get_logger

logger = get_logger(__name__)


async def run_benchmark(
    model: str = 'gpt-4o-mini',
    dataset: str = 'validate',
    num_samples: int | None = None,
    verbose: bool = False,
    output_file: str | None = None
) -> dict:
    '''
    Run HotpotQA benchmark with specified configuration.
    
    Args:
        model: LLM model to use
        dataset: Dataset split to use ('train', 'validate', 'test')
        num_samples: Number of samples to evaluate (None for all)
        verbose: Whether to print detailed progress
        output_file: Path to save results (optional)
        
    Returns:
        Dictionary with benchmark results
    '''
    logger.info('Starting HotpotQA benchmark')
    logger.info(f'  Model: {model}')
    logger.info(f'  Dataset: {dataset}')
    logger.info(f'  Samples: {num_samples or "all"}')
    logger.info(f'  Verbose: {verbose}')
    logger.info(f'  Output file: {output_file or "none"}')
    
    # Initialize the benchmark
    dataset_type = DatasetType.from_value(dataset)
    
    benchmark = HotpotQA(dataset_type=dataset_type)
    
    # Initialize the agent
    config = AgentConfig(
        model=model,
        max_steps=5,  # Single-step for basic agent
        verbose=verbose
    )
    agent = HotpotQAAgent(config=config)
    
    # Create the agent function for benchmark
    async def agent_fn(question: str, context) -> str:
        return await agent.run(question=question, context=context)
    
    # Run the benchmark
    logger.info('Running benchmark...')
    results = await benchmark.run(
        callback=agent_fn,
        dataset=dataset,
        num_samples=num_samples,
        verbose=verbose
    )
    
    # Add metadata
    results['metadata'] = {
        'model': model,
        'dataset': dataset,
        'timestamp': datetime.now().isoformat(),
        'num_samples': num_samples
    }
    
    # Print summary
    metrics = results['metrics']
    logger.info('=' * 60)
    logger.info('Benchmark Results:')
    logger.info(f'  Exact Match (EM): {metrics["exact_match"]:.4f}')
    logger.info(f'  F1 Score: {metrics["f1"]:.4f}')
    logger.info(f'  Samples: {metrics["num_samples"]}')
    logger.info(f'  Passed: {metrics["num_passed"]}/{metrics["num_samples"]}')
    logger.info('=' * 60)
    
    # Save results if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f'Results saved to {output_path}')
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run HotpotQA benchmark')
    parser.add_argument(
        '--model', 
        type=str,
        default='gpt-4o-mini',
        help='LLM model to use (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='validate',
        choices=['train', 'validate', 'test'],
        help='Dataset split to use (default: validate)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        # required=True,
        help='Number of samples to evaluate (default: all)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path for results (JSON)'
    )
    
    args = parser.parse_args()
    
    # Run the benchmark
    asyncio.run(run_benchmark(
        model=args.model,
        dataset=args.dataset,
        num_samples=args.num_samples,
        verbose=args.verbose,
        output_file=args.output
    ))


if __name__ == '__main__':
    main()
