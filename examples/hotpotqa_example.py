'''
Simple example demonstrating HotpotQA adaptive agent workflow.

This example shows:
1. How to create an adaptive agent that automatically handles simple and complex questions
2. How the agent adapts between single-step and multi-step reasoning
3. How to run benchmark evaluation
'''

import asyncio
from src.agents import HotpotQAAgent, AgentConfig
from src.benchmarks import HotpotQA, DatasetType


async def example_simple_question():
    '''Example 1: Simple question - agent will use single-step automatically.'''
    print('=' * 60)
    print('Example 1: Simple Question (Automatic Single-Step)')
    print('=' * 60)
    
    # Create adaptive agent
    config = AgentConfig(
        model='gemini-3-pro-preview',
        max_steps=5,  # Allow multi-step, but agent will decide
        verbose=True
    )
    agent = HotpotQAAgent(config=config)
    
    # Simple question
    question = 'What is the capital of France?'
    context = [
        ['France', ['France is a country in Europe.', 'The capital of France is Paris.']]
    ]
    
    print(f'\nQuestion: {question}')
    print('Agent analyzing...\n')
    
    answer = await agent.run(question=question, context=context)
    
    print(f'\n{"="*60}')
    print(f'Answer: {answer}')
    print(f'Steps used: {len(agent.state.steps)} (agent chose single-step)')
    print(f'{"="*60}\n')
    
    return answer


async def example_complex_multihop_question():
    '''Example 2: Complex multi-hop question - agent will use multi-step automatically.'''
    print('=' * 60)
    print('Example 2: Complex Multi-Hop Question (Automatic Multi-Step)')
    print('=' * 60)
    
    # Same agent configuration
    config = AgentConfig(
        model='gemini-3-pro-preview',
        max_steps=5,
        verbose=True
    )
    agent = HotpotQAAgent(config=config)
    
    # Complex multi-hop question
    question = 'Which film director was born first, Christopher Nolan or the director of The Grand Budapest Hotel?'
    
    # Information spread across multiple documents
    context = [
        ['Christopher Nolan', [
            'Christopher Edward Nolan (born 30 July 1970) is a British-American film director, producer, and screenwriter.',
            'He is known for making films with complex narratives and non-linear storytelling.'
        ]],
        ['The Grand Budapest Hotel', [
            'The Grand Budapest Hotel is a 2014 comedy-drama film written and directed by Wes Anderson.',
            'The film received widespread acclaim and was nominated for nine Academy Awards.'
        ]],
        ['Wes Anderson', [
            'Wesley Wales Anderson (born May 1, 1969) is an American filmmaker.',
            'His films are known for their distinctive visual and narrative style.'
        ]]
    ]
    
    print(f'\nQuestion: {question}')
    print('Agent analyzing...\n')
    
    answer = await agent.run(question=question, context=context)
    
    print(f'\n{"="*60}')
    print(f'Answer: {answer}')
    print(f'Steps used: {len(agent.state.steps)} (agent chose multi-step)')
    
    # Show reasoning steps
    if agent.state and len(agent.state.steps) > 1:
        print(f'\nReasoning process:')
        for i, step in enumerate(agent.state.steps, 1):
            if step.get('reasoning'):
                print(f'  Step {i}: {step["reasoning"][:100]}...')
            elif step.get('answer'):
                print(f'  Step {i}: Final Answer - {step["answer"]}')
    print(f'{"="*60}\n')
    
    return answer


async def example_adaptive_demonstration():
    '''Example 3: Demonstrate automatic adaptation with multiple questions.'''
    print('=' * 60)
    print('Example 3: Adaptive Agent - Multiple Questions')
    print('=' * 60)
    print('Same agent configuration, different question complexities\n')
    
    config = AgentConfig(
        model='gemini-3-pro-preview',
        max_steps=5,
        verbose=False  # Less verbose for comparison
    )
    
    questions = [
        {
            'question': 'Who wrote the novel "1984"?',
            'context': [['1984', ['The novel "1984" was written by George Orwell in 1949.']]],
            'expected': 'simple'
        },
        {
            'question': 'What is the birthplace of the author of "1984"?',
            'context': [
                ['1984', ['The novel "1984" was written by George Orwell.']],
                ['George Orwell', ['George Orwell was born in Motihari, British India.']]
            ],
            'expected': 'multi-hop'
        },
        {
            'question': 'What is the capital of the country where the Eiffel Tower is located?',
            'context': [
                ['Eiffel Tower', ['The Eiffel Tower is located in Paris, France.']],
                ['France', ['France is a country in Europe with capital city Paris.']]
            ],
            'expected': 'multi-hop'
        }
    ]
    
    results = []
    for i, q in enumerate(questions, 1):
        agent = HotpotQAAgent(config=config)
        answer = await agent.run(question=q['question'], context=q['context'])
        steps = len(agent.state.steps)
        results.append({
            'question': q['question'][:60],
            'answer': answer,
            'steps': steps,
            'expected': q['expected']
        })
    
    print(f'{"="*60}')
    print('Results Summary:')
    print(f'{"="*60}')
    for i, r in enumerate(results, 1):
        print(f'\n{i}. {r["question"]}')
        print(f'   Answer: {r["answer"]}')
        print(f'   Steps: {r["steps"]} ({"single" if r["steps"] == 1 else "multi"}-step)')
        print(f'   Expected: {r["expected"]}')
    print(f'\n{"="*60}')
    print('[OK] Agent automatically adapts to question complexity!')
    print(f'{"="*60}\n')
    
    return results


async def example_benchmark_evaluation():
    '''Example 4: Run benchmark evaluation.'''
    print('=' * 60)
    print('Example 4: Benchmark Evaluation')
    print('=' * 60)
    
    # Load benchmark
    benchmark = HotpotQA(dataset_type=DatasetType.VALIDATE)
    
    # Create agent
    config = AgentConfig(
        model='gemini-3-pro-preview',
        max_steps=3,  # Allow some multi-step reasoning
        verbose=False
    )
    agent = HotpotQAAgent(config=config)
    
    # Define agent function
    async def agent_fn(question: str, context) -> str:
        return await agent.run(question=question, context=context)
    
    # Run evaluation on first 5 samples
    print('\nRunning evaluation on 5 samples...')
    results = await benchmark.run(
        callback=agent_fn,
        dataset='validate',
        num_samples=5,
        verbose=True
    )
    
    # Print results
    print('\n' + '-' * 60)
    print('Benchmark Results:')
    print(f'  Exact Match: {results["metrics"]["exact_match"]:.4f}')
    print(f'  F1 Score: {results["metrics"]["f1"]:.4f}')
    print(f'  Passed: {results["metrics"]["num_passed"]}/{results["metrics"]["num_samples"]}')
    print('-' * 60)
    
    return results


async def main():
    '''Run all examples.'''
    # Example 1: Simple question
    await example_simple_question()
    
    # Example 2: Complex multi-hop question
    await example_complex_multihop_question()
    
    # Example 3: Adaptive demonstration
    await example_adaptive_demonstration()
    
    # Example 4: Benchmark evaluation (commented out - requires API)
    await example_benchmark_evaluation()


if __name__ == '__main__':
    asyncio.run(main())
