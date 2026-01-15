'''HotpotQA Agent for answering multi-hop questions.'''

from __future__ import annotations

import re
from typing import Any, List

from pydantic import Field, PrivateAttr

from src.utils import get_logger
from .base import Agent, AgentConfig, AgentState

logger = get_logger(__name__)


# Unified adaptive reasoning prompt
ADAPTIVE_REASONING_PROMPT = '''Question: {question}

Context Information:
{context}

{previous_steps}

Instructions:
Analyze the question and available context carefully. You can either:
1. If you can confidently answer the question now, respond with: 'Final Answer: [your answer]'
2. If you need to reason through multiple pieces of information, explain your reasoning step by step

Think carefully about what information is needed and how to connect different facts.
Then provide your response.'''


class HotpotQAAgent(Agent):
    '''Adaptive agent for HotpotQA that automatically handles single-step and multi-step reasoning.'''
    
    _state: AgentState | None = PrivateAttr(default=None)
    
    @property
    def state(self) -> AgentState | None:
        '''Current agent state.'''
        return self._state
    
    async def run(self, question: str, context: List[List[str]] | str = '', **kwargs: Any) -> str:
        '''
        Run adaptive reasoning to answer a HotpotQA question.
        
        Automatically switches between single-step (simple questions) and 
        multi-step reasoning (complex multi-hop questions) based on the 
        question complexity and agent's analysis.
        
        Args:
            question: The question to answer
            context: Supporting context, either as a list of [title, sentences] or a formatted string
            **kwargs: Additional arguments
            
        Returns:
            The agent's answer to the question
        '''
        # Format context if provided as list
        formatted_context = self._format_context(context) if isinstance(context, list) else context
        
        # Initialize state
        self._state = AgentState(
            question=question,
            context=formatted_context,
            steps=[],
            answer='',
            finished=False
        )
        
        # Run adaptive reasoning (may finish in 1 step for simple questions, or use multiple steps)
        for step_num in range(self.config.max_steps):
            if self.config.verbose:
                logger.info(f'\n{"="*60}')
                logger.info(f'Step {step_num + 1}/{self.config.max_steps}')
                logger.info(f'{"="*60}')
                
            self._state = await self.step(self._state)
            
            if self._state.finished:
                if self.config.verbose and len(self._state.steps) > 1:
                    logger.info(f'\n{"="*60}')
                    logger.info(f'Completed in {len(self._state.steps)} steps')
                    logger.info(f'{"="*60}')
                break
                
        return self._state.answer
    
    async def step(self, state: AgentState) -> AgentState:
        '''
        Execute one adaptive reasoning step.
        
        The agent will decide whether to:
        1. Provide a final answer immediately (single-step)
        2. Continue reasoning through the problem (multi-step)
        '''
        # Build prompt with reasoning history
        previous_steps_text = self._format_previous_steps(state.steps)
        
        prompt = ADAPTIVE_REASONING_PROMPT.format(
            question=state.question,
            context=state.context,
            previous_steps=previous_steps_text
        )
        
        # Call LLM
        try:
            response = await self.llm(prompt)
            
            if self.config.verbose:
                logger.info(f'Response: {response}')
            
            # Check if we have a final answer
            is_final, answer = self._extract_answer(response)
            
            # Record the step
            new_steps = state.steps + [{
                'step_num': len(state.steps) + 1,
                'reasoning': response if not is_final else None,
                'answer': answer if is_final else None,
                'full_response': response
            }]
            
            if is_final:
                # Got final answer - finish
                state = state.model_copy(update={
                    'steps': new_steps,
                    'answer': answer,
                    'finished': True
                })
                
                if self.config.verbose:
                    logger.info(f'✅ Final Answer: {answer}')
            else:
                # Continue reasoning
                state = state.model_copy(update={
                    'steps': new_steps
                })
                
                if self.config.verbose:
                    logger.info(f'🤔 Reasoning: {response[:100]}...')
                
        except Exception as e:
            logger.error(f'Error in agent step: {e}')
            state = state.model_copy(update={
                'answer': 'Error occurred',
                'finished': True
            })
            
        return state
    
    def _format_context(self, context: List[List[str]]) -> str:
        '''
        Format HotpotQA context from list format to string.
        
        HotpotQA context format: [[title1, [sent1, sent2, ...]], [title2, [sent1, sent2, ...]], ...]]
        '''
        if not context:
            return ''
            
        formatted_parts = []
        for item in context:
            if len(item) >= 2:
                title = item[0]
                sentences = item[1] if isinstance(item[1], list) else [item[1]]
                text = ' '.join(sentences)
                formatted_parts.append(f'[{title}]\n{text}')
                
        return '\n\n'.join(formatted_parts)
    
    def _format_previous_steps(self, steps: List[dict]) -> str:
        '''Format previous reasoning steps for prompt.'''
        if not steps:
            return ''
        
        formatted = ['\nPrevious reasoning steps:']
        for step in steps:
            step_num = step.get('step_num', 0)
            reasoning = step.get('reasoning')
            if reasoning:
                formatted.append(f'Step {step_num}: {reasoning[:200]}...')
        
        return '\n'.join(formatted) if len(formatted) > 1 else ''
    
    def _extract_answer(self, response: str) -> tuple[bool, str]:
        '''
        Extract answer from LLM response.
        
        Returns:
            (is_final, answer): is_final=True if found 'Final Answer', otherwise False
        '''
        # Try to find 'Final Answer: ...' pattern
        patterns = [
            r'Final Answer:\s*(.+?)(?:\n|$)',
            r'final answer:\s*(.+?)(?:\n|$)',
            r'Answer:\s*(.+?)(?:\n|$)',
            r'answer:\s*(.+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(1).strip()
                # Clean up the answer - remove quotes and backslashes
                answer = answer.strip('\\"\'')
                return True, answer
        
        # No final answer found - still reasoning
        return False, ''
    
    def reset(self) -> None:
        '''Reset the agent state.'''
        self._state = None
        self._llm = None
