"""Unit tests for HotpotQA Agent."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pydantic import ValidationError

from src.agents.hotpotqa_agent import HotpotQAAgent
from src.agents.agent import AgentConfig, AgentState


class TestHotpotQAAgent:
    """Test cases for HotpotQAAgent class."""

    @pytest.fixture
    def agent_config(self):
        """Create test agent configuration."""
        return AgentConfig(
            model="gpt-4o-mini",
            max_steps=3,
            verbose=False
        )

    @pytest.fixture
    def agent(self, agent_config):
        """Create test agent instance."""
        return HotpotQAAgent(config=agent_config)
    
    @pytest.fixture
    def sample_context(self):
        """Sample context data for testing."""
        return [
            ["France", ["Paris is the capital of France.", "France is in Europe."]],
            ["Paris", ["Paris is a major city.", "It has famous landmarks."]]
        ]

    # Initialization Tests
    
    def test_init(self, agent, agent_config):
        """Test agent initialization."""
        assert agent.config == agent_config
        assert agent.state is None
    
    def test_init_with_custom_config(self):
        """Test agent initialization with custom configuration."""
        config = AgentConfig(model="gpt-4", max_steps=10, temperature=0.8)
        agent = HotpotQAAgent(config=config)
        assert agent.config.model == "gpt-4"
        assert agent.config.max_steps == 10
        assert agent.config.temperature == 0.8
    
    # Context Formatting Tests
    
    def test_format_context_empty(self, agent):
        """Test formatting empty context."""
        result = agent._format_context([])
        assert result == ""

    def test_format_context_single_item(self, agent):
        """Test formatting single context item."""
        context = [["Title1", ["Sentence 1.", "Sentence 2."]]]
        result = agent._format_context(context)
        assert "[Title1]" in result
        assert "Sentence 1." in result
        assert "Sentence 2." in result

    def test_format_context_multiple_items(self, agent, sample_context):
        """Test formatting multiple context items."""
        result = agent._format_context(sample_context)
        assert "[France]" in result
        assert "[Paris]" in result
        assert "Paris is the capital" in result
        assert "major city" in result
    
    def test_format_context_with_single_sentence(self, agent):
        """Test formatting context item with single sentence (not a list)."""
        context = [["Title", "Single sentence here."]]
        result = agent._format_context(context)
        assert "[Title]" in result
        assert "Single sentence here." in result
    
    # Answer Extraction Tests

    def test_extract_answer_with_final_answer(self, agent):
        """Test extracting answer with 'Final Answer:' pattern."""
        response = "Let me think... Final Answer: Paris"
        is_final, answer = agent._extract_answer(response)
        assert is_final is True
        assert answer == "Paris"

    def test_extract_answer_lowercase(self, agent):
        """Test extracting answer with lowercase pattern."""
        response = "Based on analysis, final answer: London"
        is_final, answer = agent._extract_answer(response)
        assert is_final is True
        assert answer == "London"
    
    def test_extract_answer_with_answer_pattern(self, agent):
        """Test extracting answer with 'Answer:' pattern."""
        response = "After reasoning, Answer: Tokyo"
        is_final, answer = agent._extract_answer(response)
        assert is_final is True
        assert answer == "Tokyo"

    def test_extract_answer_with_quotes(self, agent):
        """Test extracting answer removes quotes."""
        response = "Final Answer: \"New York\""
        is_final, answer = agent._extract_answer(response)
        assert is_final is True
        assert answer == "New York"
    
    def test_extract_answer_multiline(self, agent):
        """Test extracting answer from multiline response."""
        response = """Let me reason through this.
        Based on the facts provided,
        Final Answer: Berlin
        This is the correct answer."""
        is_final, answer = agent._extract_answer(response)
        assert is_final is True
        assert answer == "Berlin"

    def test_extract_answer_no_pattern(self, agent):
        """Test extracting answer when no pattern matches (still reasoning)."""
        response = "The answer is clearly Berlin but I need to verify..."
        is_final, answer = agent._extract_answer(response)
        assert is_final is False
        assert answer == ""
    
    # Previous Steps Formatting Tests
    
    def test_format_previous_steps_empty(self, agent):
        """Test formatting with no previous steps."""
        result = agent._format_previous_steps([])
        assert result == ""
    
    def test_format_previous_steps_with_reasoning(self, agent):
        """Test formatting previous reasoning steps."""
        steps = [
            {'step_num': 1, 'reasoning': 'First, I need to identify...', 'answer': None},
            {'step_num': 2, 'reasoning': 'Next, I should connect...', 'answer': None}
        ]
        result = agent._format_previous_steps(steps)
        assert "Previous reasoning steps:" in result
        assert "Step 1:" in result
        assert "Step 2:" in result
    
    def test_format_previous_steps_skips_final_answer(self, agent):
        """Test that final answer step without reasoning is skipped."""
        steps = [
            {'step_num': 1, 'reasoning': 'Analyzing the context...', 'answer': None},
            {'step_num': 2, 'reasoning': None, 'answer': 'Paris'}
        ]
        result = agent._format_previous_steps(steps)
        assert "Step 1:" in result
        assert "Step 2:" not in result
    
    # State Management Tests

    def test_reset(self, agent):
        """Test agent reset."""
        agent._state = AgentState(question="test")
        agent._llm = AsyncMock()
        assert agent.state is not None
        
        agent.reset()
        
        assert agent.state is None
        assert agent._llm is None
    
    # Integration Tests with Mocked LLM

    @pytest.mark.asyncio
    async def test_run_single_step_answer(self, agent, sample_context):
        """Test run with immediate answer (single-step)."""
        mock_llm = AsyncMock(return_value="Final Answer: Paris")
        agent._llm = mock_llm
        
        result = await agent.run(
            question="What is the capital of France?",
            context=sample_context
        )
        
        assert result == "Paris"
        assert agent.state.finished is True
        assert len(agent.state.steps) == 1
        mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_multi_step_reasoning(self, agent, sample_context):
        """Test run with multi-step reasoning."""
        # Simulate multi-step reasoning
        mock_llm = AsyncMock(side_effect=[
            "I need to analyze the context first...",
            "Based on the information, Final Answer: Paris"
        ])
        agent._llm = mock_llm
        
        result = await agent.run(
            question="What is the capital of France?",
            context=sample_context
        )
        
        assert result == "Paris"
        assert agent.state.finished is True
        assert len(agent.state.steps) == 2
        assert mock_llm.call_count == 2

    @pytest.mark.asyncio
    async def test_run_with_string_context(self, agent):
        """Test run with string context instead of list."""
        mock_llm = AsyncMock(return_value="Final Answer: Tokyo")
        agent._llm = mock_llm
        
        result = await agent.run(
            question="What is the capital of Japan?",
            context="Tokyo is the capital of Japan."
        )
        
        assert result == "Tokyo"
        assert agent.state.finished is True
    
    @pytest.mark.asyncio
    async def test_run_max_steps_reached(self, agent):
        """Test that run stops at max_steps."""
        # Never provide final answer
        mock_llm = AsyncMock(return_value="I'm still thinking...")
        agent._llm = mock_llm
        
        result = await agent.run(
            question="Complex question?",
            context="Some context"
        )
        
        # Should stop after max_steps (3)
        assert mock_llm.call_count == 3
        assert len(agent.state.steps) == 3
    
    @pytest.mark.asyncio
    async def test_run_empty_context(self, agent):
        """Test run with empty context."""
        mock_llm = AsyncMock(return_value="Final Answer: Unknown")
        agent._llm = mock_llm
        
        result = await agent.run(
            question="Question?",
            context=[]
        )
        
        assert result == "Unknown"
        assert agent.state.context == ""
    
    # Step Execution Tests

    @pytest.mark.asyncio
    async def test_step_records_history(self, agent):
        """Test that step records action history correctly."""
        mock_llm = AsyncMock(return_value="Final Answer: Berlin")
        agent._llm = mock_llm
        
        state = AgentState(
            question="What is the capital of Germany?",
            context="Berlin is the capital of Germany.",
            steps=[],
            answer="",
            finished=False
        )
        
        new_state = await agent.step(state)
        
        assert len(new_state.steps) == 1
        assert new_state.steps[0]['step_num'] == 1
        assert new_state.steps[0]['answer'] == 'Berlin'
        assert new_state.finished is True
        assert new_state.answer == "Berlin"
    
    @pytest.mark.asyncio
    async def test_step_continues_reasoning(self, agent):
        """Test step that continues reasoning without final answer."""
        mock_llm = AsyncMock(return_value="I need to connect these facts...")
        agent._llm = mock_llm
        
        state = AgentState(
            question="Complex question?",
            context="Some facts here.",
            steps=[],
            answer="",
            finished=False
        )
        
        new_state = await agent.step(state)
        
        assert len(new_state.steps) == 1
        assert new_state.steps[0]['reasoning'] is not None
        assert new_state.steps[0]['answer'] is None
        assert new_state.finished is False
        assert new_state.answer == ""
    
    @pytest.mark.asyncio
    async def test_step_with_previous_steps(self, agent):
        """Test step execution with previous reasoning history."""
        mock_llm = AsyncMock(return_value="Final Answer: Result")
        agent._llm = mock_llm
        
        state = AgentState(
            question="Question?",
            context="Context",
            steps=[
                {'step_num': 1, 'reasoning': 'First thought...', 'answer': None}
            ],
            answer="",
            finished=False
        )
        
        new_state = await agent.step(state)
        
        # Should have 2 steps now
        assert len(new_state.steps) == 2
        assert new_state.steps[1]['step_num'] == 2
        
        # Check that prompt included previous steps
        call_args = mock_llm.call_args[0][0]
        assert "Previous reasoning steps:" in call_args
    
    # Error Handling Tests

    @pytest.mark.asyncio
    async def test_run_handles_llm_error(self, agent):
        """Test run handles LLM errors gracefully."""
        mock_llm = AsyncMock(side_effect=Exception("API Error"))
        agent._llm = mock_llm
        
        result = await agent.run(
            question="Test question?",
            context="Some context"
        )
        
        assert result == "Error occurred"
        assert agent.state.finished is True
    
    @pytest.mark.asyncio
    async def test_step_handles_exception(self, agent):
        """Test step handles exceptions and marks as finished."""
        mock_llm = AsyncMock(side_effect=ValueError("Invalid input"))
        agent._llm = mock_llm
        
        state = AgentState(question="Q?", context="C", steps=[], answer="", finished=False)
        new_state = await agent.step(state)
        
        assert new_state.finished is True
        assert new_state.answer == "Error occurred"


class TestAgentConfig:
    """Test cases for AgentConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AgentConfig()
        assert config.model == "gpt-4o-mini"
        assert config.max_steps == 5
        assert config.verbose is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AgentConfig(
            model="gpt-4",
            max_steps=10,
            verbose=True,
            temperature=0.5
        )
        assert config.model == "gpt-4"
        assert config.max_steps == 10
        assert config.verbose is True
        assert config.temperature == 0.5

    def test_config_validation_max_steps(self):
        """Test that max_steps must be >= 1."""
        with pytest.raises(ValidationError):
            AgentConfig(max_steps=0)

    def test_config_validation_temperature_range(self):
        """Test that temperature must be between 0 and 2."""
        with pytest.raises(ValidationError):
            AgentConfig(temperature=-0.1)
        with pytest.raises(ValidationError):
            AgentConfig(temperature=2.5)

    def test_config_forbids_extra_fields(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            AgentConfig(unknown_field="value")


class TestAgentState:
    """Test cases for AgentState class."""

    def test_default_state(self):
        """Test default state values."""
        state = AgentState()
        assert state.question == ""
        assert state.context == ""
        assert state.steps == []
        assert state.answer == ""
        assert state.finished is False

    def test_custom_state(self):
        """Test custom state values."""
        state = AgentState(
            question="Test?",
            context="Context",
            answer="Answer",
            finished=True
        )
        assert state.question == "Test?"
        assert state.context == "Context"
        assert state.answer == "Answer"
        assert state.finished is True

    def test_state_model_copy(self):
        """Test state immutable update via model_copy."""
        state = AgentState(question="Original")
        new_state = state.model_copy(update={"answer": "New Answer"})
        
        assert state.answer == ""  # Original unchanged
        assert new_state.answer == "New Answer"
        assert new_state.question == "Original"

    def test_state_serialization(self):
        """Test state can be serialized to dict."""
        state = AgentState(question="Test?", answer="Answer")
        data = state.model_dump()
        
        assert data['question'] == "Test?"
        assert data['answer'] == "Answer"
