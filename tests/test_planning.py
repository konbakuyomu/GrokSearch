import pytest

from smart_search.planning import PlanningEngine, PHASE_NAMES, _split_csv


class TestSplitCsv:
    def test_basic(self):
        assert _split_csv("a, b, c") == ["a", "b", "c"]

    def test_empty(self):
        assert _split_csv("") == []

    def test_whitespace_only(self):
        assert _split_csv("  ,  ,  ") == []

    def test_single(self):
        assert _split_csv("hello") == ["hello"]


class TestPlanningEngine:
    def setup_method(self):
        self.engine = PlanningEngine()

    def test_intent_creates_session(self):
        result = self.engine.process_phase(
            phase="intent_analysis",
            thought="Analyzing user intent",
            phase_data={
                "core_question": "What is X?",
                "query_type": "factual",
                "time_sensitivity": "irrelevant",
            },
        )
        assert "session_id" in result
        assert "intent_analysis" in result["completed_phases"]
        assert result["plan_complete"] is False

    def test_phase_ordering_enforced(self):
        """验证必须先完成前驱阶段"""
        result = self.engine.process_phase(
            phase="complexity_assessment",
            thought="Skip intent",
            phase_data={"level": 1},
        )
        assert "error" in result
        assert "intent_analysis" in result["error"]

    def test_full_level1_flow(self):
        """验证 Level 1 完整流程 (3 阶段)"""
        r1 = self.engine.process_phase(
            phase="intent_analysis",
            thought="Analyzing",
            phase_data={
                "core_question": "What is X?",
                "query_type": "factual",
                "time_sensitivity": "irrelevant",
            },
        )
        sid = r1["session_id"]

        r2 = self.engine.process_phase(
            phase="complexity_assessment",
            thought="Simple query",
            session_id=sid,
            phase_data={"level": 1, "estimated_sub_queries": 1, "estimated_tool_calls": 1, "justification": "simple"},
        )
        assert r2["complexity_level"] == 1

        r3 = self.engine.process_phase(
            phase="query_decomposition",
            thought="Single query",
            session_id=sid,
            phase_data={"id": "sq1", "goal": "Find X", "expected_output": "Definition", "boundary": "none"},
        )
        assert r3["plan_complete"] is True
        assert "executable_plan" in r3

    def test_level1_blocks_extra_phases(self):
        """验证 Level 1 不允许进入 phase 4-6"""
        r1 = self.engine.process_phase(
            phase="intent_analysis", thought="t",
            phase_data={"core_question": "Q", "query_type": "factual", "time_sensitivity": "irrelevant"},
        )
        sid = r1["session_id"]

        self.engine.process_phase(
            phase="complexity_assessment", thought="t", session_id=sid,
            phase_data={"level": 1, "estimated_sub_queries": 1, "estimated_tool_calls": 1, "justification": "s"},
        )
        self.engine.process_phase(
            phase="query_decomposition", thought="t", session_id=sid,
            phase_data={"id": "sq1", "goal": "G", "expected_output": "E", "boundary": "B"},
        )

        result = self.engine.process_phase(
            phase="search_strategy", thought="t", session_id=sid,
            phase_data={"approach": "targeted", "search_terms": []},
        )
        assert "error" in result

    def test_accumulative_phases(self):
        """验证 query_decomposition 累积多次调用"""
        r1 = self.engine.process_phase(
            phase="intent_analysis", thought="t",
            phase_data={"core_question": "Q", "query_type": "factual", "time_sensitivity": "irrelevant"},
        )
        sid = r1["session_id"]

        self.engine.process_phase(
            phase="complexity_assessment", thought="t", session_id=sid,
            phase_data={"level": 3, "estimated_sub_queries": 2, "estimated_tool_calls": 4, "justification": "complex"},
        )

        self.engine.process_phase(
            phase="query_decomposition", thought="first", session_id=sid,
            phase_data={"id": "sq1", "goal": "G1", "expected_output": "E1", "boundary": "B1"},
        )
        r = self.engine.process_phase(
            phase="query_decomposition", thought="second", session_id=sid,
            phase_data={"id": "sq2", "goal": "G2", "expected_output": "E2", "boundary": "B2"},
        )

        session = self.engine.get_session(sid)
        data = session.phases["query_decomposition"].data
        assert isinstance(data, list)
        assert len(data) == 2

    def test_reset_clears_sessions(self):
        """验证 reset() 清空所有会话"""
        r = self.engine.process_phase(
            phase="intent_analysis", thought="t",
            phase_data={"core_question": "Q", "query_type": "factual", "time_sensitivity": "irrelevant"},
        )
        sid = r["session_id"]
        assert self.engine.get_session(sid) is not None

        self.engine.reset()
        assert self.engine.get_session(sid) is None

    def test_unknown_phase(self):
        result = self.engine.process_phase(
            phase="nonexistent_phase", thought="t", phase_data={},
        )
        assert "error" in result
