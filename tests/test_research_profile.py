import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from paper2slides.core.paths import (
    get_config_name,
    get_rag_checkpoint,
    get_summary_checkpoint,
)
from paper2slides.generator.config import (
    GenerationConfig,
    GenerationInput,
    OutputType,
    SlideProfile,
    StyleType,
)
from paper2slides.generator.content_planner import ContentPlanner
from paper2slides.rag.query import get_paper_queries
from paper2slides.summary.models import OriginalElements
from paper2slides.summary.paper import PaperContent


class TestResearchProfile(unittest.TestCase):
    def test_research_profile_has_deeper_default_range_and_custom_bounds(self):
        research = GenerationConfig(
            output_type=OutputType.SLIDES,
            profile=SlideProfile.RESEARCH,
            style=StyleType.ACADEMIC,
        )
        self.assertEqual(research.get_page_range(), (24, 36))

        custom = GenerationConfig(
            output_type=OutputType.SLIDES,
            profile=SlideProfile.RESEARCH,
            min_slides=18,
            max_slides=28,
        )
        self.assertEqual(custom.get_page_range(), (18, 28))
        with self.assertRaises(ValueError):
            GenerationConfig(min_slides=12, max_slides=4)

    def test_research_checkpoints_are_separate_from_standard(self):
        base = Path("/tmp/paper2slides-test")
        standard = {"fast_mode": False, "profile": "standard"}
        research = {"fast_mode": False, "profile": "research"}
        self.assertNotEqual(get_rag_checkpoint(base, standard), get_rag_checkpoint(base, research))
        self.assertNotEqual(get_summary_checkpoint(base, standard), get_summary_checkpoint(base, research))
        research_appendix = {"fast_mode": False, "profile": "research", "include_appendix": True}
        self.assertNotEqual(
            get_rag_checkpoint(base, research),
            get_rag_checkpoint(base, research_appendix),
        )
        self.assertIn("research", get_config_name({"output_type": "slides", "profile": "research"}))

    def test_research_queries_add_technical_categories(self):
        standard = get_paper_queries("standard")
        research = get_paper_queries("research")
        self.assertNotIn("formulation", standard)
        self.assertIn("data_generation", research)
        self.assertIn("formulation", research)
        self.assertIn("implementation", research)
        self.assertNotIn("appendix", research)
        self.assertIn("appendix", get_paper_queries("research", include_appendix=True))
        self.assertIn("appendix", get_paper_queries("standard", include_appendix=True))

    def test_research_planner_uses_research_prompt_and_metadata(self):
        fake_client = SimpleNamespace(settings=SimpleNamespace(provider="openai_compatible"))
        planner = ContentPlanner(llm_client=fake_client)
        planner._load_figure_images = Mock(return_value=[])
        planner._call_multimodal_llm = Mock(return_value='{"slides": [{"id": "s1", "title": "Title", "content": "Details", "tables": [], "figures": []}]}')

        config = GenerationConfig(
            output_type=OutputType.SLIDES,
            profile=SlideProfile.RESEARCH,
            include_appendix=True,
            min_slides=20,
            max_slides=24,
        )
        generation_input = GenerationInput(
            config=config,
            content=PaperContent(solution="equation and method details"),
            origin=OriginalElements(),
        )
        plan = planner.plan(generation_input)
        prompt = planner._call_multimodal_llm.call_args.args[0]
        self.assertIn("research-seminar", prompt)
        self.assertIn("data / benchmark construction", prompt.lower())
        self.assertIn("20-24", prompt)
        self.assertIn("include the appendix", prompt.lower())
        self.assertEqual(plan.metadata["profile"], "research")
        self.assertEqual(plan.metadata["requested_slide_range"], (20, 24))
        self.assertFalse(plan.metadata["quality_checks"]["slide_count_in_requested_range"])
        self.assertEqual(plan.metadata["quality_checks"]["uncategorized_slides"], 0)

    def test_explicit_planner_context_limit_is_honored(self):
        fake_client = SimpleNamespace(settings=SimpleNamespace(provider="openai_compatible"))
        with patch.dict("os.environ", {"PLANNER_LLM_CONTEXT_CHARS": "1234"}):
            planner = ContentPlanner(llm_client=fake_client)
        self.assertEqual(planner.context_chars, 1234)
        self.assertTrue(planner.context_chars_explicit)


if __name__ == "__main__":
    unittest.main()
