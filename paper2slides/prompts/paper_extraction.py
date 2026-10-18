"""
LLM prompts for extracting structured content from paper RAG results
"""
from typing import Dict

# LLM prompts for structured extraction
EXTRACT_PROMPTS: Dict[str, str] = {
    "motivation": """Organize the following research motivation text into a structured format.
IMPORTANT: Keep ALL information, do NOT summarize or omit any details.

Text:
{content}

Output format (use exact headers):
## RESEARCH PROBLEM
[Complete description of the main research problem with full context]

## LIMITATIONS OF EXISTING METHODS
[List ALL limitations mentioned, with full details]
- [limitation 1 with complete explanation]
- [limitation 2 with complete explanation]
...

## RESEARCH GAP
[Complete description of the gap being addressed]

## KEY CHALLENGES
[List all challenges mentioned with details]
- [challenge 1 with explanation]
- [challenge 2 with explanation]
...

## BACKGROUND CONTEXT
[Any relevant background information mentioned]""",

    "solution": """Organize the following methodology text into a structured format.
CRITICAL: Keep ALL technical details, ALL formulas/equations, and ALL component descriptions. Do NOT summarize.

Text:
{content}

Output format (use exact headers):
## FRAMEWORK OVERVIEW
[Complete description of the framework/approach]

## KEY COMPONENTS
[List ALL components with their FULL descriptions]
1. **[Component Name]**: [Complete description of how it works, including sub-components]
2. **[Component Name]**: [Complete description of how it works, including sub-components]
...

## MATHEMATICAL FORMULATIONS
[List ALL formulas and equations exactly as they appear]
- Formula 1: [exact formula]
  - Notation: [explain each symbol]
  - Purpose: [what this formula computes]
- Formula 2: [exact formula]
  - Notation: [explain each symbol]
  - Purpose: [what this formula computes]
...

## TECHNICAL PIPELINE
[Describe the complete processing pipeline step by step]
1. [Step 1 with details]
2. [Step 2 with details]
...

## IMPLEMENTATION DETAILS
[Any specific parameters, settings, or implementation notes mentioned]

## KEY INNOVATIONS
[What makes this approach novel or different from prior work]""",

    "data_generation": """Organize the following data-generation and benchmark-construction evidence.
Keep every exact count, resolution, source, split, sampling rule, preprocessing step, and quality-control detail. Do not invent missing values.

Text:
{content}

Output format (use exact headers):
## DATA SOURCES
[Where the data comes from, including source families, simulations, measurements, or repositories]

## GENERATION / COLLECTION PIPELINE
[Step-by-step description of how examples, scenarios, fields, or trajectories are generated or collected]

## PREPROCESSING AND QUALITY CONTROL
[Normalization, filtering, alignment, deduplication, validity checks, and other processing]

## SPLITS AND COVERAGE
[Exact training/validation/test/held-out counts, resolutions, time windows, categories, and cross-reference design]

## REPRODUCIBILITY VALUES
[All parameters and exact numbers needed to reproduce the dataset or benchmark]""",

    "formulation": """Extract the mathematical, physical, and numerical formulation from the text below.
Preserve equations exactly where possible, define every symbol, and retain assumptions, boundary/initial conditions, discretization, solver, and objective details. Do not replace formulas with vague prose.

Text:
{content}

Output format (use exact headers):
## PROBLEM FORMULATION
[Precise inputs, outputs, governing problem, and assumptions]

## GOVERNING EQUATIONS / OBJECTIVES
[Every important equation in LaTeX or the source notation]

## VARIABLES AND PARAMETERS
[Define all symbols, units, domains, and parameter values]

## NUMERICAL DISCRETIZATION
[Grid/mesh, time stepping, boundary and initial conditions, solver, tolerances, and stability choices]

## ALGORITHM / UPDATE RULE
[Step-by-step numerical or optimization procedure]

## ASSUMPTIONS AND LIMITATIONS
[All stated modeling assumptions and their implications]""",

    "implementation": """Extract all implementation and reproducibility details from the text below.
Keep exact values and software/hardware names. Do not summarize away settings that another researcher would need to repeat the work.

Text:
{content}

Output format (use exact headers):
## MODEL / SYSTEM CONFIGURATION
[Architecture, components, parameter counts, representations, and input/output shapes]

## TRAINING / SOLVER SETTINGS
[Optimizer, learning rate, schedule, batch size, epochs, seeds, stopping rules, tolerances, and solver settings]

## COMPUTATIONAL ENVIRONMENT
[Hardware, software, libraries, parallelism, runtime, memory, and cost]

## END-TO-END WORKFLOW
[Preprocessing, training, inference, postprocessing, and evaluation steps]

## REPRODUCTION CHECKLIST
[Concrete values or procedures needed to reproduce the experiment]""",

    "appendix": """Extract the appendix and supplementary technical material below.
Preserve derivations, proofs, extra experiments, diagnostics, tables, equations, implementation notes, and exact values. Clearly distinguish supplementary evidence from the main paper. Do not invent content.

Text:
{content}

Output format (use exact headers):
## DERIVATIONS / PROOFS
[All relevant derivations or proof details]

## SUPPLEMENTARY EXPERIMENTS
[Additional experiments, diagnostics, sensitivity analyses, and exact results]

## EXTRA TABLES / FIGURES
[Describe or reproduce supplementary tables and figures with exact values]

## ADDITIONAL REPRODUCIBILITY NOTES
[Any implementation, data, or evaluation details only present in the appendix]""",

    "results": """Organize the following experimental results into a structured format.
CRITICAL: Keep ALL numbers, ALL percentages, ALL table data EXACTLY as they appear. Do NOT round or omit any values.
IMPORTANT: Use HTML table format (<table>) for ALL tables, NOT markdown format (| xxx |).

Text:
{content}

Output format (use exact headers):
## DATASET / BENCHMARK
[If available, complete dataset information]
- Name: [exact name]
- Size: [exact statistics - samples, videos, hours, etc.]
- Categories/Splits: [if available]
  - [Category 1]: [exact numbers]
  - [Category 2]: [exact numbers]
  ...
- Other details: [any other dataset info]

## EVALUATION METRICS
[List all metrics used with definitions]
- [Metric 1]: [what it measures]
- [Metric 2]: [what it measures]
...

## MAIN RESULTS
[Reproduce the COMPLETE performance comparison table with ALL methods and ALL metrics]
[Use HTML table format as shown below - this is REQUIRED]
<table>
<tr><th>Method</th><th>Metric1</th><th>Metric2</th><th>...</th></tr>
<tr><td>Method1</td><td>value</td><td>value</td><td>...</td></tr>
<tr><td>Method2</td><td>value</td><td>value</td><td>...</td></tr>
</table>

## PERFORMANCE BY CATEGORY/SPLIT
[If available, show detailed breakdown by category/task/dataset using HTML table]
<table>
<tr><th>Category</th><th>Metric1</th><th>Metric2</th><th>...</th></tr>
<tr><td>Category1</td><td>value</td><td>value</td><td>...</td></tr>
</table>

## ABLATION STUDY
[If available, complete ablation results with ALL numbers using HTML table]
<table>
<tr><th>Variant</th><th>Metric1</th><th>Metric2</th><th>...</th></tr>
<tr><td>-Component</td><td>value</td><td>value</td><td>...</td></tr>
</table>

## DETAILED FINDINGS
[List ALL findings with specific numbers]
- [Finding 1 with exact numbers and percentages]
- [Finding 2 with exact numbers and percentages]
...

## COMPARISON ANALYSIS
[If available, detailed comparison with baseline methods, explaining performance differences]""",

    "contributions": """Extract all contributions, novelty claims, limitations, and future directions from the text.
Keep ALL details, do NOT summarize.

Text:
{content}

Output format (use exact headers):
## MAIN CONTRIBUTIONS
[List ALL contributions with complete explanations]
1. [Contribution 1]: [Full description of what was contributed]
2. [Contribution 2]: [Full description of what was contributed]
...

## NOVELTY & INNOVATIONS
[Detailed explanation of what's new]
- [Innovation 1 with comparison to prior work]
- [Innovation 2 with comparison to prior work]
...

## LIMITATIONS
[All acknowledged limitations]
- [Limitation 1 with details]
- [Limitation 2 with details]
...

## FUTURE DIRECTIONS
[All suggested future work]
- [Direction 1 with explanation]
- [Direction 2 with explanation]
...

## BROADER IMPACT
[If mentioned, any discussion of impact, applications, or societal implications]""",
}
