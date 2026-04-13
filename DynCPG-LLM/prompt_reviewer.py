SYSTEM_PROMPT = """You are a professional security code reviewer. Your task is to analyze a given function and review the code for security defects"""

ANALYSIS_RULES_PROMPT = """ANALYSIS RULES:
Each step of your reasoning and hypothesis requires evidence, and you can request additional code context to support it. You can only request one missing piece of context at each step. After steps of reasoning, you can finally obtain the security defect detection result for the code. Now you need to generate the output for the next step."""


OUTPUT_FORMAT_PROMPT = """OUTPUT FORMAT: strictly in JSON
If you need additional context, output:
{
  "thought": "<clear, reasoning explaining what has been analyzed and why context is required>",
  "missing_context": "<concise description of ONE required context, should be retrievable via a CPG query, including definition, data flow, control flow, caller relationship, etc.(e.g., “data flow from A to B”)>"
}

If you can make the final answer, output:
{
    "vulnerability": [
    {
        "vulnerability_type": "vulner_1",
        "cwe_id": "cwe_id_1",
        "location": "code_line_number_1",
        "explanation": "vulner_desc_and_evidence_1"
    },
    {
        "vulnerability_type": "vulner_2",
        "cwe_id": "cwe_id_2",
        "location": "code_line_number_2",
        "explanation": "vulner_desc_and_evidence_2"
    }
  ]
}
If no vulnerability is found, output:
{
    "vulnerability": []
}
"""
OUTPUT_FORMAT_PROMPT_CRITIC = """OUTPUT FORMAT: strictly in JSON
If you need additional context, output:
{
  "thought": "<clear, reasoning explaining what has been analyzed and why context is required>",
  "missing_context": "<concise description of ONE required context, should be retrievable via a CPG query, including definition, data flow, control flow, caller relationship, etc.(e.g., “data flow from A to B”)>"
}

If you can make the final answer, output:
{
    "vulnerability": [
    {
        "vulnerability_type": "vulner_1",
        "cwe_id": "cwe_id_1",
        "location": "code_line_number_1",
        "explanation": "vulner_desc_and_evidence_1"
    },
    {
        "vulnerability_type": "vulner_2",
        "cwe_id": "cwe_id_2",
        "location": "code_line_number_2",
        "explanation": "vulner_desc_and_evidence_2"
    }
  ]
}
If no NEW vulnerabilities are found, output:
{
    "vulnerability": []
}"""