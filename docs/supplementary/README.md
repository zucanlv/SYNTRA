# Accompanying Supplementary Information files

- `shared_prompt_templates.json`: complete expanded shared templates, including the optional document filter, alternative annotation validator and query-generation ablation. Named display blocks in the PDF are expanded here; runtime placeholders remain unfilled.
- `task_specific_instruction_examples.json`: complete selected FiQA, ArguAna, NFCorpus and TheoremQA instruction records, plus the initial FiQA instructions. These illustrate refinement and are not a complete production-instruction inventory.
- `task_registry_descriptors.json`: all 24 task descriptors and available registered-example metadata from the supplementary source.
- `fiqa_instruction_refinement_case.json`: the complete initial/refined query and annotation instructions, source examples and recorded label comparison shown in the FiQA case study.

The corresponding source builders and input formatters are in `Main_Pipeline/HighLevel_Generator.py`, `IdAttr_Generator.py`, `QueryInstr_Generator.py`, `DiverseQuery_Generator.py`, `AnnoInstr_Generator.py`, `AnnoCalibration_Generator.py`, `Batch_Candidate_Annotator.py`, `Few_Shot_Formatter.py`, `Few_Shot_Example.py` and `DocType_Filter.py`.

Prompt text is preserved rather than retrospectively edited. These JSON files retain punctuation and branch-specific output contracts; outer blank-line spacing follows the displayed archive. The source builders specify exact whitespace. The PDF uses typographic wrapping and shared display blocks to avoid repeating identical content.
