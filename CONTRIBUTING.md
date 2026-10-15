# Contributing to SYNTRA

Thank you for your interest in SYNTRA. Bug reports, questions, reproductions and results on new retrieval tasks are all welcome.

## Reporting problems and asking questions

- Open an [issue](https://github.com/zucanlv/SYNTRA/issues/new/choose) and pick the matching template.
- For pipeline errors, include the stage, the full traceback, the command, the relevant configuration (without secrets or private paths) and your environment.
- Many common questions (inputs, outputs, compute, limitations) are answered in the [project page FAQ](https://zucanlv.github.io/SYNTRA/#faq).

## Sharing results on new tasks

If you apply SYNTRA to a new corpus, domain or task, we would like to hear about it, including cases where it did not help. Useful details are the task description, the number of few-shot examples, the LLM used for synthesis, the embedding model, and retrieval scores before and after fine-tuning.

## Pull requests

1. Keep changes focused; open an issue first for larger changes.
2. Run the existing delivery checks described in [docs/DELIVERY_VALIDATION.md](docs/DELIVERY_VALIDATION.md) where relevant.
3. Describe what changed and how you tested it.

Contributions are accepted under the repository's [MIT License](LICENSE). Third-party code keeps its upstream license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citing

If SYNTRA is useful in your work, please cite the paper; see the [Citation section of the README](README.md#citation) or [CITATION.cff](CITATION.cff).
