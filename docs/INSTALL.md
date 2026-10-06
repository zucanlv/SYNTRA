# Synthesis environment

The initial supported target is Linux and Python 3.10, matching the research
environment. The original CLI and pipeline modules are retained. Training and
benchmark evaluation have separate dependencies and are not needed for synthesis.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1
python -m pip install -r requirements-synthesis.txt
```

Install **one** compatible FAISS build. The research environment used
`faiss-gpu==1.7.3`; its installation depends on the platform/CUDA environment.
For CPU FAISS with Python 3.10, the delivery candidate is:

```bash
python -m pip install faiss-cpu==1.7.4
python -m pip check
```

Do not install CPU and GPU FAISS packages into the same environment. The supplied
delivery template sets both FAISS GPU switches to false. Set the GPU switches and
device IDs explicitly if you use GPU FAISS. Embedding encoding may still use an
available GPU; `CUDA_VISIBLE_DEVICES=""` forces the documented CPU route.

The direct Python dependency pins come from the original environment; they are
not a transitive lockfile or a claim that a new installation has been reproduced.
See [validation](DELIVERY_VALIDATION.md) for checks actually performed. GPU and
full-corpus runtime requirements depend on corpus/model size; no universal memory
or runtime promise is made.

Qwen-Agent must expose `parallel_exec` and `serial_exec` in
`qwen_agent.utils.parallel_executor`. The import check in the workflow checks the
actual pipeline imports. Do not install an unrelated similarly named package.

Supply an OpenAI-compatible endpoint and a model available there. Keep the API key
in `OPENAI_API_KEY`; keep YAML `api_key` fields null. `OPENAI_BASE_URL` supplies the
global endpoint when `llm_model_server` is null. YAML strings such as `${DATA_ROOT}`
are **not** expanded by the original loader. Model names must match your endpoint;
provider-prefixed names from archived experiments may not work at another endpoint.

For training/evaluation, use the recorded versions in `environment_snapshot.json`
and inspect the relevant recipe. The bundled FlagEmbedding has local research
changes; substituting an arbitrary PyPI version is not equivalent. Those archived
recipes still need their data/model paths and environment adapted.
