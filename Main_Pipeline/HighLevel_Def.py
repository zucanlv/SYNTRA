# (query type, doc type, relevance definition)
HighLevel_Task_Definition = {
    "msmarco": (
        "Web query",
        "Web passage",
        "Given a query (web query) and a document (web passage), the document is relevant to the query if the critical concepts or theories discussed in the document can provide references for domain experts to draft an answer to the query."
    ),
    "arguana": (
        "Argumentative Claim",
        "Counter-argument Passage",
        "Given a query (argumentative claim) and a document (counter-argument passage), the document is relevant if it provides a valid, logically structured counter-argument that directly challenges, refutes, or offers a conflicting perspective to the central premise of the query."
    ),
    "trec-covid": (
        "Covid-19 domain query",
        "Covid-19 related scientific passage",
        "Given a query (Covid-19 domain query) and a document (Covid-19 related scientific passage), the document is relevant if it contains specific biomedical evidence, clinical findings, or epidemiological data that can directly support medical experts in forming evidence-based conclusions or public health responses regarding the pandemic."
    ),
    "apps": (
        "Code contest problem description",
        "Code contest problem solution",
        "Given a query (Code contest problem description) and a document (Code contest problem solution), the document is relevant if it is a complete and correct implementation of the problem."
    ),
    "covid_retrieval": (
        "Chinese-language Administrative Policy Query",
        "Chinese-language Official Information Notice about Covid-19",
        "Given a query (Chinese-language Administrative Policy Query) and a document (Chinese-language Official Information Notice about Covid-19), the document is relevant if it provides precise, official information or procedural details that directly address the specific COVID-19-related policy, measure, or administrative question stated in the query."
    ),
     "climate-fever": (
        "Climate change claim",
        "Informational encyclopedia passage",
        "Given a query (climate change claim) and a document (informational encyclopedia passage), the document is relevant if it provides verifiable factual evidence, scientific definitions, or contextual data that allows an evaluator to support, refute, or clarify the specific claims, statistics, or phenomena mentioned."
    ),
     "dbpedia": (
        "Entity-centric search query",
        "Encyclopedic entity description",
        "Given a query (entity-centric search query) and a document (encyclopedic entity description), the document is relevant if it provides a factual summary of a specific entity, instance, or concept that directly satisfies the categorical constraints and attributes specified in the query."
    ),
    "touche2020": (
        "Controversial Argumentative Question",
        "Argumentative passage with reasoning",
        "Given a query (Controversial Argumentative Question) and a document (Argumentative passage with reasoning), the document is relevant if it provides a comprehensive, logically structured argument or rebuttal that takes a clear stance on the topic, supporting its position with specific contentions, evidence, or reasoning."
    ),
    "cosqa": (
        "Natural language web query for code search",
        "Python function implementation with optional documentation/comments",
        "Given a query (Natural language web query for code search) and a document (Python function implementation with optional documentation/comments), the document is relevant if the code implementation and its internal documentation (docstrings) provide a direct, functional solution to the specific programming task or logic described in the query."
    ),
    "fiqa": (
        "Personal finance or investment question",
        "Financial answer passage",
        "Given a query (personal finance or investment question) and a document (financial answer passage), the document is relevant if it substantially addresses the specific financial scenario, decision, or concept raised by the query and provides information that would materially help answer the user's financial information need."
    ),
    "scidocs": (
        "Scientific paper title",
        "Scholarly article passage",
        "Given a query (scientific paper title) and a document (scholarly article passage), the document is relevant if it describes a scientific work or foundational source that the query paper cites or would reasonably cite, including closely related prior work, methods, datasets, benchmark systems, theoretical background, or seminal work that directly supports, motivates, contextualizes, or is compared with the query paper."
    ),
    "scirepeval-search": (
        "Scientific literature search query",
        "Scientific paper title, abstract, venue, and publication year",
        "Given a query (a scientific literature search query) and a document "
        "(a scientific paper represented by its title, abstract, venue, and "
        "publication year), the document is relevant if it directly satisfies the "
        "scientific information need expressed by the query, such as identifying a "
        "requested work or addressing the specified research topic, method, "
        "phenomenon, dataset, population, intervention, or outcome. Venue and year "
        "may help resolve publication-specific constraints, but do not by themselves "
        "make an otherwise topically unrelated paper relevant. A document that merely "
        "shares broad terminology while studying a different question or failing the "
        "query's specific constraints is insufficient."
    ),
    "nfcorpus": (
        "Non-technical health or nutrition query",
        "Scientific medical article or biomedical research abstract",
        "Given a query (non-technical health or nutrition query) and a document (scientific medical article or biomedical research abstract), the document is relevant if it provides scientific evidence, clinical or epidemiological findings, experimental results, or biomedical explanations that directly address or substantially inform the health, nutrition, disease, or dietary intervention topic raised by the query."
    ),
    "synthetic-text2sql": (
        "Natural Language Database Query",
        "SQL Query",
        "Given a query (Natural Language Database Query) and a document (SQL Query), the document is relevant if it accurately and completely implements the database operation described in the query."
    ),
    "codetrans-dl": (
        "TensorFlow Code Snippet",
        "PaddlePaddle Code Snippet",
        "Given a query (TensorFlow Code Snippet) and a document (PaddlePaddle Code Snippet), the document is relevant if it implements the same deep learning functionality as the query using PaddlePaddle APIs and conventions, making the two snippets functionally equivalent cross-framework counterparts."
    ),
    "bright-documents-economics": (
        "Reasoning-intensive economics question",
        "Economics reference passage",
        "Given a query (reasoning-intensive economics question) and a document (economics reference passage), the document is relevant if it supplies the economic concepts, theories, definitions, empirical evidence, historical background, or examples required to infer the appropriate analytical framework and construct a well-supported answer, even if the document is not lexically similar to the query or does not directly answer it."
    ),
    "bright-documents-pony": (
        "Pony programming task with template",
        "Pony language documentation passage",
        "Given a query (Pony programming task with template) and a document (Pony language documentation passage), the document is relevant only if it provides solution-critical, Pony-specific knowledge that materially determines how to implement the task, such as the exact API, data type, reference-capability rule, language construct, or non-obvious semantic behavior needed by the core solution. A document is not relevant merely because the final code could contain the syntax or feature it describes. Generic documentation about functions, variables, loops, conditionals, ordinary operators, arithmetic, or other basic language mechanics is irrelevant unless the task specifically hinges on a Pony-specific behavior of that feature and the document provides actionable information needed to choose or implement the solution."
    ),
    "bright-documents-theoremqa_questions": (
        "Theorem-based STEM word problem",
        "Solved STEM problem",
        "Given a query (theorem-based STEM word problem) and a document (solved STEM problem), the document is relevant if it uses the same or a closely related theorem, principle, formula, or reasoning pattern required to solve the query, providing a worked example that can guide the solution."  
    ),
    "bright-documents-theoremqa_theorems": (
        "Mathematical word problem",
        "Mathematical theorem, definition, or proof",
        "Given a query (mathematical word problem) and a document "
        "(mathematical theorem, definition, or proof), the document is relevant "
        "if it states, defines, or proves a theorem or mathematical principle "
        "that is used to derive the solution to the query."
    ),
    "legalbench-rag": (
        "Natural-language question about a legal contract or privacy policy",
        "Legal contract or privacy policy passage",
        "Given a query (a natural-language question about a legal contract or "
        "privacy policy) and a document (a legal contract or privacy policy "
        "passage), the document is relevant if it contains one or more specific "
        "clauses or textual statements that directly answer the question or provide "
        "the evidence needed to determine the answer. Evidence that establishes a "
        "negative answer is also relevant; merely discussing the same legal topic, "
        "agreement, party, or policy without resolving the question is insufficient."
    ),
    "lecardv2": (
        "Chinese Criminal Case Description",
        "Chinese Criminal Judgment",
        "Given a query (a Chinese criminal case description) and a document "
        "(a candidate Chinese criminal judgment), the document is relevant if "
        "legal experts consider it legally comparable to the query, taking into "
        "account the criminal charge and similarities in critical facts, "
        "constitutive elements of the offense, sentencing circumstances, "
        "procedural controversies, and overall legal relevance."
    ),
    "cqadupstack-stats": (
        "Statistics Q&A question",
        "Statistics Q&A forum post",
        "Given a query (Statistics Q&A question) and a document (Statistics Q&A forum post), the document is relevant if it asks or addresses the same underlying statistical question or problem as the query. It should express an equivalent or near-duplicate information need, even if the wording, notation, dataset, or concrete example differs. Merely mentioning the same statistical method, analytical tool, or broad topic is not sufficient unless the document helps answer the query's specific concern."
    ),
    "cqadupstack-physics": (
        "Physics Q&A question",
        "Physics Q&A forum post",
        "Given a query (Physics Q&A question) and a document (Physics Q&A forum post), the document is relevant if it asks or addresses the same underlying physics question or problem as the query. It should express an equivalent or near-duplicate information need, even if the wording, notation, physical system, or concrete example differs. Merely mentioning the same physical concept, theory, or formula is not sufficient unless the document helps answer the query's specific concern."
    ),
    "cqadupstack-mathematica": (
        "Mathematica Q&A question",
        "Mathematica Q&A forum post",
        "Given a query (Mathematica Q&A question) and a document (Mathematica "
        "Q&A forum post), the document is relevant if it asks or addresses the same "
        "underlying Mathematica or Wolfram Language question or problem as the query. "
        "It should express an equivalent or near-duplicate information need even if "
        "the wording, code, data, or concrete example differs. Merely mentioning the "
        "same function, symbol, or broad topic is insufficient unless the document "
        "helps resolve the query's specific concern."
    ),
    "nq": (
        "Natural-language factoid question",
        "Wikipedia passage",
        "Given a query (natural-language factoid question) and a document (Wikipedia "
        "passage), the document is relevant if it contains factual evidence that "
        "directly answers the question or provides the specific information needed "
        "to derive the answer. Merely discussing the same entity or broad topic "
        "without resolving the question is insufficient."
    ),
    "medicalretrieval": (
        "Chinese-language medical or health question",
        "Chinese-language medical answer passage",
        "Given a query (Chinese-language medical or health question) and a document "
        "(Chinese-language medical answer passage), the document is relevant if it "
        "provides a direct and medically pertinent answer to the question or otherwise "
        "helps resolve the health concern expressed in the query."
    ),
    "evidencebench": (
        "Biomedical scientific hypothesis",
        "Sentence from a biomedical research paper",
        "Given a query (a biomedical scientific hypothesis) and a document "
        "(a sentence from a biomedical research paper), the document is relevant "
        "if it reports a finding, observation, analysis, or conclusion that directly "
        "supports, contradicts, qualifies, or otherwise provides evidence about the "
        "relationship or outcome asserted by the hypothesis. Merely mentioning the "
        "same biomedical entities or broad topic is insufficient."
    ),
    "medical_qa": (
        "English-language medical question",
        "English-language medical answer passage",
        "Given a query (an English-language medical question) and a document "
        "(an English-language medical answer passage), the document is relevant if "
        "it directly provides the specific medical information needed to answer the "
        "question, such as information about a disease, condition, symptom, drug, "
        "medical test, procedure, or treatment. Merely mentioning the same medical "
        "topic or entity without addressing the specific aspect asked about in the "
        "query is insufficient."
    ),
    "spartqa-mchoice": (
        "Textual spatial scene followed by a multiple-choice question",
        "Candidate spatial answer phrase",
        "Given a query (a textual spatial scene followed by a multiple-choice "
        "question) and a document (a candidate spatial answer phrase), the document "
        "is relevant if it denotes an answer supported by the scene's stated spatial "
        "relations and valid inferences across objects and containers. If both "
        "choices satisfy the queried relation, each choice and 'both of them' are "
        "relevant; if neither does, 'none of them' is relevant. Mere object mention "
        "or lexical overlap is insufficient.",
    ),
    "winogrande": (
        "Commonsense fill-in-the-blank sentence",
        "Candidate answer phrase",
        "Given a query (a sentence containing one blank marked by an underscore) "
        "and a document (a candidate answer phrase), the document is relevant if "
        "inserting it into the blank yields the intended completion under the causal, "
        "physical, social, or temporal commonsense constraints of the full sentence. "
        "Grammatical compatibility or lexical association alone is insufficient.",
    ),
    "birco-relic": (
        "Literary analysis excerpt with a masked quotation",
        "Candidate passage from a literary work",
        "Given a query (an excerpt of literary analysis containing a quotation "
        "replaced by '[masked sentence(s)]') and a document (a candidate passage "
        "from the analyzed literary work), the document is relevant if it can be "
        "naturally inserted at the masked position and directly supports at least "
        "one of the literary claims made in the surrounding context. Sharing the "
        "same work, characters, themes, or vocabulary without supporting the "
        "surrounding interpretation, or merely repeating content already present "
        "in the query, is insufficient.",
    ),
}
