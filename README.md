<div align="center">
<h1> AI Engineer Roadmap & Resources 🤖 </h1>

<p align="center">
<a href="https://www.youtube.com/channel/UCH-xwLTKQaABNs2QmGxK2bQ/">Follow me on YouTube</a> •  <a href="https://twitter.com/dswharshit"> X  </a> •
<a href="https://www.linkedin.com/in/tyagiharshit/">LinkedIn </a> •
<a href="https://dswharshit.substack.com/"> Subscribe to my Newsletter </a>
</p>
</div>


---
The AI Engineering Roadmap categorizes the journey into 
1. `Beginner` - Master the skill of developing applications with LLMs, APIs, and Prompt Engineering.
2. `Intermediate` - Build expertise in handling longer context, building RAG applications, and building agents.
3. `Advanced` (Entering AI Research territory) - Learn to fine-tune LLMs, build automated and orchestrated pipelines with model evals and testing frameworks.

I'd suggest you go deep in each stage, build projects, POCs or better yet, functional products and then move to the next stage.

![AI Engineer Roadmap Cover](./imgs/ai_engg_roadmap.png)


## Learning Resources & References
The following table enlists learning resources and references that I found helpful and plan to use myself to build.


| Step                               | Resources |
|------------------------------------|-----------|
| **Beginner**                       |           |
| Working with LLM APIs              | - Commonly used LLM APIs: [OpenAI](https://www.notion.so/Resources-34b331afa220479889666fb6e0f245f7?pvs=21), [Anthropic(Claude)](https://docs.anthropic.com/claude/reference/getting-started-with-the-api), [Hugging Face](https://huggingface.co/inference-api) |
| Prompt Engineering                 | - [DeepLearning.AI Course on ChatGPT Prompt Engineering for Developers](http://DeepLearning.AI)<br>- [Prompt Engineering Guide](https://www.promptingguide.ai/): A detailed resource encapsulating the latest papers, advanced prompting techniques, learning guides, model-specific prompting guides, lectures, references, new LLM capabilities, and tools related to prompt engineering.<br>- [Awesome ChatGPT Prompts](https://github.com/f/awesome-chatgpt-prompts): A compilation of great prompts to be used with ChatGPT models. |
| Running and working with Open Source LLMs | - [Deeplearning.AI course on Open Source Models with Hugging Face](http://Deeplearning.AI)<br>- Open Source LLMs can be accessed via [Hugging Face Hub](https://huggingface.co/models) and you can play with a few of them in [Hugging Face Spaces](https://huggingface.co/spaces)<br>- [OpenRouter Docs](https://openrouter.ai/docs#quick-start)<br>- Run LLMs on your local machine using [LM Studio](https://lmstudio.ai/) |
| Chain of Operations - LangChain    | - [Quickstart guide](https://python.langchain.com/docs/get_started/quickstart/) on how to build an application with LangChain.<br>- [Deeplearning.AI course on LangChain for LLM Application Development](http://Deeplearning.AI) |
| Code / Image / Audio Generation    | - This is covered in parts in the [Open Source Models with HF](https://www.deeplearning.ai/short-courses/open-source-models-hugging-face/) course on DeepLearning.AI<br>- [Building Generative AI Applications with Gradio](https://www.deeplearning.ai/short-courses/building-generative-ai-applications-with-gradio/)<br>- Code Gen: Check out these resources on code generation - [gpt-engineer](https://github.com/gpt-engineer-org/gpt-engineer), [Tabby](https://tabby.tabbyml.com/), [gpt-migrate](https://github.com/joshpxyne/gpt-migrate) to migrate your codebase from one framework to another or one language to another.<br>- Audio Gen: [text to speech by openAI](https://platform.openai.com/docs/guides/text-to-speech), [resemble.ai](https://www.resemble.ai/), [elevenlabs API](https://elevenlabs.io/docs/api-reference/text-to-speech)<br>- Image Gen: [Image generation by Open AI](https://platform.openai.com/docs/guides/images?context=node), [creating images using Stable Diffusion API](https://replicate.com/docs/get-started/discord-bot) |
| **Intermediate**                   |           |
| Working with Vector Databases      | - [Text chunking and splitting by LangChain](https://python.langchain.com/docs/modules/data_connection/document_transformers/)<br>- [Course on vector databases](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/): Learn what are embeddings and how to store them. Build applications.<br>- Another course on building applications with vector databases using Pinecone<br>- Learn to compute sentence, text, and image embeddings using Framework like [SentenceTransformers](https://www.sbert.net/).<br>- Check out top embedding models [here](https://huggingface.co/spaces/mteb/leaderboard). |
| Building RAG Applications           | - RAG applications are all about building connections between tools, databases, context lengths, embeddings, memories, etc. You need frameworks like [LangChain](https://python.langchain.com/docs/get_started/introduction), [LlamaIndex](https://docs.llamaindex.ai/en/stable/), [FastRAG](https://github.com/IntelLabs/fastRAG) to build these.<br>- [Step-by-step tutorial to build a Q&A RAG pipeline by LangChain](https://python.langchain.com/docs/use_cases/question_answering/quickstart)<br>- [LangChain’s RAG from Scratch](https://www.youtube.com/watch?v=wd7TZ4w1mSw&list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x&ab_channel=LangChain) playlist on YouTube is pretty detailed and amazing. |
| Advanced RAGs                       | - Check out Jerrry Liu’s course on [Building and evaluating Advanced RAG Application](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) on [DeepLearning.AI](http://DeepLearning.AI) for best practices and improving your RAG pipeline’s performance.<br>- [Cheat Sheet and some recipes for building Advanced RAG](https://www.llamaindex.ai/blog/a-cheat-sheet-and-some-recipes-for-building-advanced-rag-803a9d94c41b)<br>- [A comprehensive guide on building RAG-based LLM application by AnyScale](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1) |
| Evaluating RAGs                     | - [Hugging Face Cookbook on How to evaluate RAG system](https://huggingface.co/learn/cookbook/en/rag_evaluation)<br>- [Evaluating all components of your RAG pipeline](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1#evaluator)<br>- [RAGAS framework](https://docs.ragas.io/en/stable/) to evaluate RAG pipelines. |
| Building Agents                     | - [Quickstart guide by LangChain](https://python.langchain.com/docs/modules/agents/) to build agents to have a sequence of actions taken to do a job or multiple jobs.<br>- [Course on Functions Tools and Agents with LangChain](https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/) by Harrison Chase on DeepLearning.AI<br>- [Autogen](https://microsoft.github.io/autogen/docs/Getting-Started): Framework to develop LLM applications using multiple agents that can converse with each other to solve tasks.<br>- [Crew AI](https://www.crewai.com/): AI Agents for real use cases. |
| Deploying Apps                      | - **Local deployment**: Running open source LLMs on local machines ([LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.ai/), [oobabooga](https://github.com/oobabooga/text-generation-webui), [kobold.cpp](https://github.com/LostRuins/koboldcpp), etc.)<br>- Building POCs and demo applications using frameworks like [Gradio](https://www.gradio.app/) and [Streamlit](https://docs.streamlit.io/)<br>- Deploying LLMs at scale on cloud technologies like [vLLM](https://github.com/vllm-project/vllm/tree/main) and [SkyPilot](https://skypilot.readthedocs.io/en/latest/).<br>- [Deploying LangChain applications](https://python.langchain.com/docs/langserve/) (runnables and chains) as a REST API. |
| **Advanced**                        |           |
| Fine-tuning for specific use cases  | - [DeepLearning.AI course on finetuning LLMs](http://DeepLearning.AI)<br>- [A Beginner’s Guide to LLM Fine-Tuning](https://mlabonne.github.io/blog/posts/A_Beginners_Guide_to_LLM_Finetuning.html) is a detailed guide on finetuning LLMs<br>- A very detailed and simplified read on [how to fine-tune LLMs with Hugging Face by Philipp Schmid](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl)<br>- 4-part [blog series by Anyscale](https://www.anyscale.com/blog/how-to-fine-tune-and-serve-llms) is a comprehensive guide on fine tuning and serving LLMs.<br>- [Fine-Tune Your Own Llama 2 Model in a Colab Notebook](https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html) |
| LLMOps                              | - [Deeplearning.AI Course on LLMOPs](http://Deeplearning.AI) is a good starting place for advanced practitioners.<br>- [GPU Inference optimization techniques](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one) like FlashAttention and FlashAttention-2<br>- [LLMOps guide by DataBricks](https://www.databricks.com/glossary/llmops)<br>- [Efficiently serving LLMs course on DeepLearning.AI](https://www.deeplearning.ai/short-courses/efficiently-serving-llms/). |
| Multi-modal applications            | - [Building hybrid search apps with vector databases like Pinecone](https://www.deeplearning.ai/short-courses/building-applications-vector-databases/)<br>- Cookbook for [multimodal RAG pipelines](https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_and_multi_modal_RAG.ipynb). |
| Model Evals and benchmarking        | - [How to evaluate LLMs with Hugging Face Lighteval](https://www.philschmid.de/sagemaker-evaluate-llm-lighteval)<br>- Course on [Automated Testing for LLMOps](https://www.deeplearning.ai/short-courses/automated-testing-llmops/): Learn to test and evaluate LLM application using an evaluation LLM. |
| AI Security                         | - [Red Teaming LLM Applications](https://www.deeplearning.ai/short-courses/red-teaming-llm-applications/) - learn to identify and evaluate vulnerabilities in LLM apps.<br>- [Planning red teaming for large language models (LLMs) and their applications](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/red-teaming)<br>- A detailed list of resources on [LLM security](https://llmsecurity.net/) highlighting all potential risks and vulnerabilities in AI applications. |


## Functional Product / Project Ideas

Here's the content of your HTML table converted into a well-formatted markdown table:

| S.No. | Project Idea | Skills required / to be developed |
|-------|--------------|-----------------------------------|
|       | **Beginner** |                                   |
| 1.    | Building a chatGPT / Midjourney -like bot for a niche persona / use-case and integrate it with WhatsApp or Slack or Discord or build an app using Streamlit / Gradio. | `LLMs`, `Prompt Engineering`, `open-source LLMs`, `Working with APIs`, `Python` |
| 2.    | Chrome extension to summarise / ideate / extract takeaways / research with web pages. | `LLM APIs`, `Prompt Engineering`, `open-source LLMs`, `Working with API`, `JS`, `HTML`,`CSS` |
| 3.    | Create a news aggregator for a targeted persona (PMs, AI Engg., ) | `Web Parsing`, `Prompt Engineering`, `LLMs` |
| 4.    | Multi-modal generation via Discord. | All of above and `Multi-modal generation` |
|       | **Intermediate** |                                   |
| 5.    | `Q&A bot for specific persona` / department - Doctor / Lawyer / Engineer, Marketer, Sales etc. | `text splitting`, `chunking`, `indexing`, `working with vector databases`, `building RAG pipelines`, `deploying apps` |
| 6.    | `StackOverflow-like search engine` that provides multiple response from multiple documentation sources - Advanced RAG pipeline to answer complex query using multiple data sources. | All above and `vector retrieval`, `summary retrieval`, `sub-question query engine`, `deploying apps` |
| 7.    | AI Interview Buddy - an AI agent that helps in preparing for specific roles in an interactive manner. | `Building agents`, `agentic design`, `text-to-speech`, `speech-to-text`, `deployment` |
|       | **Advanced** |                                   |
| 8.    | Personal assistant fine-tuned on domain-specific data. Personal Healthcare, Country’s Law, Finance, etc. | `Dataset curation`, `Data engineering`, `model evaluation`, `benchmarking`, `PEFT`, `LoRA`, `QLoRA` |
| 9.    | An e2e e-commerce product assistant. | `Advanced RAG pipeline`, `LLMOps` - `Data Engineering`, `CI/CD`, `Scaling`, `Monitoring`, `Observability` |
| 10.   | InvestGPT - fine-tuned LLM to classify personal investments based on technical (algorithmic) and fundamental (LLMs) analysis. | `fine tuning`, `evals and benchmarking`, `AI Security` |

You can extrapolate each of these ideas into full-fledged products / MVPs for different domains, industries and verticals.

If you liked this roadmap, support the work by starring the repo.

You can follow me on [YouTube](https://www.youtube.com/channel/UCH-xwLTKQaABNs2QmGxK2bQ/), [Twitter](https://twitter.com/dswharshit), [LinkedIn](https://www.linkedin.com/in/tyagiharshit/), [Substack](https://dswharshit.substack.com/), [GitHub](https://github.com/tyagi-harshit/).


