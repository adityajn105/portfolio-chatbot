---
title: "Aditya Jain"
url: https://adityajain.me
---

MSCS (Honors) @ USC · ML / AI Engineer

Hi, I'm Aditya Jain.

I build machine learning that ships.

I'm a machine learning and software engineer on the Search team at Salesforce, where I build data and ML systems at scale — from Search Analytics, which streams tens of millions of rows per org from Apache Iceberg into customers' Data Cloud, to entity-prediction models on Salesforce's open-source ml4ir. I enjoy turning research ideas into production features people actually use.

I earned my MS in Computer Science with Honors (4.0 GPA) from USC, where I also TA'd Applied NLP (CSCI-544). Before that I spent two years as a Data Scientist at Cognizant working on search-ad click prediction and healthcare analytics. My interests span NLP, information retrieval, and computer vision — especially applications at the intersection of language and vision.

View résumé Get in touch

Writing

## Blog

### How to Actually Work With AI Agents: A Field Guide

The bottleneck in agentic development isn't the model — it's you. A practical field guide to prompting that lands, engineering the context window, the plan → act → verify loop, and the modern tooling (MCP, subagents, skills, memory) that separates people who ship with agents from people who fight them.

- Agents
- LLM
- Prompting
- Productivity

Sep 10, 2026

### Guardrails for LLM Apps and Agents: What Breaks and How to Contain It

An LLM is gullible and, as an agent, powerful — a dangerous combination once it's in production. A field guide to what goes wrong (prompt injection, PII leakage, excessive agency, insecure output, denial-of-wallet and more, mapped to the OWASP LLM Top 10), which guardrail contains each one, and the tools — NeMo Guardrails, Guardrails AI, Llama Guard, Presidio, Bedrock/Azure guardrails, Garak — you use to build them.

- LLM
- Agents
- Security
- Guardrails

Sep 6, 2026

### Using LLMs Efficiently: The Right Model for the Job

The cheapest token is the one you never sent to the biggest model. A practical guide to using LLMs efficiently — how to match a model tier to each task, what model routers like OpenRouter and Cursor's Auto mode actually do, the reuse/context/throughput levers that cut cost without touching quality, and the 'local metric trap' that makes over-optimizing backfire.

- LLM
- Cost
- Routing
- Systems

Sep 5, 2026

### From Query to Next Token: How LLM Inference Gets Fast

How does a model with a million-token context produce the next token in milliseconds? Trace the full journey — query → tokens → forward pass → sampled token — then see why decode is memory-bandwidth-bound, not compute-bound, and how KV caching, GQA, PagedAttention, FlashAttention, continuous batching, speculative decoding, quantization, distillation, MoE, and tensor/pipeline parallelism each buy back speed.

- LLM
- Inference
- Performance
- Systems

Sep 4, 2026

### Building the Chatbot on This Site: from the series to a live assistant

The capstone: a full, file-by-file walkthrough of the assistant on this site — crawler, from-scratch RAG, a ReAct agent, tools over MCP, SSE streaming, an embeddable Shadow-DOM widget, and the safety-and-cost work tutorials skip. Every step points at the real code on GitHub and the production tool you'd swap in. It's live — go talk to it.

- LLM
- RAG
- Agents
- Production

Sep 3, 2026

View all posts

Toolbox

## Skills

### Languages

- Python
- Scala
- Java
- C / C++
- SQL
- JavaScript
- HTML / CSS

### ML / AI

- Machine Learning
- Deep Learning
- Reinforcement Learning
- Statistical Modelling
- Descriptive & Inferential Statistics

### Frameworks & Libraries

- Keras / TensorFlow
- scikit-learn
- pandas
- matplotlib / seaborn
- NLTK
- pySpark

### Tools & Platforms

- Git
- Docker / Swarm
- gRPC
- MongoDB
- Linux
- Web Development
- Android

Selected work

## Projects

### Portfolio Chatbot — Agentic RAG

A retrieval-augmented, agentic assistant for my site, built from scratch (no LangChain): a ReAct agent that searches a crawl of my blog and can email me, with tools exposed over MCP and streamed to an embeddable chat widget. Runs on a fully-free stack — FastAPI on Render, Gemini for generation and embeddings.

RAG · Agents · MCP · FastAPI · Gemini

Demo
Code

### TradeBuddy — Stock Analysis

A client-side stock-analysis tool: enter a US ticker to get support/resistance zones, entry/stop/target levels, risk:reward, and position sizing. All math runs in the browser and shows its work — the formula and source behind every number. Not financial advice.

JavaScript · Technical Analysis · Charts

Demo
Code

### Checkers AI — Alpha-Beta Minimax

An AI agent using Minimax with alpha-beta pruning to play checkers. Built for the CSCI-561 "Foundations of AI" course, where it competed against other students' agents.

AI · Minimax · Game

Code

### MLfromScratch

Classification, regression, and clustering algorithms — plus metrics, preprocessing, and model-selection helpers — implemented from scratch with NumPy for a deeper understanding of how they work.

Machine Learning · NumPy · Python

Code

### Brain Tumor Segmentation (MRI)

A U-Net (from "U-Net: Convolutional Networks for Biomedical Image Segmentation") built in Keras to segment brain tumors in MRI scans.

Deep Learning · Segmentation · Keras

Code

### NER / POS Tagging App

An LSTM-based seq2seq model that tags every word of a paragraph with its Named Entity or Part of Speech. Served with Flask and Docker.

NLP · LSTM · Flask · Docker

Code

View all projects

Career

## Experience

- May 2023 — Present

### MTS Software Engineer

Salesforce, Inc.

San Francisco, California
- Jan 2023 — Apr 2023

### Software Engineer

TaxBit, Inc.

Seattle, Washington
- Aug 2022 — Dec 2022

### Teaching Assistant — Applied NLP

USC Viterbi School of Engineering

Los Angeles, California
- May 2022 — Aug 2022

### Software Engineering Intern

Salesforce, Inc.

San Francisco, California
- Feb 2021 — May 2022

### Student Research Assistant

USC Institute for Creative Technologies

Los Angeles, California
- Sep 2018 — Dec 2020

### Associate Data Scientist

Cognizant Technology Solutions

Bengaluru, India
- Apr 2016 — Jul 2016

### Intern — MEAN Stack Developer

Heelium Sports Pvt. Ltd.

Pune, India

Academics

## Education

- Jan 2021 — Dec 2022

### M.S. in Computer Science (Honors)

University of Southern California

Los Angeles, California
- Aug 2014 — Jun 2018

### B.E. in Computer Science

Maharashtra Institute of Technology

Pune, India

Say hello

## Get in touch

Have an opportunity, a question, or just want to talk ML? Drop a message and I'll get
back to you.

- adityajn105@gmail.com
- Sunnyvale, CA
