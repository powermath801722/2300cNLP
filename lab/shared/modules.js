// Registry of NLP Lab modules. The hub (/lab/) and every module header read from this file.
// tier: 1 Beginner · 2 Intermediate · 3 Advanced · 4 Expert · 5 Graduate · 6 Research · 7 Seminar · 8 Frontier
export const TIERS = [
  { n: 1, name: 'Beginner', blurb: 'Before any model, a computer has to decide what a word is. Here you split real text into tokens, watch Zipf’s law appear, cut words down to stems, and build a language model out of nothing but counts.' },
  { n: 2, name: 'Intermediate', blurb: 'The tools behind the spell checker, the search box and the review score. Each one here works on real data: 60,000 word frequencies, every U.S. inaugural address, and 2,000 labelled movie reviews.' },
  { n: 3, name: 'Advanced', blurb: 'Models that find structure nobody labelled for them: grammatical roles from an HMM, themes in Reuters news from LDA, word meaning as geometry, and names of people and places pulled out of text.' },
  { n: 4, name: 'Expert', blurb: 'Pretrained transformers doing real jobs in your browser: translating, answering questions from a passage, sorting text into labels they never trained on, and searching by meaning instead of keywords.' },
  { n: 5, name: 'Graduate', blurb: 'What happens inside the models you used one tier up: the same word getting different vectors in different sentences, and how the choice of decoding strategy changes what a language model writes.' },
  { n: 6, name: 'Research', blurb: 'Instruments researchers use to check models: whether word vectors carry human biases, how surprised GPT-2 is by each word, whether a classifier’s confidence can be trusted, and how easily it is fooled.' },
  { n: 7, name: 'Seminar', blurb: 'Topics from a graduate seminar, built small enough to run here: a transformer you train yourself, retrieval that keeps a generator honest, and learning from human preferences.' },
  { n: 8, name: 'Frontier', blurb: 'Open questions about large language models, reproduced at toy scale or computed from published results: marking machine text, generating faster without changing the output, and how loss falls with compute.' },
];

// kind: 'classic' runs pure JS on shipped data; 'transformer' downloads a pretrained model from Hugging Face on first use.
export const MODULES = [
  // Tier 1
  { slug: 'tokenize', tier: 1, kind: 'classic', title: 'Tokenization & Zipf’s Law', summary: 'Split text into sentences and tokens three ways, count types vs tokens, and watch Zipf’s law appear on a log–log plot.' },
  { slug: 'stemming', tier: 1, kind: 'classic', title: 'Stemming & Lemmatization', summary: 'Run the Porter stemmer step by step next to a rule-based lemmatizer and see where each one breaks.' },
  { slug: 'ngram', tier: 1, kind: 'classic', title: 'N-gram Language Models', summary: 'Train unigram, bigram and trigram models, generate text, inspect next-word probabilities and compare smoothing by perplexity.' },
  // Tier 2
  { slug: 'spell', tier: 2, kind: 'classic', title: 'Edit Distance & Spelling Correction', summary: 'Fill in the Levenshtein table cell by cell, then correct misspellings with a noisy-channel model.' },
  { slug: 'search', tier: 2, kind: 'classic', title: 'TF-IDF & BM25 Search', summary: 'A tiny search engine: inverted index, TF-IDF vs BM25 ranking, term weights and keyword extraction.' },
  { slug: 'sentiment', tier: 2, kind: 'classic', title: 'Sentiment Analysis', summary: 'Lexicon rules vs Naive Bayes vs logistic regression vs a fine-tuned transformer, with per-word explanations.' },
  { slug: 'textrank', tier: 2, kind: 'classic', title: 'Extractive Summarization (TextRank)', summary: 'Build a sentence-similarity graph, run PageRank on it and pick the summary sentences.' },
  // Tier 3
  { slug: 'pos', tier: 3, kind: 'classic', title: 'HMM Part-of-Speech Tagging', summary: 'A hidden Markov model trained on the Penn Treebank sample, decoded with Viterbi; inspect the trellis.' },
  { slug: 'parsing', tier: 3, kind: 'classic', wip: true, title: 'Syntactic Parsing (Dependency & CKY)', summary: 'A transition-based dependency parser trained on Universal Dependencies, with its arc-standard stack and buffer stepped out, plus a PCFG parsed by CKY.' },
  { slug: 'topics', tier: 3, kind: 'classic', title: 'Topic Modeling (LDA)', summary: 'Latent Dirichlet Allocation topics learned from a news corpus, and topic inference for your own text.' },
  { slug: 'word2vec', tier: 3, kind: 'classic', title: 'Word2Vec Embeddings', summary: 'Skip-gram vectors trained on 17M words of Wikipedia: nearest neighbors, analogies, 2D and 3D PCA.', href: '/word2vec/' },
  { slug: 'ner', tier: 3, kind: 'transformer', title: 'Named Entity Recognition', summary: 'A BERT token classifier finds people, places and organizations; see BIO tags and per-token confidence.' },
  // Tier 4
  { slug: 'tokenizers', tier: 4, kind: 'transformer', title: 'Subword Tokenizers (BPE, WordPiece)', summary: 'Train byte-pair encoding from scratch merge by merge, then compare GPT-2, BERT and T5 tokenizers.' },
  { slug: 'semantic-search', tier: 4, kind: 'transformer', title: 'Semantic Search', summary: 'Sentence embeddings vs BM25 on the same documents: when meaning beats keyword overlap, and when it doesn’t.' },
  { slug: 'qa', tier: 4, kind: 'transformer', title: 'Extractive Question Answering', summary: 'A SQuAD-trained model highlights the answer span in a passage, with start/end scores.' },
  { slug: 'zero-shot', tier: 4, kind: 'transformer', title: 'Zero-shot Classification (NLI)', summary: 'Classify text into labels the model never trained on by recasting it as natural language inference.' },
  { slug: 'translation', tier: 4, kind: 'transformer', title: 'Neural Machine Translation', summary: 'An encoder–decoder MarianMT model translates English to French, German or Spanish in the browser.' },
  { slug: 'metrics', tier: 4, kind: 'transformer', wip: true, title: 'Evaluating Generated Text', summary: 'BLEU, ROUGE, chrF and BERTScore computed side by side, with the cases where n-gram overlap and human judgment disagree.' },
  // Tier 5
  { slug: 'contextual', tier: 5, kind: 'transformer', title: 'Contextual Embeddings & Masked LM', summary: 'BERT fills in [MASK]s, and the same word gets different vectors in different sentences, unlike Word2Vec.' },
  { slug: 'generation', tier: 5, kind: 'transformer', title: 'Decoding Strategies for Text Generation', summary: 'Greedy, beam, temperature, top-k and nucleus sampling on GPT-2, with the probability of every step.' },
  { slug: 'probing', tier: 5, kind: 'transformer', wip: true, title: 'Probing Classifiers & Control Tasks', summary: 'Train linear probes on contextual embeddings to read out part of speech, then test whether the probe or the representation did the work.' },
  { slug: 'attribution', tier: 5, kind: 'transformer', wip: true, title: 'Feature Attribution (Occlusion, LIME, SHAP)', summary: 'Which words made the classifier decide? Three model-agnostic explanation methods on DistilBERT, compared and sanity-checked.' },
  // Tier 6
  { slug: 'bias', tier: 6, kind: 'classic', title: 'Measuring Bias in Embeddings (WEAT)', summary: 'Word Embedding Association Tests with effect sizes and permutation p-values, plus bias-direction projections.' },
  { slug: 'surprisal', tier: 6, kind: 'transformer', title: 'Surprisal, Perplexity & Machine-Text Forensics', summary: 'Per-token surprisal from GPT-2, as used in psycholinguistics, and GLTR-style rank coloring for spotting generated text.' },
  { slug: 'calibration', tier: 6, kind: 'transformer', wip: true, title: 'Calibration & Uncertainty', summary: 'Reliability diagrams, expected calibration error, temperature scaling and selective prediction on a real classifier and held-out data.' },
  { slug: 'adversarial', tier: 6, kind: 'transformer', wip: true, title: 'Adversarial Robustness & Behavioral Testing', summary: 'Word-substitution attacks that flip a classifier, and CheckList-style tests for negation, vocabulary and invariance.' },
  // Tier 7
  { slug: 'transformer', tier: 7, kind: 'classic', wip: true, title: 'A Transformer from Scratch & Induction Heads', summary: 'Train a tiny attention-only transformer in your browser and watch induction heads form, the circuit behind in-context copying.' },
  { slug: 'rag', tier: 7, kind: 'transformer', wip: true, title: 'Retrieval-Augmented Generation', summary: 'Retrieve passages, generate an answer from them, and check every claim against its sources with an entailment model.' },
  { slug: 'preference', tier: 7, kind: 'classic', wip: true, title: 'Preference Learning: Reward Models, RLHF & DPO', summary: 'Fit a Bradley–Terry reward model to pairwise preferences, then optimize a policy with KL-regularized RL and with DPO, at toy scale.' },
  // Tier 8
  { slug: 'watermark', tier: 8, kind: 'transformer', wip: true, title: 'Watermarking Language Model Output', summary: 'Generate with a green-list watermark on GPT-2, detect it with a z-test, and see how edits and paraphrase wear it down.' },
  { slug: 'speculative', tier: 8, kind: 'transformer', wip: true, title: 'Speculative Decoding', summary: 'A small draft model proposes tokens and a larger model verifies them in one pass, giving the same output distribution with fewer large-model calls.' },
  { slug: 'scaling', tier: 8, kind: 'classic', wip: true, title: 'Neural Scaling Laws', summary: 'Published loss-versus-compute laws made interactive: compute-optimal model and data sizes, and how the Kaplan and Chinchilla fits differ.' },
];

// Modules that have passed tests and are linked from the hub and pagers. Others show as "In development"
// even if their files are on main, so unfinished work never appears as a working page.
export const LIVE = new Set(MODULES.filter((m) => !m.wip).map((m) => m.slug));
export const isLive = (m) => LIVE.has(m.slug);

export const tierOf = (n) => TIERS.find((t) => t.n === n);
export const moduleBySlug = (slug) => MODULES.find((m) => m.slug === slug);
export const hrefOf = (m) => m.href || `/lab/${m.slug}/`;
