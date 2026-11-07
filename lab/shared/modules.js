// Registry of NLP Lab modules. The hub (/lab/) and every module header read from this file.
// tier: 1 Beginner · 2 Intermediate · 3 Advanced · 4 Expert · 5 Graduate · 6 Research
export const TIERS = [
  { n: 1, name: 'Beginner', blurb: 'What text looks like to a computer: tokens, counts, word forms, and the first language models.' },
  { n: 2, name: 'Intermediate', blurb: 'Classic pipelines that still power real products: spelling, search, sentiment, summaries.' },
  { n: 3, name: 'Advanced', blurb: 'Sequence models, latent structure and learned representations.' },
  { n: 4, name: 'Expert', blurb: 'Pretrained transformers in your browser: subwords, semantic search, reading comprehension, zero-shot.' },
  { n: 5, name: 'Graduate', blurb: 'How contextual models represent meaning and how decoding shapes what they say.' },
  { n: 6, name: 'Research', blurb: 'Measurement tools from the literature: bias in embeddings, surprisal and machine-text forensics.' },
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
  { slug: 'topics', tier: 3, kind: 'classic', title: 'Topic Modeling (LDA)', summary: 'Latent Dirichlet Allocation topics learned from a news corpus, and topic inference for your own text.' },
  { slug: 'word2vec', tier: 3, kind: 'classic', title: 'Word2Vec Embeddings', summary: 'Skip-gram vectors trained on 17M words of Wikipedia: nearest neighbors, analogies, 2D and 3D PCA.', href: '/word2vec/' },
  { slug: 'ner', tier: 3, kind: 'transformer', title: 'Named Entity Recognition', summary: 'A BERT token classifier finds people, places and organizations; see BIO tags and per-token confidence.' },
  // Tier 4
  { slug: 'tokenizers', tier: 4, kind: 'transformer', title: 'Subword Tokenizers (BPE, WordPiece)', summary: 'Train byte-pair encoding from scratch merge by merge, then compare GPT-2, BERT and T5 tokenizers.' },
  { slug: 'semantic-search', tier: 4, kind: 'transformer', title: 'Semantic Search', summary: 'Sentence embeddings vs BM25 on the same documents: when meaning beats keyword overlap, and when it doesn’t.' },
  { slug: 'qa', tier: 4, kind: 'transformer', title: 'Extractive Question Answering', summary: 'A SQuAD-trained model highlights the answer span in a passage, with start/end scores.' },
  { slug: 'zero-shot', tier: 4, kind: 'transformer', title: 'Zero-shot Classification (NLI)', summary: 'Classify text into labels the model never trained on by recasting it as natural language inference.' },
  { slug: 'translation', tier: 4, kind: 'transformer', title: 'Neural Machine Translation', summary: 'An encoder–decoder MarianMT model translates English to French, German or Spanish in the browser.' },
  // Tier 5
  { slug: 'contextual', tier: 5, kind: 'transformer', title: 'Contextual Embeddings & Masked LM', summary: 'BERT fills in [MASK]s, and the same word gets different vectors in different sentences, unlike Word2Vec.' },
  { slug: 'generation', tier: 5, kind: 'transformer', title: 'Decoding Strategies for Text Generation', summary: 'Greedy, beam, temperature, top-k and nucleus sampling on GPT-2, with the probability of every step.' },
  // Tier 6
  { slug: 'bias', tier: 6, kind: 'classic', title: 'Measuring Bias in Embeddings (WEAT)', summary: 'Word Embedding Association Tests with effect sizes and permutation p-values, plus bias-direction projections.' },
  { slug: 'surprisal', tier: 6, kind: 'transformer', title: 'Surprisal, Perplexity & Machine-Text Forensics', summary: 'Per-token surprisal from GPT-2, as used in psycholinguistics, and GLTR-style rank coloring for spotting generated text.' },
];

// Modules that have passed tests and are linked from the hub and pagers. Others show as "In development"
// even if their files are on main, so unfinished work never appears as a working page.
export const LIVE = new Set(['word2vec', 'tokenize', 'stemming', 'ngram', 'spell', 'search', 'textrank', 'sentiment', 'pos', 'topics', 'ner', 'qa', 'zero-shot', 'translation', 'generation', 'surprisal']);
export const isLive = (m) => LIVE.has(m.slug);

export const tierOf = (n) => TIERS.find((t) => t.n === n);
export const moduleBySlug = (slug) => MODULES.find((m) => m.slug === slug);
export const hrefOf = (m) => m.href || `/lab/${m.slug}/`;
