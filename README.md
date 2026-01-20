# Vocabulary expanding of multilingual language models

- **Training corpus for target languages**: [MADLAD-400](https://huggingface.co/datasets/allenai/MADLAD-400), [wiki data](https://dumps.wikimedia.org/)
- **testing set**: [Universial Treebanks for POS Tagging](https://universaldependencies.org/), [wikiann for named entity recognition](https://huggingface.co/datasets/unimelb-nlp/wikiann)

Use ```train_random.py``` script to continue to train multilingual BERT (mBERT) model with extended vocabulary for target languages being randomly represented;<br>
Use ```train_sim.py``` script to continue to train multilingual BERT (mBERT) model with extended vocabulary for target languages represented using our method;<br>
Use ```[XTREME](https://dl.acm.org/doi/10.5555/3524938.3525348)``` to evaluate the extended models.

