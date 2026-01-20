# Vocabulary expanding of multilingual language models

- **Training corpus for target languages**: [MADLAD-400](https://huggingface.co/datasets/allenai/MADLAD-400), [wiki data](https://dumps.wikimedia.org/)
- **testing set**: [Universial Treebanks for POS Tagging](https://universaldependencies.org/), [wikiann for named entity recognition](https://huggingface.co/datasets/unimelb-nlp/wikiann)

Use the ```train_random.py``` script to further pre-train multilingual BERT (mBERT) with an extended vocabulary which are randomly represented;<br>
Use the ```train_sim.py``` script to further pre-train multilingual BERT (mBERT) with an extended vocabulary which are initialized represented using our method;<br>
valuate the extended models on the [XTREME](https://dl.acm.org/doi/10.5555/3524938.3525348) benchmark. 

