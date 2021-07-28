`cluster_summary.py` and `process_sentence_embedding.py` is used for extractive summary. I have already put the processed data in `processed`, so no need to do this again.

The classification system mainly consists of:
- data.py: define the dataset and the dataloader
- model.py: define the model
- main_post_clf.py: concat the posts as one flat input and run classification
    - can use `runme7.sh` to start a sample run
- main_hier_clf.py: use hierarchical model for classification
    - can use `runme_hier_trans5.sh` to start a sample run

The system is written with pytorch lightning, which supports early stopping and tensorboard visualization. You can use `tensorboard --logdir=lightning_logs --reload_multifile=true` to see the result plots in the webpage.