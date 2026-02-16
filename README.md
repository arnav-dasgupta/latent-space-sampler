# latent-space-sampler

This project maps the high-dimensional geometry of how LLMs actually "see" text, into a 3D space to visualize the relationship between 2 sample sentences.

Link : [`latent-space-sampler`](https://latent-space-sampler.streamlit.app/)

the app converts the input sentences into a 768-dimension vectors using the [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) model. 

Since humans can't visualize 768 dimensions, the app uses PCA (Principal Component Analysis) to crunch that data down into a 3-dimensional system with $(x, y, z)$ coordinates. 

The app calculates:
- **Cosine Similarity**: How closely the two vectors are pointing in the same direction (0-100%).

- **Angular Distance**: The radian gap between the two sentences in the model's "brain".

To provide a relative context for comparison during plotting, the app uses 3 anchor statements. 

Stack : `python` `tailwind-css` `streamlit` `plotly` `sentence-transformers` `scikit-learn`