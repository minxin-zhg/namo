# Adam Improves Muon: Adaptive Moment Estimation with Orthogonalized Momentum

This repo contains code for NAMO and NAMO-D optimizers. 

### Install dependencies

```
conda create -n namo python=3.11
conda activate namo
pip install -e .[dev]
```

### Run the model

Data preparation can be found in `src/nanogpt/data/openwebtext`. Training scripts can be found in `scripts/train.sh`.



### External Library
- [nanoGPT](https://github.com/karpathy/nanoGPT) commit: 93a43d9a5c22450bbf06e78da2cb6eeef084b717

### Citation

If you find our paper and code useful, please consider citing:

```bibtex
@article{zhang2026namo,
  title={Adam Improves Muon: Adaptive Moment Estimation with Orthogonalized Momentum},
  author={Zhang, Minxin and Liu, Yuxuan and Schaeffer, Hayden},
  journal={arXiv preprint arXiv:2602.17080},
  year={2026}
}