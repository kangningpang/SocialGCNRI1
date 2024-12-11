# SocailGCNRI

This is an implemention for our Information Sciences paper based on Pytorch

[A Social Recommendation Model Based on Adaptive Residual Graph Convolution Networks ]
by Rui Chen, Kangning Pang, Qingfang Liu, Lei Zhang, Hao Wu, Cundong Tang, and Pu Li

# Dataset
We provide two datasets: [LastFM](https://grouplens.org/datasets/hetrec-2011/) and [Ciao](https://www.cse.msu.edu/~tangjili/datasetcode/truststudy.htm).

# Example to run the codes
1. Environment: I have tested this code with python3.9 Pytorch=2.0.1 CUDA=11.0
2. Run SocialLGN

    `python main.py --model=SocailGCNRI --dataset=lastfm --decay=1e-4 --lr=0.001 --layer=4 --seed=2020 --topks="[10,20]" --recdim=128 --bpr_batch=2048`
