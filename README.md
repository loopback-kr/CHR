# SIXray：A Large-scale Security Inspection X-ray Benchmark for  Prohibited Item Discovery in Overlapping Images

[[Paper]](https://arxiv.org/pdf/1901.00303.pdf) [[dataset]](https://github.com/MeioJane/SIXray)

![Illustration](Illustration.png)

## Requirements

* Python 3.10 or newer
* PyTorch: 2.0.0 or newer

## Usage
1. Clone the CHR repository:
    ```bash
    git clone https://github.com/loopback-kr/CHR.git
    ```

1. Create new container:
    ```bash
    docker compose up -d
    ```

1. Run the training demo in the container:
    ```bash
    python CHR/main.py
    ```

## Checkpoint
If you only want to test images, you can download [here](https://pan.baidu.com/s/19wuNL8KaZ5vm-yiJfu2CZA?pwd=tunq).
## Citation 
If you use the code in your research, please cite:

```bibtex
@INPROCEEDINGS{Miao2019SIXray,
    author = {Miao, Caijing and Xie, Lingxi and Wan, Fang and Su, chi and Liu, Hongye and Jiao, jianbin and Ye, Qixiang },
    title = {SIXray: A Large-scale Security Inspection X-ray Benchmark for Prohibited Item Discovery in Overlapping Images},
    booktitle = {CVPR},
    year = {2019}
}
```

## Acknowledgement
This project was forked from [MeioJane/CHR](https://github.com/MeioJane/CHR) and refactored to PyTorch 2.0 or newer.


