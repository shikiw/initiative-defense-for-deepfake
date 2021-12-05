## PyTorch Implementation for "Initiative Defense against Facial Manipulation"

This repository provides the official PyTorch implementation of the following paper:
> **Initiative Defense against Facial Manipulation (AAAI 2021)**<br>
> https://ojs.aaai.org/index.php/AAAI/article/view/16254/16061 <br>
>


```bash

# Test with the noise generator defense
python main.py --mode test --dataset CelebA --image_size 128 \
               --c_dim 5 --g_repeat_num 9 --batch_size 1 \
               --selected_attrs Black_Hair Gray_Hair Pale_Skin No_Beard Eyeglasses \
               --celeba_image_dir ./clean_faces --eps 0.03

# Test without the noise generator defense
python main.py --mode test --dataset CelebA --image_size 128 \
               --c_dim 5 --g_repeat_num 9 --batch_size 1 \
               --selected_attrs Black_Hair Gray_Hair Pale_Skin No_Beard Eyeglasses \
               --celeba_image_dir ./clean_faces --eps 0.03 --use_PG False
```


## Citation
If you find this work useful for your research, please cite our [paper](https://ojs.aaai.org/index.php/AAAI/article/view/16254/16061):
```
@inproceedings{huang2021initiative,
author={Qidong Huang and Jie Zhang and Wenbo Zhou and Weiming Zhang and Nenghai Yu},
title={Initiative Defense against Facial Manipulation},
booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
year={2021}
}
```

## Acknowledgements
This work is heavily based on [StarGAN](https://github.com/yunjey/stargan).
