
![ex1](asset/logo.png)

# G-LLaVA: Solving Geometric Problem with Multi-Modal Large Language Model

This repository contains the code and data for the paper titled "G-LLaVA: Solving Geometric Problem with Multi-Modal Large
Language Model".

[Paper](https://arxiv.org/pdf/2312.11370.pdf), [Dataset](https://huggingface.co/datasets/Luckyjhg/Geo170K/tree/main) , [Model](https://huggingface.co/renjiepi/G-LLaVA-7B/tree/main)


![ex1](asset/geollava_mainfigure.png)


## Install Packages
```
cd G-LLaVA
conda create -n gllava python=3.10 -y
conda activate gllava
pip install -e .
```
## Enable Deepspeed
```
pip install deepspeed
```

## Data Preparation

[comment]: <> (Download the COCO dataset from [huggingface] &#40;To be published&#41;.)
Download our [dataset](https://huggingface.co/datasets/Luckyjhg/Geo170K/tree/main).

Place the data under playground/data.
Here is the data structure:
```
playground/data/
├── images/
│   ├── geo3k/
│   ├── geoqa_plus/
│   ├── test/
├── alignment.json
├── qa_tuning.json
├── test_question.jsonl
├── test_answers.jsonl
```
"test_question.jsonl" and "test_answers.jsonl" correspond to the test set of GeoQA. 

## First Stage Alignment
This stage enables the model to better interpret the content of geometric figures.
```
bash scripts/run_alignment.sh
```

## Second Stage Instruction Tuning
This stage equips the model with stronger ability for solving geometry problems.

```
bash scripts/run_qa.sh
```

## Evaluation
Generate responses from the model.
```
bash scripts/eval_multi.sh /
                path-to-model /
                playground/data/test_questions.jsonl /
                path-to-output /
                path-to-image-folder /
                num_gpus /
                temperature
```
Run automatic evaluation to calculate the accuracy.
```
python scripts/geo_acc_calculate.py /
             --ground_truth_file playground/data/test_answers.jsonl /
             --predictions_file path-to-output-file
```

Here are some example scripts:
```
bash scripts/eval_multi.sh /path/to/checkpoint/ playground/data/test_questions.jsonl results_try/Gllava-test playground/data/images/ 8 0

python scripts/geo_acc_calculate.py  --ground_truth_file playground/data/test_answers.jsonl --predictions_file results_try/Gllava-test_merged.jsonl
```

## Acknowledgement
The project is built on top of the amazing [LLaVA](https://github.com/haotian-liu/LLaVA) repository. Thanks for their great work!


If you find our code and dataset helpful to your research, please consider citing us with this BibTeX:
```bibtex
@misc{gao2023gllava,
      title={G-LLaVA: Solving Geometric Problem with Multi-Modal Large Language Model}, 
      author={Jiahui Gao and Renjie Pi and Jipeng Zhang and Jiacheng Ye and Wanjun Zhong and Yufei Wang and Lanqing Hong and Jianhua Han and Hang Xu and Zhenguo Li and Lingpeng Kong},
      year={2023},
      eprint={2312.11370},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
