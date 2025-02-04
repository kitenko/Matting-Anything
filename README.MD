# Matting Anything (Fork with SAM2)

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

**Note:** This fork uses **SAM2** as the backbone instead of the original SAM, resulting in improved performance and stability.

[[`ArXiv`](https://arxiv.org/abs/2306.05399)]

![](./assets/teaser_arxiv_v2.png)

## Fork Features

- **SAM2 Integration.** This fork replaces the original SAM with SAM2, providing enhanced results and stability.
- **Training & Demo Scripts.** Clear instructions and scripts are provided to facilitate both training and demo execution.

## Data Preparation

To train the model, you must first prepare the datasets:

- **Training Data:** We use foregrounds from AIM, Distinctions-646, AM2K, Human-2K, and RefMatte to ensure a diverse range of instance classes. The background images come from COCO and BG20K, providing a mix of real-world and synthetic backgrounds.
- **Evaluation Data:** MAM is tested on multiple image matting benchmarks, including:
  - **Semantic Image Matting Benchmarks:** PPM-100, AM2K, PM-10K
  - **Instance Image Matting Benchmarks:** RWP636, HIM2K
  - **Referring Image Matting Benchmark:** RefMatte-RW100

Make sure the datasets are correctly formatted and placed in the appropriate directory before training or evaluation.

## Training

To start training the model, run the following command from the project root:

```bash
./run.sh python main.py --phase train --config config/MAM-ViTB-8gpu.toml
```

This command will automatically build a container with all required dependencies, so there is no need to install anything manually.

## Demo

To launch the demo, execute the following command:

```bash
./run.sh python gradio_app.py --model-weights path_to_weights.pth
```

After starting the demo, you will see output similar to the following in your terminal:

```
Running on local URL:  http://0.0.0.0:7860
```

If the demo is not running locally, a public URL (via a tunnel) will also be provided, for example:

```
Running on public URL: https://5b65a2e5c4a842a350.gradio.live
```

**Important:** Copy the URL provided in the logs and open it in your browser to access the demo.

### Gradio Interface Examples

#### Image Matting Interface
This example shows the **Gradio UI for image matting**:

<div align="center">
  <img src="./assets/gradio_image_demo.png" width="80%" alt="Gradio Image Matting Interface"/>
</div>

#### Video Matting Interface
This example shows the **Gradio UI for video matting**:

<div align="center">
  <img src="./assets/gradio_video_demo.png" width="80%" alt="Gradio Video Matting Interface"/>
</div>

## Getting Started

Detailed instructions on data preparation, training, and inference are available in the [Getting Started](GETTING_STARTED.md) guide.

## Citation

```bibtex
@article{li2023matting,
      title={Matting Anything},
      author={Jiachen Li and Jitesh Jain and Humphrey Shi},
      journal={arXiv: 2306.05399}, 
      year={2023}
}
```

## Acknowledgement

We thank the developers of [SAM2](https://github.com/facebookresearch/segment-anything) and related projects, as well as those behind [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [MGMatting](https://github.com/yucornetto/MGMatting), and [InstMatt](https://github.com/nowsyn/InstMatt/tree/main) for releasing their codebases.
