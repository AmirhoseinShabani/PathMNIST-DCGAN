# PathMNIST DCGAN: An Educational Baseline for Histopathology Image Generation

> **Project status:** Educational proof of concept — unconditional DCGAN, single tissue class  
> **Framework:** TensorFlow / Keras  
> **Dataset:** PathMNIST (MedMNIST v2), 28 × 28 RGB patches  
> **Target class:** 8 — colorectal adenocarcinoma epithelium  
> **Training:** 30 epochs, batch size 32

## Project overview

This repository documents an end-to-end implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) trained on a single category of colorectal histopathology images from PathMNIST.

The project is a practical learning and implementation exercise. It introduces the core GAN workflow—dataset preparation, generator and discriminator design, adversarial losses, gradient computation, optimization, checkpointing, and visual inspection—using a small, standardized biomedical image dataset.

The notebook is intended to serve as a foundation for a subsequent conditional GAN exercise and a separate medical-imaging GAN project. It is **not** intended to establish clinical validity or demonstrate that synthetic images improve a diagnostic or classification system.

**Notebook:** [`01_PathMNIST_DCGAN.ipynb`](01_PathMNIST_DCGAN.ipynb) — the documented, publish-ready version. It adds section introductions, rationale, results interpretation, limitations, references, an explicit execution order, and the code fixes from the debugging pass.

## Why this project?

Generative Adversarial Networks (GANs) provide a framework for learning to generate samples that resemble a training-data distribution. In a GAN, a generator creates synthetic samples while a discriminator learns to distinguish real samples from generated ones. The two networks are optimized adversarially.

Learning this workflow on a small, labeled histopathology dataset gives a concrete way to practice deep generative modeling in a biomedical image setting. The resulting implementation provides reusable experience with convolutional generative models before moving to larger or more complex medical-image datasets.

Potential research applications of medical-image generation include controlled data synthesis and investigation of training-data augmentation. Those applications require separate validation. This notebook does **not** test whether generated patches improve downstream model performance, preserve clinically meaningful morphology, or generalize to external data.

## Project objectives

1. Understand the generator–discriminator adversarial learning setup.
2. Implement a DCGAN-style generator that maps a latent noise vector to a 28 × 28 RGB image.
3. Implement a discriminator that maps an image to a real/fake logit.
4. Implement binary cross-entropy losses and separate Adam optimizers.
5. Train the networks with a unified `tf.GradientTape` training step.
6. Monitor generated images using fixed latent inputs across training epochs.
7. Save model weights and training state so the experiment can be restored without retraining.
8. Establish a baseline for a later conditional DCGAN and a separate medical-GAN project.

## Dataset

### PathMNIST

PathMNIST is part of MedMNIST v2, a collection of standardized biomedical image datasets designed for lightweight machine-learning research and education. PathMNIST contains 28 × 28 RGB image patches from colorectal histopathology and has nine tissue categories.

The PathMNIST data are derived from the NCT-CRC-HE-100K collection of H&E-stained colorectal tissue patches. The original NCT-CRC-HE-100K patches are 224 × 224 pixels; the standard PathMNIST release uses downsampled 28 × 28 images. The reduced resolution makes this dataset convenient for experiments but limits the fine cellular and tissue morphology that can be represented.

### Dataset splits and scope used here

The official PathMNIST split contains 89,996 training images, 10,004 validation images, and 7,180 test images. This notebook trains the GAN using only the **training split**. Validation and test images are not used for generator or discriminator optimization.

For the first unconditional experiment, the notebook selects class 8 only:

| Label | Tissue category | Training examples in the full PathMNIST training split |
|---:|---|---:|
| 0 | Adipose | 9,366 |
| 1 | Background | 9,509 |
| 2 | Debris | 10,360 |
| 3 | Lymphocytes | 10,401 |
| 4 | Mucus | 8,006 |
| 5 | Smooth muscle | 12,182 |
| 6 | Normal colon mucosa | 7,886 |
| 7 | Cancer-associated stroma | 9,401 |
| 8 | Colorectal adenocarcinoma epithelium | **12,885** |

The selected class therefore contains 12,885 training patches. Restricting the first experiment to one category makes the learning problem narrower than unconditional generation across all nine tissue types.

### Preprocessing

- Input image shape: `28 × 28 × 3` (RGB).
- Source pixel type: `uint8` with values in `[0, 255]`.
- Converted to `float32`.
- Pixel values scaled to `[-1, 1]` using `(image - 127.5) / 127.5`.
- The generator's final activation is `tanh`, so its output range is aligned with the normalized training images.
- Batch size: 32, with `drop_remainder=True`.

For the selected class, this produces 402 complete batches per epoch (`12,885 // 32`).

## Model architecture

The implementation is DCGAN-inspired and follows the general convolutional generator/discriminator pattern demonstrated in the TensorFlow DCGAN tutorial, adapted for RGB PathMNIST patches and a single tissue class.

### Generator

The generator receives a 100-dimensional latent vector sampled from a standard normal distribution and produces a `28 × 28 × 3` image.

| Layer | Output shape | Main settings |
|---|---|---|
| Input | `(None, 100)` | Latent vector |
| Dense | `(None, 12544)` | 7 × 7 × 256 projection, no bias |
| BatchNormalization + LeakyReLU | `(None, 12544)` | Feature normalization and nonlinearity |
| Reshape | `(None, 7, 7, 256)` | Spatial feature map |
| Conv2DTranspose | `(None, 7, 7, 128)` | 5 × 5 kernel, stride 1, same padding, no bias |
| BatchNormalization + LeakyReLU | `(None, 7, 7, 128)` |  |
| Conv2DTranspose | `(None, 14, 14, 64)` | 5 × 5 kernel, stride 2, same padding, no bias |
| BatchNormalization + LeakyReLU | `(None, 14, 14, 64)` |  |
| Conv2DTranspose + Tanh | `(None, 28, 28, 3)` | 5 × 5 kernel, stride 2, same padding |

Reported model summary: 2,334,144 total parameters, including 2,308,672 trainable and 25,472 non-trainable parameters.

### Discriminator

The discriminator receives a real or generated RGB image and outputs one raw logit. It does not include a sigmoid activation because the binary cross-entropy loss is configured with `from_logits=True`.

| Layer | Output shape | Main settings |
|---|---|---|
| Input | `(None, 28, 28, 3)` | RGB patch |
| Conv2D + LeakyReLU + Dropout | `(None, 14, 14, 64)` | 5 × 5 kernel, stride 2, same padding; Dropout 0.3 |
| Conv2D + LeakyReLU + Dropout | `(None, 7, 7, 128)` | 5 × 5 kernel, stride 2, same padding; Dropout 0.3 |
| Flatten | `(None, 6272)` | Flatten feature maps |
| Dense | `(None, 1)` | Raw real/fake logit |

Reported model summary: 216,065 trainable parameters.

## Training setup

| Setting | Value |
|---|---:|
| Latent dimension | 100 |
| Batch size | 32 |
| Epochs | 30 |
| Batches per epoch | 402 |
| Optimizer | Adam (separate instance for each network) |
| Learning rate | 0.0001 |
| Adam `beta_1` | 0.5 |
| Loss | Binary cross-entropy with `from_logits=True` |
| Generator target for generated images | 1 (real) |
| Discriminator target for real images | 1 |
| Discriminator target for generated images | 0 |

### Training step

For each real minibatch:

1. Sample a batch of latent noise vectors.
2. Generate one batch of synthetic images.
3. Pass real and generated images through the discriminator in training mode.
4. Compute generator loss using a target of 1 for generated images.
5. Compute discriminator loss as real-image BCE plus generated-image BCE.
6. Use two `tf.GradientTape` instances to calculate gradients for the corresponding networks.
7. Apply generator gradients with the generator optimizer and discriminator gradients with the discriminator optimizer.

The same generated-image batch is used for both adversarial objectives in a training step. The optimizer updates are separate; the generator loss backpropagates through the discriminator, but its gradients are applied only to generator variables.

## Training results

The notebook trains in two stages — epochs 1–20 in the first loop and epochs 21–30 in a second loop that continues from the in-memory weights without reloading a checkpoint. The epoch-level mean losses recorded in the published run were:

| Epoch | Generator loss | Discriminator loss |
|---:|---:|---:|
| 1 | 0.7502 | 1.3339 |
| 2 | 0.8143 | 1.2176 |
| 3 | 0.7842 | 1.2792 |
| 4 | 0.8523 | 1.2251 |
| 5 | 0.8849 | 1.1786 |
| 6 | 0.7681 | 1.3295 |
| 7 | 0.7531 | 1.3433 |
| 8 | 0.7624 | 1.3327 |
| 9 | 0.7681 | 1.3283 |
| 10 | 0.7529 | 1.3408 |
| 11 | 0.7557 | 1.3444 |
| 12 | 0.7371 | 1.3595 |
| 13 | 0.7173 | 1.3852 |
| 14 | 0.7039 | 1.3951 |
| 15 | 0.7018 | 1.3946 |
| 16 | 0.7011 | 1.3936 |
| 17 | 0.7037 | 1.3895 |
| 18 | 0.7008 | 1.3910 |
| 19 | 0.7035 | 1.3881 |
| 20 | 0.7010 | 1.3890 |
| 21 | 0.7043 | 1.3889 |
| 22 | 0.7006 | 1.3864 |
| 23 | 0.7044 | 1.3826 |
| 24 | 0.7060 | 1.3839 |
| 25 | 0.7023 | 1.3855 |
| 26 | 0.7052 | 1.3832 |
| 27 | 0.7030 | 1.3874 |
| 28 | 0.7024 | 1.3835 |
| 29 | 0.7037 | 1.3826 |
| 30 | 0.7038 | 1.3859 |

These values are transcribed from the executed notebook cells, not from a separate run. Losses approached approximately 0.693 for the generator and 1.386 for the discriminator. Those are useful reference values when a discriminator outputs probabilities near 0.5, but similar losses alone do not prove that the generated distribution matches the real distribution. Visual inspection and independent evaluation remain necessary.

The same values are stored in `training_history.npz` (`epochs`, `generator_loss`, `discriminator_loss`), and the notebook's second training loop writes this file on a fresh run.

During visual inspection, the project author reported that samples after 20 epochs looked recognizably like histopathology patches and were substantially better than earlier experimental runs, although some samples remained imperfect. This is a qualitative observation, not a blinded expert assessment or a quantitative validation result.

## Results and interpretation

The main outcome is a functioning, trained, single-class DCGAN that produces 28 × 28 RGB samples with a visually recognizable resemblance to the selected PathMNIST tissue category, according to the author's qualitative inspection.

The experiment demonstrates the implementation workflow and provides a baseline checkpoint. It does not establish:

- Clinical or diagnostic validity of generated patches.
- Improvement in a downstream classifier from synthetic-data augmentation.
- Generalization to external institutions or whole-slide images.
- Robust coverage of all nine PathMNIST tissue categories.
- Absence of training-image memorization.
- High-resolution cellular morphology generation.

The 28 × 28 resolution is a major limitation: it is useful for learning and rapid experimentation but cannot retain the same level of fine detail as the original 224 × 224 source patches.

A pixel-space nearest-neighbour check on eight generated samples did not show unusually close copies of training images. Generated-to-nearest-real distances were 8.1–15.6 (mean 11.9), while randomly selected real patches were 7.3–13.2 from their nearest *other* real patch. The nearest neighbours share coarse staining and texture but are visibly different images. This is weak evidence against exact duplication and not a formal memorization or privacy assessment.

The sample grids are in `outputs/` (one per epoch from 1 to 20, then 25 and 30), the loss curve is in `outputs/loss_curve.png`, and the nearest-neighbour panel is in `outputs/nearest_neighbours.png`.

## What was learned

- How a generator maps latent vectors into images through dense projection and transposed convolutions.
- How a discriminator maps images to a real/fake logit.
- Why real and generated images use different target labels in discriminator training.
- Why the generator uses a target of 1 for generated images.
- How binary cross-entropy with logits is used without a sigmoid output layer.
- How two gradient tapes can calculate separate generator and discriminator gradients in one training step.
- Why gradients can propagate through the discriminator while only generator variables are updated by the generator optimizer.
- Why `training=True` and `training=False` affect layers such as BatchNormalization and Dropout.
- Why GAN losses should not be interpreted as direct image-quality metrics.
- Why fixed latent inputs and saved checkpoints are useful for monitoring and reproducibility.

## Reproducibility and execution

The notebook uses a fixed random seed for Python, NumPy and TensorFlow where configured. Exact reproducibility may still depend on TensorFlow/Keras versions, hardware, deterministic-operation support, and data-pipeline behavior.

Recommended notebook execution order:

1. Imports, seed and configuration.
2. Dataset loading and class-8 filtering.
3. Preprocessing and `tf.data` pipeline.
4. Generator and discriminator definitions.
5. Loss functions and optimizers.
6. Unified `train_step` definition.
7. Fixed-noise image-generation function.
8. First training loop (epochs 1–20), then the continuation loop (epochs 21–30).
9. Save the full checkpoint (models and optimizer state).
10. Restore the checkpoint or export generator weights, then generate samples without retraining.

The two training loops are deliberate: the first stops at 20 epochs for inspection, and the second continues from the in-memory weights. Running the continuation loop a second time trains 10 additional epochs; it does not resume from epoch 21.

### Restoring the published checkpoint

Training saves every five epochs to `pathmnist_dcgan_checkpoints/` and then exports the final state to the baseline directory:

- `checkpoints/pathmnist_baseline/` — the final 30-epoch baseline (`ckpt-30`).
- `pathmnist_dcgan_checkpoints/` — training snapshots. The files committed here are from the latest run and contain `ckpt-10`, `ckpt-15`, `ckpt-20`, `ckpt-25`, and `ckpt-30`; `max_to_keep=5` keeps the five most recent.

Both managers share the same `tf.train.Checkpoint` object, so there is no duplicated training state. Each checkpoint is about 29 MB, so the two directories total roughly 175 MB.

Because the checkpoint contains optimizer slots, the simplest reliable restore builds the same object graph that was saved:

```python
generator = build_generator()
discriminator = build_discriminator()
generator_optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)

checkpoint = tf.train.Checkpoint(
    generator=generator,
    discriminator=discriminator,
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
)
checkpoint.restore("checkpoints/pathmnist_baseline/ckpt-30").expect_partial()

samples = generator(fixed_noise, training=False)
```

For image generation only, exporting generator weights avoids the optimizer dependency:

```python
generator.save_weights("pathmnist_generator_epoch30.weights.h5")

# later, after rebuilding the generator:
_ = generator(tf.zeros((1, 100)), training=False)  # build the model
generator.load_weights("pathmnist_generator_epoch30.weights.h5")
```

Weight files are smaller but contain no training state, so they cannot be used to resume training. Adjust the paths if the notebook is stored in a subdirectory or if the checkpoints are downloaded separately.

## Limitations and future work

1. **Low resolution:** 28 × 28 images are small and lose fine tissue morphology.
2. **Single-class scope:** the unconditional model was trained only on colorectal adenocarcinoma epithelium (class 8); it is not a nine-class generator.
3. **Qualitative evaluation:** visual inspection was performed, but no validated pathology feature-space metric was reported.
4. **Memorization assessment:** the nearest-neighbour check retrieves true nearest neighbours. For eight generated samples, the nearest-real distances were 8.1–15.6 (mean 11.9), compared with 7.3–13.2 for randomly selected real patches and their nearest *other* real patch. This is weak evidence against exact memorization, not a formal privacy or memorization guarantee.
5. **Downstream utility:** synthetic-image augmentation was not tested in a controlled classifier experiment.

Planned educational extension: a conditional DCGAN using PathMNIST labels to request specific tissue categories. The broader project will then apply the learned GAN workflow to a separate medical-imaging dataset.

## References

1. Goodfellow IJ, Pouget-Abadie J, Mirza M, et al. **Generative Adversarial Nets.** *Advances in Neural Information Processing Systems.* 2014;27. https://papers.nips.cc/paper/2014/hash/f033ed80deb0234979a61f95710dbe25-Abstract.html
2. Radford A, Metz L, Chintala S. **Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.** ICLR 2016. arXiv:1511.06434. https://arxiv.org/abs/1511.06434
3. TensorFlow. **Deep Convolutional Generative Adversarial Network (DCGAN) tutorial.** https://www.tensorflow.org/tutorials/generative/dcgan
4. Yang J, Shi R, Wei D, et al. **MedMNIST v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image classification.** *Scientific Data.* 2023;10:41. https://doi.org/10.1038/s41597-022-01721-8
5. Kather JN, Halama N, Marx A. **100,000 histological images of human colorectal cancer and healthy tissue (NCT-CRC-HE-100K).** Zenodo. https://doi.org/10.5281/zenodo.1214456
6. MedMNIST project. **Dataset documentation, source links, license and citation instructions.** https://medmnist.com/ and https://github.com/MedMNIST/MedMNIST

## Acknowledgements and data attribution

This notebook uses PathMNIST from the MedMNIST project. Please cite the MedMNIST v2 paper and the source dataset documentation when reusing the data. MedMNIST states that its datasets are licensed under CC BY 4.0, with DermaMNIST as an exception; users should check the current license and attribution details for the exact data release they use. This notebook does not redistribute the dataset.

## Disclaimer

This is an educational machine-learning project. The generated images are synthetic research/learning outputs and must not be used for diagnosis, clinical decision-making, patient care, or as a substitute for expert pathology review.

## Suggested repository layout

```text
PathMNIST-DCGAN/
├── README.md
├── LICENSE
├── 01_PathMNIST_DCGAN.ipynb        # documented, publish-ready notebook
├── Medical DCGAN.ipynb             # original working notebook (optional)
├── requirements.txt
├── .gitignore
├── training_history.npz            # 30-epoch losses from the recorded run
├── outputs/
│   ├── epoch_01_samples.png … epoch_20_samples.png   # one grid per epoch
│   ├── epoch_25_samples.png
│   ├── epoch_30_samples.png
│   ├── loss_curve.png
│   └── nearest_neighbours.png
├── checkpoints/
│   └── pathmnist_baseline/         # final 30-epoch checkpoint (~29 MB)
│       ├── checkpoint
│       ├── ckpt-30.index
│       └── ckpt-30.data-00000-of-00001
└── pathmnist_dcgan_checkpoints/    # five most recent snapshots (~145 MB)
    ├── checkpoint
    ├── ckpt-10.*
    ├── ckpt-15.*
    ├── ckpt-20.*
    ├── ckpt-25.*
    └── ckpt-30.*
```

This repository is currently flat (not split into per-notebook folders). That is fine for a first publication; if the project later grows to several notebooks, move each notebook and its outputs into the numbered folders shown in earlier drafts. Update relative checkpoint paths if you move the notebook.

The notebook can also be published individually on Kaggle. The GitHub repository can serve as the source-controlled home for the notebooks, documentation, experiment outputs selected for publication, and future stages.

## Publication checklist

Before publishing the notebook:

- [x] Use a clean runtime and execute the notebook from top to bottom.
- [ ] Confirm that all data are loaded from a documented Kaggle input or the official MedMNIST package/source, not from a personal local path.
- [ ] Record the actual TensorFlow, Keras, Python and accelerator versions used for the final run.
- [x] Keep the training loop clearly separated from inference so users can generate samples without retraining.
- [x] Include sample grids from multiple epochs, with captions identifying the epoch and tissue class.
- [x] Include the epoch-level loss plot and the 30-epoch loss table.
- [x] Save the generator weights/checkpoint and training history as outputs if you intend readers to reuse them.
- [x] Cite MedMNIST and the source histopathology dataset.
- [x] State clearly that the images are 28 × 28 and that this is an educational, non-clinical experiment.
- [x] Remove stale cell outputs, duplicate experiments, debugging traces, and unused imports.
- [x] Verify that a reader can run the notebook without relying on hidden notebook state.
- [x] Check that no single file exceeds GitHub's 100 MB limit and that the repository size is reasonable.
- [x] Add a `LICENSE` file if you want to define reuse terms for the code.
- [ ] Add a short repository description and topics on GitHub (for example: `dcgan`, `pathmnist`, `medmnist`, `histopathology`, `tensorflow`).

Two practical notes for this repository:

- The documented notebook keeps the recorded cell outputs, so it is about 2.5 MB. That is acceptable for GitHub.
- The two checkpoint directories total roughly 175 MB. Consider committing only `checkpoints/pathmnist_baseline/` and adding `pathmnist_dcgan_checkpoints/` to `.gitignore`, or keep both if readers need the intermediate snapshots.

Kaggle notebooks are versioned, and **Save & Run All** executes the notebook in a clean session. Use that final run to check that the notebook is self-contained and reproducible in Kaggle's environment. See the [Kaggle Notebooks documentation](https://www.kaggle.com/docs/notebooks).

## Environment record

The recorded run used **Python 3.11.15** and a **Keras 3** model-summary format. The exact TensorFlow, Keras, and accelerator versions were not captured when the saved outputs were produced. Before publishing, run the following in the environment used for the final run and paste the output into this section:

```python
import platform
import tensorflow as tf

print("Python:", platform.python_version())
print("TensorFlow:", tf.__version__)
print("Keras:", tf.keras.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))
```

Do not copy version numbers from another tutorial or another machine; record the output from the environment used for the final notebook run.

## License

The code in this repository is released under the **MIT License**; see [`LICENSE`](LICENSE). Copyright (c) 2026 The PathMNIST DCGAN contributors.

The PathMNIST data are **not** covered by this license and are not redistributed here. MedMNIST releases its datasets under CC BY 4.0 (DermaMNIST is an exception); cite and follow the MedMNIST terms for the data itself.
