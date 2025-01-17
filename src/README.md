# Trackformer DETR-Only Mode

Train and test your model using this repository.

## **Inference**
To perform inference:
1. Clone commit `958cf89a3d912182e0a67327b493c34232ca8ec4` from the [official Trackformer repository](https://github.com/timmeinhardt/trackformer) as mentioned in [Issue #47](https://github.com/timmeinhardt/trackformer/issues/47).
2. Follow the installation steps from the original repository.

---

## **Setup**
1. Clone this repository.
2. Replace the `src` folder in the official Trackformer repository with the `src` folder from this repository.
3. Download the MOTS17 dataset (or any dataset of your choice):
   - Place your dataset folder containing images in `/data`.
   - Update `data_root_dir` to point to your dataset folder.
   - Set `obj_detect_checkpoint_file` to the path of your trained model.

---

## **Install Deformable Attention Module**
Run the following command to build and install the deformable attention module:
```bash
python src/trackformer/models/ops/setup.py build --build-base=src/trackformer/models/ops/ install
