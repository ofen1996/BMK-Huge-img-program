# Large-Scale Image Alignment and Stitching

A Python computer-vision prototype for array localisation, geometric registration, and multi-channel merging on high-resolution images. It combines ONNX keypoint detection, regular-grid templates, and libvips processing for memory-conscious large-image workflows.

## Highlights

- ONNX/YOLO-style keypoint detection
- Regular array-template construction and geometric matching
- Large-area array presets
- libvips-based multi-channel image merging
- Configuration-driven scanner, tile, and array parameters

## Project structure

| Path | Purpose |
| --- | --- |
| `match_imgs.py` | Registration, template construction, and image matching |
| `merge_channels_pics_by_libvips.py` | Multi-channel image merging |
| `need/KpDetectByYolo.py` | ONNX inference and non-maximum suppression |
| `need/BmTiffLib.py` | TIFF and large-image helper functions |
| `model/` | Keypoint-detection models |
| `setting/` | Image-acquisition and array presets |

## Environment

Python 3.9+ is recommended:

```bash
pip install numpy opencv-python tifffile pillow pyvips onnxruntime torch scikit-learn
```

Install libvips separately and ensure its `bin` directory is on PATH. Some scripts retain development-machine paths; replace them with local paths or environment-based configuration.

## Usage guidance

1. Select a `setting/*.ini` preset closest to the acquisition setup.
2. Confirm that the ONNX models remain available at their expected relative paths.
3. Validate keypoint detection and alignment on a cropped sample.
4. Run registration or merge workflows on the full dataset, writing results to a dedicated output directory.

## Notes

This is an engineering prototype for specialised high-resolution image workflows rather than a packaged general-purpose CLI. Production deployment should add fixed environments, automated tests, and dataset-specific validation. Large-image workloads require substantial memory, disk space, and compute time.

## License

No open-source license is currently declared. Contact the repository owner before reuse.