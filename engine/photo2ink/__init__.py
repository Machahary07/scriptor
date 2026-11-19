"""Photos of handwritten pages to online ink (strokes) with text labels.

    page.py      deskew, crop to the paper, contrast, binarise
    inksight.py  Google InkSight Small-p: word image -> pen strokes (TensorFlow)
    layout.py    word checks, page coordinates, grouping words into lines
    preview.py   photo vs recovered strokes, one PNG per page
    run.py       command line: folder of photos -> writer_samples.json

inksight.py and run.py need TensorFlow and docTR and run in their own Python
environment (setup_env.sh) so they never share a process with PyTorch. The
other modules only need numpy, Pillow and matplotlib.
"""
