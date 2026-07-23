# CPU inference image for the SatNOGS waterfall classifier.
# The pinned requirements.txt targets the CUDA training box; this image
# installs CPU wheels instead so it runs anywhere.
#
#   docker build -t satnogs-classifier .
#   docker run --rm -v /path/to/waterfalls:/data satnogs-classifier --dir /data
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && pip install --no-cache-dir numpy pillow

COPY dataset.py train.py infer.py make_synthetic.py model.pt ./

ENTRYPOINT ["python", "infer.py"]
CMD ["--help"]
