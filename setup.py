from setuptools import setup, find_packages

setup(
    name='ego4d_forecasting',
	version='1.0.0',
	description='EGO4D Forecasting Benchmark code',
    packages=find_packages(exclude=('scripts',)),
    url='https://github.com/EGO4D/forecasting',
    install_requires=[
		"torch==1.9.0",
		"torchvision==0.10.0",
		"pytorchvideo==0.1.5",
		"pytorch-lightning==1.5.6",
		"editdistance==0.6.0",
		"scikit-learn==0.24.2",
		"psutil==5.9.0",
		"opencv-python==4.5.3.56",
		"einops==0.3.0",
		"decord==0.6.0",
		"lmdb==1.2.1",
		"imutils==0.5.4",
		"submitit==1.3.3",
		"pandas==1.3.5",
		"detectron2@git+https://github.com/facebookresearch/detectron2.git@9258799e4e72786edd67940872e0ed2c4387aac5"
        ]

)
