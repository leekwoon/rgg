from setuptools import setup


setup(
    name='rgg',
    packages=['rgg'],
    install_requires=[
        'gdown',
        'grad-cam',
        'opencv-python==4.5.5.64',
        'mediapy==1.1.3',
        'einops==0.6.0',
        'typed-argument-parser',
        'gitpython',
        'patchelf',
        'scikit-video==1.1.11',
        'scikit-image==0.17.2',
        'matplotlib==3.6.2',
        'free-mujoco-py',
        'torch==1.9.1',
        'tqdm==4.64.1',
        'gym==0.18.0',
        'd4rl @ git+https://github.com/Farama-Foundation/d4rl@f2a05c0d66722499bf8031b094d9af3aea7c372b#egg=d4rl',
        'diffusers @ git+https://github.com/huggingface/diffusers@374ec161707c5a6a6abdaa6a87117fba0e8d58d4#egg=diffusers'
        'numpy==1.24.1',
    ]
)