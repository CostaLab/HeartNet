#  HearNet

```

Install :
git clone [https://github.com/ORGANIZATION_NAME/polyploid-heart.git](https://github.com/ORGANIZATION_NAME/heartnet.git)
cd heartnet


# Create the environment
conda env create -f envs/environment.heartnet.yml

# Activate the environment
conda activate heartnet
# Install the Package

To make the heartnet code accessible in notebooks, install the package in editable mode:

pip install -e .

Note: This command links the code in src/ to your environment. It allows you to import the model using from heartnet.model import ...