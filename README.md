# 🖋️ English Word Recognition

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](http://forthebadge.com)

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=shields)](https://github.com/supercoolwizard/english_word_recognition/pulls)

A small Python program for recognizing handwritten text using a neural network model trained on the [EMNIST dataset](https://www.kaggle.com/datasets/crawford/emnist).


## 💻 Installing
Install [git](https://git-scm.com/), [python](https://www.python.org/) and [pip](https://pip.pypa.io/en/stable/):
```shell script
sudo apt update -y
sudo apt install -y git python3 python3-pip
```

Clone the repo:
```shell script
git clone --depth 1 https://github.com/supercoolwizard/english_word_recognition
```

Create and activate a [Python Virtual Environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/):
```shell script
cd english_word_recognition
python3 -m venv .venv
source .venv/bin/activate
```

Install requirements and [Jupyter Notebook](https://jupyter.org/) via [pip](https://pip.pypa.io/en/stable/):
```shell script
pip install -r requirements.txt
pip install notebook
```
## 🚀 Usage
0. **Activate the Virtual Environment**  
   Before running any scripts or notebooks, ensure you activate the Python Virtual Environment previously set up for this project. Navigate to the root directory of the project and run:
   ```shell script
   source .venv/bin/activate
   ```
1. **Generate Handwritten Text Image**  
   Run the `main.py` to create a handwritten text image. The output will be saved as `handwriting.jpg`.
2. **Perform Prediction**  
   - For quick results, execute the `clustering_and_prediction.py` script.  
   - To visualize the entire process step-by-step, launch `clustering_and_prediction.ipynb`. To run the notebook:  
     1. Start Jupyter Notebook by running `jupyter notebook` in your terminal.  
     2. Open the `clustering_and_prediction.ipynb` file in your browser.
     3. Click on "**Kernel**" in the top menu, then select "**Restart Kernel and Run All Cells**" 
3. **Train the Neural Network (Optional)**  
   If you wish to train the neural network model, use the `model_training.ipynb` notebook. By default, the model is configured to train for 10 epochs. You can adjust this setting by modifying the value in cell 22 of the notebook. Launch the notebook using the same Jupyter Notebook steps described above.  

## 📜 License
The MIT License (MIT) 2025 - [supercoolwizard](https://github.com/supercoolwizard). Please have a look at the [LICENSE.md](LICENSE) for more details.