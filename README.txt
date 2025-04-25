create an enviroment as follows:
conda env create -f environment.yml


FILES TO RUN:

TPE_optimization.py
- To get the best hyperparameters for the
loss function coefficients


train_inpainting.py
- To train the ensemble model and ablation models
for face inpainting (we recommend to
run it from console as follows
python train_inpainting.py -- model
the possible parameters for model are:
a) Reconstruction
b) ablationv1
c) ablationv2
d) ablationv3


train_selfsupervised.py
- To train the model for face inpainting where the
input data has no binary masks but objects and
synthetic occluders.


train_general_inpainting.py
- To train the model for a general inpainting
task with a smaller dataset

inference.py
- To perform inferences with the trained models
as well as calculating the metrics for validation set

inference_UMAP.py
- To obtain the UMAP plot and metrics based on UMAP 

inference_CKA.py
- To get CKA value between encoder's submodels features.

inference_time.py
- To get time statistics of the model's inference.


NOTE:
We recommend training from the Spyder GUI if not told from the console to keep track of the inpainting process during training.
If training from the console gives issues, it could be due to show_images function in the engine* file used in the train* code. It could be solved simply by commenting such lines in engine*.