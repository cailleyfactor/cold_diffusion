## Description
The aim of this project was to build and train different types of diffusion models on MNIST in PyTorch using non-Gaussian noise degradation for a "cold" diffusion model.

# Usage
First clone the repository from git.

## Saved models & samples
Cloning the repository includes cloning the saved models for different parts of the coursework,
which are stored in the "results" folder.
The saved models and sampling outputs for the first part of the coursework for models with a degradation of adding Gaussian noise are stored in subfolders called "Default_model" and "Model_extra_hidden" for the default model and model with an extra hidden layer, respectively.
The saved models and sampling outputs for the cold diffusion part of the coursework are stored in subfolders called "Cold_3_0.1" and "Cold_13_7.0" for the model with a kernel size of 3 and std of 0.1, and the model with a kernel size of 13 and std of 7.0, respectively.

Within each of the folders, the models are stored as ".pth" files.
The loss curves are stored in ".csv" files and plotted in "loss_curves.png" files.

For the "Default_model" and "Model_extra_hidden", the samples are labelled in the form "ddpm_sample_XXXX.png" where XXXX is the epoch number.
For the cold diffusion models, there are two types of samples stored in the form "ddpm_output_algo2_XXXX.png" and "ddpm_recon_XXXX.png",
where XXXX is epoch number. "ddpm_output_algo2_XXXX.png" represents the output of the sampling procedure and is the equivalent of
"ddpm_sample_XXXX.png" in the previous naming format. "ddpm_recon_XXXX.png" is the output of the CNN for each epoch.

## Running the code for the Gaussian noise model
The model is set to the default hyperparameters for n_hidden in the main.py file, which are n_hidden = (16, 32, 32, 16).

To run the code, a dockerfile is provided in the root directory with the environment needed to run the code, provided in the ML_environment.yml file.
To run the code from the terminal navigate to the root directory and use, e.g.,
$docker build -t [image name of choice] -f dockerfile_main .
$docker run -v .:/cf593_doxy -t [image name of choice]
(Make sure to include the periods!)

With the appropriate environment activated based on ML_environment.yml, the code can also be run from the terminal
by navigating into the root directory of the cloned git repository and running the code with the following command
$ python main.py

## Running the code for the cold diffusion model
In the ddpm2.py module, the model is set to running the code with self.kernel_std = 0.1
and self.kernel_size = 3.

To run the code, a dockerfile is provided in the root directory with the environment needed to run the code, provided in the environment.yml file.
To run the code from the terminal navigate to the root directory and use, e.g.,
$docker build -t [image name of choice] -f dockerfile_cold .
$docker run -v .:/cf593_cold -t [image name of choice]
(Make sure to include the periods!)

With the appropriate environment activated based on ML_enviroment.yml, the code can also be run from the terminal
by navigating into the root directory of the cloned git repository and running the code with the following command
$ python main2.py

## Running the code for an additional plotting function
To run the code, a dockerfile is provided in the root directory with the environment needed to run the code, provided in the environment.yml file.
To run the code from the terminal navigate to the root directory and use, e.g.,
$docker build -t [image name of choice] -f dockerfile_plot .
$docker run -v .:/cf593_plot -t [image name of choice]

With the appropriate environment activated based on MI_environment.yml, the code can also be run from the terminal
by navigating into the root directory of the cloned git repository and running the code with the following command
$ python plot.py

# Documentation
Detailed documentation is available by running the Doxyfile using doxygen in the docs file in the root directory.
This can be run by navigating in the docs file and running doxygen with:
$doxygen

# Citations
## References
Gaussian blurring was selected as the arbitrary image transformation based on the Bansal et al. paper and code from `deblurring-diffusion-pytorch` from their repository was modified to apply iterative Gaussian blurring on the MNIST images and used to inform the development of the forward process. The detailed references for the paper and the repository are available in the report located in the `report` folder.

## License
Released 2024 by Cailley Factor.
The License is included in the LICENSE.txt file.
