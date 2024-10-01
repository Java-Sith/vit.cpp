# Code instrumentation used to measure model inference performance and plotting results

## :clipboard: Installation
The first thing we need to do is to clone the repository and set it up, to do so, open a new terminal and type the following commands:
```
git clone https://github.com/ECASLab/llama.cpp
```
```
cd llama.cpp
```
```
git checkout feature/add-instrumentation

```
In order to use the GPU, we need to export the following env variable:

```
export HIP_VISIBLE_DEVICES=1
```
After this, we will proceed to make the project according to our specifications. There's a flag that allows for the count and execution time of each operation to be counted, and the project needs to be built with this flag to take measures, otherwise it executes normally:

```
LLAMA_CLOCK=1
LLAMA_CUDA_CLOCK=1
LLAMA_XRT_CLOCK=1
```

- CPU
```
make LLAMA_CLOCK=1 -j
```
- GPU
```
make LLAMA_HIPBLAS=1 LLAMA_CLOCK=1 -j
```
- FPGA
```
make LLAMA_XRT=1 LLAMA_XRT_CLOCK=1 -j
```
Before executing the FPGA implementation though, it is first necessary to synthetize the kernels used for the Xilinx Runtime Library implementation. First navigate to the folder where the XRT scripts are located:
```
cd ecas-scripts/HW
```
Once you're at this directory, execute the following command to build the Xclbin file. Compilation and synthesis can take up to a few hours, so please be patient.
```
make all TARGET=hw PLATFORM=xilinx_u250_gen3x16_xdma_4_1_202210_1
```
## :computer: Run the program

Now we need to acquire a model, for testing purposes and in order to save time, we use any of the pre-quantized Llama models found at [TheBloke](https://huggingface.co/TheBloke). The model used in this project's initial tests was [LLaMA 2 7B base](https://huggingface.co/TheBloke/Llama-2-7B-GGUF). Execute the following commands to acquire the model:

```
cd models
```
```
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_S.gguf
```
```
cd ..
```
Now we execute the model inference with GGML, for that we have to execute this command in the llama.cpp folder:

```
./main -m ./models/llama-2-7b.Q4_K_S.gguf -n 128 | tee output.txt
```
For this command, -n is the number of tokens, -m is the path and tee saves the console output in a text file. The previous command works for CPU and FPGA implementations. If we want to use the GPU instead, we need to specify the previously defined env variable:
```
HIP_VISIBLE_DEVICES=1 ./main -m ./models/llama-2-7b.Q4_K_S.gguf -n 128 | tee output.txt
```
Alternatively, if CUBLAS is being used, then the GPU will be automatically detected, and the user must input the amount of GPU layers the inference will use:
```
./main -m ./models/llama-2-7b.Q4_K_S.gguf -n 128 -ngl 32 | tee output.txt
```
It is recommended that the output text files be moved to this folder for better execution of the plotter scripts, but it is not mandatory.

## :computer: Run the plotter scripts
Open a terminal in this folder and the type the following command
```
python3 FileParsing.py <output text file>
```
This will apply the plotter generation to a text file passed as an argument. Once finished, there will be a histogram of the number of times each operation is executed, a boxplot of the execution times of said operations and a histogram of the execution times of the matrix multiplication, which is the most costly operation.
