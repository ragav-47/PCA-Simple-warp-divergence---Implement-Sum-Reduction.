# PCA-Simple-warp-divergence---Implement-Sum-Reduction.
Refer to the kernel reduceUnrolling8 and implement the kernel reduceUnrolling16, in which each thread handles 16 data blocks. Compare kernel performance with reduceUnrolling8 and use the proper metrics and events with nvprof to explain any difference in performance.

## Aim:
To implement the kernel reduceUnrolling16 and comapare the performance of kernal reduceUnrolling16 with kernal reduceUnrolling8 using proper metrics and events with nvprof.

## Procedure:
### Step 1 :
Include the required files and library.

### Step 2 :
Introduce a function named 'recursiveReduce' to implement Interleaved Pair Approach and function 'reduceInterleaved' to implement Interleaved Pair with less divergence.

### Step 3 :
Introduce a function named 'reduceNeighbored' to implement Neighbored Pair with divergence and function 'reduceNeighboredLess' to implement Neighbored Pair with less divergence.

### Step 4 :
Introduce optimizations such as unrolling to reduce divergence.

### Step 5 :
Declare three global function named 'reduceUnrolling2' , 'reduceUnrolling4' , 'reduceUnrolling8' , 'reduceUnrolling16' and then set the thread ID , convert global data pointer to the local pointer of the block , perform in-place reduction in global memory ,finally write the result of the block to global memory in all the three function respectively.

### Step 6 :
Declare functions to unroll the warp. Declare a global function named 'reduceUnrollWarps8' and then set the thread ID , convert global data pointer to the local pointer of the block , perform in-place reduction in global memory , unroll the warp ,finally write the result of the block to global memory infunction .

### Step 7 :
Declare Main method/function . In the Main method , set up the device and initialise the size and block size. Allocate the host memory and device memory and then call the kernals decalred in the function.

### Step 8 :
Atlast , free the host and device memory then reset the device and check for results.

### Program:
## kernel reduceUnrolling8

```python3
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <windows.h>
#include <device_launch_parameters.h>
#include <windows.h>
__global__ void reduceUnrolling8(int *g_idata, int *g_odata, unsigned int n)
{
 // Set thread ID
 unsigned int tid = threadIdx.x;
 unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
 // Convert global data pointer to the local pointer of this block
 int *idata = g_idata + blockIdx.x * blockDim.x * 8;
 // Unrolling 8
 if (idx + 7 * blockDim.x < n)
 {
 int a1 = g_idata[idx];
 int a2 = g_idata[idx + blockDim.x];
 int a3 = g_idata[idx + 2 * blockDim.x];
 int a4 = g_idata[idx + 3 * blockDim.x];
 int b1 = g_idata[idx + 4 * blockDim.x];
 int b2 = g_idata[idx + 5 * blockDim.x];
 int b3 = g_idata[idx + 6 * blockDim.x];
 int b4 = g_idata[idx + 7 * blockDim.x];
 g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
 }
 __syncthreads();
 // In-place reduction in global memory
 for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
 {
 if (tid < stride)
 {
 idata[tid] += idata[tid + stride];
 }
 // Synchronize within threadblock
 __syncthreads();
 }
 // Write result for this block to global memory
 if (tid == 0)
 {
 g_odata[blockIdx.x] = idata[0];
 }
}
// Function to calculate elapsed time in milliseconds
double getElapsedTime(struct timeval start, struct timeval end)
{
 long seconds = end.tv_sec - start.tv_sec;
 long microseconds = end.tv_usec - start.tv_usec;
 double elapsed = seconds + microseconds / 1e6;
 return elapsed * 1000; // Convert to milliseconds
}
int main()
{
 // Input size and host memory allocation
 unsigned int n = 1 << 20; // 1 million elements
 size_t size = n * sizeof(int);
 int *h_idata = (int *)malloc(size);
 int *h_odata = (int *)malloc(size);
 // Initialize input data on the host
 for (unsigned int i = 0; i < n; i++)
 {
 h_idata[i] = 1;
 }
 // Device memory allocation
 int *d_idata, *d_odata;
 cudaMalloc((void **)&d_idata, size);
 cudaMalloc((void **)&d_odata, size);
 // Copy input data from host to device
 cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice);
 // Define grid and block dimensions
 dim3 blockSize(256); // 256 threads per block
 dim3 gridSize((n + blockSize.x * 8 - 1) / (blockSize.x * 8));
 // Start CPU timer
 
	// Start CPU timer
	cudaEvent_t start, stop;
	float elapsedTimeCPU, elapsedTimeGPU;

	// Measure time for CPU Matrix Multiplication
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	int sum_cpu = 0;
	for (unsigned int i = 0; i < n; i++)
	{
		sum_cpu += h_idata[i];
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTimeCPU, start, stop);
	
	// Compute the sum on the CPU
	
	// Start GPU timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	reduceUnrolling8 << <gridSize, blockSize >> > (d_idata, d_odata, n);
	
	
	// Copy the result from device to host
	cudaMemcpy(h_odata, d_odata, size, cudaMemcpyDeviceToHost);
	// Compute the final sum on the GPU
	int sum_gpu = 0;
	for (unsigned int i = 0; i < gridSize.x; i++)
	{
		sum_gpu += h_odata[i];
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTimeGPU, start, stop);
	// Stop GPU timer
	// Print the results and elapsed times
	printf("CPU Sum: %d\n", sum_cpu);
	printf("GPU Sum: %d\n", sum_gpu);
	printf("CPU Time: %f ms\n", elapsedTimeCPU);
	printf("GPU Time: %f ms\n", elapsedTimeGPU);
 // Free memory
 free(h_idata);
 free(h_odata);
 cudaFree(d_idata);
 cudaFree(d_odata);
 return 0;
}
```
## unroll16
```
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <windows.h>
#include <device_launch_parameters.h>
#include <windows.h>
// Kernel function declaration
__global__ void reduceUnrolling16(int *g_idata, int *g_odata, unsigned int n);
// Function to calculate elapsed time in milliseconds
double getElapsedTime(struct timeval start, struct timeval end)
{
 long seconds = end.tv_sec - start.tv_sec;
 long microseconds = end.tv_usec - start.tv_usec;
 double elapsed = seconds + microseconds / 1e6;
 return elapsed * 1000; // Convert to milliseconds
}
int main()
{
 // Input size and host memory allocation
 unsigned int n = 1 << 20; // 1 million elements
 size_t size = n * sizeof(int);
 int *h_idata = (int *)malloc(size);
 int *h_odata = (int *)malloc(size);
 // Initialize input data on the host
 for (unsigned int i = 0; i < n; i++)
 {
 h_idata[i] = 1;
 }
 // Device memory allocation
 int *d_idata, *d_odata;
 cudaMalloc((void **)&d_idata, size);
 cudaMalloc((void **)&d_odata, size);
 // Copy input data from host to device
 cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice);
 // Define grid and block dimensions
 dim3 blockSize(256); // 256 threads per block
 dim3 gridSize((n + blockSize.x * 16 - 1) / (blockSize.x * 16));
 // Start CPU timer
 
	// Start CPU timer
	cudaEvent_t start, stop;
	float elapsedTimeCPU, elapsedTimeGPU;

	// Measure time for CPU Matrix Multiplication
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	int sum_cpu = 0;
	for (unsigned int i = 0; i < n; i++)
	{
		sum_cpu += h_idata[i];
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTimeCPU, start, stop);
	
	// Compute the sum on the CPU
	
	// Start GPU timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	reduceUnrolling8 << <gridSize, blockSize >> > (d_idata, d_odata, n);
	
	
	// Copy the result from device to host
	cudaMemcpy(h_odata, d_odata, size, cudaMemcpyDeviceToHost);
	// Compute the final sum on the GPU
	int sum_gpu = 0;
	for (unsigned int i = 0; i < gridSize.x; i++)
	{
		sum_gpu += h_odata[i];
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTimeGPU, start, stop);
	// Stop GPU timer
	// Print the results and elapsed times
	printf("CPU Sum: %d\n", sum_cpu);
	printf("GPU Sum: %d\n", sum_gpu);
	printf("CPU Time: %f ms\n", elapsedTimeCPU);
	printf("GPU Time: %f ms\n", elapsedTimeGPU);
 // Free memory
 free(h_idata);
 free(h_odata);
 cudaFree(d_idata);
 cudaFree(d_odata);
 return 0;
}
__global__ void reduceUnrolling16(int *g_idata, int *g_odata, unsigned int n)
{
 // Set thread ID
 unsigned int tid = threadIdx.x;
 unsigned int idx = blockIdx.x * blockDim.x * 16 + threadIdx.x;
 // Convert global data pointer to the local pointer of this block
 int *idata = g_idata + blockIdx.x * blockDim.x * 16;
 // Unrolling 16
 if (idx + 15 * blockDim.x < n)
 {
 int a1 = g_idata[idx];
 int a2 = g_idata[idx + blockDim.x];
 int a3 = g_idata[idx + 2 * blockDim.x];
 int a4 = g_idata[idx + 3 * blockDim.x];
 int a5 = g_idata[idx + 4 * blockDim.x];
 int a6 = g_idata[idx + 5 * blockDim.x];
 int a7 = g_idata[idx + 6 * blockDim.x];
 int a8 = g_idata[idx + 7 * blockDim.x];
 int b1 = g_idata[idx + 8 * blockDim.x];
 int b2 = g_idata[idx + 9 * blockDim.x];
 int b3 = g_idata[idx + 10 * blockDim.x];
 int b4 = g_idata[idx + 11 * blockDim.x];
 int b5 = g_idata[idx + 12 * blockDim.x];
 int b6 = g_idata[idx + 13 * blockDim.x];
 int b7 = g_idata[idx + 14 * blockDim.x];
 int b8 = g_idata[idx + 15 * blockDim.x];
 g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + b1 + b2 + b3 + b4 + b5 + b6 + b7 +
b8;
 }
 __syncthreads();
 // In-place reduction in global memory
 for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
 {
 if (tid < stride)
 {
 idata[tid] += idata[tid + stride];
 }
 // Synchronize within thread block
 __syncthreads();
 }
 // Write result for this block to global memory
 if (tid == 0)
 {
 g_odata[blockIdx.x] = idata[0];
 }
}
```


## Output:
### kernel reduceUnrolling8
```python3
root@MidPC:/home/student/Desktop# nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243

root@MidPC:/home/student/Desktop# nvcc first.cu
root@MidPC:/home/student/Desktop# ./a.out
./a.out starting reduction at device 0: NVIDIA GeForce GTX 1660 SUPER     with array size 16777216  grid 32768 block 512
cpu reduce      elapsed 0.032312 sec cpu_sum: 2139353471
gpu Neighbored  elapsed 0.002421 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Neighbored2 elapsed 0.001394 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Interleaved elapsed 0.001276 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Unrolling2  elapsed 0.000832 sec gpu_sum: 2139353471 <<<grid 16384 block 512>>>
gpu Unrolling4  elapsed 0.000478 sec gpu_sum: 2139353471 <<<grid 8192 block 512>>>
gpu Unrolling8  elapsed 0.000357 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
gpu UnrollWarp8 elapsed 0.000299 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
gpu Cmptnroll8  elapsed 0.000300 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
gpu Cmptnroll   elapsed 0.000419 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>

root@MidPC:/home/student/Desktop# nvprof ./a.out
==9061== NVPROF is profiling process 9061, command: ./a.out
./a.out starting reduction at device 0: NVIDIA GeForce GTX 1660 SUPER     with array size 16777216  grid 32768 block 512
cpu reduce      elapsed 0.032495 sec cpu_sum: 2139353471
gpu Neighbored  elapsed 0.002494 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Neighbored2 elapsed 0.001471 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Interleaved elapsed 0.001340 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Unrolling2  elapsed 0.000831 sec gpu_sum: 2139353471 <<<grid 16384 block 512>>>
gpu Unrolling4  elapsed 0.000499 sec gpu_sum: 2139353471 <<<grid 8192 block 512>>>
gpu Unrolling8  elapsed 0.000293 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
gpu UnrollWarp8 elapsed 0.000316 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
gpu Cmptnroll8  elapsed 0.000310 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
gpu Cmptnroll   elapsed 0.000387 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
==9061== Profiling application: ./a.out
==9061== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   87.52%  52.764ms         9  5.8626ms  5.7580ms  6.3870ms  [CUDA memcpy HtoD]
                    4.06%  2.4481ms         1  2.4481ms  2.4481ms  2.4481ms  reduceNeighbored(int*, int*, unsigned int)
                    2.40%  1.4443ms         1  1.4443ms  1.4443ms  1.4443ms  reduceNeighboredLess(int*, int*, unsigned int)
                    2.18%  1.3124ms         1  1.3124ms  1.3124ms  1.3124ms  reduceInterleaved(int*, int*, unsigned int)
                    1.22%  735.63us         1  735.63us  735.63us  735.63us  reduceUnrolling2(int*, int*, unsigned int)
                    0.67%  401.22us         1  401.22us  401.22us  401.22us  reduceUnrolling4(int*, int*, unsigned int)
                    0.48%  291.12us         1  291.12us  291.12us  291.12us  reduceUnrollWarps8(int*, int*, unsigned int)
                    0.48%  288.46us         1  288.46us  288.46us  288.46us  void reduceCompleteUnroll<unsigned int=512>(int*, int*, unsigned int)
                    0.47%  285.97us         1  285.97us  285.97us  285.97us  reduceCompleteUnrollWarps8(int*, int*, unsigned int)
                    0.44%  266.80us         1  266.80us  266.80us  266.80us  reduceUnrolling8(int*, int*, unsigned int)
                    0.09%  52.608us         9  5.8450us  2.3680us  11.136us  [CUDA memcpy DtoH]
      API calls:   55.91%  128.94ms         2  64.470ms  75.385us  128.86ms  cudaMalloc
                   23.03%  53.115ms        18  2.9508ms  19.839us  6.4340ms  cudaMemcpy
                   16.86%  38.883ms         1  38.883ms  38.883ms  38.883ms  cudaDeviceReset
                    3.64%  8.3899ms        18  466.11us  76.045us  2.4483ms  cudaDeviceSynchronize
                    0.12%  273.27us         1  273.27us  273.27us  273.27us  cuDeviceTotalMem
                    0.12%  266.01us        97  2.7420us     260ns  111.43us  cuDeviceGetAttribute
                    0.10%  237.67us         2  118.84us  47.677us  190.00us  cudaFree
                    0.10%  230.59us         1  230.59us  230.59us  230.59us  cudaGetDeviceProperties
                    0.10%  228.39us         9  25.377us  21.579us  42.537us  cudaLaunchKernel
                    0.02%  36.037us         1  36.037us  36.037us  36.037us  cuDeviceGetName
                    0.00%  5.2000us         1  5.2000us  5.2000us  5.2000us  cuDeviceGetPCIBusId
                    0.00%  3.0200us         1  3.0200us  3.0200us  3.0200us  cudaSetDevice
                    0.00%  1.9000us         3     633ns     290ns  1.2700us  cuDeviceGetCount
                    0.00%     900ns         2     450ns     280ns     620ns  cuDeviceGet
                    0.00%     370ns         1     370ns     370ns     370ns  cuDeviceGetUuid
root@MidPC:/home/student/Desktop#
```
![238111396-a455d9ab-c0f3-49b3-97c2-db52edce1fbb](https://github.com/ragav-47/PCA-Simple-warp-divergence---Implement-Sum-Reduction./assets/75235488/db593775-e6bd-46e3-a8b9-a3926d32d611)
![238111400-ebadaa00-2775-41d3-a081-d54408ffa334](https://github.com/ragav-47/PCA-Simple-warp-divergence---Implement-Sum-Reduction./assets/75235488/dc231371-c547-4e9a-901d-9e58935c6424)

### kernel reduceUnrolling16
```python3
Password: 
root@MidPC:/home/student# cd Desktop
root@MidPC:/home/student/Desktop# nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243

root@MidPC:/home/student/Desktop# nvcc first.cu
root@MidPC:/home/student/Desktop# ./a.out
./a.out starting reduction at device 0: NVIDIA GeForce GTX 1660 SUPER     with array size 16777216  grid 32768 block 512
cpu reduce      elapsed 0.032371 sec cpu_sum: 2139353471
gpu Neighbored  elapsed 0.002427 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Neighbored2 elapsed 0.001394 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Interleaved elapsed 0.001279 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Unrolling2  elapsed 0.000799 sec gpu_sum: 2139353471 <<<grid 16384 block 512>>>
gpu Unrolling4  elapsed 0.000478 sec gpu_sum: 2139353471 <<<grid 8192 block 512>>>
gpu Unrolling8  elapsed 0.000282 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
gpu Unrolling16 elapsed 0.000338 sec gpu_sum: 2139353471 <<<grid 2048 block 512>>>
gpu UnrollWarp8 elapsed 0.000300 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
gpu Cmptnroll8  elapsed 0.000375 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
gpu Cmptnroll   elapsed 0.000371 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>

root@MidPC:/home/student/Desktop# nvprof ./a.out
==7688== NVPROF is profiling process 7688, command: ./a.out
./a.out starting reduction at device 0: NVIDIA GeForce GTX 1660 SUPER     with array size 16777216  grid 32768 block 512
cpu reduce      elapsed 0.036847 sec cpu_sum: 2139353471
gpu Neighbored  elapsed 0.002443 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Neighbored2 elapsed 0.001541 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Interleaved elapsed 0.001410 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Unrolling2  elapsed 0.000833 sec gpu_sum: 2139353471 <<<grid 16384 block 512>>>
gpu Unrolling4  elapsed 0.000494 sec gpu_sum: 2139353471 <<<grid 8192 block 512>>>
gpu Unrolling8  elapsed 0.000293 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
gpu Unrolling16 elapsed 0.000280 sec gpu_sum: 2139353471 <<<grid 2048 block 512>>>
gpu UnrollWarp8 elapsed 0.000315 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
gpu Cmptnroll8  elapsed 0.000383 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
gpu Cmptnroll   elapsed 0.000305 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
==7688== Profiling application: ./a.out
==7688== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   88.35%  58.554ms        10  5.8554ms  5.7410ms  6.3002ms  [CUDA memcpy HtoD]
                    3.64%  2.4101ms         1  2.4101ms  2.4101ms  2.4101ms  reduceNeighbored(int*, int*, unsigned int)
                    2.18%  1.4422ms         1  1.4422ms  1.4422ms  1.4422ms  reduceNeighboredLess(int*, int*, unsigned int)
                    1.98%  1.3134ms         1  1.3134ms  1.3134ms  1.3134ms  reduceInterleaved(int*, int*, unsigned int)
                    1.11%  734.65us         1  734.65us  734.65us  734.65us  reduceUnrolling2(int*, int*, unsigned int)
                    0.60%  395.34us         1  395.34us  395.34us  395.34us  reduceUnrolling4(int*, int*, unsigned int)
                    0.44%  289.27us         1  289.27us  289.27us  289.27us  reduceUnrollWarps8(int*, int*, unsigned int)
                    0.43%  287.73us         1  287.73us  287.73us  287.73us  reduceCompleteUnrollWarps8(int*, int*, unsigned int)
                    0.43%  282.13us         1  282.13us  282.13us  282.13us  void reduceCompleteUnroll<unsigned int=512>(int*, int*, unsigned int)
                    0.40%  267.73us         1  267.73us  267.73us  267.73us  reduceUnrolling8(int*, int*, unsigned int)
                    0.37%  248.24us         1  248.24us  248.24us  248.24us  reduceUnrolling16(int*, int*, unsigned int)
                    0.08%  54.045us        10  5.4040us  1.7280us  11.071us  [CUDA memcpy DtoH]
      API calls:   61.31%  178.75ms         2  89.376ms  119.78us  178.63ms  cudaMalloc
                   20.21%  58.919ms        20  2.9460ms  19.809us  6.3449ms  cudaMemcpy
                   15.05%  43.863ms         1  43.863ms  43.863ms  43.863ms  cudaDeviceReset
                    3.01%  8.7779ms        20  438.89us  71.676us  2.4101ms  cudaDeviceSynchronize
                    0.09%  255.26us        10  25.525us  20.529us  30.148us  cudaLaunchKernel
                    0.08%  246.24us        97  2.5380us     230ns  105.41us  cuDeviceGetAttribute
                    0.08%  243.18us         1  243.18us  243.18us  243.18us  cuDeviceTotalMem
                    0.08%  220.22us         2  110.11us  47.098us  173.12us  cudaFree
                    0.07%  210.51us         1  210.51us  210.51us  210.51us  cudaGetDeviceProperties
                    0.01%  42.258us         1  42.258us  42.258us  42.258us  cuDeviceGetName
                    0.00%  5.0400us         2  2.5200us     240ns  4.8000us  cuDeviceGet
                    0.00%  4.9200us         1  4.9200us  4.9200us  4.9200us  cuDeviceGetPCIBusId
                    0.00%  2.8700us         1  2.8700us  2.8700us  2.8700us  cudaSetDevice
                    0.00%  2.6700us         3     890ns     240ns  2.0700us  cuDeviceGetCount
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetUuid
root@MidPC:/home/student/Desktop#
 ```
![238111471-0aef7741-c7a8-4c34-b4b3-a701fb4e68fd](https://github.com/ragav-47/PCA-Simple-warp-divergence---Implement-Sum-Reduction./assets/75235488/65561132-4200-4d4c-8fba-456b21bcb984)
![238111473-4dbde075-1712-423b-a29d-adaeed5cef3f](https://github.com/ragav-47/PCA-Simple-warp-divergence---Implement-Sum-Reduction./assets/75235488/e221f8ee-1d00-403a-b2ed-c22bd4b3c338)
The time taken by the kernel reduceUnrolling16 is comparatively less to the kernal reduceUnrolling8 as each thread in the kernel reduceUnrolling16 handles 16 data blocks.

## Result:
  Implementation of the kernel reduceUnrolling16 is done and the performance of kernal reduceUnrolling16 is comapared with kernal reduceUnrolling8 using proper metrics and events with nvprof.
