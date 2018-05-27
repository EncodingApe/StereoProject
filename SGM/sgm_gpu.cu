// GPU VERSION
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <sm_20_atomic_functions.h>

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "gaussian.h"

#define BLUR_RADIUS 3
#define PATHS_PER_SCAN 4
#define MAX_SHORT 65535
#define SMALL_PENALTY 3
#define LARGE_PENALTY 20

struct path {
	short rowDiff;
	short colDiff;
	short index;
};


void CUDA_CHECK_RETURN(cudaError_t status, int i = 0) {
	if (status != cudaSuccess) {
		std::cout << cudaGetErrorString(status) << i << std::endl;
		exit(1);
	}
}

__global__ void calculatePixelCostBT(int *param, uchar *im_left, uchar *im_right, unsigned short *tmp_C) {
	int rows = param[0], cols = param[1], disps = param[2];
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	if (id >= rows * cols * disps) return;

	int row = (id / (disps * cols)) % rows;
	int leftCol = (id / disps) % cols;
	int d = id % disps;
	int rightCol = leftCol - d;

	char leftValue, rightValue, beforeRightValue, afterRightValue, rightValueMinus, rightValuePlus, rightValueMin, rightValueMax;

	// Interpolation on the right image.
	int col1 = leftCol, col2 = rightCol;
	if (col1 < 0)
		leftValue = 0;
	else
		leftValue = im_left[row * cols + col1];

	if (col2 < 0)
		rightValue = 0;
	else
		rightValue = im_right[row * cols + col2];

	if (col2 > 0) {
		beforeRightValue = im_right[row * cols + col2 - 1];
	}
	else {
		beforeRightValue = rightValue;
	}

	if (col2 + 1 < cols && col2>0) {
		afterRightValue = im_right[row * cols + col2 + 1];
	}
	else {
		afterRightValue = rightValue;
	}

	// Use the median value to interpolate
	rightValueMinus = round((rightValue + beforeRightValue) / 2.f);
	rightValuePlus = round((rightValue + afterRightValue) / 2.f);

	char tmp;
	rightValueMin = rightValue < (tmp = (rightValueMinus < rightValuePlus ? rightValueMinus : rightValuePlus)) ? rightValue : tmp;
	rightValueMax = rightValue >(tmp = (rightValueMinus > rightValuePlus ? rightValueMinus : rightValuePlus)) ? rightValue : tmp;

	unsigned short firstVal = (0 > ((leftValue - rightValueMax) > (rightValueMin - leftValue) ? (leftValue - rightValueMax) : (rightValueMin - leftValue)) ? 0 : ((leftValue - rightValueMax) > (rightValueMin - leftValue) ? (leftValue - rightValueMax) : (rightValueMin - leftValue)));

	// Interpolation on the left image
	col1 = rightCol; col2 = leftCol;
	if (col1 < 0)
		leftValue = 0;
	else
		leftValue = im_right[row * cols + col1];

	if (col2 < 0)
		rightValue = 0;
	else
		rightValue = im_left[row * cols + col2];

	if (col2 > 0) {
		beforeRightValue = im_left[row * cols + col2 - 1];
	}
	else {
		beforeRightValue = rightValue;
	}

	if (col2 + 1 < cols && col2>0) {
		afterRightValue = im_left[row * cols + col2 + 1];
	}
	else {
		afterRightValue = rightValue;
	}

	rightValueMinus = round((rightValue + beforeRightValue) / 2.f);
	rightValuePlus = round((rightValue + afterRightValue) / 2.f);

	rightValueMin = rightValue < (tmp = (rightValueMinus < rightValuePlus ? rightValueMinus : rightValuePlus)) ? rightValue : tmp;
	rightValueMax = rightValue >(tmp = (rightValueMinus > rightValuePlus ? rightValueMinus : rightValuePlus)) ? rightValue : tmp;

	unsigned short secondVal = 0 > ((leftValue - rightValueMax) > (rightValueMin - leftValue) ? (leftValue - rightValueMax) : (rightValueMin - leftValue)) ? 0 : ((leftValue - rightValueMax) > (rightValueMin - leftValue) ? (leftValue - rightValueMax) : (rightValueMin - leftValue));

	tmp_C[id] = (firstVal < secondVal ? firstVal : secondVal);
}

void calculatePixelCost(cv::Mat &firstImage, cv::Mat &secondImage, int disparityRange, unsigned short ***C) {
	int row = firstImage.rows;
	int col = firstImage.cols;
	int size = row * col;
	uchar *im_left, *im_right;

	unsigned short *Cc;
	Cc = (unsigned short *)malloc(sizeof(unsigned short) * size * disparityRange);

	unsigned short *tmp_C;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&tmp_C, sizeof(unsigned short) * size * disparityRange), -1);

	// Allocate GPU memory for left image and right image.
	CUDA_CHECK_RETURN(cudaMalloc((void**)&im_left, sizeof(uchar) * size), 1);
	CUDA_CHECK_RETURN(cudaMalloc((void**)&im_right, sizeof(uchar) * size), 2);

	CUDA_CHECK_RETURN(cudaMemcpy(im_left, firstImage.ptr<uchar>(), size * sizeof(uchar), cudaMemcpyHostToDevice), 3);
	CUDA_CHECK_RETURN(cudaMemcpy(im_right, secondImage.ptr<uchar>(), size * sizeof(uchar), cudaMemcpyHostToDevice), 4);


	// Because a warp in CUDA is 32, block is supposed to be 32 * a
	dim3 block_size;
	block_size.x = 3;

	// (Total + perblock - 1) / perblock
	dim3 grid_size;
	grid_size.x = (size * disparityRange + block_size.x - 1) / block_size.x;

	// basic params
	int *passed_para;
	int param[3] = {row, col, disparityRange};

	CUDA_CHECK_RETURN(cudaMalloc((void**)&passed_para, sizeof(int) * 3), 5);
	CUDA_CHECK_RETURN(cudaMemcpy(passed_para, param, 3 * sizeof(int), cudaMemcpyHostToDevice), 6);

	calculatePixelCostBT << <grid_size, block_size >> >(passed_para, im_left, im_right, tmp_C);

	// synchronize
	CUDA_CHECK_RETURN(cudaDeviceSynchronize(), 7);

	// copy fron GPU to CPU
	CUDA_CHECK_RETURN(cudaMemcpy(C[0][0], tmp_C, sizeof(unsigned short) * size * disparityRange, cudaMemcpyDeviceToHost), 8);

	cudaFree(im_left);
	cudaFree(im_right);
	cudaFree(passed_para);
	cudaFree(tmp_C);
}

// pathCount can be 1, 2, 4, or 8
void initializeFirstScanPaths(std::vector<path> &paths, unsigned short pathCount) {
	for (unsigned short i = 0; i < pathCount; ++i) {
		paths.push_back(path());
	}

	if (paths.size() >= 1) {
		paths[0].rowDiff = 0;
		paths[0].colDiff = -1;
		paths[0].index = 1;
	}

	if (paths.size() >= 2) {
		paths[1].rowDiff = -1;
		paths[1].colDiff = 0;
		paths[1].index = 2;
	}

	if (paths.size() >= 4) {
		paths[2].rowDiff = -1;
		paths[2].colDiff = 1;
		paths[2].index = 4;

		paths[3].rowDiff = -1;
		paths[3].colDiff = -1;
		paths[3].index = 7;
	}

	if (paths.size() >= 8) {
		paths[4].rowDiff = -2;
		paths[4].colDiff = 1;
		paths[4].index = 8;

		paths[5].rowDiff = -2;
		paths[5].colDiff = -1;
		paths[5].index = 9;

		paths[6].rowDiff = -1;
		paths[6].colDiff = -2;
		paths[6].index = 13;

		paths[7].rowDiff = -1;
		paths[7].colDiff = 2;
		paths[7].index = 15;
	}
}

// pathCount can be 1, 2, 4, or 8
void initializeSecondScanPaths(std::vector<path> &paths, unsigned short pathCount) {
	for (unsigned short i = 0; i < pathCount; ++i) {
		paths.push_back(path());
	}

	if (paths.size() >= 1) {
		paths[0].rowDiff = 0;
		paths[0].colDiff = 1;
		paths[0].index = 0;
	}

	if (paths.size() >= 2) {
		paths[1].rowDiff = 1;
		paths[1].colDiff = 0;
		paths[1].index = 3;
	}

	if (paths.size() >= 4) {
		paths[2].rowDiff = 1;
		paths[2].colDiff = 1;
		paths[2].index = 5;

		paths[3].rowDiff = 1;
		paths[3].colDiff = -1;
		paths[3].index = 6;
	}

	if (paths.size() >= 8) {
		paths[4].rowDiff = 2;
		paths[4].colDiff = 1;
		paths[4].index = 10;

		paths[5].rowDiff = 2;
		paths[5].colDiff = -1;
		paths[5].index = 11;

		paths[6].rowDiff = 1;
		paths[6].colDiff = -2;
		paths[6].index = 12;

		paths[7].rowDiff = 1;
		paths[7].colDiff = 2;
		paths[7].index = 14;
	}
}

__global__ void aggregate(int row, int col, int cols, int rows, int disparityRange, int rowDiff, int colDiff, unsigned short *C, unsigned short *temp, unsigned short *temp1, int *S) {
	int d = threadIdx.x;

	unsigned aggregatedCost = 0;
	aggregatedCost += C[row * cols * disparityRange + col * disparityRange + d];

	if (row + rowDiff < 0 || row + rowDiff >= rows || col + colDiff < 0 || col + colDiff >= cols) {
		// border
		temp[d] = aggregatedCost;
		atomicAdd(S + d, (int)temp[d]);
		return;
	}

	unsigned short minPrev, minPrevOther, prev, prevPlus, prevMinus, tmp;
	prev = minPrev = minPrevOther = prevPlus = prevMinus = MAX_SHORT;

	// traverse all disparity
	for (int disp = 0; disp < disparityRange; ++disp) {
		tmp = temp1[disp];
		if (minPrev > tmp) {
			minPrev = tmp;
		}

		if (disp == d) {
			prev = tmp;
		}
		else if (disp == d + 1) {
			prevPlus = tmp;
		}
		else if (disp == d - 1) {
			prevMinus = tmp;
		}
		else {
			if (minPrevOther > tmp) {
				minPrevOther = tmp;
			}
		}
	}

	// Caculate Lr
	int tmp1 = (int)prevPlus + SMALL_PENALTY < (int)prevMinus + SMALL_PENALTY ? (int)prevPlus + SMALL_PENALTY : (int)prevMinus + SMALL_PENALTY;
	int tmp2 = (int)prev < (int)minPrevOther + LARGE_PENALTY ? (int)prev : (int)minPrevOther + LARGE_PENALTY;
	int s = tmp1 < tmp2 ? tmp1 : tmp2;
	aggregatedCost = aggregatedCost + s;
	aggregatedCost -= minPrev;

	// record for DP
	temp[d] = aggregatedCost;

	// atomic operation
	atomicAdd(S + d, (int)temp[d]);
}

__global__ void kernel_aggregateCosts(int *cuda_rowDiff, int *cuda_colDiff, int *params, unsigned short *C, int *S) {
	// get basic params
	int rows = params[0]; int cols = params[1]; int disparityRange = params[2];
	int id = blockIdx.x;
	int rowDiff = cuda_rowDiff[id]; int colDiff = cuda_colDiff[id];

	// the array used for DP
	unsigned short *temp, *temp1;
	temp = (unsigned short *)malloc(sizeof(unsigned short) * disparityRange);
	temp1 = (unsigned short *)malloc(sizeof(unsigned short) * disparityRange);

	int row, col, i, j, k;

	// DP in different path would rely on different "last" value.
	if (id == 0) {
		//printf("%d\n", id);
		for (i = 0; i < rows; i++) {
			for (j = 0; j < cols; j++) {
				row = i;
				col = j;
				aggregate << <1, disparityRange >> > (row, col, cols, rows, disparityRange, rowDiff, colDiff, C, temp, temp1, S + row * cols*disparityRange + col * disparityRange);
				cudaError_t e = cudaDeviceSynchronize();
				if (e != cudaSuccess) {
					printf("%s\n", cudaGetErrorString(e));
				}
				memcpy(temp1, temp, sizeof(unsigned short) * disparityRange);
			}
		}
		//printf("%d\n", id);
	} else if (id == 1) {
		//printf("%d\n", id);
		for (i = 0; i < cols; i++) {
			for (j = 0; j < rows; j++) {
				row = j;
				col = i;
				aggregate << <1, disparityRange >> > (row, col, cols, rows, disparityRange, rowDiff, colDiff, C, temp, temp1, S + row * cols*disparityRange + col * disparityRange);
				cudaError_t e = cudaDeviceSynchronize();
				if (e != cudaSuccess) {
					printf("%s\n", cudaGetErrorString(e));
				}
				memcpy(temp1, temp, sizeof(unsigned short) * disparityRange);
			}
		}
		//printf("%d\n", id);
	} else if (id == 3) {
		//printf("%d\n", id);
		for (i = cols - 1; i >= 1 - rows; i--) {
			for (j = 0; i + j < cols && j < rows; j++) {
				if (i + j < 0) continue;
				row = j;
				col = i + j;
				aggregate << <1, disparityRange >> > (row, col, cols, rows, disparityRange, rowDiff, colDiff, C, temp, temp1, S + row * cols*disparityRange + col * disparityRange);
				cudaError_t e = cudaDeviceSynchronize();
				if (e != cudaSuccess) {
					printf("%s\n", cudaGetErrorString(e));
				}
				memcpy(temp1, temp, sizeof(unsigned short) * disparityRange);
			}
		}
		//printf("%d\n", id);
	} else if (id == 2) {
		//printf("%d\n", id);
		for (i = 0; i < rows + cols - 1; i++) {
			for (j = 0; i - j >= 0 && j < rows; j++) {
				if (i - j >= cols) continue;
				row = j;
				col = i - j;
				aggregate << <1, disparityRange >> > (row, col, cols, rows, disparityRange, rowDiff, colDiff, C, temp, temp1, S + row * cols*disparityRange + col * disparityRange);
				cudaError_t e = cudaDeviceSynchronize();
				if (e != cudaSuccess) {
					printf("%s\n", cudaGetErrorString(e));
				}
				memcpy(temp1, temp, sizeof(unsigned short) * disparityRange);
			}
		}
		//printf("%d\n", id);
	} else if (id == 4) {
		//printf("%d\n", id);
		for (i = 0; i < rows; i++) {
			for (j = cols - 1; j >= 0; j--) {
				row = i;
				col = j;
				aggregate << <1, disparityRange >> > (row, col, cols, rows, disparityRange, rowDiff, colDiff, C, temp, temp1, S + row * cols*disparityRange + col * disparityRange);
				cudaError_t e = cudaDeviceSynchronize();
				if (e != cudaSuccess) {
					printf("%s\n", cudaGetErrorString(e));
				}
				memcpy(temp1, temp, sizeof(unsigned short) * disparityRange);
			}
		}
		//printf("%d\n", id);
	} else if (id == 5) {
		//printf("%d\n", id);
		for (i = 0; i < cols; i++) {
			for (j = rows - 1; j >= 0; j--) {
				row = j;
				col = i;
				aggregate << <1, disparityRange >> > (row, col, cols, rows, disparityRange, rowDiff, colDiff, C, temp, temp1, S + row * cols*disparityRange + col * disparityRange);
				cudaError_t e = cudaDeviceSynchronize();
				if (e != cudaSuccess) {
					printf("%s\n", cudaGetErrorString(e));
				}
				memcpy(temp1, temp, sizeof(unsigned short) * disparityRange);
			}
		}
		//printf("%d\n", id);
	} else if (id == 6) {
		//printf("%d\n", id);
		for (i = 1 - cols; i <= rows - 1; i++) {
			for (j = cols - 1; i + j >= 0 && j >= 0; j--) {
				if (i + j >= cols) continue;
				row = i + j;
				col = j;
				aggregate << <1, disparityRange >> > (row, col, cols, rows, disparityRange, rowDiff, colDiff, C, temp, temp1, S + row * cols*disparityRange + col * disparityRange);
				cudaError_t e = cudaDeviceSynchronize();
				if (e != cudaSuccess) {
					printf("%s\n", cudaGetErrorString(e));
				}
				memcpy(temp1, temp, sizeof(unsigned short) * disparityRange);
			}
		}
		//printf("%d\n", id);
	} else if (id == 7) {
		//printf("%d\n", id);
		for (i = 0; i < rows + cols - 1; i++) {
			for (j = 0; i - j >= 0 && j < cols; j++) {
				if (i - j >= rows) continue;
				row = i - j;
				col = j;
				aggregate << <1, disparityRange >> > (row, col, cols, rows, disparityRange, rowDiff, colDiff, C, temp, temp1, S + row * cols*disparityRange + col * disparityRange);
				cudaError_t e = cudaDeviceSynchronize();
				if (e != cudaSuccess) {
					printf("%s\n", cudaGetErrorString(e));
				}
				memcpy(temp1, temp, sizeof(unsigned short) * disparityRange);
			}
		}
		//printf("%d\n", id);
	}
}

void aggregateCosts(int rows, int cols, int disparityRange, unsigned short ***C, unsigned short ***S) {

	std::vector<path> firstScanPaths;
	std::vector<path> secondScanPaths;

	// the scan path to aggregate
	initializeFirstScanPaths(firstScanPaths, PATHS_PER_SCAN);
	initializeSecondScanPaths(secondScanPaths, PATHS_PER_SCAN);

	int rowDiff[PATHS_PER_SCAN * 2], colDiff[PATHS_PER_SCAN * 2];
	for (int i = 0; i<firstScanPaths.size(); i++) {
		rowDiff[i] = firstScanPaths[i].rowDiff;
		colDiff[i] = firstScanPaths[i].colDiff;
	}

	for (int i = 0; i<secondScanPaths.size(); i++) {
		rowDiff[i + PATHS_PER_SCAN] = secondScanPaths[i].rowDiff;
		colDiff[i + PATHS_PER_SCAN] = secondScanPaths[i].colDiff;
	}

	// the paths used in GPU
	int *cuda_rowDiff, *cuda_colDiff;

	CUDA_CHECK_RETURN(cudaMalloc((void**)&cuda_colDiff, sizeof(int) * PATHS_PER_SCAN * 2));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&cuda_rowDiff, sizeof(int) * PATHS_PER_SCAN * 2));

	CUDA_CHECK_RETURN(cudaMemcpy(cuda_rowDiff, rowDiff, sizeof(int) * PATHS_PER_SCAN * 2, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(cuda_colDiff, colDiff, sizeof(int) * PATHS_PER_SCAN * 2, cudaMemcpyHostToDevice));

	// the pixel cost transferred to GPU
	unsigned short *tmp_C;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&tmp_C, sizeof(unsigned short) * rows * cols * disparityRange));
	CUDA_CHECK_RETURN(cudaMemcpy(tmp_C, C[0][0], sizeof(unsigned short) * rows * cols * disparityRange, cudaMemcpyHostToDevice));

	// the aggregate cost calculated in parallel
	int *tmp_S;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&tmp_S, sizeof(int) * rows * cols * disparityRange));
	CUDA_CHECK_RETURN(cudaMemset(tmp_S, 0, sizeof(int) * rows * cols * disparityRange));

	int *Ss;
	Ss = (int*)malloc(sizeof(int)*rows * cols * disparityRange);

	// basic params
	int param[3] = {rows, cols, disparityRange};
	int *passed_para;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&passed_para, sizeof(int) * 3), 5);
	CUDA_CHECK_RETURN(cudaMemcpy(passed_para, param, 3 * sizeof(int), cudaMemcpyHostToDevice), 6);

	kernel_aggregateCosts << <PATHS_PER_SCAN * 2, 1 >> > (cuda_rowDiff, cuda_colDiff, passed_para, tmp_C, tmp_S);

	// synchronize
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	// copy from GPU to CPU
	CUDA_CHECK_RETURN(cudaMemcpy(Ss, tmp_S, sizeof(int)*rows * cols * disparityRange, cudaMemcpyDeviceToHost));

	// convert unsigned short to int
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			for (int k = 0; k < disparityRange; k++) {
				S[i][j][k] = (unsigned short)Ss[i*cols*disparityRange + j * disparityRange + k];
			}
		}
	}

	cudaFree(cuda_rowDiff);
	cudaFree(cuda_colDiff);
	cudaFree(tmp_C);
	cudaFree(passed_para);
	cudaFree(tmp_S);
}

void computeDisparity(unsigned short ***S, int rows, int cols, int disparityRange, cv::Mat &disparityMap) {
	unsigned int disparity = 0, minCost;
	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			minCost = MAX_SHORT;
			// Ñ¡ÔñSÖµ×îÐ¡µÄÉî¶È
			for (int d = disparityRange - 1; d >= 0; --d) {
				if (minCost > S[row][col][d]) {
					minCost = S[row][col][d];
					disparity = d;
				}
			}
			disparityMap.at<uchar>(row, col) = disparity;
		}
	}
}

void saveDisparityMap(cv::Mat &disparityMap, int disparityRange, char* outputFile) {
	double factor = 256.0 / disparityRange;
	for (int row = 0; row < disparityMap.rows; ++row) {
		for (int col = 0; col < disparityMap.cols; ++col) {
			disparityMap.at<uchar>(row, col) *= factor;
		}
	}
	cv::imwrite(outputFile, disparityMap);
}

int main(int argc, char** argv) {
	// left image. right image. output image.
	char *firstFileName = "left.png";
	char *secondFileName = "right.png";
	char *outFileName = "out.png";

	cv::Mat firstImage;
	cv::Mat secondImage;
	// read the grayscale image
	firstImage = cv::imread(firstFileName, CV_LOAD_IMAGE_GRAYSCALE);
	secondImage = cv::imread(secondFileName, CV_LOAD_IMAGE_GRAYSCALE);

	if (!firstImage.data || !secondImage.data) {
		std::cerr << "Could not open or find one of the images!" << std::endl;
		return -1;
	}

	// the range of disparity
	unsigned int disparityRange = 20;
	unsigned short ***C; // pixel cost array W x H x D
	unsigned short ***S; // aggregated cost array W x H x D

	clock_t begin = clock();

	std::cout << "Allocating space..." << std::endl;

	C = (unsigned short ***)malloc(sizeof(unsigned short **) * firstImage.rows);
	C[0] = (unsigned short **)malloc(sizeof(unsigned short *) * firstImage.rows * firstImage.cols);
	C[0][0] = (unsigned short *)malloc(sizeof(unsigned short) * firstImage.rows * firstImage.cols * disparityRange);

	S = (unsigned short ***)malloc(sizeof(unsigned short **) * firstImage.rows);
	S[0] = (unsigned short **)malloc(sizeof(unsigned short *) * firstImage.rows * firstImage.cols);
	S[0][0] = (unsigned short *)malloc(sizeof(unsigned short) * firstImage.rows * firstImage.cols * disparityRange);

	//  allocate cost arrays make sure the memory is continuous
	for (int row = 1; row<firstImage.rows; row++) {
		C[row] = C[row - 1] + firstImage.cols;
		S[row] = S[row - 1] + firstImage.cols;
	}

	for (int row = 0; row < firstImage.rows; ++row) {
		if (row != 0) {
			C[row][0] = C[row - 1][firstImage.cols - 1] + disparityRange;
			S[row][0] = S[row - 1][firstImage.cols - 1] + disparityRange;
		}
		for (int col = 0; col < firstImage.cols; ++col) {
			if (col > 0) {
				C[row][col] = C[row][col - 1] + disparityRange;
				S[row][col] = S[row][col - 1] + disparityRange;
			}
		}
	}

	std::cout << "Smoothing images..." << std::endl;
	grayscaleGaussianBlur(firstImage, firstImage, BLUR_RADIUS);
	grayscaleGaussianBlur(secondImage, secondImage, BLUR_RADIUS);

	std::cout << "Calculating pixel cost for the image..." << std::endl;
	calculatePixelCost(firstImage, secondImage, disparityRange, C);


	std::cout << "Aggregating costs..." << std::endl;
	aggregateCosts(firstImage.rows, firstImage.cols, disparityRange, C, S);

	cv::Mat disparityMap = cv::Mat(cv::Size(firstImage.cols, firstImage.rows), CV_8UC1, cv::Scalar::all(0));

	std::cout << "Computing disparity..." << std::endl;
	computeDisparity(S, firstImage.rows, firstImage.cols, disparityRange, disparityMap);

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	printf("Done in %.2lf seconds.\n", elapsed_secs);

	saveDisparityMap(disparityMap, disparityRange, outFileName);

	return 0;
}
