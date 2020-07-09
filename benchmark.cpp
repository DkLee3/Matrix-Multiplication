#define SIZE 4096

#include <iostream>
#include <string.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <string>
#include <cstdio>
#include <pthread.h>
#include <stdlib.h>
#include <x86intrin.h>

using namespace std;

typedef struct{

        float (*a);
        float (*b);
        float (*result);
        int start_;
        int end_;

}matrix_info;


long long wall_clock_time();
void zero_matrix(float *m);
void transpose_matrix(float *m);
void *matrix_multiply(void *);
void print_matrix(float *m);
void random_matrix(float *m);
void print_simd(__m128 var);


void zero_matrix(float *m) {
	for (int i = 0; i < SIZE; i++) 
		for (int j = 0; j < SIZE; j++) 
			m[i * SIZE + j] = 0;
}

void transpose_matrix(float *m) {
	for (int i = 0; i < SIZE; i++) 
		for (int j = i + 1; j < SIZE; j++) 
			swap(m[i * SIZE + j], m[j * SIZE + i]);
}

void matrix_multiply_unoptimized(float *m1, float *m2, float *result) {
	long long start, end;

	start = wall_clock_time();
	for (int i = 0; i < SIZE; i++) 
		for (int j = 0; j < SIZE; j++) 
			for (int k = 0; k < SIZE; k++) 
				result[i * SIZE + j] += m1[i * SIZE + k] * m2[k * SIZE + j];
	end = wall_clock_time();
	fprintf(stderr, "Unoptimized Matrix multiplication took %1.2f seconds\n", ((float)(end - start))/1000000000);
}


/* 
* Edit mat mul algorithm.
*/
void matrix_multiply_optimized(float *m1, float *m2, float *result) {
	long long start, end;
	start = wall_clock_time();
	pthread_t thr[16];
	matrix_info matrix_inf[16];

	for(int i = 0 ; i<16; i++){
		
		matrix_inf[i].a = m1;
		matrix_inf[i].b = m2;
		matrix_inf[i].result = result;
		matrix_inf[i].start_ = (SIZE/16)*i;
		matrix_inf[i].end_ = (SIZE/16)*(i+1);
		pthread_create(&thr[i], NULL, matrix_multiply, (void *)(&matrix_inf[i]));
	}

	
	for(int j = 0 ; j<16; j++)
		pthread_join(thr[j], NULL);

	// Main mat mul algorithm end
	

	end = wall_clock_time();
	fprintf(stderr, "Optimized Matrix multiplication took %1.2f seconds\n", ((float)(end - start))/1000000000);
}

void *matrix_multiply(void (*matrix_inf)){

	float (*m1) = ((matrix_info*)matrix_inf)->a;
	float (*m2) = ((matrix_info*)matrix_inf)->b;

	//print_matrix(m1);
	//cout<<"/////////////////////////"<<endl;
	//print_matrix(m2);

	float (*result) = ((matrix_info*)matrix_inf)->result;
	int start = ((matrix_info*)matrix_inf)->start_;
	int end = ((matrix_info*)matrix_inf)->end_;

        for (int i = start; i < end; i++){
                for (int k = 0; k < SIZE; k++){
			for(int j = 0; j < SIZE; j++){
                        	result[i * SIZE + j] += m1[i * SIZE + k] * m2[k * SIZE + j];	
			}
		}
	}
}

/* 
* 2. Use your matrix_multiply_optimized.
* 	 Edit here with your code.
*/

void print_matrix(float *m) {
	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			cout << m[i * SIZE + j] << " ";
		} 
		cout << endl;
	}
}

void random_matrix(float *m) {
	for (int i = 0; i < SIZE; i++) 
		for (int j = 0; j < SIZE; j++) 
			m[i * SIZE + j] = rand() % 10;
}

void print_simd(__m128 var){
	float val[4];
	memcpy(val, &var, sizeof(val));
	printf("values are  %g, %g, %g, %g\n", val[0], val[1], val[2], val[3]);
}

int main() {
	// init matrixes
	float *m1 __attribute__((aligned(16))) = new float[SIZE * SIZE];
	float *m2 __attribute__((aligned(16))) = new float[SIZE * SIZE];
	float *result_unoptimized __attribute__((aligned(16))) = new float[SIZE * SIZE];
	float *result_optimized1 __attribute__((aligned(16))) = new float[SIZE * SIZE]; // 1-D multithread
	random_matrix(m1);
	random_matrix(m2);
	zero_matrix(result_unoptimized);
	zero_matrix(result_optimized1);

	// do matrix multiply

	matrix_multiply_optimized(m1, m2, result_optimized1);
	matrix_multiply_unoptimized(m1, m2, result_unoptimized);

	// print result
	int wrong = 0;
	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			if (result_optimized1[i * SIZE + j] != result_unoptimized[i * SIZE + j]) {
				//cout<<"different index of i and j are "<<i <<" "<<j<<endl;
				wrong++;
			}
		}
	}

	if (wrong == 0) {
		cout << "SUCCESS: Both optimized and unoptimized gave same results" << endl;
	} else {
		cout << "FAIL: Optimized and unoptimized gave different results" << endl;
	}

	delete m1;
	delete m2;
	delete result_unoptimized;
	delete result_optimized1;

	return 0;
}




/********************************************
 * Helpers
 *******************************************/
long long wall_clock_time() {
	#ifdef __linux__
		struct timespec tp;
		clock_gettime(CLOCK_REALTIME, &tp);
		return (long long)(tp.tv_nsec + (long long)tp.tv_sec * 1000000000ll);
	#else
		#warning "Your timer resoultion might be too low. Compile on Linux and link with librt"
		struct timeval tv;
		gettimeofday(&tv, NULL);
		return (long long)(tv.tv_usec * 1000 + (long long)tv.tv_sec * 1000000000ll);
	#endif
}
