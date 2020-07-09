#define SIZE 4096
#define BSIZE 32

#include <iostream>
#include <string>
#include <cstdio>
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
void print_matrix(float *m);
void random_matrix(float *m);
void *matrix_multiply(void *);


void zero_matrix(float *m) {
	for (int i = 0; i < SIZE; i++) 
		for (int j = 0; j < SIZE; j++) 
			m[i * SIZE + j] = 0;
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

/////////// miss rate 개선전&후 각 레벨 miss rate, end-to-end time 비교/////////
inline void *matrix_multiply(void (*matrix_inf)){
	
	float (*m1) = ((matrix_info*)matrix_inf)->a;
	float (*m2) = ((matrix_info*)matrix_inf)->b;

	float (*result) = ((matrix_info*)matrix_inf)->result;
	int start = ((matrix_info*)matrix_inf)->start_; 
	int end = ((matrix_info*)matrix_inf)->end_;


	int a_index_00 = 0;
	int a_index_01 = 0;
	int a_index_10 = 0;
	int a_index_11 = 0;
	int b_index_00 = 0;
	int b_index_01 = 0;
       	int b_index_10 = 0;
	int b_index_11 = 0; 	

	for(int i = start; i < end; i += BSIZE){
		for(int k = 0; k < SIZE; k += BSIZE){
			for(int j = 0; j < SIZE; j += BSIZE){
				for(int x = i; x < i + BSIZE ; x+=2){
					for(int y = j;  y < j + BSIZE; y+=2){
						register int result_index_00 = x*SIZE + y;
						register int result_index_01 = result_index_00+1;
						register int result_index_10 = (x+1)*SIZE+y;
						register int result_index_11 = result_index_10+1;
						for(int z = k; z < k + BSIZE; z+=2){
							a_index_00 = x*SIZE+z;
							a_index_01 = a_index_00+1;
							a_index_10 = (x+1)*SIZE+z;
							a_index_11 = a_index_10+1;
							b_index_00 = z*SIZE+y;
							b_index_01 = b_index_00+1;
							b_index_10 =(z+1)*SIZE+y;
							b_index_11 = b_index_10+1;
							result[result_index_00] += m1[a_index_00]*m2[b_index_00] + m1[a_index_01]*m2[b_index_10];
							result[result_index_01] += m1[a_index_00]*m2[b_index_01] + m1[a_index_01]*m2[b_index_11];
							result[result_index_10] += m1[a_index_10]*m2[b_index_00] + m1[a_index_11]*m2[b_index_10];
							result[result_index_11] += m1[a_index_10]*m2[b_index_01] + m1[a_index_11]*m2[b_index_11];
						}

					}

				}

			}
		}
	}




	//////////////////"naive ikj"///////////////////
	/*
	for (int i = start; i < end; ++i)
		for (int k = 0; k < SIZE; ++k)
	        	for (int j = 0; j < SIZE; j++){
		        	result[i * SIZE + j] += m1[i * SIZE + k] * m2[k * SIZE + j];
			}	
	*/
	//////////////////////////////////////////////////
}


/* 
* Edit mat mul algorithm.
*/
inline void matrix_multiply_optimized(float *m1, float *m2, float *result) {
	long long start, end;
	start = wall_clock_time();
	// Main mat mul algorithm begin
	
	pthread_t thr[16];
	matrix_info matrix_inf[16];

	for(int i = 0 ; i<16; i++){
		matrix_inf[i].a = m1;
		matrix_inf[i].b = m2;
		matrix_inf[i].result = result;
		matrix_inf[i].start_ =(SIZE/16)*i;
		matrix_inf[i].end_ = (SIZE/16)*(i+1);
		pthread_create(&thr[i], NULL, matrix_multiply, (void *)(&matrix_inf[i]));
	}
	
	// Main mat mul algorithm end
	
	for(int j = 0 ; j<16; j++)
		pthread_join(thr[j], NULL);
	end = wall_clock_time();

	fprintf(stderr, "Optimized Matrix multiplication took %1.2f seconds\n", ((float)(end - start))/1000000000);
}

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

int main() {
	// init matrixes
	float *m1 __attribute__((aligned(64))) = new float[SIZE * SIZE];
	float *m2 __attribute__((aligned(64))) = new float[SIZE * SIZE];
	float *result_unoptimized __attribute__((aligned(64))) = new float[SIZE * SIZE];
	float *result_optimized __attribute__((aligned(64))) = new float[SIZE * SIZE];
	random_matrix(m1);
	random_matrix(m2);
	zero_matrix(result_unoptimized);
	zero_matrix(result_optimized);

	// do matrix multiply
	matrix_multiply_optimized(m1, m2, result_optimized);
	matrix_multiply_unoptimized(m1, m2, result_unoptimized);

	// print result
	
	//print_matrix(result_optimized);
	//cout<<"/////////////"<<endl;
	//print_matrix(result_unoptimized);

	
	int wrong = 0;
	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			if (result_optimized[i * SIZE + j] != result_unoptimized[i * SIZE + j]) {
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
	delete result_optimized;

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
