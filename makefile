all: matrix

matrix: matrix.cpp
	g++ -O3 -msse3 -fstrict-aliasing matrix.cpp -o matrix -lrt
