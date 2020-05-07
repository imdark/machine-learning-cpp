
build:
	-mkdir out
	g++ main.cpp -o ./out/machine-learning-cpp
run: build
	./out/machine-learning-cpp
