default: all

all: run

run: run-cifar10
train: train-cifar10
analyze: analyze-cifar10


# for Cifar10
run-cifar10: train-cifar10 analyze-cifar10

train-cifar10:
	python -m influ_examples.executables.run_train --dataset-type=cifar10

analyze-cifar10:
	python -m influ_examples.executables.analyze_influence --dataset-type=cifar10 --result-file=result/analyzed_influence-cifar10.gz


# for MNIST
run-mnist: train-mnist analyze-mnist

train-mnist:
	python -m influ_examples.executables.run_train --dataset-type=mnist

analyze-mnist:
	python -m influ_examples.executables.analyze_influence --dataset-type=mnist --result-file=result/analyzed_influence-mnist.gz


clean:
	rm -rf build *.egg-info
