default: all

all: run

run:
	python -m influ_examples.executables.run_train

clean:
	rm -rf build influence_function.egg-info