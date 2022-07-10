#//////////////////////////////////////////////////////////////
#//   ____                                                   //
#//  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___  //
#//  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __| //
#//  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__  //
#//  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___| //
#//                             |_|             |_|          //
#//////////////////////////////////////////////////////////////
#//                                                          //
#//  Script, 2022                                            //
#//  Created: 19, April, 2022                                //
#//  Modified: 04, July, 2022                                //
#//  file: -                                                 //
#//  -                                                       //
#//  Source:                                                 //
#//  OS: ALL                                                 //
#//  CPU: ALL                                                //
#//                                                          //
#//////////////////////////////////////////////////////////////

PARALLEL := 1

GENERATOR := Ninja

CTEST_TIMEOUT := 1500
CTEST_OPTIONS := --output-on-failure --verbose

.PHONY: build
build: base

.PHONY: all
all: release debug minsizerel coverage relwithdebinfo minsizerel relwithdebinfo release-clang debug-clang base base-clang sanitize sanitize-clang gprof

.PHONY: base
base:
	cmake --preset=$@ -G $(GENERATOR)
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --timeout $(CTEST_TIMEOUT) --parallel $(PARALLEL) --test-dir build/$@

.PHONY: base-clang
base-clang:
	cmake --preset=$@ -G $(GENERATOR)
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --timeout $(CTEST_TIMEOUT) --parallel $(PARALLEL) --test-dir build/$@

.PHONY: release
release:
	cmake -B build/$@ -S . -G $(GENERATOR) --preset=base -DCMAKE_BUILD_TYPE=Release
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --timeout $(CTEST_TIMEOUT) --parallel $(PARALLEL) --test-dir build/$@

.PHONY: release-clang
release-clang:
	cmake -B build/$@ -S . -G $(GENERATOR) --preset=base -DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --timeout $(CTEST_TIMEOUT) --parallel $(PARALLEL) --test-dir build/$@

.PHONY: debug
debug:
	cmake -B build/$@ -S . -G $(GENERATOR) --preset=dev -DCMAKE_BUILD_TYPE=Debug
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --timeout $(CTEST_TIMEOUT) --parallel $(PARALLEL) --test-dir build/$@

.PHONY: debug-clang
debug-clang:
	cmake -B build/$@ -S . -G $(GENERATOR) --preset=dev -DCMAKE_BUILD_TYPE=Debug \
	-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --timeout $(CTEST_TIMEOUT) --parallel $(PARALLEL) --test-dir build/$@

.PHONY: graph
graph:
	cmake --preset=base -G $(GENERATOR) --graphviz=build/base/graph/graph.dot
	cmake --build build/base
	dot -Tpng -o build/base/graph/graph.png build/base/graph/graph.dot

.PHONY: coverage
coverage:
	cmake -B build/$@ -S . -G $(GENERATOR) --preset=dev-coverage -DCMAKE_BUILD_TYPE=Coverage
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --timeout $(CTEST_TIMEOUT) --parallel $(PARALLEL) --test-dir build/$@
	cmake --build build/$@ --target coverage

.PHONY: sanitize
sanitize:
	cmake -B build/$@ -S . -G $(GENERATOR) --preset=ci-sanitize
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --timeout $(CTEST_TIMEOUT) --parallel $(PARALLEL) --test-dir build/$@

sanitize-clang:
	cmake -B build/$@ -S . -G $(GENERATOR) --preset=ci-sanitize \
	-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --timeout $(CTEST_TIMEOUT) --parallel $(PARALLEL) --test-dir build/$@

.PHONY: minsizerel
minsizerel:
	cmake -B build/$@ -S . -G $(GENERATOR) --preset=dev -DCMAKE_BUILD_TYPE=MinSizeRel
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --timeout $(CTEST_TIMEOUT) --parallel $(PARALLEL) --test-dir build/$@

.PHONY: relwithdebinfo
relwithdebinfo:
	cmake -B build/$@ -S . -G $(GENERATOR) --preset=dev -DCMAKE_BUILD_TYPE=RelWithDebInfo
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --timeout $(CTEST_TIMEOUT) --parallel $(PARALLEL) --test-dir build/$@

.PHONY: gprof
gprof:
	cmake --preset=$@ -G $(GENERATOR)
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --timeout $(CTEST_TIMEOUT) --parallel $(PARALLEL) --test-dir build/$@
	echo "Run executable and after gprof <exe> gmon.out | less"

.PHONY: lint
lint:
	cmake -D FORMAT_COMMAND=clang-format -P cmake/lint.cmake
	cmake -P cmake/spell.cmake

.PHONY: format
format:
	time find . -regex '.*\.\(cpp\|cxx\|hpp\|hxx\|c\|h\|cu\|cuh\|cuhpp\|tpp\)' -not -path '*/build/*' | parallel clang-format -style=file -i {} \;

.PHONY: update
update:
	git pull --recurse-submodules --all --progress

.PHONY: clean
clean:
	rm -rf build/*
