#//////////////////////////////////////////////////////////////
#//                                                          //
#//  GTASA_alternative_code, 2023                               //
#//  Created: 04, June, 2021                                 //
#//  Modified: 18, November, 2023                            //
#//  file: -                                                 //
#//  -                                                       //
#//  Source:                                                 //
#//  OS: ALL                                                 //
#//  CPU: ALL                                                //
#//                                                          //
#//////////////////////////////////////////////////////////////

PROJECT_NAME ?= GTASA_alternative_code

PARALLEL ?= 1

GENERATOR ?= Ninja
PROJECT_ROOT ?= .

CTEST_TIMEOUT ?= 1500
CTEST_OPTIONS ?= --output-on-failure --timeout $(CTEST_TIMEOUT) --parallel $(PARALLEL) --verbose

#-DRUN_HAVE_STD_REGEX=1
CMAKE_ADDITONAL_ARGS ?= -DHAVE_STD_REGEX=ON

# LANG := en
# LANG=$(LANG)
# -Werror=float-equal

.PHONY: build
build: base

.PHONY: all
all: release debug minsizerel coverage relwithdebinfo minsizerel relwithdebinfo release-clang \
	debug-clang base base-clang sanitize sanitize-clang gprof $(DOCKCROSS_IMAGE) docker valgrind gdb

.PHONY: base
base:
	cmake -B build/$@ -S $(PROJECT_ROOT) -G $(GENERATOR) --preset=$@ $(CMAKE_ADDITONAL_ARGS)
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --test-dir build/$@

.PHONY: base-clang
base-clang:
	cmake -B build/$@ -S $(PROJECT_ROOT) -G $(GENERATOR) --preset=$@ $(CMAKE_ADDITONAL_ARGS)
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --test-dir build/$@

.PHONY: base-debug
base-debug:
	cmake -B build/$@ -S $(PROJECT_ROOT) -G $(GENERATOR) --preset=$@ $(CMAKE_ADDITONAL_ARGS)
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --test-dir build/$@

.PHONY: release
release:
	cmake -B build/$@ -S $(PROJECT_ROOT) -G $(GENERATOR) --preset=base -DCMAKE_BUILD_TYPE=Release $(CMAKE_ADDITONAL_ARGS)
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --test-dir build/$@

.PHONY: release-clang
release-clang:
	cmake -B build/$@ -S $(PROJECT_ROOT) -G $(GENERATOR) --preset=base -DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ $(CMAKE_ADDITONAL_ARGS)
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --test-dir build/$@

.PHONY: debug
debug:
	cmake -B build/$@ -S $(PROJECT_ROOT) -G $(GENERATOR) --preset=dev -DCMAKE_BUILD_TYPE=Debug $(CMAKE_ADDITONAL_ARGS)
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --test-dir build/$@

.PHONY: debug-clang
debug-clang:
	cmake -B build/$@ -S $(PROJECT_ROOT) -G $(GENERATOR) --preset=dev -DCMAKE_BUILD_TYPE=Debug \
	-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ $(CMAKE_ADDITONAL_ARGS)
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --test-dir build/$@

.PHONY: coverage
coverage:
	cmake -B build/$@ -S $(PROJECT_ROOT) -G $(GENERATOR) --preset=dev-coverage -DCMAKE_BUILD_TYPE=Coverage $(CMAKE_ADDITONAL_ARGS)
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --test-dir build/$@
	cmake --build build/$@ --target $@

.PHONY: sanitize
sanitize:
	cmake -B build/$@ -S $(PROJECT_ROOT) -G $(GENERATOR) --preset=ci-sanitize $(CMAKE_ADDITONAL_ARGS)
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --test-dir build/$@

sanitize-clang:
	cmake -B build/$@ -S $(PROJECT_ROOT) -G $(GENERATOR) --preset=ci-sanitize \
	-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ $(CMAKE_ADDITONAL_ARGS)
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --test-dir build/$@

.PHONY: minsizerel
minsizerel:
	cmake -B build/$@ -S $(PROJECT_ROOT) -G $(GENERATOR) --preset=dev -DCMAKE_BUILD_TYPE=MinSizeRel $(CMAKE_ADDITONAL_ARGS)
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --test-dir build/$@

.PHONY: relwithdebinfo
relwithdebinfo:
	cmake -B build/$@ -S $(PROJECT_ROOT) -G $(GENERATOR) --preset=dev -DCMAKE_BUILD_TYPE=RelWithDebInfo $(CMAKE_ADDITONAL_ARGS)
	cmake --build build/$@
	ctest $(CTEST_OPTIONS) --test-dir build/$@

.PHONY: gprof
gprof:
	cmake --preset=$@ -G $(GENERATOR) $(CMAKE_ADDITONAL_ARGS)
	cmake --build build/$@
	@echo "Run executable and after gprof <exe> gmon.out | less"

.PHONY: perf
perf:
	cmake --preset=base -G $(GENERATOR) $(CMAKE_ADDITONAL_ARGS)
	cmake --build build/base
	perf record --all-user -e branch-misses ./build/base/bin/$(PROJECT_NAME)

.PHONY: graph
graph:
	cmake -B build/$@ -S $(PROJECT_ROOT) -G $(GENERATOR) $(CMAKE_ADDITONAL_ARGS) --graphviz=build/$@/graph.dot
	cmake --build build/base
	dot -Tpng -o build/$@/graph.png build/$@/graph.dot

.PHONY: valgrind
valgrind:
	cmake -B build/$@ -S $(PROJECT_ROOT) -G $(GENERATOR) --preset=debugger $(CMAKE_ADDITONAL_ARGS)
	cmake --build build/$@
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=build/$@/valgrind.log ./build/$@/$(PROJECT_NAME)

.PHONY: gdb
gdb:
	cmake -B build/$@ -S $(PROJECT_ROOT) -G $(GENERATOR) --preset=debugger $(CMAKE_ADDITONAL_ARGS)
	cmake --build build/$@
# gdb build/$@/bin/$(PROJECT_NAME)

.PHONY: lint
lint:
	cmake -D FORMAT_COMMAND=clang-format -P cmake/lint.cmake
	cmake -P cmake/spell.cmake

.PHONY: format
format:
	find . -regex '.*\.\(cpp\|cxx\|hpp\|hxx\|c\|h\|cu\|cuh\|cuhpp\|tpp\)' -not -path '*/build/*' -not -path '*/external/*' -not -path '.git/*' | parallel clang-format -style=file -i {} \;

.PHONY: cloc
cloc:
	cloc --fullpath --not-match-d="(build|.git|external)" --not-match-f="(.git)" .

.PHONY: update
update:
# 	git submodule update --recursive --remote --force --rebase
#	git submodule update --init --recursive
	git pull --recurse-submodules --all --progress

.PHONY: clear
clear:
	rm -rf "build/*/bin/*"

.PHONY: purge
purge:
	rm -rf "build"
