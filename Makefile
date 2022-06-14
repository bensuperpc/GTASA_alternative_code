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
#//  Modified: 14, June, 2022                                //
#//  file: -                                                 //
#//  -                                                       //
#//  Source:                                                 //
#//  OS: ALL                                                 //
#//  CPU: ALL                                                //
#//                                                          //
#//////////////////////////////////////////////////////////////

PARALLEL := 1

.PHONY: build
build: base

.PHONY: all
all: release debug minsizerel coverage relwithdebinfo minsizerel relwithdebinfo release-clang debug-clang base base-clang sanitize

.PHONY: base
base:
	cmake --preset=base
	cmake --build build/$@
	ctest --output-on-failure --verbose --parallel $(PARALLEL) --test-dir build/$@

.PHONY: base-clang
base-clang:
	cmake --preset=base-clang
	cmake --build build/$@
	ctest --output-on-failure --verbose --parallel $(PARALLEL) --test-dir build/$@

.PHONY: release
release:
	cmake -B build/$@ -S . -G Ninja --preset=dev -DCMAKE_BUILD_TYPE=Release
	ninja -C build/$@
	ctest --verbose --parallel $(PARALLEL) --test-dir build/$@

.PHONY: release-clang
release-clang:
	cmake -B build/$@ -S . -G Ninja --preset=dev -DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
	ninja -C build/$@
	ctest --verbose --parallel $(PARALLEL) --test-dir build/$@

.PHONY: debug
debug:
	cmake -B build/$@ -S . -G Ninja --preset=dev -DCMAKE_BUILD_TYPE=Debug
	ninja -C build/$@
	ctest --verbose --parallel $(PARALLEL) --test-dir build/$@

.PHONY: debug-clang
debug-clang:
	cmake -B build/$@ -S . -G Ninja --preset=dev -DCMAKE_BUILD_TYPE=Debug \
	-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
	ninja -C build/$@
	ctest --verbose --parallel $(PARALLEL) --test-dir build/$@

.PHONY: coverage
coverage:
	cmake -B build/$@ -S . -G Ninja --preset=dev-coverage -DCMAKE_BUILD_TYPE=Coverage
	ninja -C build/$@
	ctest --verbose --parallel $(PARALLEL) --test-dir build/$@
	ninja -C build/$@ coverage

.PHONY: sanitize
sanitize:
	cmake -B build/$@ -S . -G Ninja --preset=ci-sanitize
	ninja -C build/$@
	ctest --verbose --parallel $(PARALLEL) --test-dir build/$@

sanitize-clang:
	cmake -B build/$@ -S . -G Ninja --preset=ci-sanitize \
	-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
	ninja -C build/$@
	ctest --verbose --parallel $(PARALLEL) --test-dir build/$@

.PHONY: minsizerel
minsizerel:
	cmake -B build/$@ -S . -G Ninja --preset=dev -DCMAKE_BUILD_TYPE=MinSizeRel
	ninja -C build/$@
	ctest --verbose --parallel $(PARALLEL) --test-dir build/$@

.PHONY: relwithdebinfo
relwithdebinfo:
	cmake -B build/$@ -S . -G Ninja --preset=dev -DCMAKE_BUILD_TYPE=RelWithDebInfo
	ninja -C build/$@
	ctest --verbose --parallel $(PARALLEL) --test-dir build/$@

.PHONY: lint
lint:
	cmake -D FORMAT_COMMAND=clang-format -P cmake/lint.cmake
	cmake -P cmake/spell.cmake

.PHONY: format
format:
	time find . -regex '.*\.\(cpp\|cxx\|hpp\|hxx\|c\|h\|cu\|cuh\|cuhpp\|tpp\)' -not -path '*/build/*' | parallel clang-format -style=file -i {} \;

.PHONY: clean
clean:
	rm -rf build
