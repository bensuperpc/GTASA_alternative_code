# GTA_SA_cheat_finder

## _Find alternate cheat codes in Grand Theft Auto San Andreas by hash collision_

[![Continuous Integration](https://github.com/bensuperpc/GTA_SA_cheat_finder/actions/workflows/ci.yml/badge.svg)](https://github.com/bensuperpc/GTA_SA_cheat_finder/actions/workflows/base.yml) [![Dockcross CI](https://github.com/bensuperpc/GTA_SA_cheat_finder/actions/workflows/dockcross.yml/badge.svg)](https://github.com/bensuperpc/GTA_SA_cheat_finder/actions/workflows/dockcross.yml) [![tagged-release](https://github.com/bensuperpc/GTA_SA_cheat_finder/actions/workflows/release.yml/badge.svg)](https://github.com/bensuperpc/GTA_SA_cheat_finder/actions/workflows/release.yml) [![linux](https://github.com/bensuperpc/GTA_SA_cheat_finder/actions/workflows/linux.yml/badge.svg)](https://github.com/bensuperpc/GTA_SA_cheat_finder/actions/workflows/linux.yml)

![GitHub top language](https://img.shields.io/github/languages/top/bensuperpc/GTA_SA_cheat_finder) ![GitHub](https://img.shields.io/github/license/bensuperpc/GTA_SA_cheat_finder) ![GitHub all releases](https://img.shields.io/github/downloads/bensuperpc/GTA_SA_cheat_finder/total) ![GitHub Release Date](https://img.shields.io/github/release-date/bensuperpc/GTA_SA_cheat_finder) ![GitHub release (latest by date)](https://img.shields.io/github/v/release/bensuperpc/GTA_SA_cheat_finder) [![codecov](https://codecov.io/gh/bensuperpc/GTA_SA_cheat_finder/branch/main/graph/badge.svg?token=34WAC5P9TR)](https://codecov.io/gh/bensuperpc/GTA_SA_cheat_finder)

 [![Twitter](https://img.shields.io/twitter/follow/Bensuperpc?style=social)](https://img.shields.io/twitter/follow/Bensuperpc?style=social) [![Youtube](https://img.shields.io/youtube/channel/subscribers/UCJsQFFL7QW4LSX9eskq-9Yg?style=social)](https://img.shields.io/youtube/channel/subscribers/UCJsQFFL7QW4LSX9eskq-9Yg?style=social) 

## New Features

- Multi-plateform build: AMD64, I386, ARM64, ARMv7, ARMv7a, ARMv6, ARMv5, RISC-V 32/64, PPC64le, Mips, Windows 32/64, Android (Exec only), m68k... Thanks [dockcross](https://github.com/dockcross/dockcross) and [crosstool-ng](https://github.com/crosstool-ng/crosstool-ng)
- Add Qt GUI (Qt 6.6.0 and Qt Quick)
- Add CUDA support (tested with NVIDIA RTX 2070 and 3070)


## Usage (GUI)

```sh
./GTA_SA_cheat_finder
```

## Usage (CLI)

```sh
./GTA_SA_cheat_finder --min 0 --max 60000000 # (Searches codes from index 0 to 60000000, from A to: BYKYLXA alphabetic sequence)
```

```sh
./GTA_SA_cheat_finder --min 50 --max 58800000 # (Searches codes from index 50 to 58800000, from XA to: LFLQXD Alphabetic sequence)
```

```sh
./GTA_SA_cheat_finder -h # Get help (or with --help)
```

## Example

```sh
./GTA_SA_cheat_finder --min 0 --max 600000000
```

**You should get this result:**

```bash
Number of calculations: 600000000

From: A to: BYKYLXA Alphabetic sequence

Iter. N°         Code           JAMCRC value   
20810792         ASNAEB         0x555fc201       
75396850         FHYSTV         0x44b34866       
147491485        LJSPQK         0xfeda77f7       
181355281        OFVIAC         0x6c0fa650       
181961057        OHDUDE         0xe958788a       
198489210        PRIEBJ         0xf2aa0c1d       
241414872        THGLOJ         0xcaec94ee       
289334426        XICWMD         0x1a9aa3d6       
299376767        YECGAA         0x40cf761        
311365503        ZEIIVG         0x74d4fcb1       
370535590        AEDUWNV        0x9a629401       
380229391        AEZAKMI        0xe1b33eb9       
535721682        ASBHGRB        0xa7613f99       
Time: 1.19597 sec
This program execute: 501.684973 MOps/sec
```

## Benchmark

### CPU

On AMD R7 5800H (clang 16)

```bash
5429503678976 = ~4h
208827064576 = ~8 min
8031810176 = ~20 min
1544578880 = ~3 sec
308915776 = 0.6 sec
```
### GPU

On NVIDIA RTX 3070 (clang 16 and CUDA 11.3)

```bash
5429503678976 = ~45min
208827064576 = ~1 min
8031810176 = ~2 sec
1544578880 = ~0.3 sec
308915776 = 0.06 sec
```


### How does it work ?

- The algorithm will generate sequences of characters (A, B, C... AA, AB, AC...)
- It will then calculate hash (**JAMCRC**) from series of characters
- It will compare hashes, if they are equal to one of the hashes of one of the official cheat codes, this will save the sequence of characters
- When it is finished it displays the results and each code found can work as a alternative cheat code for GTA SA

More information on this video:

[![GTA SA - Alternative Cheats - Feat. Badger Goodger](https://yt-embed.herokuapp.com/embed?v=W_eFZ4HzU7Q)](https://youtu.be/W_eFZ4HzU7Q "GTA SA - Alternative Cheats - Feat. Badger Goodger")

## Building and installing

See the [BUILDING](BUILDING.md) document.

## Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) document.

## Open source projects used

- [Stbrumme CRC32](https://github.com/stbrumme/crc32)
- [dockcross](https://github.com/dockcross/dockcross)
- [crosstool-ng](https://github.com/crosstool-ng/crosstool-ng)
- [git](https://github.com/git/git)
- [cmake-init](https://github.com/friendlyanon/cmake-init)
- [buildroot](https://github.com/buildroot/buildroot)
- [CMake](https://github.com/Kitware/CMake)
- [llvm-project](https://github.com/llvm/llvm-project)
- [gcc](https://github.com/gcc-mirror/gcc)
- [docker](https://github.com/docker/docker)
- [actions](https://github.com/actions/virtual-environments)

## Licensing

[MIT License](LICENSE)
