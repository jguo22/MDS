# raven
Robot Hardware Controller

## Pcb
Color pallet works best with KiCad wDark theme

## Firmware
Built using cmake and all dependencies exist locally

Install build tools if you need to
```
sudo apt install ninja-build cmake gcc-arm-none-eabi
```

### Compiling
Compile from motor-hub folder
```
cmake . --preset=Release
cd Release
ninja
```
