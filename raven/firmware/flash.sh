#!/bin/bash
python3 stm32prog.py -d STM32G431CBUx -b ./motor-hub/Release/motor-hub.bin -rst 18 -bt0 17 -sbd 460800
