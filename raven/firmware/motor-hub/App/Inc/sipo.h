#ifndef INC_SIPO_H_
#define INC_SIPO_H_

#include "main.h"

struct Sipo_Handle {
	// configuration
	GPIO_TypeDef* clkPort;
	uint16_t clkPin;
	GPIO_TypeDef* dataPort;
	uint16_t dataPin;
	GPIO_TypeDef* loadPort;
	uint16_t loadPin;

	// internal
	uint8_t cycle;
	uint8_t data;
};

void Sipo_Init(struct Sipo_Handle* handle);

// max 200khz speed for 170mhz clock
void Sipo_Update(struct Sipo_Handle* handle);

#endif
