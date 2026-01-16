#ifndef INC_ENCODER_H_
#define INC_ENCODER_H_

#include "main.h"
#include "stdbool.h"

struct Encoder_Handle {
	// configuration
	GPIO_TypeDef* aPort;
	uint16_t aPin;
	GPIO_TypeDef* bPort;
	uint16_t bPin;

	// internal
	bool aLast;
	bool bLast;

	int32_t pos;
};

enum Encoder_Event {
	EncoderIncrease,
	EncoderDecrease,
	EncoderNoEvent,
};

void Encoder_Init(struct Encoder_Handle* handle);

// recommended 2 * (state changes / sec)
enum Encoder_Event Encoder_Update(struct Encoder_Handle* handle);

#endif
